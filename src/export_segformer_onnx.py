"""Export a trained SegFormer DLO checkpoint to ONNX and (optionally) verify
PyTorch <-> onnxruntime parity.

Reusable across backbones: the model + num_classes are read from the checkpoint
by gen_rgb_only_sota_gifs.load_model, so the same script exports the mit-b5
teacher and the (future) mit-b0 student by only swapping --ckpt.

The exported graph has one input ('input', N x 3 x H x W, ImageNet-normalised
RGB) and one output ('logits', N x num_classes x H x W, already bilinear-
upsampled to the input HxW inside the SegFormerSegmenter wrapper).

Usage (from project root, env activated):
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python src/export_segformer_onnx.py \
        --ckpt results/.../best_model.pth \
        --out  results/onnx_smoke/teacher_mit_b5.onnx --check
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "data", "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(PROJECT_ROOT, "data", "hf_cache"))

import onnxruntime as ort  # noqa: E402

from gen_rgb_only_sota_gifs import load_model  # noqa: E402


class LogitsOnly(nn.Module):
    """Force a single, clean ONNX output: SegFormerSegmenter.forward takes an
    optional ``label`` arg; calling it label-free here returns only the
    upsampled logits tensor and keeps the exported graph to one output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help=".onnx output path")
    ap.add_argument("--backbone", default=None,
                    help="HF backbone id; default None lets load_model infer from the ckpt.")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dynamic-batch", action="store_true",
                    help="mark axis 0 of input & output as dynamic (for TensorRT batch sweeps).")
    ap.add_argument("--check", action="store_true",
                    help="after export, run onnxruntime on the same input and compare to PyTorch.")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--argmax-tol", type=float, default=1e-4,
                    help="max allowed fraction of argmax-mismatch pixels for PARITY PASS. "
                         "The CUDA EP flips a handful of near-tie boundary pixels vs PyTorch "
                         "due to FP non-determinism; raw-logit allclose at 1e-3 is informational only.")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading model from {args.ckpt} on {device} ...")
    if args.backbone is None:
        base = load_model(args.ckpt, device)
    else:
        base = load_model(args.ckpt, device, backbone=args.backbone)
    model = LogitsOnly(base).eval().to(device)

    torch.manual_seed(0)
    dummy = torch.randn(1, 3, args.height, args.width, device=device)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"Exporting to {args.out} (opset={args.opset}, dynamic_batch={args.dynamic_batch}) ...")
    # dynamo=False forces the legacy TorchScript exporter: it honours the
    # requested opset (the torch>=2.6 dynamo path silently bumps to >=18) and
    # embeds weights in a single self-contained .onnx (dynamo spills them to a
    # sidecar .onnx.data), which the downstream TensorRT step needs.
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    size_mb = os.path.getsize(args.out) / (1024 * 1024)

    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()
    out_shape = tuple(torch_out.shape)
    print(f"Exported. size={size_mb:.2f} MB  torch output shape={out_shape}")

    if not args.check:
        print("\n===== EXPORT DONE (no --check) =====")
        print(f"  out          : {args.out}")
        print(f"  opset        : {args.opset}")
        print(f"  output shape : {out_shape}")
        print(f"  size MB      : {size_mb:.2f}")
        return

    sess = ort.InferenceSession(
        args.out, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    active_provider = sess.get_providers()[0]
    ort_out = sess.run(["logits"], {"input": dummy.cpu().numpy()})[0]

    max_abs = float(np.max(np.abs(torch_out - ort_out)))
    mean_abs = float(np.mean(np.abs(torch_out - ort_out)))
    close = bool(np.allclose(torch_out, ort_out, atol=args.atol, rtol=args.rtol))

    torch_argmax = np.argmax(torch_out, axis=1)
    ort_argmax = np.argmax(ort_out, axis=1)
    argmax_mismatch = float(np.mean(torch_argmax != ort_argmax))

    # PARITY gates on the segmentation-relevant criterion (argmax map), not the
    # raw-logit allclose: a deep FP32 graph with bilinear upsample + GELU drifts
    # ~1e-2 on the CUDA EP, which fails 1e-3 allclose while the argmax map stays
    # pixel-identical bar a few near-tie boundary pixels. allclose is reported FYI.
    parity_pass = argmax_mismatch <= args.argmax_tol

    print("\n===== ONNX PARITY SUMMARY =====")
    print(f"  PARITY            : {'PASS' if parity_pass else 'FAIL'}  (gate: argmax-mismatch <= {args.argmax_tol:g})")
    print(f"  out               : {args.out}")
    print(f"  opset             : {args.opset}")
    print(f"  ORT provider      : {active_provider}")
    print(f"  output shape      : {out_shape}")
    print(f"  size MB           : {size_mb:.2f}")
    print(f"  allclose (FYI)    : {close}  (atol={args.atol}, rtol={args.rtol})")
    print(f"  max-abs-diff      : {max_abs:.6e}")
    print(f"  mean-abs-diff     : {mean_abs:.6e}")
    print(f"  argmax-mismatch   : {argmax_mismatch:.6e}  (fraction of pixels)")
    print("================================")


if __name__ == "__main__":
    main()
