#!/usr/bin/env python
"""WiSE-FT weight interpolation between the QN teacher (expanded to 3 classes) and a fine-tuned
3-way checkpoint:

    theta(alpha) = (1 - alpha) * theta_QN3 + alpha * theta_FT

theta_QN3 = QN 2-class ckpt with its head expanded by bg_split (wire row = DLO row, connector row
= bg row, bg & connector biases - log 2), so alpha=0 reproduces the teacher's {bg(+conn), DLO}
marginals exactly and alpha=1 is the fine-tuned model bit-for-bit.

--interp-scope restricts interpolation to a parameter subset (encoder / chosen encoder stages);
everything outside it is copied from FT verbatim. Default 'all' = original behaviour:
all floating-point tensors are interpolated (weights, biases, BN running mean/var); integer
buffers (num_batches_tracked) are taken from the FT ckpt. ``criterion.weight`` (CE class weights,
not a model weight) is taken from the FT ckpt. Output = P3W checkpoint format
(epoch/metrics/args/model_state_dict/config) with an extra ``wise`` record, loadable by every
SegFormerSegmenter-based loader (eval_real_wires_valset.py, infer_video_rgb_only.py, ...).

Usage:
  ./env/bin/python src/wise_interp_qn3way.py --ft results/qn_backbone_3way/<run>/best_model.pth \
      --alphas 0,0.2,0.4,0.5,0.6,0.8,1.0 --out-dir results/qn_backbone_3way/<run>/wise
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_qn3way_kd import HEAD_B, HEAD_W, QN_CKPT_DEFAULT, bg_split_head  # noqa: E402


def load_sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    return ck, sd


def qn3_state(qn_path):
    _, sd = load_sd(qn_path)
    sd = dict(sd)
    assert sd[HEAD_W].shape[0] == 2, "teacher ckpt must be 2-class"
    sd[HEAD_W], sd[HEAD_B] = bg_split_head(sd[HEAD_W].float(), sd[HEAD_B].float())
    return sd


ENC_PREFIX = "model.segformer.encoder."
STAGE_GROUPS = ("block", "patch_embeddings", "layer_norm")


def parse_scope(scope):
    """'all' | 'encoder' | 'encoder_stages:3,4' (1-indexed MiT stages) -> predicate(key)->bool
    telling whether a key is interpolated (else the FT value is copied verbatim)."""
    if scope == "all":
        return lambda k: True
    if scope == "encoder":
        return lambda k: k.startswith("model.segformer.")
    if scope.startswith("encoder_stages:"):
        stages = {int(x) - 1 for x in scope.split(":", 1)[1].split(",") if x.strip()}
        assert stages and all(0 <= s <= 3 for s in stages), f"bad stages in {scope!r} (1..4)"

        def pred(k):
            if not k.startswith(ENC_PREFIX):
                return False
            parts = k[len(ENC_PREFIX):].split(".")
            return parts[0] in STAGE_GROUPS and int(parts[1]) in stages
        return pred
    raise SystemExit(f"unknown --interp-scope {scope!r}")


def interpolate(sd0, sd1, alpha, in_scope=lambda k: True):
    k0, k1 = set(sd0) - {"criterion.weight"}, set(sd1) - {"criterion.weight"}
    if k0 != k1:
        raise SystemExit(f"key sets differ: only-QN {sorted(k0 - k1)[:5]} "
                         f"only-FT {sorted(k1 - k0)[:5]}")
    out = {}
    for k, v1 in sd1.items():
        if k == "criterion.weight" or not torch.is_floating_point(v1) or not in_scope(k):
            out[k] = v1.clone()
            continue
        v0 = sd0[k]
        if v0.shape != v1.shape:
            raise SystemExit(f"shape mismatch {k}: {tuple(v0.shape)} vs {tuple(v1.shape)}")
        if alpha == 1.0:
            out[k] = v1.clone()
        elif alpha == 0.0:
            out[k] = v0.to(v1.dtype).clone()
        else:
            out[k] = ((1.0 - alpha) * v0.double() + alpha * v1.double()).to(v1.dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True, help="fine-tuned 3-class ckpt (P3W format)")
    ap.add_argument("--qn", default=QN_CKPT_DEFAULT, help="hf4-converted QN 2-class ckpt")
    ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--interp-scope", default="all",
                    help="all (default, full-model WiSE) | encoder (model.segformer.* only) | "
                         "encoder_stages:3,4 (patch_embeddings/block/layer_norm of the listed 1-indexed "
                         "MiT stages). Out-of-scope params (e.g. decode head + its BN stats) = FT verbatim.")
    args = ap.parse_args()

    ft_ck, ft_sd = load_sd(args.ft)
    assert ft_sd[HEAD_W].shape[0] == 3, "FT ckpt must be 3-class"
    q_sd = qn3_state(args.qn)
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = dict(ft_ck.get("config") or {})
    cfg.setdefault("backbone", "nvidia/mit-b5")
    cfg["num_classes"] = 3
    cfg.setdefault("class_names", ["bg", "wire", "connector"])
    cfg.setdefault("image_size", [480, 640])
    in_scope = parse_scope(args.interp_scope)
    n_in = sum(1 for k, v in ft_sd.items() if k != "criterion.weight" and torch.is_floating_point(v) and in_scope(k))
    print(f"interp-scope={args.interp_scope}: {n_in} float tensors interpolated, rest = FT verbatim")
    assert n_in > 0, "scope selects no parameters"
    for a in [float(x) for x in args.alphas.split(",")]:
        sd = interpolate(q_sd, ft_sd, a, in_scope)
        ck = {"epoch": ft_ck.get("epoch"), "metrics": {}, "args": ft_ck.get("args", {}),
              "model_state_dict": sd, "config": cfg,
              "wise": {"alpha": a, "ft": os.path.abspath(args.ft),
                       "qn": os.path.abspath(args.qn), "qn_head": "bg_split",
                       "formula": "(1-alpha)*QN3 + alpha*FT",
                       "interp_scope": args.interp_scope, "n_interpolated": n_in}}
        out = os.path.join(args.out_dir, f"wise_a{a:.2f}.pth")
        torch.save(ck, out)
        print(f"alpha={a:.2f} -> {out}")


if __name__ == "__main__":
    main()
