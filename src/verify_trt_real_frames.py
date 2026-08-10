"""CRITICAL compiled-model check: does the TensorRT-FP16 engine preserve the
REAL-WORLD DLO segmentation of the eager PyTorch student?

The 4-video real-world eval (data/dlo_real_sample_{1..4}.mp4) is the gate that
decides whether a teacher/student ships. TRT FP16 can shift outputs vs PyTorch,
so a passing PyTorch eval does NOT automatically carry over to the compiled
engine. This script measures the per-frame argmax disagreement between the
PyTorch student and the TRT-FP16 engine on real frames, sampled evenly across
all 4 videos.

Preprocessing is imported VERBATIM from src/infer_video_rgb_only.py
(center-crop to 4:3 -> resize to 640x480 BGR uint8) and the BGR->RGB swap +
ImageNet normalisation is replicated EXACTLY from
gen_rgb_only_sota_gifs.predict (the same transform the model was trained/served
with). The normalised RGB tensor is fed to BOTH:
    (a) the eager PyTorch model on GPU  -> torch_argmax
    (b) the ORT TensorrtExecutionProvider session (reusing the engine/cache
        built by src/trt_benchmark_segformer.py) -> trt_argmax
and we report the fraction of pixels where the two argmax maps differ, per
frame and aggregated.

Usage (from project root, env activated, TRT libs on LD_LIBRARY_PATH):
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python src/verify_trt_real_frames.py \
        --onnx results/optim_bench/student_final_dyn.onnx \
        --ckpt results/segformer_b0_rgb_kd/full_phase15/best_model.pth \
        --cache-dir results/trt_cache_final \
        --frames-per-video 40 \
        --out results/optim_bench/trt_real_frame_parity.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "data", "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(PROJECT_ROOT, "data", "hf_cache"))

import torch  # noqa: E402
import onnxruntime as ort  # noqa: E402

# Import the EXACT preprocessing + normalisation constants used at inference
# time, rather than re-implementing them, so this check cannot drift from the
# real serving pipeline.
from infer_video_rgb_only import preprocess, IMAGE_H, IMAGE_W  # noqa: E402
from gen_rgb_only_sota_gifs import load_model  # noqa: E402
from train_rgb_only_sota import RGB_MEAN, RGB_STD  # noqa: E402

DEFAULT_VIDEOS = [
    os.path.join(PROJECT_ROOT, "data", f"dlo_real_sample_{i}.mp4")
    for i in (1, 2, 3, 4)
]


def normalise_to_chw(frame_bgr):
    """Replicate gen_rgb_only_sota_gifs.predict's transform EXACTLY, returning a
    (3, H, W) float32 numpy array of ImageNet-normalised RGB (no batch dim).

    predict does:
        rgb_rgb = rgb_bgr[:, :, ::-1].copy()
        rgb = from_numpy(rgb_rgb.transpose(2,0,1)).unsqueeze(0).float() / 255.0
        rgb = (rgb - RGB_MEAN) / RGB_STD
    We mirror it on CPU tensors (mean/std are the same [1,3,1,1] constants) and
    return numpy so the identical array can be fed to both PyTorch and ORT.
    """
    rgb_rgb = frame_bgr[:, :, ::-1].copy()
    rgb = torch.from_numpy(rgb_rgb.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32) / 255.0
    rgb = (rgb - RGB_MEAN) / RGB_STD  # RGB_MEAN/STD are [1,3,1,1] cpu tensors
    return rgb.squeeze(0).numpy()  # (3, H, W) float32


def sample_frame_indices(n_frames, k):
    """k evenly-spaced unique frame indices across [0, n_frames-1]."""
    if n_frames <= 0:
        return []
    k = min(k, n_frames)
    idx = np.linspace(0, n_frames - 1, k, dtype=int)
    return sorted(set(idx.tolist()))


def build_trt_session(onnx_path, device_id, cache_dir):
    """ORT InferenceSession on the TensorRT EP (FP16), reusing the cached engine
    from src/trt_benchmark_segformer.py. Mirrors that script's provider options
    exactly so the SAME serialised engine is loaded (no rebuild)."""
    os.makedirs(cache_dir, exist_ok=True)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    trt_opts = {
        "device_id": int(device_id),
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
        "trt_fp16_enable": True,
        "trt_max_partition_iterations": 1000,
        "trt_timing_cache_enable": True,
    }
    providers = [("TensorrtExecutionProvider", trt_opts),
                 "CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    return sess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="dynamic-batch ONNX (matches the built engine).")
    ap.add_argument("--ckpt", required=True, help=".pth student checkpoint (PyTorch reference).")
    ap.add_argument("--cache-dir", required=True, help="TRT engine cache dir from the benchmark step.")
    ap.add_argument("--videos", nargs="*", default=DEFAULT_VIDEOS)
    ap.add_argument("--frames-per-video", type=int, default=40)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--flag-mean-pct", type=float, default=0.5,
                    help="loudly FLAG if mean argmax-mismatch exceeds this %% (FP16 shifting real outputs).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device_id}")

    print("=" * 90)
    print("TRT-FP16 vs PyTorch REAL-FRAME argmax-parity (compiled-model correctness on real videos)")
    print("=" * 90)
    print(f"  ONNX        : {args.onnx}")
    print(f"  ckpt        : {args.ckpt}")
    print(f"  cache dir   : {args.cache_dir}")
    print(f"  videos      : {len(args.videos)}")
    print(f"  frames/vid  : {args.frames_per_video}")
    print(f"  ORT version : {ort.__version__}")
    print(f"  avail EPs   : {ort.get_available_providers()}")

    # ---- PyTorch student (eager) ----------------------------------------- #
    model = load_model(args.ckpt, device)

    # ---- TRT-FP16 ORT session (reuse cached engine) ---------------------- #
    sess = build_trt_session(args.onnx, args.device_id, args.cache_dir)
    active_eps = sess.get_providers()
    trt_engaged = "TensorrtExecutionProvider" in active_eps
    print(f"  session EPs : {active_eps}  -> {'TRT EP ENGAGED' if trt_engaged else 'NOT TRT (FALLBACK!)'}")
    if not trt_engaged:
        print("  !!! WARNING: TRT EP did NOT engage; this check would NOT measure the compiled engine.")
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    per_frame = []  # list of dicts
    per_video = {}

    for vpath in args.videos:
        vname = os.path.basename(vpath)
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            raise RuntimeError(f"could not open {vpath}")
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sample_frame_indices(n_frames, args.frames_per_video)
        print(f"\n  {vname}: {n_frames} frames -> sampling {len(idxs)} "
              f"(idx {idxs[0]}..{idxs[-1]})")

        vid_mismatch = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                print(f"    frame {idx}: read failed; skipping")
                continue
            pre = preprocess(frame)                  # BGR uint8 640x480 (verbatim from infer)
            chw = normalise_to_chw(pre)              # (3,H,W) f32 ImageNet-norm RGB
            x_np = chw[None, ...].astype(np.float32)  # (1,3,H,W)

            # PyTorch eager argmax
            with torch.no_grad():
                xt = torch.from_numpy(x_np).to(device)
                torch_logits = model(xt).cpu().numpy()
            torch_argmax = np.argmax(torch_logits, axis=1)[0]  # (H,W)

            # TRT-FP16 argmax (same numpy input)
            trt_logits = sess.run([out_name], {in_name: x_np})[0]
            trt_argmax = np.argmax(trt_logits, axis=1)[0]      # (H,W)

            mismatch = float(np.mean(torch_argmax != trt_argmax))
            # also track DLO coverage so a flag can be read in context
            torch_cov = float(np.mean(torch_argmax == 1))
            trt_cov = float(np.mean(trt_argmax == 1))
            vid_mismatch.append(mismatch)
            per_frame.append({
                "video": vname, "frame_idx": int(idx),
                "argmax_mismatch": mismatch,
                "torch_dlo_cov": torch_cov, "trt_dlo_cov": trt_cov,
            })

        vm = np.array(vid_mismatch, dtype=np.float64)
        per_video[vname] = {
            "n_frames": int(vm.size),
            "mean_mismatch": float(vm.mean()) if vm.size else float("nan"),
            "max_mismatch": float(vm.max()) if vm.size else float("nan"),
            "frac_frames_gt_0p1pct": float(np.mean(vm > 1e-3)) if vm.size else float("nan"),
        }
        print(f"    -> mean {per_video[vname]['mean_mismatch']*100:.4f}%  "
              f"max {per_video[vname]['max_mismatch']*100:.4f}%  "
              f">0.1%-frames {per_video[vname]['frac_frames_gt_0p1pct']*100:.1f}%")
        cap.release()

    all_m = np.array([f["argmax_mismatch"] for f in per_frame], dtype=np.float64)
    overall = {
        "n_frames_total": int(all_m.size),
        "mean_mismatch": float(all_m.mean()) if all_m.size else float("nan"),
        "max_mismatch": float(all_m.max()) if all_m.size else float("nan"),
        "median_mismatch": float(np.median(all_m)) if all_m.size else float("nan"),
        "frac_frames_gt_0p1pct": float(np.mean(all_m > 1e-3)) if all_m.size else float("nan"),
        "frac_frames_gt_0p5pct": float(np.mean(all_m > 5e-3)) if all_m.size else float("nan"),
        "frac_frames_exact_zero": float(np.mean(all_m == 0.0)) if all_m.size else float("nan"),
    }

    flag = overall["mean_mismatch"] > (args.flag_mean_pct / 100.0)
    verdict = ("FLAG: TRT-FP16 SHIFTS real-world outputs (mean mismatch > "
               f"{args.flag_mean_pct}% pixels) -- investigate before relying on the compiled model."
               if flag else
               "PASS: TRT-FP16 preserves the real-world segmentation; the 4-video "
               "PyTorch eval carries over to the compiled engine.")

    print("\n" + "=" * 90)
    print("REAL-FRAME PARITY SUMMARY (TRT-FP16 vs PyTorch)")
    print("=" * 90)
    print(f"  TRT EP engaged       : {trt_engaged}")
    print(f"  frames total         : {overall['n_frames_total']}")
    print(f"  mean argmax-mismatch : {overall['mean_mismatch']*100:.4f}% pixels")
    print(f"  median               : {overall['median_mismatch']*100:.4f}% pixels")
    print(f"  max                  : {overall['max_mismatch']*100:.4f}% pixels")
    print(f"  frames >0.1% mismatch: {overall['frac_frames_gt_0p1pct']*100:.2f}%")
    print(f"  frames >0.5% mismatch: {overall['frac_frames_gt_0p5pct']*100:.2f}%")
    print(f"  frames exactly 0     : {overall['frac_frames_exact_zero']*100:.2f}%")
    print(f"  VERDICT              : {verdict}")
    print("=" * 90)

    out = {
        "onnx": args.onnx, "ckpt": args.ckpt, "cache_dir": args.cache_dir,
        "ort_version": ort.__version__, "fp16": True,
        "device_id": args.device_id, "trt_engaged": trt_engaged,
        "frames_per_video": args.frames_per_video,
        "image_size": [IMAGE_H, IMAGE_W],
        "overall": overall, "per_video": per_video, "per_frame": per_frame,
        "flag": flag, "verdict": verdict,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
