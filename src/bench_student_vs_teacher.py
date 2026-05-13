"""Benchmark single-image forward latency for teacher and student on the dev A40.

Outputs:
    params (M), checkpoint size (MB), peak forward-only GPU memory (MB),
    mean / std / min single-image latency (ms) over N timed iterations.

Usage:
    source env/bin/activate
    CUDA_VISIBLE_DEVICES=0 python src/bench_student_vs_teacher.py \
        --student-ckpt results/segformer_b0_rgb_kd/full_<tag>/best_model.pth \
        --out          results/segformer_b0_rgb_kd/full_<tag>/latency.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "data", "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(PROJECT_ROOT, "data", "hf_cache"))

from gen_rgb_only_sota_gifs import load_model
from train_rgb_only_sota import IMAGE_H, IMAGE_W


def benchmark(model, device, n_warmup=20, n_iter=100, batch_size=1, dtype=torch.float32):
    """Single-image forward latency (ms) on GPU."""
    x = torch.randn(batch_size, 3, IMAGE_H, IMAGE_W, device=device, dtype=dtype)
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = model(x)
        torch.cuda.synchronize()
        # Timed
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        times_ms = []
        for _ in range(n_iter):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))
    t = np.array(times_ms, dtype=np.float64)
    return {
        "batch_size": batch_size,
        "dtype": str(dtype),
        "n_warmup": n_warmup,
        "n_iter": n_iter,
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std()),
        "min_ms": float(t.min()),
        "max_ms": float(t.max()),
        "p50_ms": float(np.percentile(t, 50)),
        "p95_ms": float(np.percentile(t, 95)),
        "fps_mean": 1000.0 / float(t.mean()),
    }


def measure_one(name, ckpt_path, device):
    print(f"\n=== {name}: {ckpt_path} ===")
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
    print(f"  ckpt size: {ckpt_size_mb:.1f} MB")

    torch.cuda.reset_peak_memory_stats(device)
    model = load_model(ckpt_path, device)
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params: {params_M:.3f} M")

    # FP32
    fp32 = benchmark(model, device, dtype=torch.float32)
    print(f"  FP32: {fp32['mean_ms']:.2f} ± {fp32['std_ms']:.2f} ms  "
          f"(min {fp32['min_ms']:.2f}, p95 {fp32['p95_ms']:.2f})  → {fp32['fps_mean']:.1f} fps")

    # FP16
    model_half = model.half()
    fp16 = benchmark(model_half, device, dtype=torch.float16)
    print(f"  FP16: {fp16['mean_ms']:.2f} ± {fp16['std_ms']:.2f} ms  "
          f"(min {fp16['min_ms']:.2f}, p95 {fp16['p95_ms']:.2f})  → {fp16['fps_mean']:.1f} fps")

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"  peak GPU mem (during bench): {peak_mb:.0f} MB")

    return {
        "ckpt_path": ckpt_path,
        "ckpt_size_mb": float(ckpt_size_mb),
        "params_M": float(params_M),
        "peak_gpu_mb_during_bench": float(peak_mb),
        "fp32": fp32,
        "fp16": fp16,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-ckpt",
                    default=os.path.join(PROJECT_ROOT, "results", "segformer_b5_rgb",
                                          "full_20260430_2032", "best_model.pth"))
    ap.add_argument("--student-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=100)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    teacher = measure_one("Teacher", args.teacher_ckpt, device)
    student = measure_one("Student", args.student_ckpt, device)

    speedup_fp32 = teacher["fp32"]["mean_ms"] / student["fp32"]["mean_ms"]
    speedup_fp16 = teacher["fp16"]["mean_ms"] / student["fp16"]["mean_ms"]
    compression = teacher["params_M"] / student["params_M"]
    print(f"\n=== Summary ===")
    print(f"  Compression (params): {compression:.2f}×")
    print(f"  Speedup FP32: {speedup_fp32:.2f}×")
    print(f"  Speedup FP16: {speedup_fp16:.2f}×")

    out = {
        "device": str(device),
        "image_size": [IMAGE_H, IMAGE_W],
        "teacher": teacher,
        "student": student,
        "compression_ratio": float(compression),
        "speedup_fp32": float(speedup_fp32),
        "speedup_fp16": float(speedup_fp16),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
