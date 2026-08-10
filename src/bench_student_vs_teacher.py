"""Benchmark forward latency AND GPU peak memory for teacher and student on the dev A40.

Outputs (per model):
    backbone, params (M / count), checkpoint size (MB), and for each
    (precision in {fp32, fp16}) x (batch in --batch-sizes):
        mean / std / min / p50 / p90 latency (ms) over N timed iterations,
        fps, GPU peak allocated (MB), GPU peak reserved (MB).

GPU peak memory is measured per-(model, precision, batch) by calling
torch.cuda.reset_peak_memory_stats(device) immediately before the timed loop and
torch.cuda.max_memory_allocated(device) / max_memory_reserved(device) after. These
counters are PER-PROCESS, so a co-tenant's GPU usage on the same card does not
pollute them.

Usage:
    source env/bin/activate
    CUDA_VISIBLE_DEVICES=0 python src/bench_student_vs_teacher.py \
        --teacher-ckpt results/segformer_b5_.../best_model.pth \
        --student-ckpt results/segformer_b0_rgb_kd/full_<tag>/best_model.pth \
        --batch-sizes  1,4,8 \
        --out          results/optim_bench/pytorch_bench.json
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
    """Forward latency (ms) + per-process GPU peak memory (MB) on GPU.

    Peak memory is captured by resetting the CUDA peak-memory counters right
    before the timed loop and reading max_memory_allocated / max_memory_reserved
    after. These are per-process, so a co-tenant on the same physical GPU does
    not contribute to the reported peak.
    """
    x = torch.randn(batch_size, 3, IMAGE_H, IMAGE_W, device=device, dtype=dtype)
    with torch.no_grad():
        # Warmup (also primes cuDNN/autotuner so allocations are representative)
        for _ in range(n_warmup):
            _ = model(x)
        torch.cuda.synchronize()
        # Reset peak counters AFTER warmup so the measured peak reflects the
        # steady-state timed loop (per-(model, precision, batch)).
        torch.cuda.reset_peak_memory_stats(device)
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
        torch.cuda.synchronize()
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
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
        "p90_ms": float(np.percentile(t, 90)),
        "p95_ms": float(np.percentile(t, 95)),
        "fps_mean": 1000.0 / float(t.mean()),
        "gpu_peak_alloc_mb": float(peak_alloc_mb),
        "gpu_peak_reserved_mb": float(peak_reserved_mb),
    }


def _infer_backbone(ckpt_path, device):
    """Read the backbone id the way load_model does, for reporting."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = state.get("args") or {}
    cfg = state.get("config") or {}
    return cfg.get("backbone") or saved_args.get("backbone") or BACKBONE_DEFAULT


def measure_one(name, ckpt_path, device, batch_sizes, n_warmup=20, n_iter=100):
    print(f"\n=== {name}: {ckpt_path} ===")
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
    backbone = _infer_backbone(ckpt_path, device)
    print(f"  backbone : {backbone}")
    print(f"  ckpt size: {ckpt_size_mb:.1f} MB")

    model = load_model(ckpt_path, device)
    param_count = sum(p.numel() for p in model.parameters())
    params_M = param_count / 1e6
    print(f"  params   : {params_M:.3f} M ({param_count:,})")

    # fp32 / fp16 sub-dicts keyed by str(batch).
    results = {"fp32": {}, "fp16": {}}

    # FP32 sweep
    model_fp32 = model.float()
    for bs in batch_sizes:
        r = benchmark(model_fp32, device, n_warmup=n_warmup, n_iter=n_iter,
                      batch_size=bs, dtype=torch.float32)
        results["fp32"][str(bs)] = r
        print(f"  FP32 b={bs:<3d}: {r['mean_ms']:7.2f} ± {r['std_ms']:5.2f} ms "
              f"(p50 {r['p50_ms']:6.2f}, p90 {r['p90_ms']:6.2f}) "
              f"→ {r['fps_mean']:7.1f} fps | "
              f"peak alloc {r['gpu_peak_alloc_mb']:7.1f} MB / reserved {r['gpu_peak_reserved_mb']:7.1f} MB")

    # FP16 sweep (cast once)
    model_half = model.half()
    for bs in batch_sizes:
        r = benchmark(model_half, device, n_warmup=n_warmup, n_iter=n_iter,
                      batch_size=bs, dtype=torch.float16)
        results["fp16"][str(bs)] = r
        print(f"  FP16 b={bs:<3d}: {r['mean_ms']:7.2f} ± {r['std_ms']:5.2f} ms "
              f"(p50 {r['p50_ms']:6.2f}, p90 {r['p90_ms']:6.2f}) "
              f"→ {r['fps_mean']:7.1f} fps | "
              f"peak alloc {r['gpu_peak_alloc_mb']:7.1f} MB / reserved {r['gpu_peak_reserved_mb']:7.1f} MB")

    # Free this model's GPU memory before the next one so per-model peaks are clean.
    del model, model_fp32, model_half
    torch.cuda.empty_cache()

    return {
        "name": name,
        "ckpt_path": ckpt_path,
        "backbone": backbone,
        "ckpt_size_mb": float(ckpt_size_mb),
        "param_count": int(param_count),
        "params_M": float(params_M),
        "batch_sizes": list(batch_sizes),
        "fp32": results["fp32"],
        "fp16": results["fp16"],
    }


def _print_comparison_table(teacher, student, batch_sizes):
    """Teacher-vs-student comparison per (precision, batch) with ratio factors."""
    print("\n" + "=" * 118)
    print("COMPARISON TABLE  (teacher vs student; ratio = teacher/student)")
    print("=" * 118)
    print(f"  Teacher: {teacher['backbone']:<14s} params={teacher['params_M']:.2f}M  "
          f"ckpt={teacher['ckpt_size_mb']:.1f}MB")
    print(f"  Student: {student['backbone']:<14s} params={student['params_M']:.2f}M  "
          f"ckpt={student['ckpt_size_mb']:.1f}MB")
    print("-" * 118)
    hdr = (f"{'prec':<5} {'b':>3} | "
           f"{'teach ms':>9} {'stud ms':>9} {'speedup':>8} | "
           f"{'teach fps':>9} {'stud fps':>9} | "
           f"{'t.alloc MB':>11} {'s.alloc MB':>11} {'mem x':>6} | "
           f"{'t.resv MB':>10} {'s.resv MB':>10}")
    print(hdr)
    print("-" * 118)
    for prec in ("fp32", "fp16"):
        for bs in batch_sizes:
            tb = teacher[prec][str(bs)]
            sb = student[prec][str(bs)]
            speedup = tb["mean_ms"] / sb["mean_ms"] if sb["mean_ms"] else float("nan")
            mem_x = (tb["gpu_peak_alloc_mb"] / sb["gpu_peak_alloc_mb"]
                     if sb["gpu_peak_alloc_mb"] else float("nan"))
            print(f"{prec:<5} {bs:>3} | "
                  f"{tb['mean_ms']:>9.2f} {sb['mean_ms']:>9.2f} {speedup:>7.2f}x | "
                  f"{tb['fps_mean']:>9.1f} {sb['fps_mean']:>9.1f} | "
                  f"{tb['gpu_peak_alloc_mb']:>11.1f} {sb['gpu_peak_alloc_mb']:>11.1f} {mem_x:>5.2f}x | "
                  f"{tb['gpu_peak_reserved_mb']:>10.1f} {sb['gpu_peak_reserved_mb']:>10.1f}")
        print("-" * 118)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-ckpt",
                    default=os.path.join(PROJECT_ROOT, "results", "segformer_b5_rgb",
                                          "full_20260430_2032", "best_model.pth"))
    ap.add_argument("--student-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-sizes", default="1",
                    help="comma-separated batch sizes, e.g. '1,4,8'. Batch 1 is the headline.")
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=100)
    args = ap.parse_args()

    batch_sizes = [int(b) for b in str(args.batch_sizes).split(",") if b.strip()]
    if not batch_sizes:
        batch_sizes = [1]

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    teacher = measure_one("Teacher", args.teacher_ckpt, device, batch_sizes,
                          n_warmup=args.n_warmup, n_iter=args.n_iter)
    student = measure_one("Student", args.student_ckpt, device, batch_sizes,
                          n_warmup=args.n_warmup, n_iter=args.n_iter)

    _print_comparison_table(teacher, student, batch_sizes)

    # Headline = batch 1.
    hb = str(batch_sizes[0])
    speedup_fp32 = teacher["fp32"][hb]["mean_ms"] / student["fp32"][hb]["mean_ms"]
    speedup_fp16 = teacher["fp16"][hb]["mean_ms"] / student["fp16"][hb]["mean_ms"]
    compression = teacher["params_M"] / student["params_M"]
    ckpt_ratio = teacher["ckpt_size_mb"] / student["ckpt_size_mb"]
    mem_red_fp32 = (teacher["fp32"][hb]["gpu_peak_alloc_mb"]
                    / student["fp32"][hb]["gpu_peak_alloc_mb"])
    mem_red_fp16 = (teacher["fp16"][hb]["gpu_peak_alloc_mb"]
                    / student["fp16"][hb]["gpu_peak_alloc_mb"])

    print(f"\n=== HEADLINE (batch {hb}) ===")
    print(f"  Compression (params)     : {compression:.2f}x")
    print(f"  Compression (ckpt size)  : {ckpt_ratio:.2f}x")
    print(f"  Speedup FP32             : {speedup_fp32:.2f}x")
    print(f"  Speedup FP16             : {speedup_fp16:.2f}x")
    print(f"  GPU peak-alloc reduction FP32 : {mem_red_fp32:.2f}x")
    print(f"  GPU peak-alloc reduction FP16 : {mem_red_fp16:.2f}x")
    print("  NOTE: architecture-representative; final student weights pending.")

    out = {
        "device": str(device),
        "note": "architecture-representative; final student weights pending",
        "image_size": [IMAGE_H, IMAGE_W],
        "batch_sizes": batch_sizes,
        "headline_batch": batch_sizes[0],
        "teacher": teacher,
        "student": student,
        "compression_ratio_params": float(compression),
        "compression_ratio_ckpt": float(ckpt_ratio),
        "speedup_fp32": float(speedup_fp32),
        "speedup_fp16": float(speedup_fp16),
        "gpu_peak_alloc_reduction_fp32": float(mem_red_fp32),
        "gpu_peak_alloc_reduction_fp16": float(mem_red_fp16),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
