#!/usr/bin/env python3
"""
Final TensorRT Benchmark for DGCNN Student Model

Demonstrates REAL GPU speedup using ONNX Runtime TensorRT Execution Provider.
"""

import os
import sys
import time
import argparse
import numpy as np

import torch
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import DGCNNStudent
from src.dataset import PointWireDataset


def benchmark_pytorch(model, batch_sizes, warmup=20, runs=50):
    """Benchmark PyTorch FP32 inference"""
    times = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 2048).cuda()

        for _ in range(warmup):
            with torch.no_grad():
                _ = model(x)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()

        times[bs] = (time.perf_counter() - start) / runs * 1000

    return times


def benchmark_pytorch_fp16(model, batch_sizes, warmup=20, runs=50):
    """Benchmark PyTorch FP16 (AMP) inference"""
    times = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 2048).cuda()

        for _ in range(warmup):
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _ = model(x)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _ = model(x)
        torch.cuda.synchronize()

        times[bs] = (time.perf_counter() - start) / runs * 1000

    return times


def benchmark_onnx_tensorrt(onnx_path, batch_sizes, cache_dir, fp16=True, warmup=20, runs=50):
    """Benchmark ONNX Runtime with TensorRT EP"""
    os.makedirs(cache_dir, exist_ok=True)

    trt_provider = ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_fp16_enable': fp16,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache_dir,
    })

    sess_opts = ort.SessionOptions()

    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=[trt_provider]
    )

    input_name = session.get_inputs()[0].name
    providers_used = session.get_providers()

    if 'TensorrtExecutionProvider' not in providers_used:
        print("WARNING: TensorRT EP not active, falling back to CPU!")
        return None

    times = {}
    for bs in batch_sizes:
        x = np.random.randn(bs, 3, 2048).astype(np.float32)

        for _ in range(warmup):
            _ = session.run(None, {input_name: x})

        start = time.perf_counter()
        for _ in range(runs):
            _ = session.run(None, {input_name: x})
        elapsed = time.perf_counter() - start

        times[bs] = elapsed / runs * 1000

    return times


def main():
    parser = argparse.ArgumentParser(description='TensorRT Benchmark')
    parser.add_argument('--data-path', default='./data/set2', help='Dataset path')
    parser.add_argument('--model-path', default='./results/student_best.pth', help='Model checkpoint')
    parser.add_argument('--onnx-path', default='./results/student_dynamic.onnx', help='ONNX model path')
    parser.add_argument('--output-dir', default='./results', help='Output directory')
    args = parser.parse_args()

    print("=" * 70)
    print("TENSORRT FP16 BENCHMARK FOR DGCNN STUDENT MODEL")
    print("=" * 70)

    print(f"\nONNX Runtime: {ort.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available EPs: {ort.get_available_providers()}")

    batch_sizes = [1, 4, 8, 16]

    # Load model
    print("\n" + "-" * 70)
    print("LOADING MODEL")
    print("-" * 70)

    model = DGCNNStudent(num_classes=5, k=20, dropout=0.5).cuda()
    model.load_state_dict(torch.load(args.model_path, map_location='cuda', weights_only=True))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DGCNNStudent")
    print(f"Parameters: {n_params:,}")
    print(f"FP32 Size: {n_params * 4 / 1024 / 1024:.2f} MB")

    # PyTorch FP32
    print("\n" + "-" * 70)
    print("1. PYTORCH FP32 BASELINE")
    print("-" * 70)

    fp32_times = benchmark_pytorch(model, batch_sizes)
    for bs, t in fp32_times.items():
        print(f"   Batch {bs}: {t:.2f} ms")

    # PyTorch FP16 (AMP)
    print("\n" + "-" * 70)
    print("2. PYTORCH FP16 (AMP)")
    print("-" * 70)

    fp16_amp_times = benchmark_pytorch_fp16(model, batch_sizes)
    for bs, t in fp16_amp_times.items():
        speedup = fp32_times[bs] / t
        print(f"   Batch {bs}: {t:.2f} ms (speedup: {speedup:.2f}x)")

    # Clear GPU memory
    del model
    torch.cuda.empty_cache()

    # TensorRT FP16
    print("\n" + "-" * 70)
    print("3. TENSORRT FP16 (via ONNX Runtime)")
    print("-" * 70)

    cache_dir = os.path.join(args.output_dir, 'trt_cache_fp16')
    trt_fp16_times = benchmark_onnx_tensorrt(args.onnx_path, batch_sizes, cache_dir, fp16=True)

    if trt_fp16_times:
        for bs, t in trt_fp16_times.items():
            speedup = fp32_times[bs] / t
            print(f"   Batch {bs}: {t:.2f} ms (speedup: {speedup:.2f}x)")
    else:
        print("   TensorRT FP16 not available")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Batch':<8} {'FP32 (ms)':<12} {'FP16 AMP':<12} {'TRT FP16':<12} {'TRT Speedup':<12}")
    print("-" * 56)

    for bs in batch_sizes:
        row = f"{bs:<8} {fp32_times[bs]:<12.2f} {fp16_amp_times[bs]:<12.2f}"
        if trt_fp16_times and bs in trt_fp16_times:
            row += f" {trt_fp16_times[bs]:<12.2f} {fp32_times[bs]/trt_fp16_times[bs]:.2f}x"
        else:
            row += f" {'N/A':<12} {'N/A':<12}"
        print(row)

    # Save results
    results_file = os.path.join(args.output_dir, 'tensorrt_fp16_benchmark_results.txt')
    with open(results_file, 'w') as f:
        f.write("TensorRT FP16 Benchmark Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"ONNX Runtime: {ort.__version__}\n\n")
        f.write(f"Model: DGCNNStudent\n")
        f.write(f"Parameters: {n_params:,}\n")
        f.write(f"FP32 Size: {n_params * 4 / 1024 / 1024:.2f} MB\n\n")

        f.write("Results:\n")
        f.write(f"{'Batch':<8} {'FP32 (ms)':<12} {'TRT FP16 (ms)':<15} {'Speedup':<10}\n")
        f.write("-" * 45 + "\n")
        for bs in batch_sizes:
            if trt_fp16_times and bs in trt_fp16_times:
                f.write(f"{bs:<8} {fp32_times[bs]:<12.2f} {trt_fp16_times[bs]:<15.2f} {fp32_times[bs]/trt_fp16_times[bs]:.2f}x\n")
            else:
                f.write(f"{bs:<8} {fp32_times[bs]:<12.2f} {'N/A':<15} {'N/A':<10}\n")

        f.write(f"\n\nConclusion: TensorRT FP16 achieves {fp32_times[8]/trt_fp16_times[8]:.1f}x speedup over PyTorch FP32!\n")

    print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if trt_fp16_times:
        avg_speedup = sum(fp32_times[bs]/trt_fp16_times[bs] for bs in batch_sizes) / len(batch_sizes)
        print(f"\nTensorRT FP16 achieves {avg_speedup:.1f}x average speedup over PyTorch FP32!")
        print("\nTo use the optimized model:")
        print("  1. Export ONNX with dynamic batch: results/student_dynamic.onnx")
        print("  2. Run with ONNX Runtime TensorRT EP (see this script for example)")
        print(f"  3. TensorRT engine cached in: {cache_dir}")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
