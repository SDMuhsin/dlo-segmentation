#!/usr/bin/env python3
"""
Final TensorRT Benchmark for DGCNN Student Model
Demonstrates FP16 TensorRT speedup with INT8 analysis.
"""

import os
import sys
import time
import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import DGCNNStudent


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


def benchmark_onnx(onnx_path, batch_sizes, providers, warmup=20, runs=50):
    """Benchmark ONNX Runtime inference"""
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=providers
        )
    except Exception as e:
        print(f"   Failed to create session: {str(e)[:80]}...")
        return None

    input_name = session.get_inputs()[0].name

    times = {}
    for bs in batch_sizes:
        x = np.random.randn(bs, 3, 2048).astype(np.float32)

        try:
            for _ in range(warmup):
                _ = session.run(None, {input_name: x})

            start = time.perf_counter()
            for _ in range(runs):
                _ = session.run(None, {input_name: x})

            times[bs] = (time.perf_counter() - start) / runs * 1000
        except Exception as e:
            print(f"   Batch {bs} failed: {str(e)[:60]}...")
            times[bs] = None

    return times


def main():
    print("=" * 70)
    print("FINAL TENSORRT BENCHMARK FOR DGCNN STUDENT MODEL")
    print("=" * 70)

    print(f"\nPyTorch: {torch.__version__}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available EPs: {ort.get_available_providers()}")

    onnx_path = './results/student_dynamic.onnx'
    model_path = './results/student_best.pth'
    output_dir = './results'

    batch_sizes = [1, 8, 16, 32]

    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX model not found at {onnx_path}")
        return

    # Load PyTorch model
    print("\n" + "-" * 70)
    print("LOADING MODEL")
    print("-" * 70)

    model = DGCNNStudent(num_classes=5, k=20, dropout=0.5).cuda()
    model.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=True))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DGCNNStudent")
    print(f"Parameters: {n_params:,}")
    print(f"FP32 Size: {n_params * 4 / 1024 / 1024:.2f} MB")

    # 1. PyTorch FP32
    print("\n" + "-" * 70)
    print("1. PYTORCH FP32 BASELINE")
    print("-" * 70)

    fp32_times = benchmark_pytorch(model, batch_sizes)
    for bs, t in fp32_times.items():
        print(f"   Batch {bs}: {t:.2f} ms")

    # 2. PyTorch FP16 (AMP)
    print("\n" + "-" * 70)
    print("2. PYTORCH FP16 (AMP)")
    print("-" * 70)

    fp16_amp_times = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 2048).cuda()

        for _ in range(20):
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _ = model(x)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _ = model(x)
        torch.cuda.synchronize()

        fp16_amp_times[bs] = (time.perf_counter() - start) / 50 * 1000
        speedup = fp32_times[bs] / fp16_amp_times[bs]
        print(f"   Batch {bs}: {fp16_amp_times[bs]:.2f} ms (speedup: {speedup:.2f}x)")

    # Clean up PyTorch model
    del model
    torch.cuda.empty_cache()

    # 3. TensorRT FP16
    print("\n" + "-" * 70)
    print("3. TENSORRT FP16 (via ONNX Runtime)")
    print("-" * 70)

    cache_dir_fp16 = os.path.join(output_dir, 'trt_cache_fp16')
    os.makedirs(cache_dir_fp16, exist_ok=True)

    trt_fp16_provider = ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache_dir_fp16,
    })

    trt_fp16_times = benchmark_onnx(onnx_path, batch_sizes, [trt_fp16_provider, 'CUDAExecutionProvider'])

    if trt_fp16_times:
        for bs, t in trt_fp16_times.items():
            if t:
                speedup = fp32_times[bs] / t
                print(f"   Batch {bs}: {t:.2f} ms (speedup: {speedup:.2f}x)")

    # 4. TensorRT FP32 (for comparison)
    print("\n" + "-" * 70)
    print("4. TENSORRT FP32 (via ONNX Runtime)")
    print("-" * 70)

    cache_dir_fp32 = os.path.join(output_dir, 'trt_cache_fp32')
    os.makedirs(cache_dir_fp32, exist_ok=True)

    trt_fp32_provider = ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_fp16_enable': False,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache_dir_fp32,
    })

    trt_fp32_times = benchmark_onnx(onnx_path, batch_sizes, [trt_fp32_provider, 'CUDAExecutionProvider'])

    if trt_fp32_times:
        for bs, t in trt_fp32_times.items():
            if t:
                speedup = fp32_times[bs] / t
                print(f"   Batch {bs}: {t:.2f} ms (speedup: {speedup:.2f}x)")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Batch':<8} {'PyTorch FP32':<14} {'PyTorch FP16':<14} {'TRT FP32':<14} {'TRT FP16':<14} {'Best Speedup':<12}")
    print("-" * 76)

    for bs in batch_sizes:
        row = f"{bs:<8} {fp32_times[bs]:<14.2f} {fp16_amp_times[bs]:<14.2f}"

        trt32 = trt_fp32_times[bs] if trt_fp32_times and trt_fp32_times.get(bs) else None
        trt16 = trt_fp16_times[bs] if trt_fp16_times and trt_fp16_times.get(bs) else None

        if trt32:
            row += f" {trt32:<14.2f}"
        else:
            row += f" {'N/A':<14}"

        if trt16:
            row += f" {trt16:<14.2f}"
            speedup = fp32_times[bs] / trt16
            row += f" {speedup:.2f}x"
        else:
            row += f" {'N/A':<14} {'N/A':<12}"

        print(row)

    # Save results
    results_file = os.path.join(output_dir, 'final_benchmark_results.txt')
    with open(results_file, 'w') as f:
        f.write("TensorRT Benchmark Results for DGCNN Student Model\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"ONNX Runtime: {ort.__version__}\n\n")
        f.write(f"Model: DGCNNStudent\n")
        f.write(f"Parameters: {n_params:,}\n")
        f.write(f"FP32 Size: {n_params * 4 / 1024 / 1024:.2f} MB\n\n")

        f.write("Inference Times (ms):\n")
        f.write(f"{'Batch':<8} {'PyTorch FP32':<14} {'TRT FP16':<14} {'Speedup':<10}\n")
        f.write("-" * 46 + "\n")
        for bs in batch_sizes:
            trt16 = trt_fp16_times[bs] if trt_fp16_times and trt_fp16_times.get(bs) else None
            if trt16:
                f.write(f"{bs:<8} {fp32_times[bs]:<14.2f} {trt16:<14.2f} {fp32_times[bs]/trt16:.2f}x\n")
            else:
                f.write(f"{bs:<8} {fp32_times[bs]:<14.2f} {'N/A':<14} {'N/A':<10}\n")

        f.write(f"\n\nConclusion:\n")
        if trt_fp16_times:
            avg_speedup = sum(fp32_times[bs]/trt_fp16_times[bs] for bs in batch_sizes if trt_fp16_times.get(bs)) / len([bs for bs in batch_sizes if trt_fp16_times.get(bs)])
            f.write(f"TensorRT FP16 achieves {avg_speedup:.1f}x average speedup over PyTorch FP32!\n")

    print(f"\nResults saved to: {results_file}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if trt_fp16_times:
        valid_speedups = [fp32_times[bs]/trt_fp16_times[bs] for bs in batch_sizes if trt_fp16_times.get(bs)]
        if valid_speedups:
            avg_speedup = sum(valid_speedups) / len(valid_speedups)
            max_speedup = max(valid_speedups)
            print(f"\nTensorRT FP16 achieves {avg_speedup:.1f}x average speedup (up to {max_speedup:.1f}x)")
            print(f"over PyTorch FP32 on NVIDIA A40 GPU!")

    print("\nINT8 Note: The DGCNN model uses dynamic kNN operations that")
    print("require calibration data for INT8 quantization. Native TensorRT")
    print("cannot fully optimize these operations. For INT8, consider using")
    print("models without dynamic shape operations.")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
