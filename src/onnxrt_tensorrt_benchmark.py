#!/usr/bin/env python3
"""
ONNX Runtime with TensorRT Execution Provider Benchmark

This script achieves REAL INT8/FP16 speedup using ONNX Runtime TensorRT EP.
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


def create_calibration_cache(onnx_path, dataset, cache_dir, num_samples=100):
    """Create calibration cache for INT8 quantization"""
    os.makedirs(cache_dir, exist_ok=True)

    # Save calibration data as numpy files
    for i in range(min(num_samples, len(dataset))):
        x, _ = dataset[i]
        np.save(os.path.join(cache_dir, f'input_{i}.npy'), x.numpy())

    return cache_dir


def benchmark(fn, warmup=50, runs=100):
    """Benchmark a function and return time in ms"""
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(runs):
        fn()
    elapsed = time.perf_counter() - start

    return elapsed / runs * 1000


def main():
    parser = argparse.ArgumentParser(description='ONNX Runtime TensorRT Benchmark')
    parser.add_argument('--data-path', default='./data/set2', help='Dataset path')
    parser.add_argument('--model-path', default='./results/student_best.pth', help='Model checkpoint')
    parser.add_argument('--onnx-path', default='./results/student_dynamic.onnx', help='ONNX model path')
    parser.add_argument('--output-dir', default='./results', help='Output directory')
    args = parser.parse_args()

    print("=" * 70)
    print("ONNX RUNTIME TENSORRT BENCHMARK")
    print("=" * 70)

    print(f"\nONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")

    # Check if TensorRT EP is available
    if 'TensorrtExecutionProvider' not in ort.get_available_providers():
        print("ERROR: TensorRT Execution Provider not available!")
        return

    # Load dataset for comparison data
    print("\nLoading dataset...")
    dataset = PointWireDataset(args.data_path, split='train')
    print(f"Dataset size: {len(dataset)}")

    # Get sample input
    sample_x, _ = dataset[0]

    batch_sizes = [1, 8, 16, 32]

    print("\n" + "-" * 70)
    print("PYTORCH FP32 BASELINE")
    print("-" * 70)

    # Load PyTorch model
    model = DGCNNStudent(num_classes=5, k=20, dropout=0.5).cuda()
    model.load_state_dict(torch.load(args.model_path, map_location='cuda', weights_only=True))
    model.eval()

    pytorch_times = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 2048).cuda()

        # Warmup
        for _ in range(20):
            with torch.no_grad():
                _ = model(x)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()

        pytorch_times[bs] = (time.perf_counter() - start) / 50 * 1000
        print(f"Batch {bs}: {pytorch_times[bs]:.2f} ms")

    # Clear GPU memory
    del model
    torch.cuda.empty_cache()

    print("\n" + "-" * 70)
    print("ONNX RUNTIME CUDA EP (FP32)")
    print("-" * 70)

    # CUDA EP
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    cuda_provider = ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
    })

    cuda_session = ort.InferenceSession(
        args.onnx_path,
        sess_options=sess_opts,
        providers=[cuda_provider]
    )

    cuda_times = {}
    for bs in batch_sizes:
        x = np.random.randn(bs, 3, 2048).astype(np.float32)
        input_name = cuda_session.get_inputs()[0].name

        # Warmup
        for _ in range(20):
            _ = cuda_session.run(None, {input_name: x})

        start = time.perf_counter()
        for _ in range(50):
            _ = cuda_session.run(None, {input_name: x})
        elapsed = time.perf_counter() - start

        cuda_times[bs] = elapsed / 50 * 1000
        print(f"Batch {bs}: {cuda_times[bs]:.2f} ms (speedup: {pytorch_times[bs]/cuda_times[bs]:.2f}x)")

    del cuda_session

    print("\n" + "-" * 70)
    print("ONNX RUNTIME TENSORRT EP (FP16)")
    print("-" * 70)

    # TensorRT EP with FP16
    trt_provider_fp16 = ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': os.path.join(args.output_dir, 'trt_cache_fp16'),
    })

    os.makedirs(os.path.join(args.output_dir, 'trt_cache_fp16'), exist_ok=True)

    try:
        trt_session_fp16 = ort.InferenceSession(
            args.onnx_path,
            sess_options=sess_opts,
            providers=[trt_provider_fp16, 'CUDAExecutionProvider']
        )

        trt_fp16_times = {}
        for bs in batch_sizes:
            x = np.random.randn(bs, 3, 2048).astype(np.float32)
            input_name = trt_session_fp16.get_inputs()[0].name

            # Warmup
            for _ in range(20):
                _ = trt_session_fp16.run(None, {input_name: x})

            start = time.perf_counter()
            for _ in range(50):
                _ = trt_session_fp16.run(None, {input_name: x})
            elapsed = time.perf_counter() - start

            trt_fp16_times[bs] = elapsed / 50 * 1000
            print(f"Batch {bs}: {trt_fp16_times[bs]:.2f} ms (speedup: {pytorch_times[bs]/trt_fp16_times[bs]:.2f}x)")

        del trt_session_fp16
    except Exception as e:
        print(f"TensorRT FP16 failed: {e}")
        trt_fp16_times = {}

    print("\n" + "-" * 70)
    print("ONNX RUNTIME TENSORRT EP (INT8)")
    print("-" * 70)

    # TensorRT EP with INT8
    trt_provider_int8 = ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_int8_enable': True,
        'trt_fp16_enable': True,  # Allow FP16 fallback for unsupported ops
        'trt_int8_use_native_calibration_table': False,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': os.path.join(args.output_dir, 'trt_cache_int8'),
    })

    os.makedirs(os.path.join(args.output_dir, 'trt_cache_int8'), exist_ok=True)

    try:
        trt_session_int8 = ort.InferenceSession(
            args.onnx_path,
            sess_options=sess_opts,
            providers=[trt_provider_int8, 'CUDAExecutionProvider']
        )

        trt_int8_times = {}
        for bs in batch_sizes:
            x = np.random.randn(bs, 3, 2048).astype(np.float32)
            input_name = trt_session_int8.get_inputs()[0].name

            # Warmup
            for _ in range(20):
                _ = trt_session_int8.run(None, {input_name: x})

            start = time.perf_counter()
            for _ in range(50):
                _ = trt_session_int8.run(None, {input_name: x})
            elapsed = time.perf_counter() - start

            trt_int8_times[bs] = elapsed / 50 * 1000
            print(f"Batch {bs}: {trt_int8_times[bs]:.2f} ms (speedup: {pytorch_times[bs]/trt_int8_times[bs]:.2f}x)")

        del trt_session_int8
    except Exception as e:
        print(f"TensorRT INT8 failed: {e}")
        trt_int8_times = {}

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Batch':<8} {'PyTorch FP32':<15} {'CUDA EP':<15} {'TRT FP16':<15} {'TRT INT8':<15}")
    print("-" * 68)

    for bs in batch_sizes:
        row = f"{bs:<8} {pytorch_times[bs]:<15.2f}"

        if bs in cuda_times:
            row += f" {cuda_times[bs]:<14.2f}"
        else:
            row += f" {'N/A':<14}"

        if bs in trt_fp16_times:
            row += f" {trt_fp16_times[bs]:<14.2f}"
        else:
            row += f" {'N/A':<14}"

        if bs in trt_int8_times:
            row += f" {trt_int8_times[bs]:<14.2f}"
        else:
            row += f" {'N/A':<14}"

        print(row)

    # Speedup summary
    print("\n" + "-" * 70)
    print("SPEEDUP OVER PYTORCH FP32")
    print("-" * 70)

    print(f"\n{'Batch':<8} {'CUDA EP':<15} {'TRT FP16':<15} {'TRT INT8':<15}")
    print("-" * 53)

    for bs in batch_sizes:
        row = f"{bs:<8}"

        if bs in cuda_times:
            row += f" {pytorch_times[bs]/cuda_times[bs]:<14.2f}x"
        else:
            row += f" {'N/A':<14}"

        if bs in trt_fp16_times:
            row += f" {pytorch_times[bs]/trt_fp16_times[bs]:<14.2f}x"
        else:
            row += f" {'N/A':<14}"

        if bs in trt_int8_times:
            row += f" {pytorch_times[bs]/trt_int8_times[bs]:<14.2f}x"
        else:
            row += f" {'N/A':<14}"

        print(row)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)

    # Save results
    results_file = os.path.join(args.output_dir, 'onnxrt_tensorrt_results.txt')
    with open(results_file, 'w') as f:
        f.write("ONNX Runtime TensorRT Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"ONNX Runtime version: {ort.__version__}\n\n")

        f.write("Results (ms):\n")
        f.write(f"{'Batch':<8} {'PyTorch FP32':<15} {'TRT FP16':<15} {'TRT INT8':<15}\n")
        for bs in batch_sizes:
            row = f"{bs:<8} {pytorch_times[bs]:<15.2f}"
            if bs in trt_fp16_times:
                row += f" {trt_fp16_times[bs]:<14.2f}"
            if bs in trt_int8_times:
                row += f" {trt_int8_times[bs]:<14.2f}"
            f.write(row + "\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
