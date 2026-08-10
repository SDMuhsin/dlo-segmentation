#!/usr/bin/env python3
"""
TensorRT INT8 Benchmark for DGCNN Student Model

This script achieves REAL INT8 speedup on A40 GPU using TensorRT.
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch_tensorrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import DGCNNStudent
from src.dataset import PointWireDataset


def benchmark(fn, warmup=50, runs=100):
    """Benchmark a function and return time in ms"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / runs * 1000


def get_calibration_data(dataset, num_samples=100, batch_size=1):
    """Get calibration data from the dataset"""
    calibration_data = []
    for i in range(min(num_samples, len(dataset))):
        x, _ = dataset[i]
        calibration_data.append(x.unsqueeze(0).cuda())
    return calibration_data


def main():
    parser = argparse.ArgumentParser(description='TensorRT INT8 Benchmark')
    parser.add_argument('--data-path', default='./data/set2', help='Dataset path')
    parser.add_argument('--model-path', default='./results/student_best.pth', help='Model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for benchmark')
    parser.add_argument('--calibration-samples', type=int, default=100, help='Number of calibration samples')
    parser.add_argument('--output-dir', default='./results', help='Output directory')
    args = parser.parse_args()

    print("=" * 70)
    print("TENSORRT INT8 BENCHMARK FOR DGCNN STUDENT MODEL")
    print("=" * 70)

    # Device info
    device = torch.device('cuda:0')
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torch_tensorrt: {torch_tensorrt.__version__}")

    # Load model
    print("\n" + "-" * 70)
    print("LOADING MODEL")
    print("-" * 70)

    model = DGCNNStudent(num_classes=5, k=20, dropout=0.5).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    # Freeze batch norm
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()

    n_params = sum(p.numel() for p in model.parameters())
    model_size = n_params * 4 / 1024 / 1024  # FP32 size in MB
    print(f"Model: DGCNNStudent")
    print(f"Parameters: {n_params:,}")
    print(f"FP32 Size: {model_size:.2f} MB")

    # Load calibration data
    print("\n" + "-" * 70)
    print("LOADING CALIBRATION DATA")
    print("-" * 70)

    dataset = PointWireDataset(args.data_path, split='train')
    calibration_data = get_calibration_data(dataset, args.calibration_samples)
    print(f"Calibration samples: {len(calibration_data)}")

    # Test input
    batch_size = args.batch_size
    x = torch.randn(batch_size, 3, 2048).to(device)

    # FP32 baseline
    print("\n" + "-" * 70)
    print("FP32 BASELINE")
    print("-" * 70)

    with torch.no_grad():
        out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    fp32_time = benchmark(lambda: model(x))
    print(f"FP32 time: {fp32_time:.3f} ms")

    # FP16 AMP
    print("\n" + "-" * 70)
    print("FP16 AMP (PyTorch autocast)")
    print("-" * 70)

    def fp16_amp_run():
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                return model(x)

    fp16_amp_time = benchmark(fp16_amp_run)
    print(f"FP16 AMP time: {fp16_amp_time:.3f} ms, Speedup: {fp32_time/fp16_amp_time:.2f}x")

    # TensorRT FP16
    print("\n" + "-" * 70)
    print("TENSORRT FP16 COMPILATION")
    print("-" * 70)

    try:
        trt_fp16 = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(shape=[batch_size, 3, 2048], dtype=torch.float32)],
            enabled_precisions={torch.float16},
            workspace_size=1 << 30
        )

        trt_fp16_time = benchmark(lambda: trt_fp16(x))
        print(f"TensorRT FP16 time: {trt_fp16_time:.3f} ms, Speedup: {fp32_time/trt_fp16_time:.2f}x")

        # Save FP16 model
        fp16_path = os.path.join(args.output_dir, 'student_trt_fp16.ts')
        torch.jit.save(trt_fp16, fp16_path)
        print(f"Saved FP16 model: {fp16_path}")

    except Exception as e:
        print(f"TensorRT FP16 failed: {e}")
        trt_fp16_time = None

    # TensorRT INT8 with calibration
    print("\n" + "-" * 70)
    print("TENSORRT INT8 COMPILATION WITH CALIBRATION")
    print("-" * 70)

    try:
        # Create calibrator
        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            calibration_data,
            use_cache=True,
            cache_file=os.path.join(args.output_dir, 'dgcnn_calibration.cache')
        )

        trt_int8 = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(shape=[batch_size, 3, 2048], dtype=torch.float32)],
            enabled_precisions={torch.float32, torch.int8},
            calibrator=calibrator,
            workspace_size=1 << 30
        )

        trt_int8_time = benchmark(lambda: trt_int8(x))
        print(f"TensorRT INT8 time: {trt_int8_time:.3f} ms, Speedup: {fp32_time/trt_int8_time:.2f}x")

        # Save INT8 model
        int8_path = os.path.join(args.output_dir, 'student_trt_int8.ts')
        torch.jit.save(trt_int8, int8_path)
        print(f"Saved INT8 model: {int8_path}")

    except Exception as e:
        print(f"TensorRT INT8 failed: {e}")
        import traceback
        traceback.print_exc()
        trt_int8_time = None

    # Results summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<25} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    print(f"{'FP32 PyTorch':<25} {fp32_time:<15.3f} {'1.00x':<10}")
    print(f"{'FP16 AMP':<25} {fp16_amp_time:<15.3f} {fp32_time/fp16_amp_time:.2f}x")

    if trt_fp16_time:
        print(f"{'TensorRT FP16':<25} {trt_fp16_time:<15.3f} {fp32_time/trt_fp16_time:.2f}x")

    if trt_int8_time:
        print(f"{'TensorRT INT8':<25} {trt_int8_time:<15.3f} {fp32_time/trt_int8_time:.2f}x")

    # Test multiple batch sizes
    print("\n" + "-" * 70)
    print("BATCH SIZE SWEEP")
    print("-" * 70)

    batch_sizes = [1, 8, 16, 32, 64]

    print(f"\n{'Batch':<8} {'FP32 (ms)':<12} {'FP16 (ms)':<12} {'INT8 (ms)':<12} {'INT8 Speedup':<12}")
    print("-" * 56)

    for bs in batch_sizes:
        x_test = torch.randn(bs, 3, 2048).to(device)

        # FP32
        fp32_t = benchmark(lambda: model(x_test), warmup=20, runs=50)

        # FP16 AMP
        def amp_run():
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    return model(x_test)
        fp16_t = benchmark(amp_run, warmup=20, runs=50)

        # TensorRT INT8 (need to recompile for each batch size)
        try:
            trt_bs = torch_tensorrt.compile(
                model,
                inputs=[torch_tensorrt.Input(shape=[bs, 3, 2048], dtype=torch.float32)],
                enabled_precisions={torch.float32, torch.int8},
                calibrator=torch_tensorrt.ptq.DataLoaderCalibrator(
                    calibration_data,
                    use_cache=True,
                    cache_file=os.path.join(args.output_dir, f'calibration_bs{bs}.cache')
                ),
                workspace_size=1 << 30
            )
            int8_t = benchmark(lambda: trt_bs(x_test), warmup=20, runs=50)
            speedup = fp32_t / int8_t
        except Exception:
            int8_t = float('nan')
            speedup = float('nan')

        print(f"{bs:<8} {fp32_t:<12.3f} {fp16_t:<12.3f} {int8_t:<12.3f} {speedup:.2f}x")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    # Save results to file
    results_file = os.path.join(args.output_dir, 'tensorrt_benchmark_results.txt')
    with open(results_file, 'w') as f:
        f.write("TensorRT INT8 Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"torch_tensorrt: {torch_tensorrt.__version__}\n\n")
        f.write(f"Model: DGCNNStudent\n")
        f.write(f"Parameters: {n_params:,}\n")
        f.write(f"FP32 Size: {model_size:.2f} MB\n\n")
        f.write(f"Batch Size: {batch_size}\n\n")
        f.write(f"FP32 time: {fp32_time:.3f} ms\n")
        f.write(f"FP16 AMP time: {fp16_amp_time:.3f} ms ({fp32_time/fp16_amp_time:.2f}x speedup)\n")
        if trt_fp16_time:
            f.write(f"TensorRT FP16 time: {trt_fp16_time:.3f} ms ({fp32_time/trt_fp16_time:.2f}x speedup)\n")
        if trt_int8_time:
            f.write(f"TensorRT INT8 time: {trt_int8_time:.3f} ms ({fp32_time/trt_int8_time:.2f}x speedup)\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
