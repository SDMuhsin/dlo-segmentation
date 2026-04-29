#!/usr/bin/env python3
"""
ONNX Runtime INT8 Benchmark for DGCNN Student Model

Uses ONNX Runtime quantization with TensorRT EP for INT8 inference.
"""

import os
import sys
import time
import numpy as np

import torch
import onnx
from onnx import shape_inference
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import DGCNNStudent
from src.dataset import PointWireDataset


class PointCloudCalibrationReader(CalibrationDataReader):
    """Calibration data reader for point cloud model"""

    def __init__(self, dataset, num_samples=100):
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        self.current_idx = 0
        self.input_name = 'input'  # Will be set later

    def set_input_name(self, name):
        self.input_name = name

    def get_next(self):
        if self.current_idx >= self.num_samples:
            return None

        x, _ = self.dataset[self.current_idx]
        self.current_idx += 1

        # Return as batch of 1
        return {self.input_name: x.numpy()[np.newaxis, ...].astype(np.float32)}

    def rewind(self):
        self.current_idx = 0


def benchmark_pytorch_fp32(model, batch_sizes, warmup=20, runs=50):
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


def benchmark_onnx_trt(onnx_path, batch_sizes, cache_dir, fp16=False, int8=False, calibration_table=None, warmup=20, runs=50):
    """Benchmark ONNX Runtime with TensorRT EP"""
    os.makedirs(cache_dir, exist_ok=True)

    trt_options = {
        'device_id': 0,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache_dir,
        'trt_fp16_enable': fp16,
        'trt_int8_enable': int8,
    }

    if int8 and calibration_table and os.path.exists(calibration_table):
        trt_options['trt_int8_calibration_table_name'] = calibration_table

    trt_provider = ('TensorrtExecutionProvider', trt_options)

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=[trt_provider, 'CUDAExecutionProvider']
        )
    except Exception as e:
        print(f"   TensorRT EP failed: {str(e)[:100]}...")
        print("   Falling back to CUDA EP only")
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=['CUDAExecutionProvider']
        )

    input_name = session.get_inputs()[0].name
    providers_used = session.get_providers()

    print(f"   Active providers: {providers_used}")

    times = {}
    for bs in batch_sizes:
        x = np.random.randn(bs, 3, 2048).astype(np.float32)

        try:
            for _ in range(warmup):
                _ = session.run(None, {input_name: x})

            start = time.perf_counter()
            for _ in range(runs):
                _ = session.run(None, {input_name: x})
            elapsed = time.perf_counter() - start

            times[bs] = elapsed / runs * 1000
        except Exception as e:
            print(f"   Batch {bs} failed: {str(e)[:80]}...")
            times[bs] = None

    return times


def benchmark_onnx_cuda(onnx_path, batch_sizes, warmup=20, runs=50):
    """Benchmark ONNX Runtime with CUDA EP (no TensorRT)"""

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=['CUDAExecutionProvider']
    )

    input_name = session.get_inputs()[0].name

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


def create_int8_qdq_model(fp32_onnx_path, int8_onnx_path, dataset, num_calibration_samples=100):
    """Create INT8 quantized ONNX model with QDQ format"""
    print("\n   Creating INT8 QDQ model with calibration...")

    calibration_reader = PointCloudCalibrationReader(dataset, num_calibration_samples)

    # Get input name from the FP32 model
    sess = ort.InferenceSession(fp32_onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    calibration_reader.set_input_name(input_name)
    del sess

    print(f"   Using {num_calibration_samples} samples for calibration...")

    # Intermediate path before shape inference
    int8_temp_path = int8_onnx_path.replace('.onnx', '_temp.onnx')

    # Quantize with QDQ format (compatible with TensorRT)
    quantize_static(
        fp32_onnx_path,
        int8_temp_path,
        calibration_reader,
        quant_format=QuantFormat.QDQ,  # QDQ format for TensorRT compatibility
        per_channel=False,  # Per-tensor for better TensorRT support
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )

    print("   Running shape inference on INT8 model...")

    # Run shape inference
    model = onnx.load(int8_temp_path)
    model_with_shapes = shape_inference.infer_shapes(model)
    onnx.save(model_with_shapes, int8_onnx_path)

    # Clean up temp file
    if os.path.exists(int8_temp_path):
        os.remove(int8_temp_path)

    print(f"   INT8 model saved to: {int8_onnx_path}")

    # Get model sizes
    fp32_size = os.path.getsize(fp32_onnx_path) / 1024 / 1024
    int8_size = os.path.getsize(int8_onnx_path) / 1024 / 1024
    print(f"   FP32 model size: {fp32_size:.2f} MB")
    print(f"   INT8 model size: {int8_size:.2f} MB")
    print(f"   Size reduction: {fp32_size / int8_size:.2f}x")


def main():
    print("=" * 70)
    print("ONNX RUNTIME INT8 BENCHMARK FOR DGCNN STUDENT MODEL")
    print("=" * 70)

    print(f"\nONNX Runtime: {ort.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available EPs: {ort.get_available_providers()}")

    # Paths
    onnx_fp32_path = './results/student_dynamic.onnx'
    onnx_int8_path = './results/student_int8_qdq.onnx'
    data_path = './data/set2'
    output_dir = './results'

    batch_sizes = [1, 8, 16, 32]

    if not os.path.exists(onnx_fp32_path):
        print(f"ERROR: ONNX model not found at {onnx_fp32_path}")
        return

    # Load dataset for calibration
    print("\n" + "-" * 70)
    print("LOADING DATASET FOR CALIBRATION")
    print("-" * 70)

    dataset = PointWireDataset(data_path, split='train')
    print(f"Dataset size: {len(dataset)}")

    # Create INT8 QDQ model if not exists
    print("\n" + "-" * 70)
    print("CREATING INT8 QUANTIZED MODEL")
    print("-" * 70)

    if not os.path.exists(onnx_int8_path):
        create_int8_qdq_model(onnx_fp32_path, onnx_int8_path, dataset, num_calibration_samples=200)
    else:
        print(f"   Using existing INT8 model: {onnx_int8_path}")

    # PyTorch FP32 baseline
    print("\n" + "-" * 70)
    print("1. PYTORCH FP32 BASELINE")
    print("-" * 70)

    model = DGCNNStudent(num_classes=5, k=20, dropout=0.5).cuda()
    model.load_state_dict(torch.load('./results/student_best.pth', map_location='cuda', weights_only=True))
    model.eval()

    fp32_times = benchmark_pytorch_fp32(model, batch_sizes)
    for bs, t in fp32_times.items():
        print(f"   Batch {bs}: {t:.2f} ms")

    del model
    torch.cuda.empty_cache()

    # TensorRT FP16
    print("\n" + "-" * 70)
    print("2. TENSORRT FP16 (via ONNX Runtime)")
    print("-" * 70)

    cache_dir_fp16 = os.path.join(output_dir, 'trt_cache_fp16')
    trt_fp16_times = benchmark_onnx_trt(onnx_fp32_path, batch_sizes, cache_dir_fp16, fp16=True)

    if trt_fp16_times:
        for bs, t in trt_fp16_times.items():
            speedup = fp32_times[bs] / t
            print(f"   Batch {bs}: {t:.2f} ms (speedup: {speedup:.2f}x)")

    # TensorRT INT8 with QDQ model
    print("\n" + "-" * 70)
    print("3. TENSORRT INT8 (QDQ Model via ONNX Runtime)")
    print("-" * 70)

    cache_dir_int8 = os.path.join(output_dir, 'trt_cache_int8')
    # The QDQ model already contains quantization info, so we run it through TRT EP
    # TensorRT will automatically detect QDQ nodes and use INT8 kernels
    trt_int8_times = benchmark_onnx_trt(onnx_int8_path, batch_sizes, cache_dir_int8, fp16=False, int8=False)

    if trt_int8_times:
        for bs, t in trt_int8_times.items():
            speedup = fp32_times[bs] / t
            print(f"   Batch {bs}: {t:.2f} ms (speedup: {speedup:.2f}x)")

    # CUDA EP with INT8 QDQ model
    print("\n" + "-" * 70)
    print("4. CUDA EP WITH INT8 QDQ MODEL (Reference)")
    print("-" * 70)

    cuda_int8_times = benchmark_onnx_cuda(onnx_int8_path, batch_sizes)

    if cuda_int8_times:
        for bs, t in cuda_int8_times.items():
            speedup = fp32_times[bs] / t
            print(f"   Batch {bs}: {t:.2f} ms (speedup: {speedup:.2f}x)")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Batch':<8} {'PyTorch FP32':<15} {'TRT FP16':<15} {'TRT INT8':<15} {'INT8 Speedup':<15}")
    print("-" * 68)

    for bs in batch_sizes:
        row = f"{bs:<8} {fp32_times[bs]:<15.2f}"
        if trt_fp16_times:
            row += f" {trt_fp16_times[bs]:<15.2f}"
        else:
            row += f" {'N/A':<15}"
        if trt_int8_times:
            row += f" {trt_int8_times[bs]:<15.2f} {fp32_times[bs]/trt_int8_times[bs]:.2f}x"
        else:
            row += f" {'N/A':<15} {'N/A':<15}"
        print(row)

    # Save results
    results_file = os.path.join(output_dir, 'int8_benchmark_results.txt')
    with open(results_file, 'w') as f:
        f.write("INT8 TensorRT Benchmark Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"ONNX Runtime: {ort.__version__}\n\n")

        f.write("Results:\n")
        f.write(f"{'Batch':<8} {'FP32 (ms)':<12} {'FP16 (ms)':<12} {'INT8 (ms)':<12} {'INT8 Speedup':<15}\n")
        f.write("-" * 59 + "\n")
        for bs in batch_sizes:
            fp16_str = f"{trt_fp16_times[bs]:.2f}" if trt_fp16_times else "N/A"
            int8_str = f"{trt_int8_times[bs]:.2f}" if trt_int8_times else "N/A"
            speedup_str = f"{fp32_times[bs]/trt_int8_times[bs]:.2f}x" if trt_int8_times else "N/A"
            f.write(f"{bs:<8} {fp32_times[bs]:<12.2f} {fp16_str:<12} {int8_str:<12} {speedup_str:<15}\n")

    print(f"\nResults saved to: {results_file}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if trt_int8_times:
        avg_int8_speedup = sum(fp32_times[bs]/trt_int8_times[bs] for bs in batch_sizes) / len(batch_sizes)
        print(f"\nINT8 achieves {avg_int8_speedup:.1f}x average speedup over PyTorch FP32!")

    if trt_fp16_times:
        avg_fp16_speedup = sum(fp32_times[bs]/trt_fp16_times[bs] for bs in batch_sizes) / len(batch_sizes)
        print(f"FP16 achieves {avg_fp16_speedup:.1f}x average speedup over PyTorch FP32!")

    if trt_int8_times and trt_fp16_times:
        avg_int8_vs_fp16 = sum(trt_fp16_times[bs]/trt_int8_times[bs] for bs in batch_sizes) / len(batch_sizes)
        print(f"INT8 is {avg_int8_vs_fp16:.2f}x faster than FP16!")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
