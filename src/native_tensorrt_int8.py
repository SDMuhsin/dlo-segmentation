#!/usr/bin/env python3
"""
Native TensorRT INT8 Engine Builder

Uses the TensorRT Python API directly for INT8 calibration and engine building.
This is the most reliable way to achieve INT8 speedup on GPU.
"""

import os
import sys
import time
import numpy as np

import torch
import tensorrt as trt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import PointWireDataset


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 Calibrator using entropy calibration"""

    def __init__(self, dataset, cache_file, batch_size=1, num_samples=100, cuda_context=None):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.cuda_context = cuda_context

        # Load calibration data
        self.data = []
        for i in range(min(num_samples, len(dataset))):
            x, _ = dataset[i]
            self.data.append(x.numpy())

        self.current_index = 0
        self.device_input = None

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.data):
            return None

        # Get batch
        batch_data = self.data[self.current_index:self.current_index + self.batch_size]
        if len(batch_data) == 0:
            return None

        batch = np.stack(batch_data, axis=0).astype(np.float32)

        import pycuda.driver as cuda

        # Push context if provided
        if self.cuda_context:
            self.cuda_context.push()

        # Allocate device memory if not done yet
        if self.device_input is None:
            self.device_input = cuda.mem_alloc(batch.nbytes)

        cuda.memcpy_htod(self.device_input, batch)

        if self.cuda_context:
            self.cuda_context.pop()

        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def build_int8_engine(onnx_path, calibrator, batch_size=1, device_id=0):
    """Build TensorRT INT8 engine from ONNX model"""

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX parsing error: {parser.get_error(error)}")
            return None, None

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 29)  # 512MB

    # Enable INT8
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    # Set input shape - calibration runs at batch_size=1, optimize for target batch
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    # Min must be >= 1, max must cover batch_size
    profile.set_shape(input_name,
                      min=(1, 3, 2048),
                      opt=(batch_size, 3, 2048),
                      max=(max(batch_size, 64), 3, 2048))
    config.add_optimization_profile(profile)

    print("Building INT8 engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build engine")
        return None, None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    # Convert IHostMemory to bytes for saving
    engine_bytes = bytes(serialized_engine)

    return engine, engine_bytes


def build_fp16_engine(onnx_path, batch_size=1, device_id=0):
    """Build TensorRT FP16 engine from ONNX model"""

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX parsing error: {parser.get_error(error)}")
            return None, None

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 29)  # 512MB

    # Enable FP16
    config.set_flag(trt.BuilderFlag.FP16)

    # Set input shape with dynamic batch support
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name,
                      min=(1, 3, 2048),
                      opt=(batch_size, 3, 2048),
                      max=(max(batch_size, 64), 3, 2048))
    config.add_optimization_profile(profile)

    print("Building FP16 engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build FP16 engine")
        return None, None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    # Convert IHostMemory to bytes for saving
    engine_bytes = bytes(serialized_engine)

    return engine, engine_bytes


def benchmark_engine(engine, batch_size, cuda_context=None, warmup=50, runs=100):
    """Benchmark TensorRT engine inference time"""
    import pycuda.driver as cuda

    if cuda_context:
        cuda_context.push()

    context = engine.create_execution_context()

    # Get input/output info
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # Set input shape for dynamic batch
    context.set_input_shape(input_name, (batch_size, 3, 2048))

    # Allocate buffers
    h_input = np.random.randn(batch_size, 3, 2048).astype(np.float32)
    h_output = np.zeros((batch_size, 5, 2048), dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Set tensor addresses
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    stream = cuda.Stream()

    # Warmup
    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

    elapsed = time.perf_counter() - start

    # Get output for verification
    cuda.memcpy_dtoh(h_output, d_output)

    if cuda_context:
        cuda_context.pop()

    return elapsed / runs * 1000  # ms


def main():
    print("=" * 70)
    print("NATIVE TENSORRT INT8 ENGINE BUILDER")
    print("=" * 70)

    print(f"\nTensorRT version: {trt.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    onnx_path = './results/student_dynamic.onnx'  # Use dynamic batch model
    data_path = './data/set2'
    output_dir = './results'
    batch_size = 32
    device_id = 0  # Use CUDA_VISIBLE_DEVICES to select the actual GPU

    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX model not found at {onnx_path}")
        return

    # Load dataset for calibration
    print("\nLoading calibration data...")
    dataset = PointWireDataset(data_path, split='train')
    print(f"Dataset size: {len(dataset)}")

    # Check if pycuda is available and initialize on device 1
    try:
        import pycuda.driver as cuda
        cuda.init()
        device = cuda.Device(device_id)
        cuda_context = device.make_context()
        print(f"PyCUDA available - using device {device_id}: {device.name()}")
    except ImportError:
        print("ERROR: PyCUDA not installed. Install with: pip install pycuda")
        return

    # Set PyTorch to use device 1
    torch.cuda.set_device(device_id)
    print(f"PyTorch using: cuda:{device_id}")

    try:
        # Build FP16 engine
        print("\n" + "-" * 70)
        print("BUILDING FP16 ENGINE")
        print("-" * 70)

        fp16_engine, fp16_serialized = build_fp16_engine(onnx_path, batch_size, device_id)

        if fp16_engine:
            fp16_engine_path = os.path.join(output_dir, 'student_fp16.trt')
            with open(fp16_engine_path, 'wb') as f:
                f.write(fp16_serialized)
            print(f"Saved FP16 engine: {fp16_engine_path}")
            print(f"Engine size: {len(fp16_serialized) / 1024 / 1024:.2f} MB")

            # Benchmark FP16
            fp16_time = benchmark_engine(fp16_engine, batch_size, cuda_context)
            print(f"FP16 inference time: {fp16_time:.3f} ms")

        # Build INT8 engine with calibration
        print("\n" + "-" * 70)
        print("BUILDING INT8 ENGINE WITH CALIBRATION")
        print("-" * 70)

        cache_file = os.path.join(output_dir, 'tensorrt_calibration.cache')
        calibrator = EntropyCalibrator(dataset, cache_file, batch_size=1, num_samples=100, cuda_context=cuda_context)

        int8_engine, int8_serialized = build_int8_engine(onnx_path, calibrator, batch_size, device_id)

        if int8_engine:
            int8_engine_path = os.path.join(output_dir, 'student_int8.trt')
            with open(int8_engine_path, 'wb') as f:
                f.write(int8_serialized)
            print(f"Saved INT8 engine: {int8_engine_path}")
            print(f"Engine size: {len(int8_serialized) / 1024 / 1024:.2f} MB")

            # Benchmark INT8
            int8_time = benchmark_engine(int8_engine, batch_size, cuda_context)
            print(f"INT8 inference time: {int8_time:.3f} ms")

            if fp16_engine:
                print(f"\nSpeedup over FP16: {fp16_time / int8_time:.2f}x")

        # Compare with PyTorch FP32
        print("\n" + "-" * 70)
        print("PYTORCH FP32 BASELINE")
        print("-" * 70)

        from src.models import DGCNNStudent

        model = DGCNNStudent(num_classes=5, k=20, dropout=0.5).cuda().eval()
        model.load_state_dict(torch.load('./results/student_best.pth', map_location='cuda', weights_only=True))

        x = torch.randn(batch_size, 3, 2048).cuda()

        # Warmup
        for _ in range(50):
            with torch.no_grad():
                _ = model(x)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()

        fp32_time = (time.perf_counter() - start) / 100 * 1000
        print(f"PyTorch FP32 time: {fp32_time:.3f} ms")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print(f"\n{'Method':<25} {'Time (ms)':<15} {'Speedup':<10}")
        print("-" * 50)
        print(f"{'PyTorch FP32':<25} {fp32_time:<15.3f} {'1.00x':<10}")

        if fp16_engine:
            print(f"{'TensorRT FP16':<25} {fp16_time:<15.3f} {fp32_time/fp16_time:.2f}x")

        if int8_engine:
            print(f"{'TensorRT INT8':<25} {int8_time:<15.3f} {fp32_time/int8_time:.2f}x")

        print("\n" + "=" * 70)

    finally:
        # Clean up CUDA context
        cuda_context.pop()
        cuda_context.detach()


if __name__ == '__main__':
    main()
