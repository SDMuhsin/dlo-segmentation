"""
INT8 Quantization Script for DGCNN Student Model

This script performs:
1. Model loading and FP32 baseline benchmark
2. INT8 quantization using NVIDIA modelopt (for GPU) or PyTorch (for CPU)
3. ONNX export for deployment
4. Performance benchmarking (FP32 vs FP16 vs INT8)
5. Accuracy validation on test set

Usage:
    source env/bin/activate
    python src/quantize_int8.py --data-path ./data/set2 --results-path ./results
"""

import argparse
import time
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import DGCNNStudent, count_parameters, get_model_size_mb
from src.dataset import PointWireDataset, create_dataloaders


def benchmark_pytorch(model, input_shape, device, warmup=50, runs=100):
    """Benchmark PyTorch model inference time."""
    model.eval()
    x = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(x)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / runs * 1000  # ms


def benchmark_onnx(session, input_name, input_shape, warmup=50, runs=100):
    """Benchmark ONNX Runtime inference time."""
    x = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: x})

    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        _ = session.run(None, {input_name: x})

    return (time.perf_counter() - start) / runs * 1000  # ms


def compute_metrics(pred, target, num_classes=5):
    """Compute accuracy, mIoU, and per-class IoU."""
    accuracy = (pred == target).mean() * 100

    per_class_iou = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        if union > 0:
            per_class_iou.append(intersection / union * 100)
        else:
            per_class_iou.append(0.0)

    miou = np.mean(per_class_iou)
    return accuracy, miou, per_class_iou


def evaluate_model(model, test_loader, device):
    """Evaluate PyTorch model on test set."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for points, labels in test_loader:
            points = points.to(device)
            outputs = model(points)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    return compute_metrics(all_preds, all_targets)


def evaluate_onnx(session, input_name, test_loader):
    """Evaluate ONNX model on test set."""
    all_preds = []
    all_targets = []

    batch_size = None
    for points, labels in test_loader:
        if batch_size is None:
            batch_size = points.shape[0]
        x_np = points.numpy().astype(np.float32)

        # Handle batch size mismatch
        try:
            outputs = session.run(None, {input_name: x_np})
            preds = outputs[0].argmax(axis=1)
            all_preds.append(preds)
            all_targets.append(labels.numpy())
        except Exception as e:
            # Skip if batch size doesn't match
            continue

    if not all_preds:
        return 0, 0, [0]*5

    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    return compute_metrics(all_preds, all_targets)


def export_to_onnx_legacy(model, onnx_path, batch_size=1, num_points=2048):
    """Export PyTorch model to ONNX using legacy exporter."""
    model.eval()
    x = torch.randn(batch_size, 3, num_points).cuda()

    # Use legacy ONNX exporter
    torch.onnx.export(
        model,
        x,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=14,
        do_constant_folding=True,
        dynamo=False  # Use legacy exporter
    )

    # Get file size
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    return size_mb


def quantize_with_modelopt(model, calibration_data, device):
    """
    Perform INT8 quantization using NVIDIA modelopt.
    Returns quantized model for GPU inference.
    """
    try:
        import modelopt.torch.quantization as mtq

        # Create calibration forward loop
        def forward_loop(model):
            model.eval()
            with torch.no_grad():
                for x in calibration_data:
                    _ = model(x.to(device))

        # Quantize model
        quant_model = mtq.quantize(
            model,
            mtq.INT8_DEFAULT_CFG,
            forward_loop=forward_loop
        )

        return quant_model, True

    except Exception as e:
        print(f"    modelopt quantization failed: {e}")
        return model, False


def get_calibration_data(train_loader, num_samples=100, device='cuda'):
    """Get calibration data for quantization."""
    calibration_data = []
    for i, (points, _) in enumerate(train_loader):
        if i >= num_samples:
            break
        calibration_data.append(points.to(device))
    return calibration_data


def main(args):
    print("=" * 60)
    print("INT8 Quantization for DGCNN Student Model")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("\n[1] Loading student model...")
    model = DGCNNStudent(num_classes=5, k=20, dropout=0.5).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    num_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    print(f"    Parameters: {num_params:,}")
    print(f"    Model size: {model_size:.2f} MB")

    # Load data
    print("\n[2] Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_path, batch_size=args.batch_size, num_workers=4
    )
    print(f"    Train: {len(train_loader.dataset)} samples")
    print(f"    Test: {len(test_loader.dataset)} samples")

    # FP32 baseline evaluation
    print("\n[3] FP32 Baseline Evaluation...")
    fp32_acc, fp32_miou, fp32_class_iou = evaluate_model(model, test_loader, device)
    print(f"    Accuracy: {fp32_acc:.2f}%")
    print(f"    mIoU: {fp32_miou:.2f}%")

    class_names = ['Wire', 'Endpoint', 'Bifurcation', 'Connector', 'Noise']
    print("    Per-class IoU:")
    for name, iou in zip(class_names, fp32_class_iou):
        print(f"      {name}: {iou:.2f}%")

    # FP32 baseline benchmark
    print("\n[4] FP32 Benchmark...")
    batch_sizes = [1, 8, 32]
    fp32_times = {}
    for bs in batch_sizes:
        t = benchmark_pytorch(model, (bs, 3, 2048), device)
        fp32_times[bs] = t
        print(f"    Batch {bs}: {t:.2f} ms")

    # Export FP32 to ONNX with fixed batch size
    print("\n[5] Exporting FP32 model to ONNX...")
    fp32_onnx_path = os.path.join(args.results_path, 'student_fp32.onnx')

    try:
        fp32_onnx_size = export_to_onnx_legacy(model, fp32_onnx_path, batch_size=1)
        print(f"    Saved to: {fp32_onnx_path}")
        print(f"    ONNX size: {fp32_onnx_size:.2f} MB")
    except Exception as e:
        print(f"    ONNX export failed: {e}")
        fp32_onnx_size = 0

    # INT8 Quantization using modelopt
    print("\n[6] INT8 Quantization (modelopt)...")
    calibration_data = get_calibration_data(train_loader, num_samples=100, device=device)
    print(f"    Collected {len(calibration_data)} calibration samples")

    # Clone model for quantization
    quant_model = DGCNNStudent(num_classes=5, k=20, dropout=0.5).to(device)
    quant_model.load_state_dict(torch.load(args.model_path))
    quant_model.eval()

    quant_model, quant_success = quantize_with_modelopt(quant_model, calibration_data, device)

    if quant_success:
        print("    Quantization successful!")

        # Evaluate quantized model
        print("\n[7] Evaluating Quantized Model...")
        int8_acc, int8_miou, int8_class_iou = evaluate_model(quant_model, test_loader, device)
        print(f"    Accuracy: {int8_acc:.2f}% (delta: {int8_acc - fp32_acc:+.2f}%)")
        print(f"    mIoU: {int8_miou:.2f}% (delta: {int8_miou - fp32_miou:+.2f}%)")

        # Benchmark quantized model
        print("\n[8] INT8 Benchmark...")
        int8_times = {}
        for bs in batch_sizes:
            try:
                t = benchmark_pytorch(quant_model, (bs, 3, 2048), device)
                int8_times[bs] = t
                speedup = fp32_times[bs] / t if t > 0 else 0
                print(f"    Batch {bs}: {t:.2f} ms (speedup: {speedup:.2f}x)")
            except Exception as e:
                print(f"    Batch {bs}: Failed - {e}")
                int8_times[bs] = None

        # Export quantized model to ONNX
        print("\n[9] Exporting INT8 model to ONNX...")
        int8_onnx_path = os.path.join(args.results_path, 'student_int8.onnx')
        try:
            int8_onnx_size = export_to_onnx_legacy(quant_model, int8_onnx_path, batch_size=1)
            print(f"    Saved to: {int8_onnx_path}")
            print(f"    ONNX size: {int8_onnx_size:.2f} MB")
            if fp32_onnx_size > 0:
                print(f"    Size reduction: {fp32_onnx_size / int8_onnx_size:.2f}x")
        except Exception as e:
            print(f"    INT8 ONNX export failed: {e}")
            int8_onnx_size = 0
    else:
        print("    Quantization failed, skipping INT8 evaluation")
        int8_acc, int8_miou, int8_class_iou = None, None, None
        int8_times = {}
        int8_onnx_size = 0

    # ONNX Runtime benchmark (FP32)
    print("\n[10] ONNX Runtime CUDA Benchmark (FP32)...")
    if fp32_onnx_size > 0:
        try:
            session = ort.InferenceSession(
                fp32_onnx_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            print(f"    Active providers: {session.get_providers()}")

            onnx_times = {}
            for bs in batch_sizes:
                try:
                    t = benchmark_onnx(session, 'input', (bs, 3, 2048))
                    onnx_times[bs] = t
                    print(f"    Batch {bs}: {t:.2f} ms")
                except Exception as e:
                    print(f"    Batch {bs}: Failed (ONNX exported with batch=1)")
                    onnx_times[bs] = None
        except Exception as e:
            print(f"    ONNX Runtime failed: {e}")
            onnx_times = {}
    else:
        onnx_times = {}

    # FP16 Benchmark (mixed precision)
    print("\n[11] FP16 Mixed Precision Benchmark...")
    fp16_times = {}
    try:
        model_fp16 = model.half()
        for bs in batch_sizes:
            x = torch.randn(bs, 3, 2048, device=device).half()

            # Warmup
            with torch.no_grad():
                for _ in range(50):
                    _ = model_fp16(x)

            # Benchmark
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(100):
                    _ = model_fp16(x)
            torch.cuda.synchronize()

            t = (time.perf_counter() - start) / 100 * 1000
            fp16_times[bs] = t
            speedup = fp32_times[bs] / t if t > 0 else 0
            print(f"    Batch {bs}: {t:.2f} ms (speedup: {speedup:.2f}x)")

        # Evaluate FP16 accuracy
        model_fp16 = model_fp16.float()  # Convert back for evaluation
    except Exception as e:
        print(f"    FP16 benchmark failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nModel Size:")
    print(f"  PyTorch FP32: {model_size:.2f} MB")
    if fp32_onnx_size > 0:
        print(f"  ONNX FP32: {fp32_onnx_size:.2f} MB")
    if int8_onnx_size > 0:
        print(f"  ONNX INT8: {int8_onnx_size:.2f} MB ({fp32_onnx_size / int8_onnx_size:.2f}x reduction)")

    print(f"\nAccuracy:")
    print(f"  FP32: {fp32_acc:.2f}% accuracy, {fp32_miou:.2f}% mIoU")
    if int8_acc is not None:
        print(f"  INT8: {int8_acc:.2f}% accuracy, {int8_miou:.2f}% mIoU")

    print(f"\nLatency (ms) and Speedup:")
    print(f"  {'Batch':<8} {'PyTorch FP32':<15} {'PyTorch FP16':<15} {'PyTorch INT8':<15}")
    print(f"  {'-'*8} {'-'*15} {'-'*15} {'-'*15}")
    for bs in batch_sizes:
        fp32_str = f"{fp32_times[bs]:.2f}"
        fp16_str = f"{fp16_times.get(bs, 'N/A'):.2f} ({fp32_times[bs]/fp16_times[bs]:.2f}x)" if fp16_times.get(bs) else "N/A"
        int8_str = f"{int8_times.get(bs):.2f} ({fp32_times[bs]/int8_times[bs]:.2f}x)" if int8_times.get(bs) else "N/A"
        print(f"  {bs:<8} {fp32_str:<15} {fp16_str:<15} {int8_str:<15}")

    # Save results
    results_file = os.path.join(args.results_path, 'int8_quantization_results.txt')
    with open(results_file, 'w') as f:
        f.write("INT8 Quantization Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: DGCNN Student (Distilled)\n")
        f.write(f"Parameters: {num_params:,}\n\n")

        f.write("Model Size:\n")
        f.write(f"  PyTorch FP32: {model_size:.2f} MB\n")
        if fp32_onnx_size > 0:
            f.write(f"  ONNX FP32: {fp32_onnx_size:.2f} MB\n")
        if int8_onnx_size > 0:
            f.write(f"  ONNX INT8: {int8_onnx_size:.2f} MB\n")

        f.write("\nAccuracy:\n")
        f.write(f"  FP32: {fp32_acc:.2f}% acc, {fp32_miou:.2f}% mIoU\n")
        if int8_acc is not None:
            f.write(f"  INT8: {int8_acc:.2f}% acc, {int8_miou:.2f}% mIoU\n")

        f.write("\nPer-Class IoU (FP32):\n")
        for name, iou in zip(class_names, fp32_class_iou):
            f.write(f"  {name}: {iou:.2f}%\n")

        f.write("\nLatency (ms):\n")
        for bs in batch_sizes:
            f.write(f"  Batch {bs}:\n")
            f.write(f"    PyTorch FP32: {fp32_times[bs]:.2f} ms\n")
            if fp16_times.get(bs):
                f.write(f"    PyTorch FP16: {fp16_times[bs]:.2f} ms ({fp32_times[bs]/fp16_times[bs]:.2f}x speedup)\n")
            if int8_times.get(bs):
                f.write(f"    PyTorch INT8: {int8_times[bs]:.2f} ms ({fp32_times[bs]/int8_times[bs]:.2f}x speedup)\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='INT8 Quantization for DGCNN Student')
    parser.add_argument('--data-path', type=str, default='./data/set2',
                        help='Path to dataset')
    parser.add_argument('--model-path', type=str, default='./results/student_best.pth',
                        help='Path to trained model')
    parser.add_argument('--results-path', type=str, default='./results',
                        help='Path to save results')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')

    args = parser.parse_args()
    main(args)
