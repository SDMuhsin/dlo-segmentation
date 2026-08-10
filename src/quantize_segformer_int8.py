"""Static INT8 (QDQ) quantization for the RGB-only SegFormer DLO student.

Produces a TensorRT-compatible QDQ INT8 ONNX graph from an FP32 SegFormer ONNX
export (src/export_segformer_onnx.py). Activations are calibrated on REAL frames
from the Phase-15 wire-free val cache, normalised EXACTLY as the model's
inference preprocessing does -- getting that wrong silently wrecks calibration.

NORMALIZATION (must match the eager model's preprocessing):
  The cache `val_rgb.npy` is uint8 stored in **BGR** channel order (it is the raw
  `cv2.imread(..., IMREAD_COLOR)` output written verbatim by
  `src/train_rgbd_seg.py:build_cache` -> `rgb_arr[i] = bgr`). The SegFormer
  wrapper, however, is fed **RGB**: `src/gen_rgb_only_sota_gifs.py:predict`
  (and `src/train_rgb_only_sota.py:__getitem__`) do
      rgb_rgb = bgr[:, :, ::-1]            # BGR -> RGB
      x = rgb_rgb.transpose(2,0,1) / 255.0
      x = (x - IMAGENET_MEAN) / IMAGENET_STD   # mean=[.485,.456,.406] std=[.229,.224,.225]
  So calibration here does: BGR->RGB swap, /255, ImageNet-normalise (RGB-order
  mean/std), HWC->CHW, float32, batch-1. (Confirmed: gen_rgb_only_sota_gifs.py:96-98,
  train_rgb_only_sota.py:147-159, train_rgbd_seg.py:106-113.)

This mirrors the quantize_static QDQ pattern in src/onnxrt_int8_benchmark.py /
src/quantize_int8.py (point-cloud DGCNN) but adapts the IO to the (N,3,H,W) image
model and the image calibration reader. Per-tensor, QInt8 weights+activations,
QDQ format -> TensorRT detects the QDQ nodes and runs INT8 kernels.

Usage (from project root):
    mkdir -p results/optim_bench
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python src/quantize_segformer_int8.py \
        --onnx results/optim_bench/student_mit_b0_dyn.onnx \
        --out  results/optim_bench/student_mit_b0_int8.onnx
"""

import argparse
import os

import numpy as np
import onnx
from onnx import shape_inference
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ImageNet stats in RGB channel order -- identical to the constants in
# src/train_rgb_only_sota.py:89-90 / src/train_rgbd_seg.py:52-53.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

DEFAULT_CALIB_NPY = os.path.join(
    PROJECT_ROOT, "data", "dformer_dataset_phase15_wirefree", "cache", "val_rgb.npy"
)


def preprocess_bgr_to_model_input(bgr_uint8):
    """Replicate the eager model's RGB preprocessing for ONE HWC uint8 BGR frame.

    bgr_uint8: (H, W, 3) uint8, BGR channel order (as stored in val_rgb.npy).
    returns:   (1, 3, H, W) float32, RGB ImageNet-normalised (NCHW, batch 1).
    """
    rgb = bgr_uint8[:, :, ::-1]                       # BGR -> RGB (model expects RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32)   # HWC -> CHW
    chw = chw / 255.0                                 # [0,1]
    chw = (chw - IMAGENET_MEAN) / IMAGENET_STD        # ImageNet normalise (RGB order)
    return chw[np.newaxis, ...].astype(np.float32)    # (1,3,H,W)


class SegFormerImageCalibrationReader(CalibrationDataReader):
    """Yields {input_name: (1,3,H,W) float32} over a slice of the val cache.

    Reads the cache with mmap so the 950 MB array is not fully loaded; only the
    calibration slice is materialised. Channel order / normalisation exactly
    matches the eager model's inference preprocessing (see module docstring)."""

    def __init__(self, calib_npy, input_name, start, count, height, width):
        self.input_name = input_name
        self.height = height
        self.width = width
        arr = np.load(calib_npy, mmap_mode="r")
        n = arr.shape[0]
        start = max(0, min(int(start), n))
        stop = min(start + int(count), n)
        if stop <= start:
            raise ValueError(
                f"empty calibration slice: start={start} count={count} "
                f"but cache has {n} images")
        # Materialise just the slice (copy out of the mmap).
        self.frames = np.asarray(arr[start:stop]).copy()
        if self.frames.shape[1:3] != (height, width):
            raise ValueError(
                f"cache frame size {self.frames.shape[1:3]} != requested "
                f"({height},{width}); re-export ONNX or pass matching --height/--width")
        self.n = self.frames.shape[0]
        self.start = start
        self.stop = stop
        self._iter = None

    def _gen(self):
        for i in range(self.n):
            yield {self.input_name: preprocess_bgr_to_model_input(self.frames[i])}

    def get_next(self):
        if self._iter is None:
            self._iter = self._gen()
        return next(self._iter, None)

    def rewind(self):
        self._iter = None


def main():
    ap = argparse.ArgumentParser(
        description="Static INT8 (QDQ) quantization for the SegFormer DLO student.")
    ap.add_argument("--onnx", required=True, help="FP32 input ONNX (SegFormer export).")
    ap.add_argument("--out", required=True, help="output INT8 QDQ ONNX path.")
    ap.add_argument("--calib-npy", default=DEFAULT_CALIB_NPY,
                    help="uint8 BGR cache (N,H,W,3) for activation calibration.")
    ap.add_argument("--calib-start", type=int, default=100,
                    help="first cache index used for calibration.")
    ap.add_argument("--calib-count", type=int, default=100,
                    help="number of calibration frames (batch 1 each).")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--calib-method", default="MinMax", choices=["MinMax", "Entropy"],
                    help="activation calibration method (MinMax is robust for seg).")
    ap.add_argument("--per-channel", action="store_true",
                    help="per-channel weight quant (default per-tensor for TRT).")
    args = ap.parse_args()

    print("=" * 90)
    print("SEGFORMER STATIC INT8 (QDQ) QUANTIZATION")
    print("=" * 90)
    print(f"  fp32 onnx     : {args.onnx}")
    print(f"  out int8 onnx : {args.out}")
    print(f"  calib npy     : {args.calib_npy}")
    print(f"  calib slice   : [{args.calib_start}:{args.calib_start + args.calib_count}] "
          f"({args.calib_count} frames, batch 1)")
    print(f"  input size    : (1, 3, {args.height}, {args.width})")
    print(f"  calib method  : {args.calib_method}  per_channel={args.per_channel}")
    print(f"  ort version   : {ort.__version__}  onnx {onnx.__version__}")

    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(f"FP32 ONNX not found: {args.onnx}")
    if not os.path.isfile(args.calib_npy):
        raise FileNotFoundError(f"calibration npy not found: {args.calib_npy}")

    # Resolve the real input tensor name from the FP32 graph (don't hard-code).
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    in_shape = sess.get_inputs()[0].shape
    del sess
    print(f"  graph input   : '{input_name}' shape={in_shape}")

    reader = SegFormerImageCalibrationReader(
        args.calib_npy, input_name, args.calib_start, args.calib_count,
        args.height, args.width)
    print(f"  calib frames  : {reader.n} loaded from cache rows "
          f"[{reader.start}:{reader.stop}]")

    # Sanity-print the normalised stats of the first calib frame so a channel /
    # /255 mistake is visible in the log (RGB-normalised values are ~[-2.1, 2.6]).
    reader.rewind()
    first = reader.get_next()[input_name]
    print(f"  norm check    : first frame stats min={first.min():.3f} "
          f"max={first.max():.3f} mean={first.mean():.3f} "
          f"per-ch mean(RGB)={[round(float(first[0, c].mean()), 3) for c in range(3)]}")
    reader.rewind()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp_out = args.out.replace(".onnx", "_pre_shapeinfer.onnx")

    calib_method = (CalibrationMethod.MinMax if args.calib_method == "MinMax"
                    else CalibrationMethod.Entropy)

    print("\n  Running quantize_static (QDQ, QInt8 weights+activations) ...")
    quantize_static(
        args.onnx,
        tmp_out,
        reader,
        quant_format=QuantFormat.QDQ,        # QDQ -> TensorRT INT8-aware
        per_channel=args.per_channel,        # per-tensor by default (TRT-friendly)
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=calib_method,
    )

    # Shape inference so the QDQ graph carries shapes (helps TRT EP partition).
    print("  Running shape inference on the INT8 graph ...")
    model = onnx.load(tmp_out)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # non-fatal; TRT can still infer
        print(f"    [shape inference warning: {type(exc).__name__}: {exc}]")
    onnx.save(model, args.out)
    if os.path.exists(tmp_out):
        os.remove(tmp_out)

    fp32_mb = os.path.getsize(args.onnx) / (1024 * 1024)
    int8_mb = os.path.getsize(args.out) / (1024 * 1024)
    # Count QuantizeLinear nodes as a proof the graph really is QDQ-quantised.
    qdq_model = onnx.load(args.out)
    n_qlin = sum(1 for n in qdq_model.graph.node if n.op_type == "QuantizeLinear")
    n_dqlin = sum(1 for n in qdq_model.graph.node if n.op_type == "DequantizeLinear")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"  FP32 onnx size : {fp32_mb:.2f} MB")
    print(f"  INT8 onnx size : {int8_mb:.2f} MB  ({fp32_mb / max(int8_mb, 1e-9):.2f}x smaller)")
    print(f"  QDQ nodes      : {n_qlin} QuantizeLinear / {n_dqlin} DequantizeLinear")
    print(f"  saved          : {args.out}")


if __name__ == "__main__":
    main()
