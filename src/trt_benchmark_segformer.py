"""SegFormer-specific TensorRT-via-ONNXRuntime benchmark.

Benchmarks a SegFormer DLO ONNX graph through ONNX Runtime's
``TensorrtExecutionProvider`` (TRT EP). Reports per-batch latency / fps, GPU peak
memory (sampled with nvidia-smi because ORT/TRT allocate OUTSIDE torch's caching
allocator), and -- if a .pth checkpoint is supplied -- the argmax-parity fraction
of the COMPILED TRT engine vs the eager PyTorch model (the critical
compiled-model correctness check).

This is distinct from the legacy point-cloud TRT scripts
(src/onnxrt_tensorrt_benchmark.py / onnxrt_int8_benchmark.py), which target a
different model (DGCNN, 3x2048 inputs). Do not conflate them. The provider-option
pattern (trt_engine_cache_enable / trt_engine_cache_path / trt_fp16_enable /
device_id) is mirrored from that script.

Input  : (N, 3, H, W) ImageNet-normalised RGB  (default 480 x 640)
Output : (N, num_classes, H, W) logits, already upsampled inside the wrapper.

Usage (from project root, env activated):
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python src/trt_benchmark_segformer.py \
        --onnx results/optim_bench/student_mit_b0_dyn.onnx \
        --ckpt results/segformer_b0_rgb_kd/p15kd_smoke/best_model.pth \
        --fp16 --batch-sizes 1,8,16 \
        --out  results/optim_bench/trt_fp16_student.json

The ONNX graph must be a DYNAMIC-BATCH export for a multi-batch sweep (axis 0
marked dynamic by src/export_segformer_onnx.py --dynamic-batch).
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "data", "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(PROJECT_ROOT, "data", "hf_cache"))

import onnxruntime as ort  # noqa: E402

# torch is imported lazily only when --ckpt is given (PyTorch parity reference).


# --------------------------------------------------------------------------- #
# nvidia-smi GPU-memory sampler
# --------------------------------------------------------------------------- #
class GpuMemSampler(threading.Thread):
    """Polls `nvidia-smi --query-gpu=memory.used` for one GPU in a background
    thread and records the peak. Used to capture ORT/TRT GPU allocations, which
    live OUTSIDE torch's caching allocator (so torch.cuda.max_memory_allocated
    would NOT see them). Peak is reported as a delta over an idle baseline so a
    co-tenant's static footprint is subtracted out."""

    def __init__(self, device_id, poll_s=0.01):
        super().__init__(daemon=True)
        self.device_id = int(device_id)
        self.poll_s = poll_s
        # NB: must NOT be named self._stop -- that shadows threading.Thread._stop,
        # which join() calls internally, raising "'Event' object is not callable".
        self._stop_evt = threading.Event()
        self.samples_mb = []

    @staticmethod
    def _query(device_id):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-i", str(device_id)],
                stderr=subprocess.DEVNULL,
            )
            return float(out.decode().strip().splitlines()[0])
        except Exception:
            return None

    def baseline(self):
        v = self._query(self.device_id)
        return v if v is not None else 0.0

    def run(self):
        while not self._stop_evt.is_set():
            v = self._query(self.device_id)
            if v is not None:
                self.samples_mb.append(v)
            time.sleep(self.poll_s)

    def stop_and_peak(self):
        self._stop_evt.set()
        self.join(timeout=2.0)
        return max(self.samples_mb) if self.samples_mb else None


# --------------------------------------------------------------------------- #
# Session construction
# --------------------------------------------------------------------------- #
def build_trt_session(onnx_path, device_id, cache_dir, fp16, int8=False):
    """ORT InferenceSession with TRT EP first, CUDA then CPU as fallbacks.

    Returns (session, providers_requested). The caller inspects
    session.get_providers() to learn which EP actually engaged.

    If the TRT EP cannot initialise (e.g. the TensorRT runtime libs --
    libnvinfer.so.* -- are not installed/on LD_LIBRARY_PATH), ORT raises during
    session construction; we catch that and rebuild on CUDA so the benchmark
    still measures the GPU-accelerated graph (clearly labelled as a CUDA-EP run).

    int8=True sets trt_int8_enable so TRT runs the QDQ graph's quantized layers in
    INT8 (non-quantized ops fall back to FP16/FP32). The FP16 path is unchanged;
    enabling both int8+fp16 is the standard QDQ recipe (INT8 kernels + FP16
    fallback for un-quantized ops).
    """
    os.makedirs(cache_dir, exist_ok=True)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    trt_opts = {
        "device_id": int(device_id),
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
        "trt_fp16_enable": bool(fp16),
        "trt_max_partition_iterations": 1000,
        "trt_timing_cache_enable": True,
    }
    if int8:
        trt_opts["trt_int8_enable"] = True
    trt_provider = ("TensorrtExecutionProvider", trt_opts)
    providers = [trt_provider, "CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        # disable ORT's own silent fallback so a broken TRT EP surfaces here and
        # we can choose CUDA (not CPU) explicitly.
        sess = ort.InferenceSession(
            onnx_path, sess_options=sess_opts, providers=providers,
            provider_options=None)
        if "TensorrtExecutionProvider" in sess.get_providers():
            return sess, "TensorrtExecutionProvider"
        # ORT fell back internally -- rebuild cleanly on CUDA below.
    except Exception as exc:
        print(f"  [TRT EP init failed: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0] if str(exc) else exc}]")

    cuda_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=cuda_providers)
    active = sess.get_providers()[0] if sess.get_providers() else "unknown"
    return sess, active


def benchmark_session(sess, x_np, n_warmup, n_iter):
    """Latency (ms) for an ORT session on a fixed input. Warmup absorbs the
    (slow) first-run TRT engine build. Returns timing dict."""
    input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # Warmup (first iter triggers/loads the TRT engine -> can be very slow).
    for _ in range(n_warmup):
        _ = sess.run([out_name], {input_name: x_np})

    times_ms = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = sess.run([out_name], {input_name: x_np})
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    t = np.array(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std()),
        "min_ms": float(t.min()),
        "max_ms": float(t.max()),
        "p50_ms": float(np.percentile(t, 50)),
        "p90_ms": float(np.percentile(t, 90)),
        "fps_mean": 1000.0 / float(t.mean()),
    }


# --------------------------------------------------------------------------- #
# PyTorch parity reference (compiled TRT vs eager PyTorch argmax)
# --------------------------------------------------------------------------- #
def pytorch_parity(ckpt_path, sess, x_np, device_id):
    """Run eager PyTorch and the ORT-TRT session on the SAME seeded input and
    return the argmax-mismatch fraction (TRT vs PyTorch) -- the compiled-model
    correctness gate. Mirrors the import style of src/eval_segformer_kd_student.py."""
    import torch
    from gen_rgb_only_sota_gifs import load_model

    device = torch.device(f"cuda:{device_id}")
    model = load_model(ckpt_path, device)
    with torch.no_grad():
        xt = torch.from_numpy(x_np).to(device)
        torch_out = model(xt).cpu().numpy()

    input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    trt_out = sess.run([out_name], {input_name: x_np})[0]

    torch_argmax = np.argmax(torch_out, axis=1)
    trt_argmax = np.argmax(trt_out, axis=1)
    argmax_mismatch = float(np.mean(torch_argmax != trt_argmax))
    max_abs = float(np.max(np.abs(torch_out - trt_out)))
    mean_abs = float(np.mean(np.abs(torch_out - trt_out)))

    del model
    torch.cuda.empty_cache()
    return {
        "argmax_mismatch": argmax_mismatch,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "torch_out_shape": list(torch_out.shape),
        "trt_out_shape": list(trt_out.shape),
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="SegFormer TensorRT-via-ORT benchmark")
    ap.add_argument("--onnx", required=True, help="ONNX graph (dynamic-batch for a sweep).")
    ap.add_argument("--ckpt", default=None,
                    help="optional .pth for the PyTorch argmax-parity reference.")
    ap.add_argument("--batch-sizes", default="1,8,16",
                    help="comma-separated batch sizes.")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--fp16", action="store_true", help="enable TRT FP16 engine.")
    ap.add_argument("--int8", action="store_true",
                    help="enable TRT INT8 (run QDQ graph's quantized layers in "
                         "INT8; combine with --fp16 for FP16 fallback on un-quantized ops).")
    ap.add_argument("--cache-dir",
                    default=os.path.join(PROJECT_ROOT, "results", "trt_cache_segformer"))
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--out", default=None, help="output JSON path.")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    batch_sizes = [int(b) for b in str(args.batch_sizes).split(",") if b.strip()]
    if not batch_sizes:
        batch_sizes = [1]

    print("=" * 90)
    print("SEGFORMER TENSORRT (via ONNX Runtime) BENCHMARK")
    print("=" * 90)
    print(f"  ONNX            : {args.onnx}")
    print(f"  ORT version     : {ort.__version__}")
    print(f"  available EPs   : {ort.get_available_providers()}")
    print(f"  FP16            : {args.fp16}")
    print(f"  INT8            : {args.int8}")
    print(f"  cache dir       : {args.cache_dir}")
    print(f"  device_id       : {args.device_id}")
    print(f"  input           : (N, 3, {args.height}, {args.width}) ImageNet-normalised")
    print(f"  batch sizes     : {batch_sizes}")

    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        print("\n!!! ERROR: TensorrtExecutionProvider NOT available in this ORT build. Aborting.")
        sys.exit(1)

    print("\nNOTE: the FIRST run of each batch builds+serialises a TRT engine "
          "(slow, minutes). Warmup iterations absorb that cost; cached engines "
          f"are reused from {args.cache_dir} on subsequent runs.")

    per_batch = {}
    trt_engaged_any = False

    for bs in batch_sizes:
        print("\n" + "-" * 90)
        print(f"BATCH {bs}")
        print("-" * 90)
        x_np = np.random.RandomState(0).randn(
            bs, 3, args.height, args.width).astype(np.float32)

        t_build0 = time.perf_counter()
        sess, active_ep = build_trt_session(args.onnx, args.device_id, args.cache_dir, args.fp16, args.int8)
        active_eps = sess.get_providers()
        # The EP that actually OWNS the nodes is the first in the returned list.
        # If TRT silently fell back / failed to init, it is absent here.
        trt_engaged = "TensorrtExecutionProvider" in active_eps
        trt_engaged_any = trt_engaged_any or trt_engaged
        if trt_engaged:
            print(f"  session providers: {active_eps}  -> TRT EP ENGAGED")
        else:
            print("  " + "!" * 80)
            print(f"  !!! WARNING: TRT EP did NOT engage (active EP: {active_ep}).")
            print(f"  !!! session providers: {active_eps}")
            print(f"  !!! Results below run on '{active_ep}', NOT TensorRT. "
                  "Likely cause: TensorRT runtime libs (libnvinfer.so.*) not installed.")
            print("  " + "!" * 80)

        # GPU-memory sampling around the warmup+timed window (captures engine
        # build allocations + steady-state inference footprint).
        sampler = GpuMemSampler(args.device_id, poll_s=0.01)
        baseline_mb = sampler.baseline()
        sampler.start()
        timing = benchmark_session(sess, x_np, args.n_warmup, args.n_iter)
        build_plus_bench_s = time.perf_counter() - t_build0
        peak_used_mb = sampler.stop_and_peak()
        peak_delta_mb = (peak_used_mb - baseline_mb) if peak_used_mb is not None else None

        print(f"  latency  : {timing['mean_ms']:.3f} ± {timing['std_ms']:.3f} ms "
              f"(p50 {timing['p50_ms']:.3f}, p90 {timing['p90_ms']:.3f}, "
              f"min {timing['min_ms']:.3f}) -> {timing['fps_mean']:.1f} fps")
        print(f"  GPU mem  : peak used {peak_used_mb} MB | baseline {baseline_mb} MB | "
              f"peak delta {peak_delta_mb} MB  (nvidia-smi, device {args.device_id})")
        print(f"  (build+bench wall time for this batch: {build_plus_bench_s:.1f} s)")

        per_batch[str(bs)] = {
            "batch_size": bs,
            "trt_engaged": trt_engaged,
            "active_ep": active_ep,
            "session_providers": active_eps,
            "gpu_peak_used_mb": peak_used_mb,
            "gpu_baseline_mb": baseline_mb,
            "gpu_peak_delta_mb": peak_delta_mb,
            "build_plus_bench_s": float(build_plus_bench_s),
            **timing,
        }

        # PyTorch parity (compiled TRT vs eager) -- once, on the smallest batch
        # session, which matches the export/parity convention in
        # export_segformer_onnx.py. Done last so the timed numbers above are clean.
        if args.ckpt and bs == batch_sizes[0]:
            print(f"\n  Running PyTorch argmax-parity reference (TRT vs eager, batch {bs}) ...")
            parity = pytorch_parity(args.ckpt, sess, x_np, args.device_id)
            per_batch[str(bs)]["parity_vs_pytorch"] = parity
            frac = parity["argmax_mismatch"]
            tag = "OK" if frac < 5e-3 else "FLAG (>0.5% pixels differ)"
            print(f"  parity   : argmax-mismatch {frac:.6e} ({frac*100:.4f}% pixels)  -> {tag}")
            print(f"             max-abs-diff {parity['max_abs_diff']:.4e}, "
                  f"mean-abs-diff {parity['mean_abs_diff']:.4e}, "
                  f"shapes torch={parity['torch_out_shape']} trt={parity['trt_out_shape']}")

        del sess

    # ----------------------------------------------------------------- table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'batch':>5} {'TRT?':>5} {'mean ms':>10} {'p50 ms':>9} {'fps':>9} "
          f"{'GPU peakΔ MB':>13}")
    print("-" * 90)
    for bs in batch_sizes:
        r = per_batch[str(bs)]
        gd = f"{r['gpu_peak_delta_mb']:.0f}" if r["gpu_peak_delta_mb"] is not None else "n/a"
        print(f"{bs:>5} {('yes' if r['trt_engaged'] else 'NO!'):>5} "
              f"{r['mean_ms']:>10.3f} {r['p50_ms']:>9.3f} {r['fps_mean']:>9.1f} {gd:>13}")
    if not trt_engaged_any:
        print("\n!!! WARNING: TRT EP did NOT engage for ANY batch -- all numbers are CUDA fallback.")

    out = {
        "onnx": args.onnx,
        "ckpt": args.ckpt,
        "ort_version": ort.__version__,
        "fp16": bool(args.fp16),
        "int8": bool(args.int8),
        "device_id": args.device_id,
        "image_size": [args.height, args.width],
        "num_classes": args.num_classes,
        "cache_dir": args.cache_dir,
        "n_warmup": args.n_warmup,
        "n_iter": args.n_iter,
        "trt_engaged_any": trt_engaged_any,
        "note": "architecture-representative; final student weights pending",
        "per_batch": per_batch,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
