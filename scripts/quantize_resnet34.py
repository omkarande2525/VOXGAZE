"""
VoxGaze Milestone M1 / Task T4: ResNet-34 ONNX Quantization & CPU Latency Benchmark

Exports/quantizes the pretrained ResNet-34 FP32 ONNX gaze model to INT8 using ONNX Runtime.
Evaluates both Dynamic Quantization and Static Calibration (QDQ) Quantization against FP32.
Compares numerical output differences (yaw/pitch MAE, max delta, vector cosine similarity)
on memory-safe synthetic face crop fixtures.
Benchmarks CPU inference latency (mean, p50, p95) on this Windows host.
"""

from dataclasses import dataclass
import json
import math
import sys
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from backend.app.perception.face_crop import extract_face_crop
from backend.app.perception.resnet34_estimator import ResNet34GazeEstimator

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
FP32_PATH = MODELS_DIR / "resnet34_gaze.onnx"
DYNAMIC_INT8_PATH = MODELS_DIR / "resnet34_gaze_dynamic_int8.onnx"
STATIC_INT8_PATH = MODELS_DIR / "resnet34_gaze_static_int8.onnx"
OBS_DIR = ROOT_DIR / "artifacts" / "observations"


@dataclass
class SyntheticLandmark:
    x: float
    y: float
    z: float = 0.0


def create_synthetic_landmarks(cx: float, cy: float, w: float, h: float, n: int = 468) -> List[SyntheticLandmark]:
    """Generates synthetic landmarks simulating an elliptical facial landmark distribution."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rx, ry = w / 2.0, h / 2.0
    return [SyntheticLandmark(x=cx + rx * np.cos(a), y=cy + ry * np.sin(a)) for a in angles]


def generate_memory_safe_test_crops() -> List[Tuple[str, np.ndarray]]:
    """
    Generates a suite of memory-safe synthetic crops in RAM without writing to disk.
    Covers varied face positions, contrast levels, and synthetic skin-like patterns.
    """
    rng = np.random.RandomState(42)
    crops: List[Tuple[str, np.ndarray]] = []

    # 1. Standard face crop via extract_face_crop
    frame_720p = np.full((720, 1280, 3), 130, dtype=np.uint8)
    # Add subtle gradients to simulate room background
    for y in range(720):
        frame_720p[y, :, 0] = int(100 + 40 * (y / 720))
        frame_720p[y, :, 1] = int(120 + 30 * (y / 720))
        frame_720p[y, :, 2] = int(140 + 20 * (y / 720))

    lm_center = create_synthetic_landmarks(0.50, 0.50, 200 / 1280, 240 / 720)
    res_center = extract_face_crop(frame_720p, lm_center, padding_scale=1.20, min_size_px=64)
    if res_center.is_valid and res_center.crop_bgr is not None:
        crops.append(("synthetic_face_center", res_center.crop_bgr))

    # 2. Left-offset face crop
    lm_left = create_synthetic_landmarks(0.35, 0.50, 190 / 1280, 230 / 720)
    res_left = extract_face_crop(frame_720p, lm_left, padding_scale=1.20, min_size_px=64)
    if res_left.is_valid and res_left.crop_bgr is not None:
        crops.append(("synthetic_face_left", res_left.crop_bgr))

    # 3. Right-offset face crop
    lm_right = create_synthetic_landmarks(0.65, 0.50, 210 / 1280, 250 / 720)
    res_right = extract_face_crop(frame_720p, lm_right, padding_scale=1.20, min_size_px=64)
    if res_right.is_valid and res_right.crop_bgr is not None:
        crops.append(("synthetic_face_right", res_right.crop_bgr))

    # 4. Varied deterministic synthetic crops (in-memory fixtures)
    c4 = rng.randint(40, 220, (224, 224, 3), dtype=np.uint8)
    crops.append(("synthetic_noise_224", c4))

    c5 = np.zeros((180, 180, 3), dtype=np.uint8)
    cv2.circle(c5, (90, 90), 60, (180, 150, 120), -1)
    cv2.circle(c5, (70, 75), 10, (50, 50, 50), -1)
    cv2.circle(c5, (110, 75), 10, (50, 50, 50), -1)
    crops.append(("synthetic_stylized_face_180", c5))

    c6 = rng.randint(90, 160, (300, 300, 3), dtype=np.uint8)
    crops.append(("synthetic_texture_300", c6))

    c7 = np.full((256, 256, 3), 128, dtype=np.uint8)
    crops.append(("synthetic_flat_gray_256", c7))

    c8 = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(c8, (20, 20), (108, 108), (200, 180, 160), -1)
    crops.append(("synthetic_high_contrast_128", c8))

    return crops


class MemorySafeCalibrationDataReader(CalibrationDataReader):
    """Feeds memory-safe standardized NCHW float32 crops into ONNX static quantizer."""

    def __init__(self, count: int = 20) -> None:
        self.count = count
        self.current = 0
        self.samples: List[np.ndarray] = []

        estimator = ResNet34GazeEstimator(model_path=FP32_PATH)
        test_crops = generate_memory_safe_test_crops()

        for _, crop in test_crops:
            tensor = estimator.preprocess(crop)
            self.samples.append(tensor)

        # Supplement with deterministic transformed crops to reach calibration target
        rng = np.random.RandomState(99)
        while len(self.samples) < self.count:
            base_crop = test_crops[len(self.samples) % len(test_crops)][1]
            jittered = np.clip(
                base_crop.astype(np.float32) + rng.normal(0, 10, base_crop.shape), 0, 255
            ).astype(np.uint8)
            tensor = estimator.preprocess(jittered)
            self.samples.append(tensor)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.current < len(self.samples):
            batch = {"input": self.samples[self.current]}
            self.current += 1
            return batch
        return None

    def rewind(self) -> None:
        self.current = 0


def quantize_resnet34_dynamic(src_path: Path, dst_path: Path) -> Dict[str, Any]:
    """Applies dynamic quantization (weight-only INT8 on MatMul/Gemm classification heads) to ResNet-34 ONNX."""
    print(f"\n[INFO] Running Dynamic Quantization (MatMul/Gemm heads)...")
    t0 = time.perf_counter()
    quantize_dynamic(
        model_input=str(src_path),
        model_output=str(dst_path),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
    )
    duration = time.perf_counter() - t0
    size_bytes = dst_path.stat().st_size
    print(f"       Dynamic quantization finished in {duration:.2f}s | Size: {size_bytes / (1024*1024):.2f} MB")
    return {
        "quantization_type": "dynamic_linear_int8",
        "output_path": str(dst_path),
        "duration_s": round(duration, 3),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
    }


def quantize_resnet34_static(src_path: Path, dst_path: Path) -> Dict[str, Any]:
    """Applies static QDQ calibration quantization (weights + activations INT8) to ResNet-34 ONNX."""
    print(f"\n[INFO] Running Static Calibration Quantization (QDQ)...")
    reader = MemorySafeCalibrationDataReader(count=20)
    t0 = time.perf_counter()
    quantize_static(
        model_input=str(src_path),
        model_output=str(dst_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    duration = time.perf_counter() - t0
    size_bytes = dst_path.stat().st_size
    print(f"       Static quantization finished in {duration:.2f}s | Size: {size_bytes / (1024*1024):.2f} MB")
    return {
        "quantization_type": "static_qdq",
        "output_path": str(dst_path),
        "duration_s": round(duration, 3),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
    }


def compute_gaze_vector_cosine_sim(yaw1_rad: float, pitch1_rad: float, yaw2_rad: float, pitch2_rad: float) -> float:
    """Computes cosine similarity between two 3D gaze unit vectors."""
    # Official convention: v = (cos(pitch)*sin(yaw), -sin(pitch), -cos(pitch)*cos(yaw))
    v1 = np.array([
        math.cos(pitch1_rad) * math.sin(yaw1_rad),
        -math.sin(pitch1_rad),
        -math.cos(pitch1_rad) * math.cos(yaw1_rad),
    ], dtype=np.float64)
    v2 = np.array([
        math.cos(pitch2_rad) * math.sin(yaw2_rad),
        -math.sin(pitch2_rad),
        -math.cos(pitch2_rad) * math.cos(yaw2_rad),
    ], dtype=np.float64)

    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return dot / norm if norm > 0 else 1.0


def compare_model_outputs(
    fp32_est: ResNet34GazeEstimator,
    quant_est: ResNet34GazeEstimator,
    test_crops: List[Tuple[str, np.ndarray]],
) -> Dict[str, Any]:
    """Compares numerical outputs between FP32 baseline and a quantized estimator across test crops."""
    yaw_diffs_deg: List[float] = []
    pitch_diffs_deg: List[float] = []
    cosine_sims: List[float] = []
    per_crop_results: List[Dict[str, Any]] = []

    for name, crop in test_crops:
        out_fp32 = fp32_est.predict(crop)
        out_q = quant_est.predict(crop)

        yaw_diff = abs(out_q.yaw_deg - out_fp32.yaw_deg)
        pitch_diff = abs(out_q.pitch_deg - out_fp32.pitch_deg)
        cos_sim = compute_gaze_vector_cosine_sim(
            out_fp32.yaw_rad, out_fp32.pitch_rad, out_q.yaw_rad, out_q.pitch_rad
        )

        yaw_diffs_deg.append(yaw_diff)
        pitch_diffs_deg.append(pitch_diff)
        cosine_sims.append(cos_sim)

        per_crop_results.append({
            "crop_name": name,
            "crop_shape": list(crop.shape),
            "fp32_yaw_deg": round(out_fp32.yaw_deg, 3),
            "fp32_pitch_deg": round(out_fp32.pitch_deg, 3),
            "quant_yaw_deg": round(out_q.yaw_deg, 3),
            "quant_pitch_deg": round(out_q.pitch_deg, 3),
            "yaw_delta_deg": round(yaw_diff, 3),
            "pitch_delta_deg": round(pitch_diff, 3),
            "cosine_similarity": round(cos_sim, 5),
        })

    return {
        "num_test_crops": len(test_crops),
        "yaw_mae_deg": round(float(np.mean(yaw_diffs_deg)), 3),
        "yaw_max_delta_deg": round(float(np.max(yaw_diffs_deg)), 3),
        "pitch_mae_deg": round(float(np.mean(pitch_diffs_deg)), 3),
        "pitch_max_delta_deg": round(float(np.max(pitch_diffs_deg)), 3),
        "mean_cosine_similarity": round(float(np.mean(cosine_sims)), 5),
        "min_cosine_similarity": round(float(np.min(cosine_sims)), 5),
        "per_crop_breakdown": per_crop_results,
    }


def benchmark_cpu_latency(
    estimator: ResNet34GazeEstimator,
    benchmark_crop: np.ndarray,
    warmup_iters: int = 10,
    benchmark_iters: int = 60,
) -> Dict[str, Any]:
    """Benchmarks inference, preprocessing, and total latency on the current CPU host."""
    # Warmup
    for _ in range(warmup_iters):
        estimator.predict(benchmark_crop)

    infer_times: List[float] = []
    pre_times: List[float] = []
    total_times: List[float] = []

    for _ in range(benchmark_iters):
        t0 = time.perf_counter()
        out = estimator.predict(benchmark_crop)
        t_tot = (time.perf_counter() - t0) * 1000.0

        if out.is_valid:
            infer_times.append(out.inference_latency_ms)
            pre_times.append(out.preprocessing_latency_ms)
            total_times.append(t_tot)

    def stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(float(np.mean(arr)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
        }

    infer_stats = stats(infer_times)
    fps_estimate = round(1000.0 / infer_stats["p50"], 1) if infer_stats["p50"] > 0 else 0.0

    return {
        "warmup_iterations": warmup_iters,
        "measured_iterations": len(infer_times),
        "inference_latency_ms": infer_stats,
        "preprocessing_latency_ms": stats(pre_times),
        "total_latency_ms": stats(total_times),
        "estimated_throughput_fps": fps_estimate,
    }


def run_quantization_and_benchmarking() -> Dict[str, Any]:
    """Full execution pipeline for Task T4."""
    if not FP32_PATH.is_file():
        raise FileNotFoundError(f"FP32 ResNet-34 model missing at: {FP32_PATH}")

    fp32_size = FP32_PATH.stat().st_size
    print("=" * 80)
    print(" VOXGAZE TASK T4: RESNET-34 ONNX INT8 QUANTIZATION & CPU BENCHMARK")
    print(f" FP32 Baseline: {FP32_PATH} ({fp32_size / (1024*1024):.2f} MB)")
    print("=" * 80)

    # 1. Run Dynamic Quantization
    dyn_meta = quantize_resnet34_dynamic(FP32_PATH, DYNAMIC_INT8_PATH)

    # 2. Run Static QDQ Quantization
    stat_meta = quantize_resnet34_static(FP32_PATH, STATIC_INT8_PATH)

    # 3. Load Estimators
    print("\n[INFO] Initializing ONNX Runtime estimators...")
    fp32_est = ResNet34GazeEstimator(model_path=FP32_PATH)
    dyn_est = ResNet34GazeEstimator(model_path=DYNAMIC_INT8_PATH)
    stat_est = ResNet34GazeEstimator(model_path=STATIC_INT8_PATH)

    # 4. Compare Numerical Output Differences on Memory-Safe Test Crops
    print("[INFO] Evaluating numerical output differences across memory-safe test crops...")
    test_crops = generate_memory_safe_test_crops()
    dyn_diff = compare_model_outputs(fp32_est, dyn_est, test_crops)
    stat_diff = compare_model_outputs(fp32_est, stat_est, test_crops)

    # 5. Benchmark Latencies
    benchmark_crop = test_crops[0][1]
    print(f"[INFO] Benchmarking CPU latencies ({benchmark_crop.shape[1]}x{benchmark_crop.shape[0]} px crop, 60 iterations)...")
    fp32_bench = benchmark_cpu_latency(fp32_est, benchmark_crop)
    dyn_bench = benchmark_cpu_latency(dyn_est, benchmark_crop)
    stat_bench = benchmark_cpu_latency(stat_est, benchmark_crop)

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": "Windows CPU",
        "fp32_baseline": {
            "path": str(FP32_PATH),
            "size_bytes": fp32_size,
            "size_mb": round(fp32_size / (1024 * 1024), 2),
            "benchmark": fp32_bench,
        },
        "dynamic_int8": {
            **dyn_meta,
            "output_differences_vs_fp32": dyn_diff,
            "benchmark": dyn_bench,
            "speedup_vs_fp32_p50": round(
                fp32_bench["inference_latency_ms"]["p50"] / dyn_bench["inference_latency_ms"]["p50"], 2
            ) if dyn_bench["inference_latency_ms"]["p50"] > 0 else 0.0,
        },
        "static_int8": {
            **stat_meta,
            "output_differences_vs_fp32": stat_diff,
            "benchmark": stat_bench,
            "speedup_vs_fp32_p50": round(
                fp32_bench["inference_latency_ms"]["p50"] / stat_bench["inference_latency_ms"]["p50"], 2
            ) if stat_bench["inference_latency_ms"]["p50"] > 0 else 0.0,
        },
    }

    # Print summary
    print("\n" + "=" * 80)
    print(" RESNET-34 QUANTIZATION & BENCHMARK SUMMARY (MEASURED NUMBERS)")
    print("=" * 80)
    print(f" FP32 Baseline:")
    print(f"   - Size: {report['fp32_baseline']['size_mb']} MB")
    print(f"   - Inference Latency: p50 = {fp32_bench['inference_latency_ms']['p50']} ms | p95 = {fp32_bench['inference_latency_ms']['p95']} ms | mean = {fp32_bench['inference_latency_ms']['mean']} ms")
    print(f"   - Throughput: {fp32_bench['estimated_throughput_fps']} FPS")

    print(f"\n Dynamic INT8:")
    print(f"   - Size: {dyn_meta['size_mb']} MB ({round((1 - dyn_meta['size_bytes']/fp32_size)*100, 1)}% reduction)")
    print(f"   - Inference Latency: p50 = {dyn_bench['inference_latency_ms']['p50']} ms | p95 = {dyn_bench['inference_latency_ms']['p95']} ms | mean = {dyn_bench['inference_latency_ms']['mean']} ms")
    print(f"   - Speedup vs FP32: {report['dynamic_int8']['speedup_vs_fp32_p50']}x")
    print(f"   - Yaw MAE vs FP32: {dyn_diff['yaw_mae_deg']} deg (Max: {dyn_diff['yaw_max_delta_deg']} deg)")
    print(f"   - Pitch MAE vs FP32: {dyn_diff['pitch_mae_deg']} deg (Max: {dyn_diff['pitch_max_delta_deg']} deg)")
    print(f"   - Mean Cosine Sim: {dyn_diff['mean_cosine_similarity']}")

    print(f"\n Static QDQ INT8:")
    print(f"   - Size: {stat_meta['size_mb']} MB ({round((1 - stat_meta['size_bytes']/fp32_size)*100, 1)}% reduction)")
    print(f"   - Inference Latency: p50 = {stat_bench['inference_latency_ms']['p50']} ms | p95 = {stat_bench['inference_latency_ms']['p95']} ms | mean = {stat_bench['inference_latency_ms']['mean']} ms")
    print(f"   - Speedup vs FP32: {report['static_int8']['speedup_vs_fp32_p50']}x")
    print(f"   - Yaw MAE vs FP32: {stat_diff['yaw_mae_deg']} deg (Max: {stat_diff['yaw_max_delta_deg']} deg)")
    print(f"   - Pitch MAE vs FP32: {stat_diff['pitch_mae_deg']} deg (Max: {stat_diff['pitch_max_delta_deg']} deg)")
    print(f"   - Mean Cosine Sim: {stat_diff['mean_cosine_similarity']}")
    print("=" * 80)

    # Save report
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OBS_DIR / f"resnet34_int8_quantization_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SUCCESS] Report persisted to: {report_path}")

    return report


if __name__ == "__main__":
    run_quantization_and_benchmarking()
