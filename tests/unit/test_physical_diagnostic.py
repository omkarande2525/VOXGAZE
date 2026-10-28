"""
Unit tests for the physical geometry diagnostic helpers in backend.app.perception.diagnostic.
Zero external OpenCV/UI dependencies.
"""

from pathlib import Path
import pytest
from backend.app.perception.diagnostic import (
    DiagnosticFrameData,
    DiagnosticTargetBuffer,
    TargetDiagnosticStats,
    compute_raw_coordinate_deltas,
    export_diagnostic_artifact,
    format_diagnostic_summary,
    verify_physical_ordering,
)


def make_frame_data(
    ts: int,
    h: float,
    v: float,
    ear: float = 0.30,
    iris_x: float = 100.0,
    iris_y: float = 100.0,
    inner_x: float = 150.0,
    inner_y: float = 100.0,
    outer_x: float = 50.0,
    outer_y: float = 100.0,
    upper_x: float = 100.0,
    upper_y: float = 85.0,
    lower_x: float = 100.0,
    lower_y: float = 115.0,
    is_valid: bool = True,
) -> DiagnosticFrameData:
    return DiagnosticFrameData(
        timestamp_ms=ts,
        h=h,
        v=v,
        ear=ear,
        iris_x=iris_x,
        iris_y=iris_y,
        inner_x=inner_x,
        inner_y=inner_y,
        outer_x=outer_x,
        outer_y=outer_y,
        upper_x=upper_x,
        upper_y=upper_y,
        lower_x=lower_x,
        lower_y=lower_y,
        is_valid=is_valid,
    )


def test_diagnostic_buffer_warmup_and_collection():
    buf = DiagnosticTargetBuffer("CENTER", warmup_sec=0.5, collection_sec=2.0)

    # 0ms - 400ms: warmup (discarded)
    for t in range(0, 500, 100):
        accepted = buf.add_frame(make_frame_data(t, 0.50, 0.50))
        assert accepted is False

    # 500ms - 2500ms: active collection
    for t in range(500, 2600, 100):
        accepted = buf.add_frame(make_frame_data(t, 0.50, 0.50))
        assert accepted is True

    # 2600ms+: window closed
    assert buf.is_complete(2600) is True
    assert buf.add_frame(make_frame_data(2700, 0.50, 0.50)) is False


def test_diagnostic_buffer_rejects_non_finite():
    buf = DiagnosticTargetBuffer("CENTER", warmup_sec=0.0, collection_sec=1.0)
    invalid_frame = make_frame_data(100, float("nan"), 0.50)
    assert buf.add_frame(invalid_frame) is False


def test_diagnostic_buffer_computes_stats():
    buf = DiagnosticTargetBuffer("CENTER", warmup_sec=0.0, collection_sec=1.0)
    # Feed 5 frames with known values
    for i, h in enumerate([0.48, 0.49, 0.50, 0.51, 0.52]):
        buf.add_frame(
            make_frame_data(
                ts=i * 100,
                h=h,
                v=0.50,
                iris_x=100.0 + (h - 0.50) * 10.0,
                iris_y=100.0,
                inner_x=150.0,
                inner_y=100.0,
                outer_x=50.0,
                outer_y=100.0,
            )
        )

    stats = buf.compute_stats()
    assert stats.sample_count == 5
    assert stats.median_h == pytest.approx(0.50)
    assert stats.mad_h == pytest.approx(0.01)
    assert stats.eye_width_px == pytest.approx(100.0)
    assert stats.eye_height_px == pytest.approx(30.0)


def test_verify_physical_ordering_pass():
    stats = {
        "CENTER": TargetDiagnosticStats(
            target="CENTER", sample_count=50, median_h=0.50, mad_h=0.01, median_v=0.50, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=100.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
        "LEFT": TargetDiagnosticStats(
            target="LEFT", sample_count=50, median_h=0.30, mad_h=0.01, median_v=0.50, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=120.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
        "RIGHT": TargetDiagnosticStats(
            target="RIGHT", sample_count=50, median_h=0.70, mad_h=0.01, median_v=0.50, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=80.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
        "UP": TargetDiagnosticStats(
            target="UP", sample_count=50, median_h=0.50, mad_h=0.01, median_v=0.30, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=100.0, median_iris_y=90.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
        "DOWN": TargetDiagnosticStats(
            target="DOWN", sample_count=50, median_h=0.50, mad_h=0.01, median_v=0.70, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=100.0, median_iris_y=110.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
    }

    ordering = verify_physical_ordering(stats)
    assert ordering["h_ordering_pass"] is True
    assert ordering["v_ordering_pass"] is True
    assert ordering["v_eyelid_ordering_pass"] is True
    assert ordering["v_canthal_ordering_pass"] is True
    assert ordering["overall_physical_ordering_pass"] is True


def test_paired_eyelid_vs_canthal_normal_ordering():
    """
    Verifies paired diagnostic ordering where eyelid retraction causes V_eyelid inversion
    while rigid canthal-normal projection preserves correct UP < CENTER < DOWN ordering.
    """
    stats = {
        "CENTER": TargetDiagnosticStats(
            target="CENTER", sample_count=50, median_h=0.50, mad_h=0.01, median_v=0.45, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=100.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0,
            median_v_eyelid=0.45, median_v_canthal=0.00, median_v_canthal_px=0.0,
        ),
        "LEFT": TargetDiagnosticStats(
            target="LEFT", sample_count=50, median_h=0.30, mad_h=0.01, median_v=0.45, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=120.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0,
            median_v_eyelid=0.45, median_v_canthal=0.00, median_v_canthal_px=0.0,
        ),
        "RIGHT": TargetDiagnosticStats(
            target="RIGHT", sample_count=50, median_h=0.70, mad_h=0.01, median_v=0.45, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=80.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0,
            median_v_eyelid=0.45, median_v_canthal=0.00, median_v_canthal_px=0.0,
        ),
        # Inverted UP for eyelid (e.g. 0.48 > 0.45) due to lid retraction, but canthal normal is -0.10 (< 0.0)
        "UP": TargetDiagnosticStats(
            target="UP", sample_count=50, median_h=0.50, mad_h=0.01, median_v=0.48, mad_v=0.01,
            median_ear=0.35, mad_ear=0.01, median_iris_x=100.0, median_iris_y=90.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=75.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=40.0,
            median_v_eyelid=0.48, median_v_canthal=-0.10, median_v_canthal_px=-10.0,
        ),
        # Inverted DOWN for eyelid (e.g. 0.40 < 0.45) due to lid droop, but canthal normal is +0.10 (> 0.0)
        "DOWN": TargetDiagnosticStats(
            target="DOWN", sample_count=50, median_h=0.50, mad_h=0.01, median_v=0.40, mad_v=0.01,
            median_ear=0.25, mad_ear=0.01, median_iris_x=100.0, median_iris_y=110.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=90.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=25.0,
            median_v_eyelid=0.40, median_v_canthal=+0.10, median_v_canthal_px=+10.0,
        ),
    }

    ordering = verify_physical_ordering(stats)
    assert ordering["h_ordering_pass"] is True
    # Eyelid ordering fails because 0.48 < 0.45 is False and 0.45 < 0.40 is False
    assert ordering["v_eyelid_ordering_pass"] is False
    # Canthal normal passes because -0.10 < 0.00 < +0.10 is True
    assert ordering["v_canthal_ordering_pass"] is True


def test_bilateral_ordering_and_disagreement():
    """
    Verifies independent ordering verification for Right eye, Left eye, and Bilateral signals,
    as well as inter-eye disagreement calculations.
    """
    stats = {
        "CENTER": TargetDiagnosticStats(
            target="CENTER", sample_count=50, sample_count_bilateral=50,
            median_h_right=0.500, mad_h_right=0.010, median_v_canthal_right=-0.0600, mad_v_canthal_right=0.005,
            median_h_left=0.510, mad_h_left=0.010, median_v_canthal_left=-0.0590, mad_v_canthal_left=0.005,
            median_h_bilateral=0.505, mad_h_bilateral=0.008, median_v_canthal_bilateral=-0.0595, mad_v_canthal_bilateral=0.004,
            median_inter_eye_h=0.010, median_inter_eye_v=0.001,
            eye_width_px=34.0, eye_height_px=10.0,
        ),
        "LEFT": TargetDiagnosticStats(
            target="LEFT", sample_count=50, sample_count_bilateral=50,
            median_h_right=0.473, mad_h_right=0.005, median_v_canthal_right=-0.0600, mad_v_canthal_right=0.005,
            median_h_left=0.480, mad_h_left=0.005, median_v_canthal_left=-0.0590, mad_v_canthal_left=0.005,
            median_h_bilateral=0.4765, mad_h_bilateral=0.004, median_v_canthal_bilateral=-0.0595, mad_v_canthal_bilateral=0.004,
            median_inter_eye_h=0.007, median_inter_eye_v=0.001,
            eye_width_px=33.0, eye_height_px=9.0,
        ),
        "RIGHT": TargetDiagnosticStats(
            target="RIGHT", sample_count=50, sample_count_bilateral=50,
            median_h_right=0.541, mad_h_right=0.005, median_v_canthal_right=-0.0600, mad_v_canthal_right=0.005,
            median_h_left=0.550, mad_h_left=0.005, median_v_canthal_left=-0.0590, mad_v_canthal_left=0.005,
            median_h_bilateral=0.5455, mad_h_bilateral=0.004, median_v_canthal_bilateral=-0.0595, mad_v_canthal_bilateral=0.004,
            median_inter_eye_h=0.009, median_inter_eye_v=0.001,
            eye_width_px=33.0, eye_height_px=11.0,
        ),
        "UP": TargetDiagnosticStats(
            target="UP", sample_count=50, sample_count_bilateral=50,
            median_h_right=0.500, mad_h_right=0.005, median_v_canthal_right=-0.0640, mad_v_canthal_right=0.005,
            median_h_left=0.510, mad_h_left=0.005, median_v_canthal_left=-0.0660, mad_v_canthal_left=0.005,
            median_h_bilateral=0.505, mad_h_bilateral=0.004, median_v_canthal_bilateral=-0.0650, mad_v_canthal_bilateral=0.004,
            median_inter_eye_h=0.010, median_inter_eye_v=0.002,
            eye_width_px=33.0, eye_height_px=12.0,
        ),
        "DOWN": TargetDiagnosticStats(
            target="DOWN", sample_count=50, sample_count_bilateral=50,
            median_h_right=0.500, mad_h_right=0.005, median_v_canthal_right=-0.0413, mad_v_canthal_right=0.005,
            median_h_left=0.510, mad_h_left=0.005, median_v_canthal_left=-0.0400, mad_v_canthal_left=0.005,
            median_h_bilateral=0.505, mad_h_bilateral=0.004, median_v_canthal_bilateral=-0.04065, mad_v_canthal_bilateral=0.004,
            median_inter_eye_h=0.010, median_inter_eye_v=0.0013,
            eye_width_px=32.0, eye_height_px=8.0,
        ),
    }

    ordering = verify_physical_ordering(stats)
    assert ordering["h_right_ordering_pass"] is True
    assert ordering["h_left_ordering_pass"] is True
    assert ordering["h_bilateral_ordering_pass"] is True
    assert ordering["v_canthal_right_ordering_pass"] is True
    assert ordering["v_canthal_left_ordering_pass"] is True
    assert ordering["v_canthal_bilateral_ordering_pass"] is True

    deltas = compute_raw_coordinate_deltas(stats)
    # Check bilateral separations
    assert deltas["LEFT"]["separation_h_bilateral"] == pytest.approx(0.505 - 0.4765, abs=1e-4)
    assert deltas["RIGHT"]["separation_h_bilateral"] == pytest.approx(0.5455 - 0.505, abs=1e-4)
    assert deltas["UP"]["separation_v_bilateral"] == pytest.approx(-0.0595 - (-0.0650), abs=1e-4)
    assert deltas["DOWN"]["separation_v_bilateral"] == pytest.approx(-0.04065 - (-0.0595), abs=1e-4)



def test_verify_physical_ordering_fail_on_inversion():
    stats = {
        "CENTER": TargetDiagnosticStats(
            target="CENTER", sample_count=50, median_h=0.50, mad_h=0.01, median_v=0.50, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=100.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
        # Inverted Left (larger H than center)
        "LEFT": TargetDiagnosticStats(
            target="LEFT", sample_count=50, median_h=0.60, mad_h=0.01, median_v=0.50, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=90.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
        "RIGHT": TargetDiagnosticStats(
            target="RIGHT", sample_count=50, median_h=0.70, mad_h=0.01, median_v=0.50, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=80.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
        "UP": TargetDiagnosticStats(
            target="UP", sample_count=50, median_h=0.50, mad_h=0.01, median_v=0.30, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=100.0, median_iris_y=90.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
        "DOWN": TargetDiagnosticStats(
            target="DOWN", sample_count=50, median_h=0.50, mad_h=0.01, median_v=0.70, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=100.0, median_iris_y=110.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
    }

    ordering = verify_physical_ordering(stats)
    assert ordering["h_ordering_pass"] is False
    assert ordering["v_ordering_pass"] is True
    assert ordering["overall_physical_ordering_pass"] is False


def test_compute_raw_coordinate_deltas():
    stats = {
        "CENTER": TargetDiagnosticStats(
            target="CENTER", sample_count=50, median_h=0.50, mad_h=0.01, median_v=0.50, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=100.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
        "LEFT": TargetDiagnosticStats(
            target="LEFT", sample_count=50, median_h=0.30, mad_h=0.01, median_v=0.50, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=120.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
    }

    deltas = compute_raw_coordinate_deltas(stats)
    assert "LEFT" in deltas
    assert deltas["LEFT"]["delta_iris_x_px"] == pytest.approx(20.0)
    assert deltas["LEFT"]["delta_iris_y_px"] == pytest.approx(0.0)
    assert deltas["LEFT"]["delta_h"] == pytest.approx(-0.20)
    assert deltas["LEFT"]["head_inner_canthus_drift_px"] == pytest.approx(0.0)


def test_diagnostic_artifact_roundtrip(tmp_path: Path):
    stats = {
        "CENTER": TargetDiagnosticStats(
            target="CENTER", sample_count=50, median_h=0.50, mad_h=0.01, median_v=0.50, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=100.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
        "LEFT": TargetDiagnosticStats(
            target="LEFT", sample_count=50, median_h=0.30, mad_h=0.01, median_v=0.50, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=120.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
        "RIGHT": TargetDiagnosticStats(
            target="RIGHT", sample_count=50, median_h=0.70, mad_h=0.01, median_v=0.50, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=80.0, median_iris_y=100.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
        "UP": TargetDiagnosticStats(
            target="UP", sample_count=50, median_h=0.50, mad_h=0.01, median_v=0.30, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=100.0, median_iris_y=90.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
        "DOWN": TargetDiagnosticStats(
            target="DOWN", sample_count=50, median_h=0.50, mad_h=0.01, median_v=0.70, mad_v=0.01,
            median_ear=0.30, mad_ear=0.01, median_iris_x=100.0, median_iris_y=110.0,
            mad_iris_x=0.5, mad_iris_y=0.5, median_inner_x=150.0, median_inner_y=100.0,
            median_outer_x=50.0, median_outer_y=100.0, median_upper_x=100.0, median_upper_y=85.0,
            median_lower_x=100.0, median_lower_y=115.0, eye_width_px=100.0, eye_height_px=30.0
        ),
    }
    ordering = verify_physical_ordering(stats)
    deltas = compute_raw_coordinate_deltas(stats)

    out_file = tmp_path / "test_diag.json"
    export_diagnostic_artifact(out_file, stats, ordering, deltas)
    assert out_file.is_file()

    summary_text = format_diagnostic_summary(stats, ordering, deltas)
    assert "VOXGAZE M1 PHYSICAL GEOMETRY DIAGNOSTIC REPORT" in summary_text
    assert "LEFT < CENTER < RIGHT" in summary_text


def test_evaluate_bilateral_diagnostic_questions_a_b_c_d():
    from scripts.evaluate_bilateral_diagnostic import audit_bilateral_dataset

    stats = {
        "CENTER": TargetDiagnosticStats(
            target="CENTER", sample_count=60, sample_count_bilateral=60, sample_count_left=60, sample_count_right=60,
            median_h_right=0.500, mad_h_right=0.004, median_v_canthal_right=-0.0601, mad_v_canthal_right=0.0004,
            median_h_left=0.508, mad_h_left=0.004, median_v_canthal_left=-0.0620, mad_v_canthal_left=0.0005,
            median_h_bilateral=0.504, mad_h_bilateral=0.003, median_v_canthal_bilateral=-0.06105, mad_v_canthal_bilateral=0.0003,
            median_inter_eye_h=0.008, median_inter_eye_v=0.0019,
        ),
        "LEFT": TargetDiagnosticStats(
            target="LEFT", sample_count=60, sample_count_bilateral=60, sample_count_left=60, sample_count_right=60,
            median_h_right=0.473, mad_h_right=0.005, median_v_canthal_right=-0.0565, mad_v_canthal_right=0.0012,
            median_h_left=0.485, mad_h_left=0.005, median_v_canthal_left=-0.0580, mad_v_canthal_left=0.0010,
            median_h_bilateral=0.479, mad_h_bilateral=0.004, median_v_canthal_bilateral=-0.05725, mad_v_canthal_bilateral=0.0008,
            median_inter_eye_h=0.012, median_inter_eye_v=0.0015,
        ),
        "RIGHT": TargetDiagnosticStats(
            target="RIGHT", sample_count=60, sample_count_bilateral=60, sample_count_left=60, sample_count_right=60,
            median_h_right=0.541, mad_h_right=0.010, median_v_canthal_right=-0.0609, mad_v_canthal_right=0.0017,
            median_h_left=0.555, mad_h_left=0.009, median_v_canthal_left=-0.0615, mad_v_canthal_left=0.0015,
            median_h_bilateral=0.548, mad_h_bilateral=0.007, median_v_canthal_bilateral=-0.0612, mad_v_canthal_bilateral=0.0012,
            median_inter_eye_h=0.014, median_inter_eye_v=0.0006,
        ),
        "UP": TargetDiagnosticStats(
            target="UP", sample_count=61, sample_count_bilateral=61, sample_count_left=61, sample_count_right=61,
            median_h_right=0.493, mad_h_right=0.003, median_v_canthal_right=-0.0640, mad_v_canthal_right=0.0015,
            median_h_left=0.500, mad_h_left=0.003, median_v_canthal_left=-0.0655, mad_v_canthal_left=0.0014,
            median_h_bilateral=0.4965, mad_h_bilateral=0.002, median_v_canthal_bilateral=-0.06475, mad_v_canthal_bilateral=0.0010,
            median_inter_eye_h=0.007, median_inter_eye_v=0.0015,
        ),
        "DOWN": TargetDiagnosticStats(
            target="DOWN", sample_count=60, sample_count_bilateral=60, sample_count_left=60, sample_count_right=60,
            median_h_right=0.518, mad_h_right=0.008, median_v_canthal_right=-0.0413, mad_v_canthal_right=0.0022,
            median_h_left=0.522, mad_h_left=0.007, median_v_canthal_left=-0.0425, mad_v_canthal_left=0.0020,
            median_h_bilateral=0.520, mad_h_bilateral=0.005, median_v_canthal_bilateral=-0.0419, mad_v_canthal_bilateral=0.0016,
            median_inter_eye_h=0.004, median_inter_eye_v=0.0012,
        ),
    }

    report = audit_bilateral_dataset(stats)
    # Check questions
    assert "question_a_left_vs_right" in report
    assert "question_b_bilateral_outperforms_better" in report
    assert "question_c_mad_reduction" in report
    assert "question_d_gate_reach" in report
    assert "question_e_inter_eye_disagreement" in report

    # Ordering is preserved
    assert report["ordering"]["H_B"] is True
    assert report["ordering"]["V_B"] is True

    # Check that separations < 0.060 correctly fail Question D
    assert report["question_d_gate_reach"]["all_reach"] is False

    # Check variance reduction
    assert report["question_c_mad_reduction"]["reduced_target_count"] > 0

