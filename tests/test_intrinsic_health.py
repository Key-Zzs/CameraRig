from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from camera_rig.calibration.intrinsic_health import (
    IntrinsicHealthObservation,
    IntrinsicHealthThresholds,
    evaluate_intrinsic_health,
    validate_intrinsic_health_report,
)
from camera_rig.calibration.pose.dependencies import cv2_module
from camera_rig.core.errors import ContractError
from camera_rig.core.intrinsics import CameraIntrinsics


def _intrinsics(
    *,
    fx: float = 600.0,
    fy: float = 605.0,
    cx: float = 320.0,
    cy: float = 240.0,
    distortion_model: str = "none",
    distortion_coeffs: tuple[float, ...] = (),
) -> CameraIntrinsics:
    return CameraIntrinsics(
        frame="camera/color_optical",
        width=640,
        height=480,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion_model=distortion_model,
        distortion_coeffs=distortion_coeffs,
    )


def _observations() -> tuple[IntrinsicHealthObservation, ...]:
    cv2 = cv2_module()
    truth = _intrinsics()
    matrix = np.asarray([[truth.fx, 0.0, truth.cx], [0.0, truth.fy, truth.cy], [0, 0, 1]])
    object_points = np.asarray(
        [(x * 0.03, y * 0.03, 0.0) for y in range(5) for x in range(6)],
        dtype=np.float64,
    )
    result = []
    for index in range(20):
        row, column = divmod(index, 5)
        rvec = np.asarray([0.05 * (row - 1.5), 0.06 * (column - 2), 0.02 * row])
        tvec = np.asarray([0.055 * (column - 2), 0.045 * (row - 1.5), 0.65 + 0.03 * row])
        image, _jacobian = cv2.projectPoints(object_points, rvec, tvec, matrix, np.zeros((5, 1)))
        result.append(
            IntrinsicHealthObservation(
                pose_id=f"pose_{index}",
                object_points_m=object_points,
                image_points_px=image.reshape(-1, 2),
            )
        )
    return tuple(result)


def _evaluate(factory: CameraIntrinsics):
    return evaluate_intrinsic_health(
        _observations(),
        factory,
        train_pose_ids=tuple(f"pose_{index}" for index in range(16)),
        holdout_pose_ids=tuple(f"pose_{index}" for index in range(16, 20)),
        thresholds=IntrinsicHealthThresholds(minimum_image_centroid_span_fraction=0.10),
        camera_identity_sha256="a" * 64,
        target_identity_sha256="b" * 64,
        provenance={"source": "fixture"},
    )


def test_correct_factory_intrinsics_pass_and_remain_immutable() -> None:
    factory = _intrinsics()
    before = factory.to_dict()
    report = _evaluate(factory)
    assert report["status"] == "PASS"
    assert factory.to_dict() == before
    assert report["factory_intrinsics_immutable"] is True
    validate_intrinsic_health_report(report, require_pass=True)


def test_known_focal_error_is_suspect_on_holdout() -> None:
    report = _evaluate(_intrinsics(fx=560.0, fy=565.0))
    assert report["status"] == "SUSPECT"
    holdout = report["holdout"]
    assert isinstance(holdout, dict)
    assert holdout["absolute_improvement_px"] > 0.005


@pytest.mark.parametrize(
    "factory",
    [
        _intrinsics(fy=565.0),
        _intrinsics(cx=300.0),
        _intrinsics(cy=220.0),
        _intrinsics(
            distortion_model="brown-conrady",
            distortion_coeffs=(0.08, -0.03, 0.0, 0.0, 0.0),
        ),
        _intrinsics(
            distortion_model="brown-conrady",
            distortion_coeffs=(0.0, 0.0, 0.02, -0.015, 0.0),
        ),
    ],
)
def test_independent_intrinsic_errors_are_detected(factory: CameraIntrinsics) -> None:
    assert _evaluate(factory)["status"] in {"SUSPECT", "INSUFFICIENT_EVIDENCE"}


def test_weak_pose_diversity_is_insufficient() -> None:
    repeated = tuple(
        IntrinsicHealthObservation(
            pose_id=f"pose_{index}",
            object_points_m=_observations()[0].object_points_m,
            image_points_px=_observations()[0].image_points_px,
        )
        for index in range(20)
    )
    report = evaluate_intrinsic_health(
        repeated,
        _intrinsics(),
        train_pose_ids=tuple(f"pose_{index}" for index in range(16)),
        holdout_pose_ids=tuple(f"pose_{index}" for index in range(16, 20)),
    )
    assert report["status"] == "INSUFFICIENT_EVIDENCE"


def test_insufficient_holdout_is_fail_closed() -> None:
    report = evaluate_intrinsic_health(
        _observations(),
        _intrinsics(),
        train_pose_ids=tuple(f"pose_{index}" for index in range(16)),
        holdout_pose_ids=("pose_16",),
        camera_identity_sha256="a" * 64,
        target_identity_sha256="b" * 64,
    )
    assert report["status"] == "INSUFFICIENT_EVIDENCE"


def test_intrinsic_health_rejects_semantically_forged_holdout() -> None:
    report = _evaluate(_intrinsics())
    holdout = report["holdout"]
    assert isinstance(holdout, dict)
    holdout["absolute_improvement_px"] = 99.0
    with pytest.raises(ContractError, match="holdout aggregate"):
        validate_intrinsic_health_report(report)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["diversity"].update(pose_count=1),
        lambda report: report["holdout"]["factory"].update(sample_count=0),
        lambda report: report["holdout"]["factory"].update(p95_px=-1.0),
        lambda report: report["holdout"]["factory"].update(per_pose_p95_px={}),
        lambda report: report["per_pose_corner_counts"].update(pose_0=4),
    ],
)
def test_intrinsic_health_rejects_rehashed_evidence_summary_forgery(mutation) -> None:
    report = deepcopy(_evaluate(_intrinsics()))
    mutation(report)
    with pytest.raises(ContractError):
        validate_intrinsic_health_report(report, require_pass=True)
