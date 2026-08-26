"""Consume a validated fixed-camera artifact through the stable public API."""

from __future__ import annotations

import argparse

import numpy as np

from camera_rig.api import load_provisioned_camera_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", help="camera-rig provision fixed artifact directory")
    args = parser.parse_args()

    bundle = load_provisioned_camera_bundle(args.artifact)
    fixed = bundle.fixed_mount_calibration
    if fixed is None:
        raise RuntimeError("camera is not fixed-calibrated")

    transform = fixed.T_parent_from_camera_reference
    print(transform.source_frame)
    print(transform.target_frame)
    points_camera = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    points_workspace = transform.transform_points(points_camera)
    print(points_workspace)


if __name__ == "__main__":
    main()
