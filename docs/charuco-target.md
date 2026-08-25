# ChArUco target workflow

Install the optional target dependencies:

```bash
python -m pip install -e ".[charuco]"
```

The standard public configuration is
`configs/targets/charuco_a4_v1.yaml`. It defines a 7 by 5 board using
`DICT_5X5_100`, 30 mm squares, 22 mm markers, and 24 detectable internal ChArUco
corners. Unknown configuration fields are rejected.

Generate and inspect the local printable artifact:

```bash
camera-rig target generate \
  --config configs/targets/charuco_a4_v1.yaml \
  --output .local/targets/charuco_a4_v1

camera-rig target inspect \
  --target .local/targets/charuco_a4_v1/target_spec.json
```

The board raster comes from OpenCV `CharucoBoard.generateImage()`. CameraRig embeds that
raster at exactly 210 by 150 mm in a deterministic A4-landscape PDF. The PDF also has
independent horizontal and vertical 100.00 mm rulers. Print the PDF at 100% / Actual
Size with all fit, shrink, and automatic scaling options disabled. A generated board is
not a substitute for measured print-scale acceptance.

The resolved `target_spec.json` freezes the dictionary, pattern mode, OpenCV version,
marker IDs, and every ChArUco corner ID-to-canonical-point mapping. Detection of an old
artifact looks up those persisted points; it does not reinterpret its object geometry
through the currently installed OpenCV version.

Detect one RGB or grayscale image:

```bash
camera-rig target detect \
  --target .local/targets/charuco_a4_v1/target_spec.json \
  --image image.png \
  --output .local/reports/detection.json \
  --overlay .local/overlays/detection.png
```

Detect every color frame in an existing snapshot artifact:

```bash
camera-rig target validate-artifact \
  --target .local/targets/charuco_a4_v1/target_spec.json \
  --artifact .local/artifacts/charuco-live \
  --stream color \
  --report .local/reports/charuco-live.json \
  --overlays .local/overlays/charuco-live
```

Reports persist complete `TargetObservation` records, aggregate 2D detection quality,
and static-board temporal pixel jitter. Overlays are diagnostics only and never feed
the detector or later calibration. Capture acceptance uses a 5% median coverage floor,
matching the detector's per-frame floor. Coverage remains the convex-hull area of the
detected ChArUco corners divided by the full image area; the acceptance report records
the applied threshold alongside every check.
