# ChArUco target workflow

CameraRig keeps the original A4 v1 route and adds two fail-closed deployment routes:

- `camera-rig.target.charuco.v2` supports A4, A3, or custom pages, board-only output,
  and a separate scale-check PDF;
- `camera-rig.target.charuco-resolved.v2` supports generated targets and existing
  physical boards without inventing printable source files.

An existing board is never identified from dimensions alone. Scan every supported
dictionary, both square orientations, both legacy modes, and border bits 1 and 2.
Image-only scans are preliminary and can never authorize registration. Final
identification requires consistent accepted frames from at least two distinct capture
artifacts whose camera identities differ; only privacy-safe identity hashes are persisted:

```bash
camera-rig target identify-existing \
  --artifact .local/captures/camera_a \
  --artifact .local/captures/camera_b \
  --board-width-mm 500 \
  --board-height-mm 700 \
  --square-length-mm 100 \
  --marker-length-mm 75 \
  --output .local/target/identification.json
```

Register only a report with one unique dictionary, orientation, legacy mode, border-bit
setting, marker layout, and independently consistent multi-camera match. Registration
recomputes every ranking and acceptance invariant instead of trusting editable conclusion
fields:

```bash
camera-rig target register-existing \
  --identification .local/target/identification.json \
  --target-name charuco_existing_v1 \
  --target-frame charuco_target \
  --output .local/target/charuco_existing_v1
```

That artifact contains only `target_spec.json`, `registration_report.json`, and
`checksums.sha256`. It retains evidence hashes but no source images and no fake print
PDF. Ambiguous results preserve the complete ranking and return
`PAUSED_FOR_USER_VALIDATION`; resolve them from the original PDF or generator metadata.
Pass that local source with `--authoritative-source` plus the relevant
`--authoritative-dictionary`, `--authoritative-legacy-pattern`,
`--authoritative-border-bits`, or `--authoritative-orientation` constraints. The report
stores only the source SHA-256, never its path or contents, and still requires every
vision gate to pass before the authoritative constraint may break a tie.

Run pose-free deployment preflight before fixed provisioning:

```bash
camera-rig target preflight \
  --camera-config .local/configs/camera.yaml \
  --target .local/target/charuco_existing_v1/target_spec.json \
  --frames 60 \
  --policy pose_validated \
  --report .local/reports/target_preflight.json \
  --overlays .local/overlays/target_preflight
```

`pose_validated` makes 5% coverage advisory while 1% absolute coverage, 12 corners,
50% corner fraction, spans, and marker scale stay hard-gated. Final calibration still hard-gates
cheirality, printed-face orientation, reprojection, pose repeatability, split-half
stability, and native-depth sanity. `legacy_strict` retains the original 5% hard gate.
The release CLI requires exactly 60 preflight frames. Fixed provisioning records the
selected target detection policy explicitly in `target.detection_policy`; existing-board
provisioning additionally requires native-depth sanity to finish with status `PASS`.

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
