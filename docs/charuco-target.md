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

Register only a report with one unique observational identity class: dictionary,
orientation, border-bit setting, spatial marker-ID layout, canonical ChArUco corner-ID
mapping, and independently consistent multi-camera evidence must agree. A differing
`legacy_pattern` value may share that identity class only when CameraRig proves identical
OpenCV marker IDs, marker corners, chessboard corners, canonical corner mapping, and a
deterministically generated binary board image. Non-equivalent legacy candidates remain
separate and fail closed. Registration recomputes every ranking, equivalence proof, and
acceptance invariant instead of trusting editable conclusion fields:

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
`PAUSED_FOR_USER_VALIDATION`; resolve them from the original PDF, generator metadata, or
authoritative board-owner metadata. Pass a local source with `--authoritative-source`, or
record user-provided metadata directly, plus the relevant
`--authoritative-dictionary`, `--authoritative-legacy-pattern`,
`--authoritative-border-bits`, or `--authoritative-orientation` constraints. The report
stores a canonical SHA-256 receipt, never a local path or source contents, and still
requires every vision gate to pass before an authoritative constraint may break a tie.
If an authoritative dictionary has no visual support while another dictionary passes
both capture sources, identification returns
`USER_AUTHORITATIVE_DICTIONARY_CONFLICTS_WITH_VISUAL_EVIDENCE`; it never falls back to
the visually preferred dictionary.

Run pose-observability deployment preflight before fixed provisioning:

```bash
camera-rig target preflight \
  --camera-config .local/configs/camera.yaml \
  --target .local/target/charuco_existing_v1/target_spec.json \
  --frames 60 \
  --policy uncertainty_validated \
  --report .local/reports/target_preflight.json \
  --overlays .local/overlays/target_preflight
```

`uncertainty_validated` makes coverage and image span advisory. It retains at least 12 corners,
50% corner fraction, finite two-dimensional target geometry, and the marker pixel-scale floor,
then requires PnP success, a full-rank scaled projection Jacobian, bounded translation/rotation
uncertainty, bounded condition number, and no materially different statistically competitive IPPE
alternative. Sixty-frame acceptance uses solve/observable ratios and p95 uncertainty; it does not
require every frame to pass. Low coverage is never an automatic pass: weak, noisy, clustered, or
ambiguous low-coverage observations still fail.

`pose_validated` is unchanged: 5% coverage is advisory while 1% absolute coverage, image spans,
12 corners, 50% corner fraction, and marker scale remain hard-gated. `legacy_strict` retains its
original 5% coverage hard gate. Final calibration under the new policy additionally requires
per-frame observability and final shared-pose observability while retaining reprojection, pose
repeatability, split-half stability, and native-depth sanity.
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
  --policy uncertainty_validated \
  --report .local/reports/charuco-live.json \
  --overlays .local/overlays/charuco-live
```

Reports persist complete `TargetObservation` records, aggregate 2D detection quality,
static-board temporal pixel jitter, and—for `uncertainty_validated`—per-frame and aggregate pose
observability. Overlays are diagnostics only and never feed the detector or later calibration.
Coverage remains the convex-hull area of detected ChArUco corners divided by the full image area.
The new policy records `coverage.hard_gate: false`, the observed value, and the 5% recommendation.

For a 500 x 700 mm, 5 x 7 target with 100 mm squares, 75 mm markers, and `DICT_4X4_50`,
`uncertainty_validated` is the recommended single-camera policy when the camera must stay outside
the robot workspace or view the board obliquely. The large board does not automatically pass:
marker pixel scale, corner localization, uncertainty, ambiguity, repeatability, split-half, and
native-depth checks remain mandatory.
