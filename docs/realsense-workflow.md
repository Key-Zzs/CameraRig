# RealSense Single-Camera Workflow

Install optional hardware and preview support:

```bash
python -m pip install -e ".[dev,realsense,viz]"
```

Copy the public strict configuration to an ignored private location and replace only the
serial placeholder. Device selection always uses that exact serial; no command selects
the first visible device implicitly.

```bash
camera-rig device inspect --config .local/configs/d435i.yaml --show-profiles
camera-rig calibration factory export \
  --config .local/configs/d435i.yaml \
  --output .local/artifacts/factory_calibration.json
camera-rig capture validate-streams \
  --config .local/configs/d435i.yaml --frames 300 \
  --report .local/reports/stream-validation.json
camera-rig capture snapshot \
  --config .local/configs/d435i.yaml --frames 30 \
  --output .local/artifacts/sequence
camera-rig replay validate --artifact .local/artifacts/sequence
```

The live path requests RGB8 color, raw Z16 depth, and Y8 infrared indices 1 and 2. It
does not align depth, filter pixels, alter exposure or emitter settings, build point
clouds, or write persistent device state. NumPy buffers are copied before the SDK can
reuse them. Replay validates hashes and paths before returning the copied raw arrays.
