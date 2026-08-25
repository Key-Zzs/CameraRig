# Target coordinate conventions

CameraRig target artifacts use a stable right-handed canonical frame independent of a
detector library's internal board coordinates.

Viewed from the printed front face:

```text
                 +Y (board up)
                  ↑
                  |
                  |
outer bottom-left ●────────────→ +X (board right)

+Z points out of the printed face toward the viewer.
+Z = +X × +Y.
```

Lengths are meters. All target object points lie on `z = 0`. The board outer boundary,
not the first internal chessboard corner, defines the origin.

OpenCV ChArUco board-local geometry is treated only as generator input. For the explicit
non-legacy pattern used here, OpenCV enumerates internal corners row-major in a frame
whose image-aligned Y direction points downward from the outer top-left. At artifact
generation CameraRig applies:

```text
x_canonical = x_opencv
y_canonical = board_height - y_opencv
z_canonical = 0
```

For the 7 by 5 standard board with 30 mm squares, examples are:

| Corner ID | Canonical point `(x, y, z)` m |
|---:|---:|
| 0 | `(0.030, 0.120, 0.000)` |
| 5 | `(0.180, 0.120, 0.000)` |
| 18 | `(0.030, 0.030, 0.000)` |
| 23 | `(0.180, 0.030, 0.000)` |

All 24 mappings are persisted in `target_spec.json`. A detector matches an observed ID
directly to this persisted mapping. It must not call OpenCV geometry APIs to reinterpret
an already generated target artifact.
