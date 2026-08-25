# Coordinate Conventions

CameraRig uses column vectors, right-handed coordinate frames, meters for length, and
nanoseconds for time. A rigid transform is a homogeneous 4 x 4 SE(3) matrix named
`T_target_from_source`.

For a point represented in the source frame:

```text
p_target = T_target_from_source @ p_source
```

Every persisted transform therefore includes the two explicit strings `source_frame`
and `target_frame`. The `RigidTransform` contract validates the homogeneous last row,
finite values, rotation orthonormality, and a rotation determinant of +1. A transform
whose source and target have the same name must be identity.

## Composition

Composition is permitted only when the intermediate frame names match. For example:

```text
T_workspace_from_color @ T_color_from_ir_left
    = T_workspace_from_ir_left
```

In the explicit API this is:

```python
T_workspace_from_ir_left = T_workspace_from_color.compose(T_color_from_ir_left)
```

The following attempted chain is rejected even though both values contain 4 x 4
matrices:

```text
T_workspace_from_color @ T_depth_from_ir_left
```

Here the left transform expects its source to be `color`, while the right transform
targets `depth`. `TransformGraph` applies the same compatibility rule while resolving
direct, inverse, and multi-hop paths deterministically.
