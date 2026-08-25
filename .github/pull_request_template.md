## Summary

Describe the user-visible behavior and contract changes.

## Verification

- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `mypy src`
- [ ] `pytest -m "not hardware"`
- [ ] `python -m build`

## Boundary

- [ ] The change preserves one physical camera per CameraSession/CameraDriver instance.
- [ ] The core package remains importable without optional hardware SDKs.
- [ ] Persisted format changes are versioned and tested.
