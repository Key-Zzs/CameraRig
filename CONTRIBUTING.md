# Contributing

Thank you for contributing to CameraRig.

## Development setup

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

Before opening a pull request, run:

```bash
ruff check .
ruff format --check .
mypy src
pytest -m "not hardware"
python -m build
```

Keep changes within CameraRig's single-physical-camera responsibility. Do not add a
hardware dependency to the package's core import path. New persisted formats require an
explicit schema version and tests.
