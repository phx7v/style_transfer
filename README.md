# Van Gogh Style Transfer Streamlit App

A web application built with **Streamlit** that performs image style transfer.  
The project is fully containerized with **Docker** and supports separate **development** and **production** environments.

---

## Tech Stack

### Runtime
- Python 3.11
- Streamlit
- Onnx
- PyTorch, torchvision
- Numpy

### Devtools
- Docker & Docker Compose
- Make
- Poetry
- Ruff

---

## Quick Start

The project uses a **Makefile** as a single entry point for both local development and CI/CD.

### Development

#### App up and down

Runs the app in development mode with mounted source code.
```bash
make dev_up
```
Then open in your browser
http://localhost:8501

To stop and remove containers.
```bash
make dev_down
```

#### Tests, linters, formatters

To run tests.
```bash
make test
```

To see linters output.
```bash
make lint
```

To fix linters suggestions.
```bash
make lint_fix
```

To see formatters suggestions.
```bash
make format
```

To see formatters suggestions with diff.
```bash
make format_diff
```

To perform formatters suggestions.
```bash
make format_done
```

To see output of lint, format and test.
```bash
make final_check
```

To perform lint_fix, format_done and test.
```bash
make final_check_fixed
```

### Dependencies control

Dependencies are controlled using poetry.

Poetry lock.
```bash
make poetry_lock
```

Poetry update - you can specify PKG (optional).
```bash
make poetry_update PKG=<name>
```

To add dependency to prod - please specify PKG - package name (required) and VER - version (optional).
```bash
make poetry_add_prod PKG=<name> [VER=<version>]
```

To add dependency to dev - please specify PKG - package name (required) and VER - version (optional).
```bash
make poetry_add_dev PKG=<name> [VER=<version>]
```

To remove dependency from prod - please specify PKG - package name (required).
```bash
make poetry_remove_prod PKG=<name>
```

To remove dependency from dev - please specify PKG - package name (required).
```bash
make poetry_remove_dev PKG=<name>
```

### Models

To export models to onnx model - [export_to_onnx.py](tools/export_to_onnx.py) can be used.

### Production

Runs the app.
```bash
make prod_up
```

To stop and remove containers.
```bash
make prod_down
```
