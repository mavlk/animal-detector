# Animal Detector

Streamlit app for detecting animals in images using a custom YOLO model.

## Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- YOLO weights file at `weights/yolo_animal_detector.pt`

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd animal-detector

# Install dependencies
poetry install

# Or, if using a manual venv:
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
pip install -e .
```

## Run

```bash
poetry run streamlit run src/animal_detector/streamlit_app.py
```

Or with a manual venv:

```bash
.venv/bin/streamlit run src/animal_detector/streamlit_app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## Linting and type checking

```bash
# Format
poetry run ruff format src/

# Lint
poetry run ruff check src/

# Type check
poetry run mypy
```
