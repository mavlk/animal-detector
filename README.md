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

## Streamlit App

```bash
poetry run streamlit run src/animal_detector/streamlit_app.py
```

Or with a manual venv:

```bash
.venv/bin/streamlit run src/animal_detector/streamlit_app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## Batch Report

Process a directory of images and generate a Markdown report that groups images
by detected object type. The report covers images where only a single object type
was detected, and includes summary statistics.

### Usage

The `scripts/run_report.sh` script handles building the Docker image and running
the report in a single command:

```bash
./scripts/run_report.sh /path/to/images # saves to out/reports/animal_detection_report_<timestamp>.md
OR
./scripts/run_report.sh /path/to/images results.md # saves to out/reports/results.md
OR
./scripts/run_report.sh /path/to/images results.md 0.15 # custom confidence
```

## Linting and type checking

```bash
# Format
poetry run ruff format src/

# Lint
poetry run ruff check src/

# Type check
poetry run mypy
```
