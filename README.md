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

## Configuration

Default values for confidence threshold and grayscale mode are defined in a
single place: `src/animal_detector/modules/config.py`. Both the Streamlit app
and the batch report CLI read their defaults from this file.

## Streamlit App

```bash
poetry run streamlit run src/animal_detector/streamlit_app.py
```

Or with a manual venv:

```bash
.venv/bin/streamlit run src/animal_detector/streamlit_app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

Use the sidebar to adjust:

- **Grayscale** — convert images to grayscale before detection
- **Min confidence** — minimum confidence threshold for displayed detections
- **Animal types** — filter which detected categories are shown

## Batch Report

Process a directory of images and generate a Markdown report that groups images
by detected object type. The report covers images where only a single object type
was detected, and includes summary statistics.

### Usage

The `scripts/run_report.sh` script handles building the Docker image and running
the report in a single command:

```bash
./scripts/run_report.sh <input_dir> <output_dir> [report_name] [confidence] [--grayscale] [--show-image-previews]
```

Examples:

```bash
./scripts/run_report.sh /path/to/images /path/to/pred/outputs # prefered one

./scripts/run_report.sh /path/to/images /path/to/pred/outputs results.md
./scripts/run_report.sh /path/to/images /path/to/pred/outputs results.md 0.15
./scripts/run_report.sh /path/to/images /path/to/pred/outputs results.md 0.15 --grayscale
./scripts/run_report.sh /path/to/images /path/to/pred/outputs results.md 0.15 --grayscale --show-image-previews
```

| Argument | Default | Description |
|---|---|---|
| `input_dir` | *(required)* | Directory containing images to process |
| `output_dir` | *(required)* | Output directory for report and detections |
| `report_name` | `animal_detection_report_<timestamp>.md` | Report filename |
| `confidence` | `0.15` | Confidence threshold (0.0–1.0) |
| `--grayscale` | off | Convert images to grayscale before detection |
| `--show-image-previews` | off | Include bbox plot previews for each image in the report |

## Linting and type checking

```bash
# Format
poetry run ruff format src/

# Lint
poetry run ruff check src/

# Type check
poetry run mypy
```
