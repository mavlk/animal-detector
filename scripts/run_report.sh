#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="animal-detector"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

usage() {
    echo "Usage: $0 <input_dir> <output_dir> [report_name] [confidence] [--grayscale] [--show-image-previews]"
    echo ""
    echo "  input_dir                Directory containing images to process"
    echo "  output_dir               Output directory for report and detections_dir/"
    echo "  report_name              Report filename (default: animal_detection_report_<timestamp>.md)"
    echo "  confidence               Confidence threshold 0.0-1.0 (default: 0.15)"
    echo "  --grayscale              Convert images to grayscale before detection"
    echo "  --show-image-previews    Include bbox plot previews in the report"
    exit 1
}

if [ $# -lt 2 ]; then
    usage
fi

INPUT_DIR="$(cd "$1" && pwd)"
OUTPUT_DIR="$(cd "$2" 2>/dev/null && pwd || mkdir -p "$2" && cd "$2" && pwd)"
REPORT_NAME="${3:-}"
CONFIDENCE="${4:-0.15}"
GRAYSCALE="${5:-}"
SHOW_PREVIEWS="${6:-}"

# Build the Docker image if it doesn't exist or if source files changed
echo "==> Checking Docker image '${IMAGE_NAME}'..."
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "==> Image not found. Building..."
    docker build -t "$IMAGE_NAME" "$PROJECT_DIR"
else
    echo "==> Image exists. Rebuilding to pick up any changes..."
    docker build -t "$IMAGE_NAME" "$PROJECT_DIR"
fi

echo "==> Running batch report..."
echo "    Input:      ${INPUT_DIR}"
echo "    Output dir: ${OUTPUT_DIR}"
echo "    Confidence: ${CONFIDENCE}"
echo "    Grayscale:  ${GRAYSCALE:+yes}"
echo ""

REPORT_ARGS=()
if [ -n "$REPORT_NAME" ]; then
    REPORT_ARGS+=(--report-name "$REPORT_NAME")
fi
if [ "$GRAYSCALE" = "--grayscale" ]; then
    REPORT_ARGS+=(--grayscale)
fi
if [ "$SHOW_PREVIEWS" = "--show-image-previews" ]; then
    REPORT_ARGS+=(--show-image-previews)
fi

docker run --rm \
    -v "${INPUT_DIR}:/data/input:ro" \
    -v "${OUTPUT_DIR}:/out_dir" \
    "$IMAGE_NAME" \
    /data/input \
    --output-dir /out_dir \
    ${REPORT_ARGS[@]+"${REPORT_ARGS[@]}"} \
    -c "$CONFIDENCE"

echo ""
echo "==> Done! Output written to: ${OUTPUT_DIR}"
