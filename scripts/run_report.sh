#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="animal-detector"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

usage() {
    echo "Usage: $0 <input_dir> [report_name] [confidence]"
    echo ""
    echo "  input_dir    Directory containing images to process"
    echo "  report_name  Report filename (default: animal_detection_report_<timestamp>.md), saved to out/reports/"
    echo "  confidence   Confidence threshold 0.0-1.0 (default: 0.15)"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

INPUT_DIR="$(cd "$1" && pwd)"
REPORT_NAME="${2:-animal_detection_report_$(date +%Y%m%d_%H%M%S).md}"
CONFIDENCE="${3:-0.15}"

# Output goes to <cwd>/out/reports/
OUTPUT_DIR="$(pwd)/out/reports"
mkdir -p "$OUTPUT_DIR"

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
echo "    Output:     ${OUTPUT_DIR}/${REPORT_NAME}"
echo "    Confidence: ${CONFIDENCE}"
echo ""

docker run --rm \
    -v "${INPUT_DIR}:/data/input:ro" \
    -v "${OUTPUT_DIR}:/data/output" \
    "$IMAGE_NAME" \
    /data/input \
    -o "/data/output/${REPORT_NAME}" \
    -c "$CONFIDENCE"

echo ""
echo "==> Done! Report written to: ${OUTPUT_DIR}/${REPORT_NAME}"
