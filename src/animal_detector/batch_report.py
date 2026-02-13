"""Batch image classification report using YOLO detector."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from PIL import Image

from animal_detector.modules.pipeline import DetectionPipeline


class DetectionResults(TypedDict):
    single_type: dict[str, list[str]]
    multi_type: list[tuple[str, set[str]]]
    no_detections: list[str]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WEIGHTS_PATH = Path(__file__).resolve().parent.parent.parent / "weights" / "yolo_animal_detector.pt"


def collect_images(input_dir: Path) -> list[Path]:
    """Return sorted list of image files in the directory."""
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_images(
    pipeline: DetectionPipeline, image_paths: list[Path]
) -> DetectionResults:
    """Run detection on all images and classify them.

    Returns a dict with:
      - single_type: dict mapping category -> list of filenames
      - multi_type: list of (filename, set of types)
      - no_detections: list of filenames
    """
    single_type: dict[str, list[str]] = defaultdict(list)
    multi_type: list[tuple[str, set[str]]] = []
    no_detections: list[str] = []

    total = len(image_paths)
    for i, path in enumerate(image_paths, 1):
        print(f"  [{i}/{total}] {path.name}", flush=True)  # noqa: T201
        image = Image.open(path).convert("RGB")
        result = pipeline.run(image)

        unique_labels = set(result.summary.keys())
        if len(unique_labels) == 0:
            no_detections.append(path.name)
        elif len(unique_labels) == 1:
            label = next(iter(unique_labels))
            single_type[label].append(path.name)
        else:
            multi_type.append((path.name, unique_labels))

    return {
        "single_type": dict(single_type),
        "multi_type": multi_type,
        "no_detections": no_detections,
    }


def generate_report(
    results: DetectionResults,
    total_images: int,
    input_dir: Path,
    confidence: float,
) -> str:
    """Build the Markdown report string."""
    single_type: dict[str, list[str]] = results["single_type"]
    multi_type: list[tuple[str, set[str]]] = results["multi_type"]
    no_detections: list[str] = results["no_detections"]

    single_count = sum(len(files) for files in single_type.values())
    multi_count = len(multi_type)
    no_det_count = len(no_detections)
    with_detections = single_count + multi_count

    lines: list[str] = []
    lines.append("# Animal Detection Report\n")
    lines.append(f"- **Date:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- **Input directory:** `{input_dir}`")
    lines.append(f"- **Confidence threshold:** {confidence}")
    lines.append(f"- **Total images processed:** {total_images}")
    lines.append("")

    # --- Summary statistics ---
    lines.append("## Summary\n")
    lines.append("| Metric | Count | % of total |")
    lines.append("|---|--:|--:|")
    lines.append(
        f"| Images with detections | {with_detections} | {_pct(with_detections, total_images)} |"
    )
    lines.append(f"| Single-type images | {single_count} | {_pct(single_count, total_images)} |")
    lines.append(f"| Multi-type images | {multi_count} | {_pct(multi_count, total_images)} |")
    lines.append(f"| No detections | {no_det_count} | {_pct(no_det_count, total_images)} |")
    lines.append("")

    # --- Category distribution ---
    if single_type:
        lines.append("## Category Distribution (single-type images)\n")
        lines.append("| Category | Images | % of total |")
        lines.append("|---|--:|--:|")
        for category in sorted(single_type, key=lambda c: len(single_type[c]), reverse=True):
            count = len(single_type[category])
            lines.append(f"| {category} | {count} | {_pct(count, total_images)} |")
        lines.append("")

    # --- Per-category detail ---
    if single_type:
        lines.append("## Images by Category\n")
        for category in sorted(single_type, key=lambda c: len(single_type[c]), reverse=True):
            files = single_type[category]
            lines.append(f"### {category} ({len(files)} images)\n")
            for fname in sorted(files):
                lines.append(f"- {fname}")
            lines.append("")

    # --- Multi-type images ---
    if multi_type:
        lines.append("## Multi-type Images\n")
        lines.append("Images where multiple object types were detected:\n")
        for fname, labels in sorted(multi_type):
            lines.append(f"- **{fname}** — {', '.join(sorted(labels))}")
        lines.append("")

    # --- No detections ---
    if no_detections:
        lines.append("## No Detections\n")
        lines.append("Images where no objects were detected:\n")
        for fname in sorted(no_detections):
            lines.append(f"- {fname}")
        lines.append("")

    return "\n".join(lines)


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100 * part / total:.1f}%"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch YOLO detection report — processes a directory of images "
        "and generates a Markdown report grouped by detected object type.",
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing images to process")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("report.md"),
        help="Output report file path (default: report.md)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.25,
        help="Minimum confidence threshold (default: 0.25)",
    )
    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    image_paths = collect_images(input_dir)
    if not image_paths:
        print(f"Error: no images found in {input_dir}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {input_dir}")  # noqa: T201
    print(f"Loading YOLO model from {WEIGHTS_PATH} ...")  # noqa: T201

    pipeline = DetectionPipeline(weights_path=WEIGHTS_PATH, confidence_threshold=args.confidence)

    print("Processing images:")  # noqa: T201
    results = process_images(pipeline, image_paths)

    report = generate_report(results, len(image_paths), input_dir, args.confidence)

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\nReport written to {output_path}")  # noqa: T201


if __name__ == "__main__":
    main()
