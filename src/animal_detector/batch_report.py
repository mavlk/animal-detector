"""Batch image classification report using YOLO detector."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import matplotlib  # type: ignore[import-untyped]

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from sklearn.manifold import TSNE  # type: ignore[import-untyped]  # noqa: E402

from animal_detector.modules.config import (
    CONFIDENCE_THRESHOLD,
    EMBEDDING_LAYERS,
    GRAYSCALE,
    SHOW_IMAGE_PREVIEWS,
    WEIGHTS_PATH,
)
from animal_detector.modules.detector import Detection
from animal_detector.modules.pipeline import DetectionPipeline


@dataclass
class ImageEmbeddingInfo:
    """Per-image data needed for embedding visualization."""

    filename: str
    embeddings: dict[str, npt.NDArray[np.float32]]  # layer_key -> embedding
    detection_counts: dict[str, int]


class DetectionResults(TypedDict):
    single_type: dict[str, list[str]]
    multi_type: list[tuple[str, set[str]]]
    no_detections: list[str]
    contains_type: dict[str, int]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Distinct colors for bbox drawing (RGB tuples)
_BBOX_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 165, 0),
    (128, 0, 128),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 128, 0),
    (70, 130, 180),
]


def collect_images(input_dir: Path) -> list[Path]:
    """Return sorted list of image files in the directory."""
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _save_bbox_plot(
    image: Image.Image,
    detections: list[Detection],
    output_path: Path,
    label_colors: dict[str, tuple[int, int, int]],
) -> None:
    """Draw bounding boxes with labels on a copy of the image and save as JPEG."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        color = label_colors.get(det.label, (255, 0, 0))
        x1, y1, x2, y2 = det.bbox

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background + text
        text = f"{det.label} {det.confidence:.0%}"
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255), font=font)

    img.save(output_path, format="JPEG", quality=90)


def _save_bbox_json(
    image_name: str,
    detections: list[Detection],
    output_path: Path,
) -> None:
    """Save detection results as JSON."""
    data = {
        "image": image_name,
        "detections": [
            {
                "label": det.label,
                "confidence": round(det.confidence, 4),
                "bbox": list(det.bbox),
            }
            for det in detections
        ],
    }
    output_path.write_text(json.dumps(data, indent=2))


def process_images(
    pipeline: DetectionPipeline,
    image_paths: list[Path],
    detections_dir: Path,
) -> tuple[DetectionResults, list[ImageEmbeddingInfo], dict[str, tuple[int, int, int]]]:
    """Run detection on all images, save per-image outputs, and classify them.

    Returns a tuple of:
      - DetectionResults dict
      - list of ImageEmbeddingInfo (per-image embeddings for t-SNE)
      - label_colors mapping (reused for t-SNE plot)
    """
    single_type: dict[str, list[str]] = defaultdict(list)
    multi_type: list[tuple[str, set[str]]] = []
    no_detections: list[str] = []
    contains_type: dict[str, int] = defaultdict(int)

    # Assign stable colors to labels as they appear
    label_colors: dict[str, tuple[int, int, int]] = {}
    embedding_infos: list[ImageEmbeddingInfo] = []

    total = len(image_paths)
    for i, path in enumerate(image_paths, 1):
        print(f"  [{i}/{total}] {path.name}", flush=True)  # noqa: T201
        image = Image.open(path).convert("RGB")
        result = pipeline.run(image)

        # Assign colors to new labels
        for det in result.detections:
            if det.label not in label_colors:
                label_colors[det.label] = _BBOX_COLORS[len(label_colors) % len(_BBOX_COLORS)]

        # Save bbox plot and JSON
        stem = path.stem
        _save_bbox_plot(
            image, result.detections, detections_dir / f"{stem}_bbox_plot.jpg", label_colors
        )
        _save_bbox_json(path.name, result.detections, detections_dir / f"{stem}_bbox_pred.json")

        # Extract embeddings from all configured layers in one forward pass
        layer_indices = [idx for idx, _key, _name in EMBEDDING_LAYERS]
        raw_embeddings = pipeline.extract_embeddings(image, layer_indices)
        embeddings_by_key = {
            key: raw_embeddings[idx] for idx, key, _name in EMBEDDING_LAYERS
        }
        embedding_infos.append(
            ImageEmbeddingInfo(
                filename=path.name,
                embeddings=embeddings_by_key,
                detection_counts=dict(result.summary),
            )
        )

        # Classify image
        unique_labels = set(result.summary.keys())

        # Track contains_type for every label present in this image
        for label in unique_labels:
            contains_type[label] += 1

        if len(unique_labels) == 0:
            no_detections.append(path.name)
        elif len(unique_labels) == 1:
            label = next(iter(unique_labels))
            single_type[label].append(path.name)
        else:
            multi_type.append((path.name, unique_labels))

    results: DetectionResults = {
        "single_type": dict(single_type),
        "multi_type": multi_type,
        "no_detections": no_detections,
        "contains_type": dict(contains_type),
    }
    return results, embedding_infos, label_colors


def _compute_tsne(
    embeddings: npt.NDArray[np.float32],
    perplexity: float = 30.0,
    random_state: int = 42,
) -> npt.NDArray[np.float64]:
    """Run t-SNE on a (n_samples, n_features) matrix and return 2D coordinates."""
    effective_perplexity = min(perplexity, max(1.0, len(embeddings) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    coords: npt.NDArray[np.float64] = tsne.fit_transform(embeddings)
    return coords



def _normalize_color(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert 0-255 RGB tuple to 0.0-1.0 for matplotlib."""
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def _plot_embedding(
    coords: npt.NDArray[np.float64],
    embedding_infos: list[ImageEmbeddingInfo],
    output_path: Path,
    label_colors: dict[str, tuple[int, int, int]],
    title: str,
    axis_prefix: str,
) -> None:
    """Create and save an embedding scatter plot with pie-chart markers for multi-class images."""
    fig, ax = plt.subplots(figsize=(12, 10))

    legend_labels_added: set[str] = set()
    data_range = max(coords.max() - coords.min(), 1e-6)
    marker_radius = data_range * 0.015

    for idx, info in enumerate(embedding_infos):
        x, y = coords[idx]
        counts = info.detection_counts

        if not counts:
            ax.plot(
                x, y, "o",
                color="gray", markersize=8,
                markeredgecolor="black", markeredgewidth=0.5,
                label="no detections" if "no detections" not in legend_labels_added else None,
            )
            legend_labels_added.add("no detections")

        elif len(counts) == 1:
            label = next(iter(counts))
            color = _normalize_color(label_colors.get(label, (128, 128, 128)))
            ax.plot(
                x, y, "o",
                color=color, markersize=8,
                markeredgecolor="black", markeredgewidth=0.5,
                label=label if label not in legend_labels_added else None,
            )
            legend_labels_added.add(label)

        else:
            total_count = sum(counts.values())
            start_angle = 0.0
            for label, count in sorted(counts.items()):
                sweep = 360.0 * count / total_count
                color = _normalize_color(label_colors.get(label, (128, 128, 128)))
                wedge = Wedge(
                    center=(x, y), r=marker_radius,
                    theta1=start_angle, theta2=start_angle + sweep,
                    facecolor=color, edgecolor="black", linewidth=0.5,
                )
                ax.add_patch(wedge)
                start_angle += sweep

                if label not in legend_labels_added:
                    ax.plot([], [], "o", color=color, label=label)
                    legend_labels_added.add(label)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{axis_prefix} 1")
    ax.set_ylabel(f"{axis_prefix} 2")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_aspect("equal")
    ax.autoscale_view()

    fig.tight_layout()
    fig.savefig(output_path, format="JPEG", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _image_preview(fname: str) -> str:
    """Return a markdown image tag linking to the bbox plot for *fname*."""
    stem = Path(fname).stem
    return f"![{fname}](detections_dir/{stem}_bbox_plot.jpg)"


def generate_report(
    results: DetectionResults,
    total_images: int,
    input_dir: Path,
    confidence: float,
    grayscale: bool = GRAYSCALE,
    show_image_previews: bool = SHOW_IMAGE_PREVIEWS,
    embedding_plots: list[tuple[str, str, str]] | None = None,
) -> str:
    """Build the Markdown report string."""
    single_type: dict[str, list[str]] = results["single_type"]
    multi_type: list[tuple[str, set[str]]] = results["multi_type"]
    no_detections: list[str] = results["no_detections"]
    contains_type: dict[str, int] = results["contains_type"]

    single_count = sum(len(files) for files in single_type.values())
    multi_count = len(multi_type)
    no_det_count = len(no_detections)
    with_detections = single_count + multi_count

    lines: list[str] = []
    lines.append("# Animal Detection Report\n")
    lines.append(f"- **Date:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- **Input directory:** `{input_dir}`")
    lines.append(f"- **Confidence threshold:** {confidence}")
    lines.append(f"- **Grayscale:** {grayscale}")
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
    all_categories = sorted(
        set(single_type) | set(contains_type),
        key=lambda c: contains_type.get(c, 0),
        reverse=True,
    )
    if all_categories:
        lines.append("## Category Distribution (single-type images)\n")
        lines.append("| Category | single_type images | contains_type images |")
        lines.append("|---|--:|--:|")
        for category in all_categories:
            st_count = len(single_type.get(category, []))
            ct_count = contains_type.get(category, 0)
            st_val = f"{st_count} ({_pct(st_count, total_images)})"
            ct_val = f"{ct_count} ({_pct(ct_count, total_images)})"
            lines.append(f"| {category} | {st_val} | {ct_val} |")
        lines.append("")

    # --- Per-category detail ---
    if single_type:
        lines.append("## Images by Category\n")
        for category in sorted(single_type, key=lambda c: len(single_type[c]), reverse=True):
            files = single_type[category]
            lines.append(f"### {category} ({len(files)} images)\n")
            for fname in sorted(files):
                lines.append(f"- {fname}")
                if show_image_previews:
                    lines.append(f"  {_image_preview(fname)}")
            lines.append("")

    # --- Multi-type images ---
    if multi_type:
        lines.append("## Multi-type Images\n")
        lines.append("Images where multiple object types were detected:\n")
        for fname, labels in sorted(multi_type):
            lines.append(f"- {fname} — {', '.join(sorted(labels))}")
            if show_image_previews:
                lines.append(f"  {_image_preview(fname)}")
        lines.append("")

    # --- No detections ---
    if no_detections:
        lines.append("## No Detections\n")
        lines.append("Images where no objects were detected:\n")
        for fname in sorted(no_detections):
            lines.append(f"- {fname}")
            if show_image_previews:
                lines.append(f"  {_image_preview(fname)}")
        lines.append("")

    # --- Embedding visualizations ---
    if embedding_plots:
        lines.append("## Embedding Visualizations\n")
        lines.append(
            "2D projections of per-image feature embeddings. "
            "Points are colored by detected class; "
            "multi-class images appear as pie charts.\n"
        )
        for layer_display_name, method_name, filename in embedding_plots:
            lines.append(f"### {layer_display_name} — {method_name}\n")
            lines.append(f"![{layer_display_name} {method_name}](detections_dir/{filename})")
            lines.append("")

    return "\n".join(lines)


def _save_embeddings_json(
    embedding_infos: list[ImageEmbeddingInfo],
    label_colors: dict[str, tuple[int, int, int]],
    output_path: Path,
) -> None:
    """Save per-image embeddings and detection metadata to a JSON file."""
    data = {
        "label_colors": {label: list(rgb) for label, rgb in label_colors.items()},
        "layers": [
            {"key": key, "layer_index": idx, "display_name": name}
            for idx, key, name in EMBEDDING_LAYERS
        ],
        "images": [
            {
                "filename": info.filename,
                "detection_counts": info.detection_counts,
                "embeddings": {
                    key: emb.tolist() for key, emb in info.embeddings.items()
                },
            }
            for info in embedding_infos
        ],
    }
    output_path.write_text(json.dumps(data, indent=2))


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
        "--output-dir",
        "-o",
        type=Path,
        default=Path("out"),
        help="Output directory for report and detections (default: out/)",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default=None,
        help="Report filename (default: animal_detection_report_<timestamp>.md)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Minimum confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        default=GRAYSCALE,
        help="Convert images to grayscale before detection",
    )
    parser.add_argument(
        "--show-image-previews",
        action="store_true",
        default=SHOW_IMAGE_PREVIEWS,
        help="Include bbox plot previews for each image in the report",
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

    output_dir: Path = args.output_dir
    detections_dir = output_dir / "detections_dir"
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_name: str = args.report_name or f"animal_detection_report_{timestamp}.md"

    print(f"Found {len(image_paths)} images in {input_dir}")  # noqa: T201
    print(f"Loading YOLO model from {WEIGHTS_PATH} ...")  # noqa: T201

    pipeline = DetectionPipeline(
        weights_path=WEIGHTS_PATH,
        confidence_threshold=args.confidence,
        grayscale=args.grayscale,
    )

    print("Processing images:")  # noqa: T201
    results, embedding_infos, label_colors = process_images(
        pipeline, image_paths, detections_dir
    )

    embeddings_json_path = detections_dir / "embeddings.json"
    _save_embeddings_json(embedding_infos, label_colors, embeddings_json_path)
    print(f"Embeddings saved to {embeddings_json_path}")  # noqa: T201

    # Generate embedding plots: 3 layers x 2 methods (t-SNE + UMAP)
    embedding_plots: list[tuple[str, str, str]] = []
    if len(embedding_infos) >= 2:
        for _layer_idx, layer_key, layer_display_name in EMBEDDING_LAYERS:
            layer_embeddings = np.stack(
                [info.embeddings[layer_key] for info in embedding_infos]
            )

            print(  # noqa: T201
                f"Computing t-SNE for {layer_display_name}...", flush=True,
            )
            coords = _compute_tsne(layer_embeddings)
            filename = f"embeddings_tsne_{layer_key}.jpg"
            _plot_embedding(
                coords,
                embedding_infos,
                detections_dir / filename,
                label_colors,
                title=f"{layer_display_name} — t-SNE",
                axis_prefix="t-SNE",
            )
            embedding_plots.append((layer_display_name, "t-SNE", filename))
            print(f"  Saved {filename}")  # noqa: T201
    else:
        print("Skipping embedding plots: need at least 2 images.")  # noqa: T201

    report = generate_report(
        results,
        len(image_paths),
        input_dir,
        args.confidence,
        grayscale=args.grayscale,
        show_image_previews=args.show_image_previews,
        embedding_plots=embedding_plots,
    )

    report_path = output_dir / report_name
    report_path.write_text(report)
    print(f"\nReport written to {report_path}")  # noqa: T201
    print(f"Detections saved to {detections_dir}")  # noqa: T201


if __name__ == "__main__":
    main()
