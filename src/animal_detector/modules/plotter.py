import base64
from collections import Counter
from io import BytesIO

import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from animal_detector.modules.detector import Detection

_PLOTLY_COLORS = px.colors.qualitative.Plotly


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _image_to_data_uri(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class InteractivePlotter:
    """Creates a Plotly figure with the image and interactive bounding boxes.

    Each bbox is a pair of traces (rectangle + text label) sharing a unique
    ``legendgroup`` with ``groupclick="togglegroup"``, so toggling a legend
    entry hides both the box and its label together.
    """

    def plot(self, image: Image.Image, detections: list[Detection]) -> go.Figure:
        w, h = image.size
        fig = go.Figure()

        # Background image — compressed JPEG data-URI instead of raw pixel array
        fig.add_layout_image(
            source=_image_to_data_uri(image),
            x=0,
            y=0,
            sizex=w,
            sizey=h,
            xref="x",
            yref="y",
            xanchor="left",
            yanchor="top",
            layer="below",
        )

        # Assign a stable colour index per label
        seen_labels: dict[str, int] = {}
        for det in detections:
            if det.label not in seen_labels:
                seen_labels[det.label] = len(seen_labels)

        for i, det in enumerate(detections):
            color_idx = seen_labels[det.label]
            hex_color = _PLOTLY_COLORS[color_idx % len(_PLOTLY_COLORS)]
            r, g, b = _hex_to_rgb(hex_color)
            rgb = f"rgb({r},{g},{b})"

            x1, y1, x2, y2 = det.bbox
            det_id = i + 1
            label_text = f"#{det_id} {det.label} {det.confidence:.0%}"
            group_id = f"bbox_{i}"

            # ── Rectangle trace ───────────────────────────────────────
            fig.add_trace(
                go.Scatter(
                    x=[x1, x2, x2, x1, x1],
                    y=[y1, y1, y2, y2, y1],
                    mode="lines",
                    line=dict(color=rgb, width=2),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.15)",
                    name=label_text,
                    legendgroup=group_id,
                    showlegend=True,
                    hoveron="fills+points",
                    hovertemplate=(
                        f"<b>{det.label}</b><br>"
                        f"Confidence: {det.confidence:.1%}<br>"
                        f"Box: ({x1}, {y1}) \u2013 ({x2}, {y2})"
                        "<extra></extra>"
                    ),
                ),
            )

            # ── Always-visible text label ─────────────────────────────
            fig.add_trace(
                go.Scatter(
                    x=[(x1 + x2) / 2],
                    y=[y1],
                    mode="text",
                    text=[f"<b>{label_text}</b>"],
                    textposition="top center",
                    textfont=dict(
                        color=rgb,
                        size=13,
                        shadow="1px 1px 2px white, -1px -1px 2px white, "
                        "1px -1px 2px white, -1px 1px 2px white",
                    ),
                    legendgroup=group_id,
                    showlegend=False,
                    hoverinfo="skip",
                ),
            )

        fig.update_layout(
            xaxis=dict(visible=False, range=[0, w], constrain="domain"),
            yaxis=dict(visible=False, range=[h, 0], scaleanchor="x"),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                title="Detections",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.85)",
                font=dict(size=12),
                groupclick="togglegroup",
            ),
            height=max(400, min(h, 700)),
        )

        return fig


def summarize_detections(detections: list[Detection]) -> dict[str, int]:
    """Returns a count of each detected label."""
    return dict(Counter(det.label for det in detections))
