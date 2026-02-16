import streamlit as st
from PIL import Image

from animal_detector.modules.config import CONFIDENCE_THRESHOLD, GRAYSCALE, WEIGHTS_PATH
from animal_detector.modules.pipeline import DetectionPipeline
from animal_detector.modules.plotter import InteractivePlotter, summarize_detections

MAX_UPLOADS = 10


@st.cache_resource
def load_pipeline(grayscale: bool = GRAYSCALE) -> DetectionPipeline:
    return DetectionPipeline(
        WEIGHTS_PATH, confidence_threshold=CONFIDENCE_THRESHOLD, grayscale=grayscale
    )


def main() -> None:
    st.set_page_config(
        page_title="Animal Detector", layout="wide", initial_sidebar_state="collapsed"
    )
    st.title("Animal Detector")

    # ── Sidebar filters ──────────────────────────────────────────────
    st.sidebar.header("Filters")
    grayscale = st.sidebar.toggle(
        "Grayscale", value=GRAYSCALE, help="Convert images to grayscale before detection"
    )
    conf_threshold = st.sidebar.slider(
        "Min confidence", 0.0, 1.0, value=CONFIDENCE_THRESHOLD, step=0.05
    )

    pipeline = load_pipeline(grayscale=grayscale)
    plotter = InteractivePlotter()

    uploaded_files = st.file_uploader(
        "Upload images (up to 10)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more images to get started.")
        return

    if len(uploaded_files) > MAX_UPLOADS:
        st.warning(f"Only the first {MAX_UPLOADS} images will be processed.")
        uploaded_files = uploaded_files[:MAX_UPLOADS]

    # ── Detect all images first ───────────────────────────────────────
    results = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        result = pipeline.run(image)
        results.append((uploaded_file.name, result))

    # Collect all unique labels across every image for the label filter
    all_labels = sorted({det.label for _, res in results for det in res.detections})

    selected_labels = st.sidebar.multiselect(
        "Animal types",
        options=all_labels,
        default=all_labels,
        help="Deselect types to hide them. You can also click the Plotly legend.",
    )

    # ── Render each image ─────────────────────────────────────────────
    for file_name, result in results:
        filtered = [
            det
            for det in result.detections
            if det.confidence >= conf_threshold and det.label in selected_labels
        ]
        summary = summarize_detections(filtered)

        st.divider()
        st.subheader(file_name)
        col_img, col_summary = st.columns([3, 1])

        with col_img:
            fig = plotter.plot(result.image, filtered)
            st.plotly_chart(fig, use_container_width=True)

        with col_summary:
            if summary:
                st.markdown("**Detected animals:**")
                for label, count in sorted(summary.items()):
                    st.metric(label=label, value=count)
                st.caption(f"Total: {len(filtered)}")
            else:
                st.write("No animals detected.")


if __name__ == "__main__":
    main()
