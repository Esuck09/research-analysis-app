import streamlit as st
import pandas as pd

# ======================
# Backend imports
# ======================
from loaders.csv_loader import load_csv
from loaders.json_loader import load_json, metrics_to_dataframe
from loaders.validate import validate_metrics_df
from loaders.normalize import normalize_experiment
from utils.io import (
    ExperimentLoadError,
    dataframe_to_csv_buffer,
)
from utils.tables import (
    build_summary_table,
    intersect_available_metrics,
)

# ======================
# Plotting imports
# ======================
from plots.curves import (
    plot_metric_curves,
    plot_metric_curves_matplotlib,
)
from plots.export import export_figure_to_png


# ======================
# Page & session setup
# ======================
def init_page() -> None:
    st.set_page_config(
        page_title="Medical Imaging Experiment Comparison",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            section[data-testid="stSidebar"] {
                min-width: 300px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    if "experiments" not in st.session_state:
        st.session_state["experiments"] = {}

    if "selected_metric" not in st.session_state:
        st.session_state["selected_metric"] = None

    if "file_metadata" not in st.session_state:
        st.session_state["file_metadata"] = {}

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []


# ======================
# Header
# ======================
def render_header() -> None:
    st.title("📊 Medical Imaging Experiment Comparison Dashboard")
    st.caption(
        "Compare classification and segmentation experiments. "
        "Visualize metrics across epochs and export publication-ready results."
    )
    st.divider()


# ======================
# Sidebar
# ======================
def render_sidebar() -> None:
    with st.sidebar:
        st.header("📥 Upload Experiments")
        st.markdown(
            "Upload experiment logs and provide metadata for comparison."
        )

        uploaded_files = st.file_uploader(
            "Upload CSV or JSON experiment files",
            type=["csv", "json"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for file in uploaded_files:
                st.markdown("---")
                st.subheader(f"📄 {file.name}")

                if file.name not in st.session_state["file_metadata"]:
                    st.session_state["file_metadata"][file.name] = {
                        "experiment_name": file.name.rsplit(".", 1)[0],
                        "model": "",
                        "dataset": "",
                        "task": "classification",
                        "notes": "",
                    }

                meta = st.session_state["file_metadata"][file.name]

                meta["experiment_name"] = st.text_input(
                    "Experiment name",
                    value=meta["experiment_name"],
                    key=f"exp_name_{file.name}",
                )

                meta["model"] = st.text_input(
                    "Model name",
                    value=meta["model"],
                    key=f"model_{file.name}",
                )

                meta["dataset"] = st.text_input(
                    "Dataset",
                    value=meta["dataset"],
                    key=f"dataset_{file.name}",
                )

                meta["task"] = st.selectbox(
                    "Task type",
                    ["classification", "segmentation"],
                    index=0 if meta["task"] == "classification" else 1,
                    key=f"task_{file.name}",
                )

                meta["notes"] = st.text_area(
                    "Notes (optional)",
                    value=meta["notes"],
                    key=f"notes_{file.name}",
                )

        st.session_state["uploaded_files"] = uploaded_files or []

        st.markdown("---")
        st.caption("All data stays local • No uploads • No tracking")


# ======================
# Safe loader integration
# ======================
def load_experiments_from_sidebar() -> None:
    uploaded_files = st.session_state.get("uploaded_files", [])
    metadata_map = st.session_state.get("file_metadata", {})

    for file in uploaded_files:
        meta = metadata_map.get(file.name)
        if not meta:
            continue

        try:
            if file.name.endswith(".csv"):
                df = load_csv(file)
            elif file.name.endswith(".json"):
                content = load_json(file)
                df = metrics_to_dataframe(content["metrics"])
            else:
                raise ExperimentLoadError("Unsupported file type.")

            df = validate_metrics_df(df)

            experiment = normalize_experiment(
                metrics_df=df,
                metadata=meta,
                source_file=file.name,
            )

            st.session_state["experiments"][experiment["id"]] = experiment

        except ExperimentLoadError as e:
            st.error(f"❌ {file.name}: {e}")


# ======================
# Metric selector
# ======================
def render_metric_selector() -> None:
    with st.sidebar:
        st.header("📊 Metric Selection")

        experiments = list(st.session_state["experiments"].values())
        if not experiments:
            st.info("Load experiments to select metrics.")
            return

        tasks = {exp["task"] for exp in experiments}
        if len(tasks) > 1:
            st.warning(
                "Metric selection disabled: mixed task types detected."
            )
            st.session_state["selected_metric"] = None
            return

        available_metrics = sorted(
            intersect_available_metrics(experiments)
        )

        if not available_metrics:
            st.warning("No common metrics available.")
            st.session_state["selected_metric"] = None
            return

        st.session_state["selected_metric"] = st.selectbox(
            "Select metric",
            available_metrics,
        )


# ======================
# Summary table
# ======================
def render_experiment_summary() -> None:
    st.header("📋 Experiment Summary")

    experiments = list(st.session_state["experiments"].values())
    metric = st.session_state.get("selected_metric")

    if not experiments:
        st.info("No experiments loaded yet.")
        return

    if not metric:
        st.info("Select a metric to view summary statistics.")
        return

    summary_df = build_summary_table(experiments, metric)
    if summary_df.empty:
        st.warning(f"No experiments contain metric '{metric}'.")
        return

    summary_df = summary_df[
        [
            "experiment_name",
            "model",
            "dataset",
            "task",
            "best_value",
            "best_epoch",
            "final_value",
        ]
    ]

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "best_value": st.column_config.NumberColumn(format="%.4f"),
            "final_value": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    csv_buffer = dataframe_to_csv_buffer(summary_df)
    st.download_button(
        "⬇️ Download summary CSV",
        csv_buffer,
        file_name=f"{metric}_summary.csv",
        mime="text/csv",
    )


# ======================
# Interactive plot
# ======================
def render_metric_plot() -> None:
    st.header("📈 Metric vs Epoch")

    experiments = list(st.session_state["experiments"].values())
    metric = st.session_state.get("selected_metric")

    if not experiments or not metric:
        st.info("Load experiments and select a metric to view plots.")
        return

    valid_experiments = [
        exp
        for exp in experiments
        if metric in exp["metrics_df"].columns
        and not exp["metrics_df"][metric].isna().all()
    ]

    if not valid_experiments:
        st.warning(f"No valid data for metric '{metric}'.")
        return

    with st.spinner("Rendering plot…"):
        fig = plot_metric_curves(valid_experiments, metric)
        st.plotly_chart(fig, use_container_width=True)


# ======================
# Comparison plot + export
# ======================
def render_comparison_plot() -> None:
    st.header("🖨️ Comparison Plot (Publication Style)")

    experiments = list(st.session_state["experiments"].values())
    metric = st.session_state.get("selected_metric")

    if not experiments or not metric:
        st.info("Load experiments and select a metric to export plots.")
        return

    fig = plot_metric_curves_matplotlib(experiments, metric)
    st.pyplot(fig)

    png_buffer = export_figure_to_png(fig)

    st.download_button(
        "⬇️ Download PNG (300 DPI)",
        png_buffer,
        file_name=f"{metric}_comparison.png",
        mime="image/png",
    )


# ======================
# Main
# ======================
def main() -> None:
    init_page()
    init_session_state()
    render_header()
    render_sidebar()
    load_experiments_from_sidebar()
    render_metric_selector()

    col1, col2 = st.columns([1.1, 1.4], gap="large")

    with col1:
        render_experiment_summary()

    with col2:
        render_metric_plot()

    st.divider()
    render_comparison_plot()


if __name__ == "__main__":
    main()
