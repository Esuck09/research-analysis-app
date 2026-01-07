import streamlit as st
from utils.tables import build_summary_table
import pandas as pd
from utils.tables import intersect_available_metrics
from plots.curves import plot_metric_curves, plot_metric_curves_matplotlib
from plots.export import export_figure_to_png

def init_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Medical Imaging Experiment Comparison",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "experiments" not in st.session_state:
        st.session_state["experiments"] = {}

    if "selected_experiment_ids" not in st.session_state:
        st.session_state["selected_experiment_ids"] = []

    if "selected_metric" not in st.session_state:
        st.session_state["selected_metric"] = None

def render_header() -> None:
    """Render application header."""
    st.title("📊 Medical Imaging Experiment Comparison Dashboard")
    st.markdown(
        """
        Upload experiment result files and compare **classification** and
        **segmentation** metrics across models and datasets.

        - Supports CSV and JSON logs  
        - Visualize metrics vs epoch  
        - Export publication-ready figures
        """
    )

def render_sidebar() -> None:
    """Render sidebar for file upload and metadata input."""
    with st.sidebar:
        st.header("📥 Upload Experiments")

        uploaded_files = st.file_uploader(
            "Upload CSV or JSON experiment files",
            type=["csv", "json"],
            accept_multiple_files=True,
        )

        if "file_metadata" not in st.session_state:
            st.session_state["file_metadata"] = {}

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
                    options=["classification", "segmentation"],
                    index=0 if meta["task"] == "classification" else 1,
                    key=f"task_{file.name}",
                )

                meta["notes"] = st.text_area(
                    "Notes (optional)",
                    value=meta["notes"],
                    key=f"notes_{file.name}",
                )

        st.session_state["uploaded_files"] = uploaded_files

def render_main_placeholders() -> None:
    """Main page placeholder sections."""
    st.header("📋 Experiment Summary")
    st.info("Summary table will be rendered here.")

    st.header("📈 Metric vs Epoch")
    st.info("Interactive and exportable plots will be rendered here.")

    st.header("📤 Export")
    st.info("Download plot PNGs and summary CSVs here.")

def render_experiment_summary() -> None:
    """Render experiment summary table."""
    st.header("📋 Experiment Summary")

    experiments = list(st.session_state.get("experiments", {}).values())
    selected_metric = st.session_state.get("selected_metric")

    if not experiments:
        st.info("No experiments loaded yet.")
        return

    if not selected_metric:
        st.info("Select a metric to view summary statistics.")
        return

    summary_df = build_summary_table(experiments, selected_metric)

    if summary_df.empty:
        st.warning(
            f"No experiments contain the metric '{selected_metric}'."
        )
        return

    # Improve column ordering for readability
    column_order = [
        "experiment_name",
        "model",
        "dataset",
        "task",
        "best_value",
        "best_epoch",
        "final_value",
    ]
    summary_df = summary_df[column_order]

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

def render_metric_selector() -> None:
    """Render metric selection controls in sidebar."""
    with st.sidebar:
        st.header("📊 Metric Selection")

        experiments = list(
            st.session_state.get("experiments", {}).values()
        )

        if not experiments:
            st.info("Load experiments to select metrics.")
            return

        # Enforce single-task selection
        tasks = {exp["task"] for exp in experiments}
        if len(tasks) > 1:
            st.warning(
                "Metric selection disabled: mixed task types detected "
                "(classification + segmentation)."
            )
            st.session_state["selected_metric"] = None
            return

        available_metrics = sorted(
            intersect_available_metrics(experiments)
        )

        if not available_metrics:
            st.warning(
                "No common metrics available across all experiments."
            )
            st.session_state["selected_metric"] = None
            return

        selected_metric = st.selectbox(
            "Select metric",
            options=available_metrics,
            index=0,
        )

        st.session_state["selected_metric"] = selected_metric

def render_metric_plot() -> None:
    """Render metric vs epoch plot."""
    st.header("📈 Metric vs Epoch")

    experiments = list(st.session_state.get("experiments", {}).values())
    selected_metric = st.session_state.get("selected_metric")

    if not experiments or not selected_metric:
        st.info("Load experiments and select a metric to view plots.")
        return

    fig = plot_metric_curves(experiments, selected_metric)

    if not fig.data:
        st.warning(
            f"No data available to plot metric '{selected_metric}'."
        )
        return

    st.plotly_chart(fig, use_container_width=True)

def render_comparison_plot() -> None:
    """Render and export publication-style comparison plot."""
    st.header("🖨️ Comparison Plot (Publication Style)")

    experiments = list(st.session_state.get("experiments", {}).values())
    selected_metric = st.session_state.get("selected_metric")

    if not experiments or not selected_metric:
        st.info(
            "Load experiments and select a metric to view and export plots."
        )
        return

    fig = plot_metric_curves_matplotlib(experiments, selected_metric)

    st.pyplot(fig)

    png_buffer = export_figure_to_png(fig)

    filename = f"{selected_metric}_comparison.png"

    st.download_button(
        label="⬇️ Download PNG (300 DPI)",
        data=png_buffer,
        file_name=filename,
        mime="image/png",
    )

def main() -> None:
    init_page()
    init_session_state()
    render_header()
    render_sidebar()
    render_metric_selector()
    render_experiment_summary()
    render_metric_plot()  
    render_comparison_plot() 


if __name__ == "__main__":
    main()
