import streamlit as st
import pandas as pd

from loaders.csv_loader import load_csv
from loaders.json_loader import load_json, metrics_to_dataframe
from loaders.validate import validate_metrics_df
from loaders.normalize import normalize_experiment

from utils.io import ExperimentLoadError, dataframe_to_csv_buffer
from utils.tables import build_summary_table, intersect_available_metrics

from plots.curves import plot_metric_curves, plot_metric_curves_matplotlib
from plots.export import export_figure_to_png


# ======================
# Page & State
# ======================
def init_page():
    st.set_page_config(
        page_title="Medical Imaging Experiment Comparison",
        page_icon="📊",
        layout="wide",
    )


def init_state():
    st.session_state.setdefault("file_registry", {})
    st.session_state.setdefault("experiments", {})
    st.session_state.setdefault("selected_metric", None)
    st.session_state.setdefault("uploader_key", 0)  # 🔑 clears uploader


# ======================
# Header
# ======================
def render_header():
    st.title("📊 Medical Imaging Experiment Comparison Dashboard")
    st.caption(
        "Upload experiments, manage metadata, compare metrics across epochs, "
        "and export publication-ready results."
    )
    st.divider()


# ======================
# File Manager
# ======================
def render_file_manager():
    st.header("📁 Manage Experiments")

    # ---- Add file (direct file picker, auto-clears) ----
    uploaded = st.file_uploader(
        "➕ Add Experiment File",
        type=["csv", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state['uploader_key']}",
    )

    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state["file_registry"]:
                st.session_state["file_registry"][f.name] = {
                    "file": f,
                    "experiment_name": f.name.rsplit(".", 1)[0],
                    "model": "",
                    "dataset": "",
                    "task": "classification",
                    "saved": False,  # 🔑 prevents premature loading
                }

        # 🔑 clear uploader UI state
        st.session_state["uploader_key"] += 1
        st.rerun()

    if not st.session_state["file_registry"]:
        st.info("No experiment files added yet.")
        return

    # ---- Experiment Rows ----
    for fname, meta in list(st.session_state["file_registry"].items()):
        with st.container(border=True):
            cols = st.columns([1.8, 2.2, 2, 2, 2, 0.7, 0.7])

            status = "✅" if meta["saved"] else "📝"
            cols[0].markdown(f"**{status} {fname}**")

            exp_name = cols[1].text_input(
                "Experiment",
                meta["experiment_name"],
                key=f"exp_{fname}",
            )

            model = cols[2].text_input(
                "Model",
                meta["model"],
                key=f"model_{fname}",
            )

            dataset = cols[3].text_input(
                "Dataset",
                meta["dataset"],
                key=f"dataset_{fname}",
            )

            task = cols[4].selectbox(
                "Task",
                ["classification", "segmentation"],
                index=0 if meta["task"] == "classification" else 1,
                key=f"task_{fname}",
            )

            has_changes = (
                exp_name != meta["experiment_name"]
                or model != meta["model"]
                or dataset != meta["dataset"]
                or task != meta["task"]
                or not meta["saved"]
            )

            save_clicked = cols[5].button(
                "💾",
                key=f"save_{fname}",
                help="Save experiment",
                disabled=not has_changes,
            )

            delete_clicked = cols[6].button(
                "🗑️",
                key=f"delete_{fname}",
                help="Delete experiment",
            )

            if save_clicked:
                meta.update(
                    {
                        "experiment_name": exp_name.strip(),
                        "model": model.strip(),
                        "dataset": dataset.strip(),
                        "task": task,
                        "saved": True,
                    }
                )
                load_experiments()
                st.rerun()

            if delete_clicked:
                del st.session_state["file_registry"][fname]
                load_experiments()
                st.rerun()


# ======================
# Loader (ONLY saved rows)
# ======================
def load_experiments():
    st.session_state["experiments"] = {}

    for meta in st.session_state["file_registry"].values():
        if not meta.get("saved", False):
            continue  # 🔑 skip unsaved rows

        try:
            file = meta["file"]

            if file.name.endswith(".csv"):
                df = load_csv(file)
            else:
                content = load_json(file)
                df = metrics_to_dataframe(content["metrics"])

            df = validate_metrics_df(df)

            exp = normalize_experiment(
                metrics_df=df,
                metadata={
                    "experiment_name": meta["experiment_name"],
                    "model": meta["model"],
                    "dataset": meta["dataset"],
                    "task": meta["task"],
                },
                source_file=file.name,
            )

            st.session_state["experiments"][exp["id"]] = exp

        except ExperimentLoadError as e:
            st.error(f"❌ {file.name}: {e}")


# ======================
# Metric Selection
# ======================
def render_metric_selection():
    st.header("📊 Metric Selection")

    experiments = list(st.session_state["experiments"].values())
    if not experiments:
        st.info("Save at least one experiment to enable metrics.")
        return

    metrics = sorted(intersect_available_metrics(experiments))
    if not metrics:
        st.warning("No common metrics available.")
        return

    st.session_state["selected_metric"] = st.selectbox(
        "Select metric",
        metrics,
    )


# ======================
# Summary
# ======================
def render_summary():
    st.header("📋 Experiment Summary")

    metric = st.session_state.get("selected_metric")
    if not metric:
        return

    df = build_summary_table(
        list(st.session_state["experiments"].values()), metric
    )

    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = dataframe_to_csv_buffer(df)
    st.download_button(
        "⬇️ Download summary CSV",
        csv,
        file_name=f"{metric}_summary.csv",
        mime="text/csv",
    )


# ======================
# Plots
# ======================
def render_plots():
    st.header("📈 Metric vs Epoch")

    metric = st.session_state.get("selected_metric")
    if not metric:
        return

    experiments = list(st.session_state["experiments"].values())

    fig = plot_metric_curves(experiments, metric)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🖨️ Export Publication Plot")

    fig_pub = plot_metric_curves_matplotlib(experiments, metric)
    png = export_figure_to_png(fig_pub)

    st.download_button(
        "⬇️ Download PNG (300 DPI)",
        png,
        file_name=f"{metric}_comparison.png",
        mime="image/png",
    )


# ======================
# Main
# ======================
def main():
    init_page()
    init_state()
    render_header()
    render_file_manager()
    st.divider()
    render_metric_selection()
    render_summary()
    render_plots()


if __name__ == "__main__":
    main()
