import json
import streamlit as st

from loaders.csv_loader import load_csv
from loaders.json_loader import load_json, metrics_to_dataframe
from loaders.validate import validate_metrics_df
from loaders.normalize import normalize_experiment

from utils.io import ExperimentLoadError
from utils.tables import intersect_available_metrics

from analysis.advanced_summary import compute_advanced_summary

from plots.curves import plot_metric_curves, plot_metric_curves_matplotlib
from plots.export import export_figure_to_png


# ======================
# Metric tooltips
# ======================
METRIC_TOOLTIPS = {
    "accuracy": "Classification accuracy (higher is better).",
    "precision": "Positive predictive value.",
    "recall": "Sensitivity / true positive rate.",
    "f1": "Harmonic mean of precision and recall.",
    "dice": "Dice similarity coefficient (segmentation overlap).",
    "iou": "Intersection over Union.",
    "loss": "Training or validation loss (lower is better).",
}


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
    st.session_state.setdefault("selected_metrics", [])
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("grid_layout", 2)

    st.session_state.setdefault(
        "filters",
        {"model": [], "dataset": [], "task": [], "tags": [], "group": []},
    )

    st.session_state.setdefault("highlight", [])
    st.session_state.setdefault("pending_session", None)
    st.session_state.setdefault("session_uploader_key", 0)


# ======================
# Header
# ======================
def render_header():
    st.title("📊 Medical Imaging Experiment Comparison Dashboard (V2)")
    st.caption(
        "Filtering, multi-metric views, advanced summaries, highlighting, "
        "session save/load, and publication-ready exports."
    )
    st.divider()


# ======================
# Helpers
# ======================
def parse_tags(raw: str) -> list[str]:
    if not raw:
        return []
    return sorted({t.strip() for t in raw.split(",") if t.strip()})


# ======================
# Session Save / Load
# ======================
def serialize_session() -> dict:
    file_meta = {}
    for fname, meta in st.session_state["file_registry"].items():
        file_meta[fname] = {
            "experiment_name": meta.get("experiment_name"),
            "model": meta.get("model"),
            "dataset": meta.get("dataset"),
            "task": meta.get("task"),
            "tags_raw": meta.get("tags_raw"),
            "group": meta.get("group"),
            "saved": meta.get("saved"),
        }

    return {
        "version": "v2",
        "file_metadata": file_meta,
        "ui_state": {
            "filters": st.session_state["filters"],
            "highlight": st.session_state["highlight"],
            "selected_metrics": st.session_state["selected_metrics"],
            "grid_layout": st.session_state["grid_layout"],
        },
    }


def apply_session(payload: dict):
    st.session_state["filters"] = payload["ui_state"].get("filters", {})
    st.session_state["highlight"] = payload["ui_state"].get("highlight", [])
    st.session_state["selected_metrics"] = payload["ui_state"].get("selected_metrics", [])
    st.session_state["grid_layout"] = payload["ui_state"].get("grid_layout", 2)

    if not st.session_state["file_registry"]:
        st.session_state["pending_session"] = payload
        return

    for fname, meta in st.session_state["file_registry"].items():
        if fname in payload["file_metadata"]:
            m = payload["file_metadata"][fname]
            meta.update(m)
            meta["tags"] = parse_tags(meta.get("tags_raw", ""))

    load_experiments()


def render_session_controls():
    with st.expander("💾 Session", expanded=False):
        c1, c2 = st.columns(2)

        with c1:
            payload = serialize_session()
            st.download_button(
                "⬇️ Download session",
                json.dumps(payload, indent=2).encode(),
                "experiment_session.json",
                "application/json",
            )

        with c2:
            f = st.file_uploader(
                "Load session",
                type=["json"],
                label_visibility="collapsed",
                key=f"session_{st.session_state['session_uploader_key']}",
            )
            if f:
                apply_session(json.load(f))
                st.session_state["session_uploader_key"] += 1
                st.rerun()


# ======================
# File Manager
# ======================
def render_file_manager():
    st.header("📁 Manage Experiments")

    uploaded = st.file_uploader(
        "Add experiment files",
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
                    "tags_raw": "",
                    "tags": [],
                    "group": "",
                    "saved": False,
                }

        if st.session_state.get("pending_session"):
            apply_session(st.session_state["pending_session"])
            st.session_state["pending_session"] = None

        st.session_state["uploader_key"] += 1
        st.rerun()

    for fname, meta in list(st.session_state["file_registry"].items()):
        with st.container(border=True):
            cols = st.columns([1.5, 2, 2, 2, 2, 2, 0.7, 0.7])

            cols[0].markdown(f"**{fname}**")
            meta["experiment_name"] = cols[1].text_input("Experiment", meta["experiment_name"], key=f"exp_{fname}")
            meta["model"] = cols[2].text_input("Model", meta["model"], key=f"model_{fname}")
            meta["dataset"] = cols[3].text_input("Dataset", meta["dataset"], key=f"data_{fname}")
            meta["task"] = cols[4].selectbox("Task", ["classification", "segmentation"], key=f"task_{fname}")
            meta["tags_raw"] = cols[5].text_input("Tags", meta["tags_raw"], key=f"tags_{fname}")
            meta["tags"] = parse_tags(meta["tags_raw"])

            save = cols[6].button("💾", key=f"save_{fname}")
            delete = cols[7].button("🗑️", key=f"del_{fname}")

            if save:
                meta["saved"] = True
                load_experiments()
                st.rerun()

            if delete:
                del st.session_state["file_registry"][fname]
                load_experiments()
                st.rerun()


# ======================
# Loader
# ======================
def load_experiments():
    st.session_state["experiments"] = {}

    for meta in st.session_state["file_registry"].values():
        if not meta["saved"]:
            continue

        try:
            f = meta["file"]
            df = load_csv(f) if f.name.endswith(".csv") else metrics_to_dataframe(load_json(f)["metrics"])
            df = validate_metrics_df(df)

            exp = normalize_experiment(
                metrics_df=df,
                metadata={
                    "experiment_name": meta["experiment_name"],
                    "model": meta["model"],
                    "dataset": meta["dataset"],
                    "task": meta["task"],
                },
                source_file=f.name,
            )

            exp["tags"] = meta["tags"]
            exp["group"] = meta["group"]

            st.session_state["experiments"][exp["id"]] = exp

        except ExperimentLoadError as e:
            st.error(str(e))


# ======================
# Filters / Highlight / Metric selection
# ======================
def render_filters(experiments):
    with st.expander("🔎 Filters & Grouping", expanded=True):
        f = st.session_state["filters"]

        models = sorted({e["model"] for e in experiments if e["model"]})
        datasets = sorted({e["dataset"] for e in experiments if e["dataset"]})
        tasks = sorted({e["task"] for e in experiments})
        tags = sorted({t for e in experiments for t in e.get("tags", [])})
        groups = sorted({e.get("group") for e in experiments if e.get("group")})

        f["model"] = st.multiselect("Model", models, f["model"])
        f["dataset"] = st.multiselect("Dataset", datasets, f["dataset"])
        f["task"] = st.multiselect("Task", tasks, f["task"])
        f["tags"] = st.multiselect("Tags", tags, f["tags"])
        f["group"] = st.multiselect("Group", groups, f["group"])

        def keep(e):
            if f["model"] and e["model"] not in f["model"]:
                return False
            if f["dataset"] and e["dataset"] not in f["dataset"]:
                return False
            if f["task"] and e["task"] not in f["task"]:
                return False
            if f["group"] and e.get("group") not in f["group"]:
                return False
            if f["tags"] and not set(e.get("tags", [])).intersection(f["tags"]):
                return False
            return True

        return [e for e in experiments if keep(e)]


def render_highlight_controls(experiments):
    with st.expander("🎯 Highlight & Emphasis", expanded=False):
        names = [e["experiment_name"] for e in experiments]
        st.session_state["highlight"] = st.multiselect("Highlight", names, st.session_state["highlight"])


def render_metric_selection(experiments):
    with st.expander("📊 Metric Selection", expanded=True):
        metrics = sorted(intersect_available_metrics(experiments))
        selected = st.multiselect("Metrics", metrics, st.session_state["selected_metrics"] or metrics[:1])
        st.session_state["selected_metrics"] = selected
        st.session_state["selected_metric"] = selected[0] if selected else None
        st.session_state["grid_layout"] = st.selectbox("Grid columns", [1, 2, 3], index=1)
        return selected


# ======================
# Summary / Plots / Export
# ======================
def render_summary(experiments, metric):
    st.subheader("📋 Advanced Summary")
    if not metric:
        return
    df = compute_advanced_summary(experiments, metric)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_metric_grid(experiments, metrics):
    highlight = st.session_state["highlight"]
    ncols = st.session_state["grid_layout"]

    for i in range(0, len(metrics), ncols):
        cols = st.columns(ncols)
        for c, m in zip(cols, metrics[i:i+ncols]):
            title = f"{m} ↓" if "loss" in m.lower() else f"{m} ↑"
            tooltip = METRIC_TOOLTIPS.get(m.lower(), "")
            with c:
                st.markdown(f"### {title}", help=tooltip)
                fig = plot_metric_curves(experiments, m, highlight=highlight)
                st.plotly_chart(fig, use_container_width=True)


def render_export_section(experiments, metrics):
    st.subheader("🖨️ Export Publication Plots")
    for m in metrics:
        fig = plot_metric_curves_matplotlib(experiments, m, highlight=st.session_state["highlight"])
        png = export_figure_to_png(fig)
        st.download_button(f"⬇️ {m}.png", png, f"{m}_comparison.png", "image/png")


# ======================
# Main
# ======================
def main():
    init_page()
    init_state()
    render_header()
    render_session_controls()
    render_file_manager()

    saved = list(st.session_state["experiments"].values())

    left, right = st.columns([1.2, 3.2], gap="large")

    with left:
        filtered = render_filters(saved)
        render_highlight_controls(filtered)
        metrics = render_metric_selection(filtered)

    with right:
        render_summary(filtered, st.session_state["selected_metric"])
        render_metric_grid(filtered, metrics)
        render_export_section(filtered, metrics)


if __name__ == "__main__":
    main()
