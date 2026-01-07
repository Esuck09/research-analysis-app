# app.py
import json
import streamlit as st

from loaders.csv_loader import load_csv
from loaders.json_loader import load_json, metrics_to_dataframe
from loaders.validate import validate_metrics_df
from loaders.normalize import normalize_experiment

from utils.io import ExperimentLoadError
from utils.tables import intersect_available_metrics

from analysis.advanced_summary import compute_advanced_summary
from analysis.aggregation import aggregate_runs
from analysis.statistics import compute_pairwise_stats
from analysis.robustness import compute_robustness_metrics
from analysis.dynamics import compute_learning_dynamics

from plots.curves import (
    plot_metric_curves,
    plot_metric_curves_matplotlib,
    plot_aggregated_curves,
)
from plots.export import export_figure_to_png


# =========================================================
# Metric tooltips
# =========================================================
METRIC_TOOLTIPS = {
    "accuracy": "Classification accuracy (higher is better).",
    "precision": "Positive predictive value.",
    "recall": "Sensitivity / true positive rate.",
    "f1": "Harmonic mean of precision and recall.",
    "dice": "Dice similarity coefficient (segmentation overlap).",
    "iou": "Intersection over Union.",
    "loss": "Training or validation loss (lower is better).",
}


# =========================================================
# Page & State
# =========================================================
def init_page():
    st.set_page_config(
        page_title="Medical Imaging Experiment Comparison",
        page_icon="📊",
        layout="wide",
    )


def init_state():
    defaults = {
        "file_registry": {},
        "experiments": {},
        "selected_metric": None,
        "selected_metrics": [],
        "uploader_key": 0,
        "grid_layout": 2,
        "filters": {"model": [], "dataset": [], "task": [], "tags": [], "group": []},
        "highlight": [],
        "pending_session": None,
        "session_uploader_key": 0,
        # V3
        "show_aggregates": False,
        "aggregation_keys": ["model", "dataset"],
        "alignment_mode": "intersection",
        "alignment_last_n": 5,
        "use_interpolation": False,
        "grid_mode": "common_range",
        "grid_step": 1,
        "show_ci": True,
        "n_boot": 1000,
        "ci_level": 0.95,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# =========================================================
# Header
# =========================================================
def render_header():
    st.title("📊 Medical Imaging Experiment Comparison Dashboard (V3)")
    st.caption(
        "Compare experiments with filtering, aggregation, alignment, "
        "robustness, learning dynamics, and statistical analysis."
    )
    st.divider()


# =========================================================
# Helpers
# =========================================================
def parse_tags(raw: str) -> list[str]:
    if not raw:
        return []
    return sorted({t.strip() for t in raw.split(",") if t.strip()})


# =========================================================
# Session controls
# =========================================================
def serialize_session():
    return {
        "version": "v3",
        "file_metadata": {
            f: {
                "experiment_name": m["experiment_name"],
                "model": m["model"],
                "dataset": m["dataset"],
                "task": m["task"],
                "tags_raw": m["tags_raw"],
                "group": m["group"],
                "saved": m["saved"],
            }
            for f, m in st.session_state["file_registry"].items()
        },
        "ui_state": {
            k: st.session_state[k]
            for k in [
                "filters",
                "highlight",
                "selected_metrics",
                "grid_layout",
                "show_aggregates",
                "aggregation_keys",
                "alignment_mode",
                "alignment_last_n",
                "use_interpolation",
                "grid_mode",
                "grid_step",
            ]
        },
    }


def apply_session(payload: dict):
    ui = payload.get("ui_state", {})
    for k, v in ui.items():
        st.session_state[k] = v

    if not st.session_state["file_registry"]:
        st.session_state["pending_session"] = payload
        return

    for fname, meta in st.session_state["file_registry"].items():
        if fname in payload.get("file_metadata", {}):
            m = payload["file_metadata"][fname]
            meta.update(m)
            meta["tags"] = parse_tags(meta.get("tags_raw", ""))

    load_experiments()

# =========================================================
# Controls (Filters / Highlight / Metric selection)
# =========================================================
def render_filters(experiments: list[dict]) -> list[dict]:
    with st.expander("🔎 Filters", expanded=True):
        if not experiments:
            st.info("Save experiments to enable filters.")
            return []

        f = st.session_state["filters"]

        models = sorted({e.get("model") for e in experiments if e.get("model")})
        datasets = sorted({e.get("dataset") for e in experiments if e.get("dataset")})
        tasks = sorted({e.get("task") for e in experiments if e.get("task")})
        tags = sorted({t for e in experiments for t in (e.get("tags") or [])})
        groups = sorted({e.get("group") for e in experiments if e.get("group")})

        # Vertical filters
        f["model"] = st.multiselect("Model", models, default=f.get("model", []))
        f["dataset"] = st.multiselect("Dataset", datasets, default=f.get("dataset", []))
        f["task"] = st.multiselect("Task", tasks, default=f.get("task", []))
        f["group"] = st.multiselect("Group", groups, default=f.get("group", []))
        f["tags"] = st.multiselect("Tags", tags, default=f.get("tags", []))

        def keep(e: dict) -> bool:
            if f["model"] and e.get("model") not in f["model"]:
                return False
            if f["dataset"] and e.get("dataset") not in f["dataset"]:
                return False
            if f["task"] and e.get("task") not in f["task"]:
                return False
            if f["group"] and (e.get("group") or "") not in f["group"]:
                return False
            if f["tags"]:
                if not set(e.get("tags") or []).intersection(set(f["tags"])):
                    return False
            return True

        filtered = [e for e in experiments if keep(e)]
        st.caption(f"Showing **{len(filtered)} / {len(experiments)}** experiments.")
        return filtered


def render_highlight_controls(experiments: list[dict]) -> None:
    with st.expander("🎯 Highlight", expanded=False):
        if not experiments:
            st.info("No experiments available.")
            st.session_state["highlight"] = []
            return

        names = [e["experiment_name"] for e in experiments]
        st.session_state["highlight"] = st.multiselect(
            "Highlight experiments",
            options=names,
            default=st.session_state.get("highlight", []),
        )


def render_metric_selection(experiments: list[dict]) -> list[str]:
    with st.expander("📊 Metrics", expanded=True):
        if not experiments:
            st.info("No experiments available.")
            st.session_state["selected_metrics"] = []
            st.session_state["selected_metric"] = None
            return []

        available = sorted(intersect_available_metrics(experiments))
        if not available:
            st.warning("No common metrics across selected experiments.")
            st.session_state["selected_metrics"] = []
            st.session_state["selected_metric"] = None
            return []

        default_sel = st.session_state.get("selected_metrics") or [available[0]]
        selected = st.multiselect("Select metrics", available, default=default_sel)
        st.session_state["selected_metrics"] = selected
        st.session_state["selected_metric"] = selected[0] if selected else None

        st.session_state["grid_layout"] = int(
            st.selectbox("Grid columns", [1, 2, 3], index=1)
        )
        return selected


# =========================================================
# V3 Controls (Alignment / Aggregation / Interpolation)
# =========================================================
def render_alignment_controls() -> None:
    with st.expander("⏱ Epoch Alignment", expanded=False):
        mode = st.selectbox(
            "Alignment strategy",
            options=["intersection", "truncate", "last_n"],
            index=["intersection", "truncate", "last_n"].index(
                st.session_state.get("alignment_mode", "intersection")
            ),
            help="How to align epochs before aggregating (no interpolation).",
        )
        st.session_state["alignment_mode"] = mode

        if mode == "last_n":
            st.session_state["alignment_last_n"] = st.number_input(
                "Last N epochs",
                min_value=1,
                step=1,
                value=int(st.session_state.get("alignment_last_n", 5)),
            )


def render_aggregation_controls() -> None:
    with st.expander("📊 Aggregation", expanded=False):
        st.session_state["show_aggregates"] = st.toggle(
            "Show aggregated curves (mean ± std)",
            value=bool(st.session_state.get("show_aggregates", False)),
        )
        st.session_state["aggregation_keys"] = st.multiselect(
            "Aggregate by",
            options=["model", "dataset", "group", "task"],
            default=st.session_state.get("aggregation_keys", ["model", "dataset"]),
        )


def render_interpolation_controls() -> None:
    with st.expander("📐 Interpolation", expanded=False):
        st.session_state["use_interpolation"] = st.toggle(
            "Enable interpolation (linear)",
            value=bool(st.session_state.get("use_interpolation", False)),
            help="Resample curves onto a shared grid. Must be disclosed in papers.",
        )

        if st.session_state["use_interpolation"]:
            st.session_state["grid_mode"] = st.selectbox(
                "Grid range",
                options=["common_range", "union_range"],
                index=0 if st.session_state.get("grid_mode", "common_range") == "common_range" else 1,
            )
            st.session_state["grid_step"] = st.number_input(
                "Grid step",
                min_value=1,
                step=1,
                value=int(st.session_state.get("grid_step", 1)),
            )


# =========================================================
# Panels (Stats / Robustness / Dynamics)
# =========================================================
def render_statistics_panel(experiments: list[dict]) -> None:
    with st.expander("🧪 Statistics", expanded=False):
        if not experiments:
            st.info("No experiments.")
            return
        metric = st.session_state.get("selected_metric")
        if not metric:
            st.info("Select a metric first.")
            return

        c1, c2, c3 = st.columns([1, 1, 1])
        group_field = c1.selectbox("Compare groups by", ["model", "dataset", "group", "task"], index=0)
        value_mode = c2.selectbox("Value mode", ["final", "best", "auc"], index=0)
        pairing_field = c3.selectbox("Pair on (optional)", ["None", "group", "dataset", "task"], index=0)
        pairing_field = None if pairing_field == "None" else pairing_field

        df = compute_pairwise_stats(
            experiments=experiments,
            metric=metric,
            group_field=group_field,
            value_mode=value_mode,
            pairing_field=pairing_field,
        )
        if df.empty:
            st.warning("Not enough data for stats with these settings.")
            return

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download stats CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"stats_{metric}_{group_field}_{value_mode}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_robustness_panel(experiments: list[dict]) -> None:
    with st.expander("🧬 Robustness", expanded=False):
        if not experiments:
            st.info("No experiments.")
            return
        metric = st.session_state.get("selected_metric")
        if not metric:
            st.info("Select a metric first.")
            return

        tail_fraction = st.slider("Late-epoch fraction", 0.10, 0.50, 0.25, 0.05)
        plateau_tol = st.number_input("Plateau tolerance", min_value=1e-6, max_value=1e-2, value=1e-3, format="%.1e")

        df = compute_robustness_metrics(
            experiments=experiments,
            metric=metric,
            tail_fraction=tail_fraction,
            plateau_tol=plateau_tol,
        )
        if df.empty:
            st.warning("Not enough data for robustness metrics.")
            return

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download robustness CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{metric}_robustness.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_learning_dynamics_panel(experiments: list[dict]) -> None:
    with st.expander("📉 Learning Dynamics", expanded=False):
        if not experiments:
            st.info("No experiments.")
            return

        base_metric = st.selectbox("Base metric", ["loss", "accuracy", "dice", "iou", "f1"], index=0)
        tail_fraction = st.slider("Tail fraction (slope window)", 0.10, 0.50, 0.25, 0.05)

        df = compute_learning_dynamics(
            experiments=experiments,
            base_metric=base_metric,
            tail_fraction=tail_fraction,
        )
        if df.empty:
            st.warning("No matching train/val columns found (expected train_<metric>, val_<metric>).")
            return

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download learning dynamics CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"learning_dynamics_{base_metric}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# Summary / Plots / Export
# =========================================================
def render_summary(experiments: list[dict], metric: str | None) -> None:
    st.subheader("📋 Advanced Summary")
    if not experiments:
        st.info("No experiments to summarize.")
        return
    if not metric:
        st.info("Select a metric to view the summary.")
        return

    df = compute_advanced_summary(experiments, metric)
    if df.empty:
        st.warning("No summary available for this metric.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_metric_grid(experiments: list[dict], metrics: list[str]) -> None:
    if not experiments:
        st.info("No experiments to plot.")
        return
    if not metrics:
        st.info("Select at least one metric.")
        return

    highlight = st.session_state.get("highlight", [])
    ncols = max(1, min(3, int(st.session_state.get("grid_layout", 2))))

    for i in range(0, len(metrics), ncols):
        row = metrics[i : i + ncols]
        cols = st.columns(len(row))

        for c, m in zip(cols, row):
            with c:
                # Title per plot
                title = f"{m} ↓" if "loss" in m.lower() else f"{m} ↑"
                tooltip = METRIC_TOOLTIPS.get(m.lower(), "")
                st.markdown(f"### {title}", help=tooltip)

                fig = plot_metric_curves(experiments, m, highlight=highlight)

                if st.session_state.get("show_aggregates"):
                    aggregated = aggregate_runs(
                        experiments=experiments,
                        metric=m,
                        group_by=st.session_state.get("aggregation_keys", []),
                        alignment_mode=st.session_state.get("alignment_mode", "intersection"),
                        last_n=int(st.session_state.get("alignment_last_n", 5)),
                        use_interpolation=bool(st.session_state.get("use_interpolation", False)),
                        grid_mode=st.session_state.get("grid_mode", "common_range"),
                        grid_step=int(st.session_state.get("grid_step", 1)),
                        interp_kind="linear",
                    )
                    plot_aggregated_curves(fig, aggregated, m)

                    if st.session_state.get("use_interpolation", False):
                        st.caption("⚠ Aggregates are **interpolated** (linear).")
                    elif st.session_state.get("alignment_mode") != "intersection":
                        st.caption(f"⚠ Aggregates use '{st.session_state['alignment_mode']}' alignment.")

                st.plotly_chart(fig, use_container_width=True)


def render_export_section(experiments: list[dict], metrics: list[str]) -> None:
    st.subheader("🖨️ Export Publication Plots")
    if not experiments or not metrics:
        st.info("Select experiments and metrics to enable export.")
        return

    highlight = st.session_state.get("highlight", [])
    for m in metrics:
        fig = plot_metric_curves_matplotlib(experiments, m, highlight=highlight)
        png = export_figure_to_png(fig)
        st.download_button(
            f"⬇️ {m}.png",
            png,
            file_name=f"{m}_comparison.png",
            mime="image/png",
            use_container_width=True,
        )


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
                use_container_width=True,
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


# =========================================================
# File manager & loader
# =========================================================
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
            st.session_state["file_registry"].setdefault(
                f.name,
                {
                    "file": f,
                    "experiment_name": f.name.rsplit(".", 1)[0],
                    "model": "",
                    "dataset": "",
                    "task": "classification",
                    "tags_raw": "",
                    "tags": [],
                    "group": "",
                    "saved": False,
                },
            )

        if st.session_state.get("pending_session"):
            apply_session(st.session_state["pending_session"])
            st.session_state["pending_session"] = None

        st.session_state["uploader_key"] += 1
        st.rerun()

    if not st.session_state["file_registry"]:
        st.info("Add experiment files to begin.")
        return

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

            if cols[6].button("💾", key=f"save_{fname}"):
                meta["saved"] = True
                load_experiments()
                st.rerun()

            if cols[7].button("🗑️", key=f"del_{fname}"):
                del st.session_state["file_registry"][fname]
                load_experiments()
                st.rerun()


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


# =========================================================
# Main
# =========================================================
def main():
    init_page()
    init_state()
    render_header()

    render_session_controls()
    render_file_manager()

    experiments = list(st.session_state["experiments"].values())
    if not experiments:
        return

    left, right = st.columns([1.2, 3.2], gap="large")

    with left:
        filtered = render_filters(experiments)
        render_highlight_controls(filtered)
        metrics = render_metric_selection(filtered)

        render_alignment_controls()
        render_aggregation_controls()
        render_interpolation_controls()

        render_statistics_panel(filtered)
        render_robustness_panel(filtered)
        render_learning_dynamics_panel(filtered)

    with right:
        render_summary(filtered, st.session_state["selected_metric"])
        render_metric_grid(filtered, metrics)
        render_export_section(filtered, metrics)


if __name__ == "__main__":
    main()
