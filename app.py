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

    # V3 aggregation
    st.session_state.setdefault("show_aggregates", False)
    st.session_state.setdefault("aggregation_keys", ["model", "dataset"])

    # V3 alignment
    st.session_state.setdefault("alignment_mode", "intersection")
    st.session_state.setdefault("alignment_last_n", 5)

    # V3 interpolation
    st.session_state.setdefault("use_interpolation", False)
    st.session_state.setdefault("grid_mode", "common_range")
    st.session_state.setdefault("grid_step", 1)


# ======================
# Header
# ======================
def render_header():
    st.title("📊 Medical Imaging Experiment Comparison Dashboard (V3)")
    st.caption(
        "Filtering, multi-metric views, advanced summaries, highlighting, "
        "session save/load, aggregation, alignment, interpolation, statistics, "
        "and publication-ready exports."
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
        "version": "v3",
        "file_metadata": file_meta,
        "ui_state": {
            "filters": st.session_state["filters"],
            "highlight": st.session_state["highlight"],
            "selected_metrics": st.session_state["selected_metrics"],
            "grid_layout": st.session_state["grid_layout"],
            "show_aggregates": st.session_state["show_aggregates"],
            "aggregation_keys": st.session_state["aggregation_keys"],
            "alignment_mode": st.session_state["alignment_mode"],
            "alignment_last_n": st.session_state["alignment_last_n"],
            "use_interpolation": st.session_state["use_interpolation"],
            "grid_mode": st.session_state["grid_mode"],
            "grid_step": st.session_state["grid_step"],
        },
    }


def apply_session(payload: dict):
    ui = payload.get("ui_state", {})

    st.session_state["filters"] = ui.get("filters", st.session_state["filters"])
    st.session_state["highlight"] = ui.get("highlight", st.session_state["highlight"])
    st.session_state["selected_metrics"] = ui.get("selected_metrics", st.session_state["selected_metrics"])
    st.session_state["grid_layout"] = ui.get("grid_layout", st.session_state["grid_layout"])

    st.session_state["show_aggregates"] = ui.get("show_aggregates", st.session_state["show_aggregates"])
    st.session_state["aggregation_keys"] = ui.get("aggregation_keys", st.session_state["aggregation_keys"])
    st.session_state["alignment_mode"] = ui.get("alignment_mode", st.session_state["alignment_mode"])
    st.session_state["alignment_last_n"] = ui.get("alignment_last_n", st.session_state["alignment_last_n"])
    st.session_state["use_interpolation"] = ui.get("use_interpolation", st.session_state["use_interpolation"])
    st.session_state["grid_mode"] = ui.get("grid_mode", st.session_state["grid_mode"])
    st.session_state["grid_step"] = ui.get("grid_step", st.session_state["grid_step"])

    if not st.session_state["file_registry"]:
        st.session_state["pending_session"] = payload
        return

    file_meta = payload.get("file_metadata", {})
    for fname, meta in st.session_state["file_registry"].items():
        if fname in file_meta:
            m = file_meta[fname]
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

    if not st.session_state["file_registry"]:
        st.info("Add experiment files to get started.")
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
        if not meta.get("saved", False):
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

            exp["tags"] = meta.get("tags", [])
            exp["group"] = meta.get("group", "")

            st.session_state["experiments"][exp["id"]] = exp

        except ExperimentLoadError as e:
            st.error(str(e))


# ======================
# Controls
# ======================
def render_filters(experiments):
    with st.expander("🔎 Filters & Grouping", expanded=True):
        if not experiments:
            st.info("Save experiments to enable filters.")
            return []

        f = st.session_state["filters"]

        models = sorted({e.get("model") for e in experiments if e.get("model")})
        datasets = sorted({e.get("dataset") for e in experiments if e.get("dataset")})
        tasks = sorted({e.get("task") for e in experiments if e.get("task")})
        tags = sorted({t for e in experiments for t in e.get("tags", [])})
        groups = sorted({e.get("group") for e in experiments if e.get("group")})

        f["model"] = st.multiselect("Model", models, f.get("model", []))
        f["dataset"] = st.multiselect("Dataset", datasets, f.get("dataset", []))
        f["task"] = st.multiselect("Task", tasks, f.get("task", []))
        f["tags"] = st.multiselect("Tags", tags, f.get("tags", []))
        f["group"] = st.multiselect("Group", groups, f.get("group", []))

        def keep(e):
            if f["model"] and e.get("model") not in f["model"]:
                return False
            if f["dataset"] and e.get("dataset") not in f["dataset"]:
                return False
            if f["task"] and e.get("task") not in f["task"]:
                return False
            if f["group"] and (e.get("group") or "") not in f["group"]:
                return False
            if f["tags"] and not set(e.get("tags", [])).intersection(f["tags"]):
                return False
            return True

        filtered = [e for e in experiments if keep(e)]
        st.caption(f"Showing **{len(filtered)} / {len(experiments)}** experiments after filtering.")
        return filtered

def render_robustness_panel(experiments: list[dict]):
    with st.expander("🧬 Robustness & Stability (V3)", expanded=False):
        if not experiments:
            st.info("Save experiments to enable robustness analysis.")
            return

        metric = st.session_state.get("selected_metric")
        if not metric:
            st.info("Select a metric first.")
            return

        tail_fraction = st.slider(
            "Late-epoch fraction",
            min_value=0.1,
            max_value=0.5,
            step=0.05,
            value=0.25,
            help="Fraction of final epochs used to assess stability.",
        )

        plateau_tol = st.number_input(
            "Plateau tolerance",
            min_value=1e-6,
            max_value=1e-2,
            value=1e-3,
            format="%.1e",
            help="Maximum per-epoch change to consider performance converged.",
        )

        df = compute_robustness_metrics(
            experiments=experiments,
            metric=metric,
            tail_fraction=tail_fraction,
            plateau_tol=plateau_tol,
        )

        if df.empty:
            st.warning("Not enough data to compute robustness metrics.")
            return

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download robustness CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{metric}_robustness.csv",
            mime="text/csv",
            use_container_width=True,
        )

def render_highlight_controls(experiments):
    with st.expander("🎯 Highlight & Emphasis", expanded=False):
        if not experiments:
            st.info("No experiments to highlight.")
            return
        names = [e["experiment_name"] for e in experiments]
        st.session_state["highlight"] = st.multiselect(
            "Highlight",
            names,
            st.session_state.get("highlight", []),
        )


def render_metric_selection(experiments):
    with st.expander("📊 Metric Selection", expanded=True):
        if not experiments:
            st.info("No experiments available.")
            st.session_state["selected_metrics"] = []
            st.session_state["selected_metric"] = None
            return []

        metrics = sorted(intersect_available_metrics(experiments))
        if not metrics:
            st.warning("No common metrics available across filtered experiments.")
            st.session_state["selected_metrics"] = []
            st.session_state["selected_metric"] = None
            return []

        selected = st.multiselect(
            "Metrics",
            metrics,
            st.session_state.get("selected_metrics") or metrics[:1],
        )
        st.session_state["selected_metrics"] = selected
        st.session_state["selected_metric"] = selected[0] if selected else None

        st.session_state["grid_layout"] = int(
            st.selectbox("Grid columns", [1, 2, 3], index=1)
        )

        return selected
    
def render_learning_dynamics_panel(experiments: list[dict]):
    with st.expander("📉 Learning Dynamics (V3)", expanded=False):
        if not experiments:
            st.info("Save experiments to enable learning dynamics analysis.")
            return

        base_metric = st.selectbox(
            "Train/Val metric base",
            options=["loss", "accuracy", "dice", "iou", "f1"],
            index=0,
            help="Looks for train_<metric> and val_<metric> columns (or similar variants).",
        )

        tail_fraction = st.slider(
            "Tail fraction (slope window)",
            min_value=0.1,
            max_value=0.5,
            step=0.05,
            value=0.25,
        )

        df = compute_learning_dynamics(
            experiments=experiments,
            base_metric=base_metric,
            tail_fraction=tail_fraction,
        )

        if df.empty:
            st.warning(
                "No experiments contain matching train/val columns for this base metric.\n\n"
                "Expected patterns: train_<metric> and val_<metric>."
            )
            return

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download learning dynamics CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"learning_dynamics_{base_metric}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_alignment_controls():
    with st.expander("⏱ Epoch Alignment (V3)", expanded=False):
        st.session_state["alignment_mode"] = st.selectbox(
            "Alignment strategy",
            options=["intersection", "truncate", "last_n"],
            index=["intersection", "truncate", "last_n"].index(
                st.session_state.get("alignment_mode", "intersection")
            ),
            help="Controls how epochs are aligned before aggregation (no interpolation).",
        )

        if st.session_state["alignment_mode"] == "last_n":
            st.session_state["alignment_last_n"] = st.number_input(
                "Last N epochs",
                min_value=1,
                step=1,
                value=int(st.session_state.get("alignment_last_n", 5)),
            )


def render_aggregation_controls():
    with st.expander("📊 Run Aggregation (V3)", expanded=False):
        st.session_state["show_aggregates"] = st.toggle(
            "Show aggregated curves (mean ± std)",
            value=st.session_state.get("show_aggregates", False),
        )

        st.session_state["aggregation_keys"] = st.multiselect(
            "Aggregate by",
            options=["model", "dataset", "group", "task"],
            default=st.session_state.get("aggregation_keys", ["model", "dataset"]),
            help="Runs sharing these fields will be aggregated.",
        )


def render_interpolation_controls():
    with st.expander("📐 Interpolation (V3)", expanded=False):
        st.session_state["use_interpolation"] = st.toggle(
            "Enable interpolation (linear)",
            value=st.session_state.get("use_interpolation", False),
            help="Opt-in. Resamples curves onto a shared epoch grid. Should be disclosed in papers.",
        )

        if st.session_state["use_interpolation"]:
            st.session_state["grid_mode"] = st.selectbox(
                "Epoch grid range",
                options=["common_range", "union_range"],
                index=0 if st.session_state.get("grid_mode", "common_range") == "common_range" else 1,
                help="common_range: overlap only (safer) | union_range: full union (more NaNs).",
            )
            st.session_state["grid_step"] = st.number_input(
                "Grid step",
                min_value=1,
                step=1,
                value=int(st.session_state.get("grid_step", 1)),
            )


def render_statistics_panel(experiments: list[dict]):
    with st.expander("🧪 Statistics (V3)", expanded=False):
        if not experiments:
            st.info("Save experiments to enable statistics.")
            return

        metric = st.session_state.get("selected_metric")
        if not metric:
            st.info("Select a metric first.")
            return

        c1, c2, c3 = st.columns([1, 1, 1])

        group_field = c1.selectbox(
            "Compare groups by",
            options=["model", "dataset", "group", "task"],
            index=0,
        )

        value_mode = c2.selectbox(
            "Comparison value",
            options=["final", "best", "auc"],
            index=0,
            help="final: final epoch value | best: best across epochs | auc: area under curve",
        )

        pairing_field = c3.selectbox(
            "Pair on (optional)",
            options=["None", "group", "dataset", "task"],
            index=0,
            help="If set, comparisons are paired by matching this key across groups.",
        )
        pairing_field = None if pairing_field == "None" else pairing_field

        stats_df = compute_pairwise_stats(
            experiments=experiments,
            metric=metric,
            group_field=group_field,
            value_mode=value_mode,
            pairing_field=pairing_field,
        )

        if stats_df.empty:
            st.warning("Not enough data to compute comparisons for the current settings.")
            return

        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download stats CSV",
            stats_df.to_csv(index=False).encode("utf-8"),
            file_name=f"stats_{metric}_{group_field}_{value_mode}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ======================
# Summary / Plots / Export
# ======================
def render_summary(experiments, metric):
    st.subheader("📋 Advanced Summary")
    if not experiments:
        st.info("No experiments to summarize.")
        return
    if not metric:
        st.info("Select a metric to view the summary.")
        return

    df = compute_advanced_summary(experiments, metric)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_metric_grid(experiments, metrics):
    if not experiments:
        st.info("No experiments to plot.")
        return
    if not metrics:
        st.info("Select at least one metric to plot.")
        return

    highlight = st.session_state.get("highlight", [])
    ncols = int(st.session_state.get("grid_layout", 2))
    ncols = max(1, min(3, ncols))

    for i in range(0, len(metrics), ncols):
        row_metrics = metrics[i:i + ncols]
        cols = st.columns(len(row_metrics))  # match count to row length

        for c, m in zip(cols, row_metrics):
            title = f"{m} ↓" if "loss" in m.lower() else f"{m} ↑"
            tooltip = METRIC_TOOLTIPS.get(m.lower(), "")

            with c:
                st.markdown(f"### {title}", help=tooltip)

                fig = plot_metric_curves(experiments, m, highlight=highlight)

                if st.session_state.get("show_aggregates"):
                    aggregated = aggregate_runs(
                        experiments=experiments,
                        metric=m,
                        group_by=st.session_state.get("aggregation_keys", []),
                        alignment_mode=st.session_state.get("alignment_mode"),
                        last_n=st.session_state.get("alignment_last_n"),
                        use_interpolation=st.session_state.get("use_interpolation", False),
                        grid_mode=st.session_state.get("grid_mode", "common_range"),
                        grid_step=int(st.session_state.get("grid_step", 1)),
                        interp_kind="linear",
                    )
                    # optional labeling: you can extend plot_aggregated_curves to show interpolated tag
                    plot_aggregated_curves(fig, aggregated, m)

                    if st.session_state.get("use_interpolation", False):
                        st.caption("⚠ Aggregated curves are **interpolated** (linear).")
                    elif st.session_state.get("alignment_mode") != "intersection":
                        st.caption(f"⚠ Aggregated curves use '{st.session_state['alignment_mode']}' alignment.")

                st.plotly_chart(fig, use_container_width=True)


def render_export_section(experiments, metrics):
    st.subheader("🖨️ Export Publication Plots")
    if not experiments or not metrics:
        st.info("Select experiments and metrics to enable export.")
        return

    for m in metrics:
        fig = plot_metric_curves_matplotlib(experiments, m, highlight=st.session_state.get("highlight", []))
        png = export_figure_to_png(fig)
        st.download_button(
            f"⬇️ {m}.png",
            png,
            f"{m}_comparison.png",
            "image/png",
            use_container_width=True,
        )


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

        render_alignment_controls()
        render_aggregation_controls()
        render_interpolation_controls()

        render_statistics_panel(filtered)
        render_robustness_panel(filtered)
        render_learning_dynamics_panel(filtered)

    with right:
        render_summary(filtered, st.session_state.get("selected_metric"))
        render_metric_grid(filtered, metrics)
        render_export_section(filtered, metrics)


if __name__ == "__main__":
    main()
