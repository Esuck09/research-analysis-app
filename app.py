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
# Page & State
# ======================
def init_page():
    st.set_page_config(
        page_title="Medical Imaging Experiment Comparison",
        page_icon="📊",
        layout="wide",
    )


def init_state():
    st.session_state.setdefault("file_registry", {})   # filename -> meta + file + saved flag
    st.session_state.setdefault("experiments", {})     # id -> normalized experiment
    st.session_state.setdefault("selected_metric", None)
    st.session_state.setdefault("selected_metrics", [])
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("grid_layout", 2)

    st.session_state.setdefault(
        "filters",
        {"model": [], "dataset": [], "task": [], "tags": [], "group": []},
    )

    st.session_state.setdefault("highlight", [])

    # Session I/O helper (when user loads session before uploading files)
    st.session_state.setdefault("pending_session", None)

    # Separate uploader key for session file to clear it after load
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
# Tags helper
# ======================
def parse_tags(raw: str) -> list[str]:
    if not raw:
        return []
    return sorted({t.strip() for t in raw.split(",") if t.strip()})


# ======================
# Session Save / Load (V2 Feature 5)
# ======================
def serialize_session() -> dict:
    """
    Serialize UI state + metadata (NOT raw uploaded file bytes).
    Files must be re-uploaded later; metadata will re-attach by filename.
    """
    file_meta = {}
    for fname, meta in st.session_state["file_registry"].items():
        file_meta[fname] = {
            "experiment_name": meta.get("experiment_name", fname.rsplit(".", 1)[0]),
            "model": meta.get("model", ""),
            "dataset": meta.get("dataset", ""),
            "task": meta.get("task", "classification"),
            "tags_raw": meta.get("tags_raw", ""),
            "group": meta.get("group", ""),
            "saved": bool(meta.get("saved", False)),
        }

    payload = {
        "version": "v2",
        "file_metadata": file_meta,
        "ui_state": {
            "filters": st.session_state.get("filters", {}),
            "highlight": st.session_state.get("highlight", []),
            "selected_metrics": st.session_state.get("selected_metrics", []),
            "grid_layout": st.session_state.get("grid_layout", 2),
        },
        "notes": "Session does not include raw files. Re-upload files to restore plots.",
    }
    return payload


def apply_session(payload: dict) -> None:
    """
    Apply session state.
    - If files already uploaded: metadata will attach to matching filenames.
    - If files not uploaded yet: store as pending_session and apply later on upload.
    """
    if not isinstance(payload, dict):
        st.error("Invalid session file format.")
        return

    file_meta = payload.get("file_metadata", {})
    ui_state = payload.get("ui_state", {})

    # Apply UI state
    st.session_state["filters"] = ui_state.get("filters", st.session_state["filters"])
    st.session_state["highlight"] = ui_state.get("highlight", st.session_state["highlight"])
    st.session_state["selected_metrics"] = ui_state.get("selected_metrics", st.session_state["selected_metrics"])
    st.session_state["grid_layout"] = int(ui_state.get("grid_layout", st.session_state["grid_layout"]))

    # If no files in registry yet, store pending to apply on later uploads
    if not st.session_state["file_registry"]:
        st.session_state["pending_session"] = payload
        st.success("Session loaded. Re-upload your experiment files to restore metadata and plots.")
        return

    # Apply metadata onto existing registry by filename
    for fname, meta in st.session_state["file_registry"].items():
        if fname in file_meta:
            m = file_meta[fname]
            meta["experiment_name"] = m.get("experiment_name", meta.get("experiment_name", fname.rsplit(".", 1)[0]))
            meta["model"] = m.get("model", meta.get("model", ""))
            meta["dataset"] = m.get("dataset", meta.get("dataset", ""))
            meta["task"] = m.get("task", meta.get("task", "classification"))
            meta["tags_raw"] = m.get("tags_raw", meta.get("tags_raw", ""))
            meta["tags"] = parse_tags(meta.get("tags_raw", ""))
            meta["group"] = m.get("group", meta.get("group", ""))
            meta["saved"] = bool(m.get("saved", meta.get("saved", False)))

    load_experiments()
    st.success("Session applied to current uploads.")


def render_session_controls():
    st.header("💾 Session")

    c1, c2 = st.columns([1, 1])

    # --- Save session ---
    with c1:
        payload = serialize_session()
        session_bytes = json.dumps(payload, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Download session (.json)",
            data=session_bytes,
            file_name="experiment_dashboard_session.json",
            mime="application/json",
            use_container_width=True,
        )

    # --- Load session ---
    with c2:
        session_file = st.file_uploader(
            "Load session (.json)",
            type=["json"],
            label_visibility="collapsed",
            key=f"session_uploader_{st.session_state['session_uploader_key']}",
        )
        if session_file is not None:
            try:
                session_file.seek(0)
                payload = json.load(session_file)
                apply_session(payload)

                # clear uploader state
                st.session_state["session_uploader_key"] += 1
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load session: {e}")


# ======================
# File Manager
# ======================
def render_file_manager():
    st.header("📁 Manage Experiments")

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
                    "tags_raw": "",
                    "tags": [],
                    "group": "",
                    "saved": False,
                }

        # If a session was loaded before files existed, apply it now
        if st.session_state.get("pending_session"):
            apply_session(st.session_state["pending_session"])
            st.session_state["pending_session"] = None

        # clear uploader UI state
        st.session_state["uploader_key"] += 1
        st.rerun()

    if not st.session_state["file_registry"]:
        st.info("No experiment files added yet.")
        return

    for fname, meta in list(st.session_state["file_registry"].items()):
        with st.container(border=True):
            cols = st.columns([1.6, 2.2, 2, 2, 2, 2.2, 1.6, 0.7, 0.7])

            status = "✅" if meta["saved"] else "📝"
            cols[0].markdown(f"**{status} {fname}**")

            exp_name = cols[1].text_input("Experiment", meta["experiment_name"], key=f"exp_{fname}")
            model = cols[2].text_input("Model", meta["model"], key=f"model_{fname}")
            dataset = cols[3].text_input("Dataset", meta["dataset"], key=f"dataset_{fname}")

            task = cols[4].selectbox(
                "Task",
                ["classification", "segmentation"],
                index=0 if meta["task"] == "classification" else 1,
                key=f"task_{fname}",
            )

            tags_raw = cols[5].text_input("Tags", meta["tags_raw"], key=f"tags_{fname}")
            group = cols[6].text_input("Group", meta["group"], key=f"group_{fname}")

            has_changes = (
                exp_name != meta["experiment_name"]
                or model != meta["model"]
                or dataset != meta["dataset"]
                or task != meta["task"]
                or tags_raw != meta["tags_raw"]
                or group != meta["group"]
                or not meta["saved"]
            )

            save_clicked = cols[7].button("💾", key=f"save_{fname}", disabled=not has_changes)
            delete_clicked = cols[8].button("🗑️", key=f"delete_{fname}")

            if save_clicked:
                meta.update(
                    {
                        "experiment_name": exp_name.strip(),
                        "model": model.strip(),
                        "dataset": dataset.strip(),
                        "task": task,
                        "tags_raw": tags_raw,
                        "tags": parse_tags(tags_raw),
                        "group": group.strip(),
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
            continue

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

            exp["tags"] = meta.get("tags", [])
            exp["group"] = meta.get("group", "")

            st.session_state["experiments"][exp["id"]] = exp

        except ExperimentLoadError as e:
            st.error(f"❌ {file.name}: {e}")


# ======================
# Filters
# ======================
def render_filters(experiments: list[dict]) -> list[dict]:
    st.header("🔎 Filters & Grouping")

    if not experiments:
        st.info("Save at least one experiment to enable filters.")
        return []

    # Available values
    all_models = sorted({e.get("model", "") for e in experiments if e.get("model")})
    all_datasets = sorted({e.get("dataset", "") for e in experiments if e.get("dataset")})
    all_tasks = sorted({e.get("task", "") for e in experiments if e.get("task")})
    all_tags = sorted({t for e in experiments for t in (e.get("tags") or [])})
    all_groups = sorted({e.get("group", "") for e in experiments if e.get("group")})

    f = st.session_state["filters"]
    st.subheader("🔎 Filters")

    selected_models = st.multiselect(
        "Model",
        all_models,
        default=f.get("model", []),
    )

    selected_datasets = st.multiselect(
        "Dataset",
        all_datasets,
        default=f.get("dataset", []),
    )

    selected_tasks = st.multiselect(
        "Task",
        all_tasks,
        default=f.get("task", []),
    )

    selected_tags = st.multiselect(
        "Tags",
        all_tags,
        default=f.get("tags", []),
    )

    selected_groups = st.multiselect(
        "Group",
        all_groups,
        default=f.get("group", []),
    )


    st.session_state["filters"] = {
        "model": selected_models,
        "dataset": selected_datasets,
        "task": selected_tasks,
        "tags": selected_tags,
        "group": selected_groups,
    }

    def passes(e: dict) -> bool:
        if selected_models and e.get("model") not in selected_models:
            return False
        if selected_datasets and e.get("dataset") not in selected_datasets:
            return False
        if selected_tasks and e.get("task") not in selected_tasks:
            return False
        if selected_groups and (e.get("group") or "") not in selected_groups:
            return False
        if selected_tags:
            tags = set(e.get("tags") or [])
            if not tags.intersection(set(selected_tags)):
                return False
        return True

    filtered = [e for e in experiments if passes(e)]
    st.caption(f"Showing **{len(filtered)} / {len(experiments)}** saved experiments after filtering.")
    return filtered


# ======================
# Highlight Controls
# ======================
def render_highlight_controls(experiments: list[dict]):
    st.header("🎯 Highlight & Emphasis")
    if not experiments:
        return

    names = [e["experiment_name"] for e in experiments]
    st.session_state["highlight"] = st.multiselect(
        "Highlight experiments",
        options=names,
        default=st.session_state.get("highlight", []),
        help="Highlighted experiments will be emphasized; others will be faded.",
    )


# ======================
# Metric Selection (multi)
# ======================
def render_metric_selection(experiments: list[dict]) -> list[str]:
    st.header("📊 Metric Selection")

    if not experiments:
        st.info("No experiments selected after filtering.")
        return []

    available = sorted(intersect_available_metrics(experiments))
    if not available:
        st.warning("No common metrics available across selected experiments.")
        return []

    c1, c2 = st.columns([3, 1])
    selected = c1.multiselect(
        "Select metrics (multi-select)",
        options=available,
        default=st.session_state.get("selected_metrics") or [available[0]],
    )
    grid_cols = c2.selectbox("Grid columns", options=[1, 2, 3], index=1)

    st.session_state["selected_metrics"] = selected
    st.session_state["selected_metric"] = selected[0] if selected else None
    st.session_state["grid_layout"] = grid_cols

    return selected


# ======================
# Summary (advanced)
# ======================
def render_summary(experiments: list[dict], metric: str | None):
    st.header("📋 Advanced Summary")

    if not experiments or not metric:
        st.info("Select a metric to view summary statistics.")
        return

    df = compute_advanced_summary(experiments, metric)
    if df.empty:
        st.warning("No valid experiments for this metric.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download advanced summary CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{metric}_advanced_summary.csv",
        mime="text/csv",
    )


# ======================
# Plot grid
# ======================
def render_metric_grid(experiments: list[dict], metrics: list[str]):
    st.header("📈 Metric Curves (Multi-Metric)")

    if not experiments or not metrics:
        st.info("Select metrics to plot.")
        return

    ncols = int(st.session_state.get("grid_layout", 2))
    ncols = max(1, min(3, ncols))

    highlight = st.session_state.get("highlight", [])

    for i in range(0, len(metrics), ncols):
        row = metrics[i : i + ncols]
        cols = st.columns(len(row))
        for c, metric in zip(cols, row):
            with c:
                fig = plot_metric_curves(experiments, metric, highlight=highlight)
                st.plotly_chart(fig, use_container_width=True)


# ======================
# Export (per metric)
# ======================
def render_export_section(experiments: list[dict], metrics: list[str]):
    st.header("🖨️ Export Publication Plots (PNG)")

    if not experiments or not metrics:
        st.info("Select metrics to export.")
        return

    highlight = st.session_state.get("highlight", [])

    cols_per_row = 3
    for i in range(0, len(metrics), cols_per_row):
        row = metrics[i : i + cols_per_row]
        cols = st.columns(len(row))
        for c, metric in zip(cols, row):
            with c:
                fig_pub = plot_metric_curves_matplotlib(experiments, metric, highlight=highlight)
                png = export_figure_to_png(fig_pub)
                st.download_button(
                    f"⬇️ {metric}.png",
                    png,
                    file_name=f"{metric}_comparison.png",
                    mime="image/png",
                    use_container_width=True,
                )


# ======================
# Main
# ======================
def main():
    init_page()
    init_state()
    render_header()

    # Feature 5: session controls
    # render_session_controls()
    # st.divider()

    # Manage experiments + load
    render_file_manager()
    st.divider()

    # Work only with saved experiments
    saved_experiments = list(st.session_state["experiments"].values())

    # ======================
    # Analysis Layout
    # ======================
    left, right = st.columns([1.1, 3.2], gap="large")

    with left:
        st.subheader("⚙️ Controls")
        st.divider()

        filtered = render_filters(saved_experiments)
        st.divider()

        render_highlight_controls(filtered)
        st.divider()

        metrics = render_metric_selection(filtered)

    with right:
        render_summary(filtered, st.session_state.get("selected_metric"))
        st.divider()

        render_metric_grid(filtered, metrics)
        st.divider()

        render_export_section(filtered, metrics)


if __name__ == "__main__":
    main()
