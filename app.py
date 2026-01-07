import streamlit as st

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

def main() -> None:
    init_page()
    init_session_state()
    render_header()
    render_sidebar()
    render_main_placeholders()

if __name__ == "__main__":
    main()
