"""
Kaggle Results Dashboard.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="Kaggle Dashboard", layout="wide")
st.title("Kaggle Competition Dashboard")

try:
    import wandb
    import pandas as pd

    st.sidebar.header("Settings")
    project = st.sidebar.text_input("W&B Project", "kaggle-competition-name")

    if st.sidebar.button("Load Runs"):
        api = wandb.Api()
        runs = api.runs(project)

        if not runs:
            st.warning("No runs found.")
        else:
            data = []
            for run in runs:
                row = {
                    "name": run.name,
                    "state": run.state,
                    "tags": ", ".join(run.tags),
                    **{k: v for k, v in run.summary.items() if not k.startswith("_")},
                }
                data.append(row)

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            # Score chart
            score_cols = [c for c in df.columns if "score" in c.lower() or "cv" in c.lower()]
            if score_cols:
                st.subheader("Score Comparison")
                st.bar_chart(df.set_index("name")[score_cols])
    else:
        st.info("Enter your W&B project name and click 'Load Runs'.")

except ImportError:
    st.error("Missing dependencies. Run: pip install wandb pandas")
