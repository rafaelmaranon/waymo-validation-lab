import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Waymo Validation Lab — Scenario Viewer")

# Load data
states = pd.read_parquet("data/silver/states.parquet")

scenario_ids = states["scenario_id"].unique()
scenario = st.selectbox("Select Scenario", scenario_ids)

df = states[states["scenario_id"] == scenario]

fig, ax = plt.subplots()

for track_id, group in df.groupby("track_id"):
    ax.plot(group["x"], group["y"], linewidth=1)

ax.set_title(f"Scenario {scenario}")
ax.set_xlabel("X")
ax.set_ylabel("Y")

st.pyplot(fig)
