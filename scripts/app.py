#!/usr/bin/env python3
"""
Waymo Validation Lab — Scenario Validation Dashboard

First validation-layer UI for scenario scoring.
Summarizes the scenario set using risk, complexity, and comfort metrics.

Run with: streamlit run scripts/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- paths ----------
PROJECT_ROOT = Path(__file__).parent.parent
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"

# ---------- helpers ----------

def load_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    """Load a parquet file if it exists, return None otherwise."""
    if path.exists():
        return pd.read_parquet(path)
    return None


def build_merged_table(
    scenarios: pd.DataFrame,
    scenario_metrics: pd.DataFrame | None,
    interaction_metrics: pd.DataFrame | None,
    risk_metrics: pd.DataFrame | None,
    comfort_metrics: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge available tables into one row-per-scenario dataframe."""
    df = scenarios[["scenario_id", "num_tracks", "data_source"]].copy()

    if scenario_metrics is not None:
        cols = ["scenario_id", "num_valid_state_rows", "avg_speed_mps", "max_speed_mps"]
        cols = [c for c in cols if c in scenario_metrics.columns]
        df = df.merge(scenario_metrics[cols], on="scenario_id", how="left")

    if interaction_metrics is not None:
        cols = [
            "scenario_id",
            "min_sdc_distance_m",
            "num_close_interactions",
            "num_unique_close_actors",
            "sdc_avg_speed_mps",
            "sdc_max_speed_mps",
            "sdc_distance_traveled_m",
            "scenario_interest_score",
        ]
        cols = [c for c in cols if c in interaction_metrics.columns]
        df = df.merge(interaction_metrics[cols], on="scenario_id", how="left")

    if risk_metrics is not None and "risk_score" in risk_metrics.columns:
        df = df.merge(
            risk_metrics[["scenario_id", "risk_score"]], on="scenario_id", how="left"
        )

    if comfort_metrics is not None and "comfort_score" in comfort_metrics.columns:
        comfort_cols = ["scenario_id", "comfort_score"]
        # Add comfort detail columns if available
        for c in ["max_acceleration_mps2", "max_deceleration_mps2", "max_jerk_mps3"]:
            if c in comfort_metrics.columns:
                comfort_cols.append(c)
        df = df.merge(comfort_metrics[comfort_cols], on="scenario_id", how="left")

    return df


def has_risk_score(df: pd.DataFrame) -> bool:
    return "risk_score" in df.columns and df["risk_score"].notna().any()


def has_comfort_score(df: pd.DataFrame) -> bool:
    return "comfort_score" in df.columns and df["comfort_score"].notna().any()


def risk_col(df: pd.DataFrame) -> str:
    return "risk_score" if has_risk_score(df) else "scenario_interest_score"


def risk_label(df: pd.DataFrame) -> str:
    if has_risk_score(df):
        return "Risk Score"
    return "Risk Proxy (scenario_interest_score)"


# ============================================================
# SECTION 1 — DATASET SUMMARY
# ============================================================

def render_dataset_summary(
    scenarios: pd.DataFrame,
    loaded_files: dict[str, bool],
):
    st.header("1 — Dataset Summary")
    st.markdown(
        "This dashboard is the first validation-layer UI for scenario scoring. "
        "It summarizes the scenario set using **risk**, **complexity**, and "
        "**comfort**-related metrics."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Scenarios", len(scenarios))
    col2.metric("Total Tracks", int(scenarios["num_tracks"].sum()))

    parser_modes = scenarios["data_source"].unique()
    col3.metric("Parser Mode", ", ".join(parser_modes))

    st.markdown("**Loaded gold tables:**")
    for name, ok in loaded_files.items():
        icon = "✅" if ok else "❌"
        st.markdown(f"- {icon} `{name}`")


# ============================================================
# SECTION 2 — TOP SCENARIOS TABLE
# ============================================================

def render_top_scenarios_table(merged: pd.DataFrame):
    st.header("2 — Top Scenarios")

    sort_col = risk_col(merged)
    display_cols = ["scenario_id"]

    if has_risk_score(merged):
        display_cols.append("risk_score")
    if "scenario_interest_score" in merged.columns:
        display_cols.append("scenario_interest_score")
    
    # Add risk detail columns if available
    for c in ["min_ttc_s", "max_closing_speed_mps", "num_ttc_below_3s"]:
        if c in merged.columns:
            display_cols.append(c)
    
    # Add comfort score if available
    if has_comfort_score(merged):
        display_cols.append("comfort_score")
    
    for c in [
        "min_sdc_distance_m",
        "num_close_interactions",
        "num_tracks",
        "data_source",
    ]:
        if c in merged.columns:
            display_cols.append(c)

    sorted_df = merged.sort_values(sort_col, ascending=False)
    st.dataframe(sorted_df[display_cols], use_container_width=True, hide_index=True)

    if not has_risk_score(merged):
        st.caption(
            "⚠️ Risk proxy = `scenario_interest_score` "
            "(true risk metrics not yet available)"
        )
    else:
        st.caption(
            "✅ Real risk score computed from TTC, closing speed, and TTC threshold breaches"
        )


# ============================================================
# SECTION 3 — RISK OVERVIEW
# ============================================================

def render_risk_overview(merged: pd.DataFrame):
    st.header("3 — Risk Overview")

    col = risk_col(merged)
    label = risk_label(merged)

    if col not in merged.columns or merged[col].isna().all():
        st.warning("No risk-related metric available.")
        return

    sorted_df = merged.sort_values(col, ascending=False).head(10)

    # A. Top scenarios by risk
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    short_ids = [sid[:8] for sid in sorted_df["scenario_id"]]
    ax1.barh(short_ids[::-1], sorted_df[col].values[::-1], color="#d94f4f")
    ax1.set_xlabel(label)
    ax1.set_title(f"Top Scenarios by {label}")
    ax1.set_xlim(0, max(1.0, sorted_df[col].max() * 1.15))
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # B. Risk distribution
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.hist(merged[col].dropna(), bins=15, color="#d94f4f", edgecolor="white", alpha=0.85)
    ax2.set_xlabel(label)
    ax2.set_ylabel("Number of Scenarios")
    ax2.set_title(f"Distribution of {label}")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.caption(
        "**X-axis** = score values · **Y-axis** = number of scenarios in each range · "
        "**Purpose** = understand overall safety/risk profile of the scenario set"
    )

    # Add note about real risk metrics
    if has_risk_score(merged):
        st.info(
            "✅ **Real risk metrics** are now computed from Time-to-Collision (TTC), "
            "closing speed, and TTC threshold breaches. "
            "Lower TTC and higher closing speed increase the risk score."
        )


# ============================================================
# SECTION 4 — COMPLEXITY OVERVIEW
# ============================================================

def render_complexity_overview(merged: pd.DataFrame):
    st.header("4 — Complexity Overview")

    col_a, col_b = st.columns(2)

    # A. Close interactions distribution
    with col_a:
        if "num_close_interactions" in merged.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(
                merged["num_close_interactions"].dropna(),
                bins=15,
                color="#3a7ca5",
                edgecolor="white",
                alpha=0.85,
            )
            ax.set_xlabel("Close Interactions (< 5 m)")
            ax.set_ylabel("Scenarios")
            ax.set_title("Interaction Complexity")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("num_close_interactions not available")

    # B. Actor density distribution
    with col_b:
        if "num_tracks" in merged.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(
                merged["num_tracks"].dropna(),
                bins=15,
                color="#3a7ca5",
                edgecolor="white",
                alpha=0.85,
            )
            ax.set_xlabel("Number of Tracks")
            ax.set_ylabel("Scenarios")
            ax.set_title("Actor Density")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("num_tracks not available")

    st.caption(
        "More close interactions generally means more interaction complexity. "
        "More tracks generally means denser scenarios."
    )


# ============================================================
# SECTION 5 — RISK VS COMPLEXITY
# ============================================================

def render_risk_vs_complexity(merged: pd.DataFrame):
    st.header("5 — Risk vs Complexity")

    x_col = risk_col(merged)
    y_col = "num_close_interactions"

    if x_col not in merged.columns or y_col not in merged.columns:
        st.warning("Required columns not available for this chart.")
        return

    plot_df = merged.dropna(subset=[x_col, y_col])
    if len(plot_df) == 0:
        st.warning("No data available for scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        plot_df[x_col],
        plot_df[y_col],
        s=80,
        color="#3a7ca5",
        edgecolors="white",
        linewidth=0.5,
        zorder=3,
    )

    # Annotate points
    n = len(plot_df)
    if n <= 30:
        annotate_df = plot_df
    else:
        annotate_df = plot_df.nlargest(10, x_col)

    for _, row in annotate_df.iterrows():
        ax.annotate(
            row["scenario_id"][:8],
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            alpha=0.8,
        )

    ax.set_xlabel(risk_label(merged))
    ax.set_ylabel("Close Interactions (< 5 m)")
    ax.set_title("Risk vs Complexity")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "**Top-right** = likely hardest scenarios (high risk + high complexity). "
        "**Bottom-left** = likely less interesting scenarios."
    )


# ============================================================
# SECTION 6 — COMFORT PANEL
# ============================================================

def render_comfort_panel(comfort_metrics: pd.DataFrame | None):
    st.header("6 — Comfort")

    if comfort_metrics is not None and "comfort_score" in comfort_metrics.columns:
        # A. Top scenarios by comfort (least comfortable first)
        sorted_df = comfort_metrics.sort_values("comfort_score", ascending=False).head(10)
        
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        short_ids = [sid[:8] for sid in sorted_df["scenario_id"]]
        ax1.barh(short_ids[::-1], sorted_df["comfort_score"].values[::-1], color="#6aaa64")
        ax1.set_xlabel("Comfort Score (higher = less comfortable)")
        ax1.set_title("Top 10 Scenarios by Comfort Score")
        ax1.set_xlim(0, max(1.0, sorted_df["comfort_score"].max() * 1.15))
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # B. Comfort distribution
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.hist(
            comfort_metrics["comfort_score"].dropna(),
            bins=15,
            color="#6aaa64",
            edgecolor="white",
            alpha=0.85,
        )
        ax2.set_xlabel("Comfort Score (higher = less comfortable)")
        ax2.set_ylabel("Scenarios")
        ax2.set_title("Distribution of Comfort Score")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # C. Detail table
        detail_cols = ["scenario_id", "comfort_score"]
        for c in ["max_acceleration_mps2", "max_deceleration_mps2", "max_jerk_mps3"]:
            if c in comfort_metrics.columns:
                detail_cols.append(c)
        
        detail_df = comfort_metrics[detail_cols].sort_values("comfort_score", ascending=False).head(10)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        
        st.caption(
            "✅ **Real comfort metrics** computed from SDC acceleration, jerk, and heading rate. "
            "Higher scores indicate less comfortable motion (more abrupt)."
        )
    else:
        st.info("**Comfort metrics not yet available.**")
        st.markdown(
            "This panel is reserved for future acceleration / jerk / "
            "ride smoothness metrics."
        )


# ============================================================
# SECTION 7 — INTERPRETATION NOTES
# ============================================================

def render_interpretation_notes():
    st.header("7 — Interpretation Notes")
    st.markdown(
        """
| Metric | Meaning |
|--------|---------|
| **Risk** | How concerning a scenario appears based on available metrics |
| **Complexity** | How many close interactions / nearby actors exist |
| **Comfort** | Future ride-quality metrics (acceleration, jerk, smoothness) |

This dashboard helps engineers understand **distributions** and identify
which scenarios deserve deeper inspection. The goal is to surface outliers
and edge cases from the scenario set.
"""
    )


# ============================================================
# SECTION 8 — HOW SCORES ARE CALCULATED
# ============================================================

def render_score_calculation_logic():
    st.header("8 — How Scores Are Calculated")
    
    # Risk Score Logic
    with st.expander("🔴 Risk Score Logic"):
        st.markdown("""
**Plain English:** Risk measures how concerning a scenario appears based on Time-to-Collision (TTC), 
closing speed, and the number of dangerous close encounters. Lower TTC and higher closing speeds 
indicate higher collision risk.
        """)
        
        st.code("""
risk_score = 0.5 * ttc_component + 0.3 * closing_component + 0.2 * breach_component

ttc_component = min(1.0, 5.0 / max(min_ttc_s, 0.1))
closing_component = min(1.0, max_closing_speed_mps / 15.0)
breach_component = min(1.0, (num_ttc_below_3s + 2 * num_ttc_below_1_5s) / 20.0)
        """, language="python")
        
        st.markdown("""
**Main Assumptions:**
- TTC calculated when objects are approaching (positive closing speed)
- TTC < 3 seconds is concerning, TTC < 1.5 seconds is critical
- Closing speeds > 15 m/s (34 mph) are high risk
- More TTC breaches indicate higher cumulative risk
- Components normalized to [0, 1] range
        """)
    
    # Complexity Score Logic
    with st.expander("🔵 Complexity Score Logic"):
        st.markdown("""
**Plain English:** Complexity measures how many interactions and nearby actors exist in a scenario. 
More close interactions and more actors generally indicate a more complex driving environment.
        """)
        
        st.code("""
# Current proxy: scenario_interest_score from interaction_metrics
scenario_interest_score = (
    0.30 * min(1.0, 10.0 / (min_sdc_distance_m + 0.1))
  + 0.25 * min(1.0, num_close_interactions / 50.0)
  + 0.20 * min(1.0, sdc_max_speed_mps / 25.0)
  + 0.15 * min(1.0, num_unique_close_actors / 10.0)
  + 0.10 * min(1.0, sdc_distance_traveled_m / 200.0)
)
        """, language="python")
        
        st.markdown("""
**Main Assumptions:**
- Close interactions defined as actors within 5 meters of SDC
- More close interactions = higher complexity
- More unique actors nearby = higher complexity
- Higher SDC speeds increase perceived complexity
- Components normalized to [0, 1] range
- Currently uses interaction_score as proxy for complexity
        """)
    
    # Comfort Score Logic
    with st.expander("🟢 Comfort Score Logic"):
        st.markdown("""
**Plain English:** Comfort measures ride quality through acceleration, jerk, and heading rate. 
Higher acceleration and jerk indicate less comfortable (more abrupt) driving experiences.
        """)
        
        st.code("""
comfort_score = 0.25 * accel_component + 0.30 * decel_component + 0.30 * jerk_component + 0.15 * heading_component

accel_component = min(1.0, max_acceleration_mps2 / 4.0)
decel_component = min(1.0, max_deceleration_mps2 / 4.0)
jerk_component = min(1.0, max_jerk_mps3 / 10.0)
heading_component = min(1.0, max_heading_rate_radps / 0.8)
        """, language="python")
        
        st.markdown("""
**Main Assumptions:**
- Comfort computed from SDC motion only (acceleration, jerk, heading rate)
- Acceleration/deceleration > 4 m/s² considered uncomfortable
- Jerk > 10 m/s³ considered uncomfortable
- Heading rate > 0.8 rad/s considered uncomfortable
- Higher comfort scores = less comfortable motion
- Components normalized to [0, 1] range
- dt = 0.1s (Waymo 10 Hz scenario rate)
        """)


# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(
        page_title="Waymo Validation Lab",
        page_icon="🚗",
        layout="wide",
    )

    st.title("🚗 Waymo Validation Lab — Scenario Dashboard")

    # ---- load data ----
    scenarios = load_parquet_if_exists(SILVER_DIR / "scenarios.parquet")
    if scenarios is None:
        st.error(
            "❌ Required file missing: `data/silver/scenarios.parquet`. "
            "Run the parser first: `python scripts/waymo_real_parser.py`"
        )
        return

    scenario_metrics = load_parquet_if_exists(GOLD_DIR / "scenario_metrics.parquet")
    interaction_metrics = load_parquet_if_exists(GOLD_DIR / "interaction_metrics.parquet")
    risk_metrics = load_parquet_if_exists(GOLD_DIR / "risk_metrics.parquet")
    comfort_metrics = load_parquet_if_exists(GOLD_DIR / "comfort_metrics.parquet")

    if interaction_metrics is None:
        st.error(
            "❌ Required file missing: `data/gold/interaction_metrics.parquet`. "
            "Run: `python scripts/compute_interaction_metrics.py`"
        )
        return

    loaded_files = {
        "scenario_metrics.parquet": scenario_metrics is not None,
        "interaction_metrics.parquet": interaction_metrics is not None,
        "risk_metrics.parquet": risk_metrics is not None,
        "comfort_metrics.parquet": comfort_metrics is not None,
    }

    merged = build_merged_table(scenarios, scenario_metrics, interaction_metrics, risk_metrics, comfort_metrics)

    # ---- render sections ----
    render_dataset_summary(scenarios, loaded_files)
    st.divider()
    render_top_scenarios_table(merged)
    st.divider()
    render_risk_overview(merged)
    st.divider()
    render_complexity_overview(merged)
    st.divider()
    render_risk_vs_complexity(merged)
    st.divider()
    render_comfort_panel(comfort_metrics)
    st.divider()
    render_interpretation_notes()
    st.divider()
    render_score_calculation_logic()


if __name__ == "__main__":
    main()
