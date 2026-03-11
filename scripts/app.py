#!/usr/bin/env python3
"""
Waymo Validation Lab — Scenario Validation Dashboard

First validation-layer UI for scenario scoring.
Summarizes the scenario set using risk, complexity, and comfort metrics.

Run with: streamlit run scripts/app.py
"""

import io
import time
import base64
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


@st.cache_data
def load_silver_for_playback(states_path: str, tracks_path: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load silver states and tracks with Streamlit caching for playback performance."""
    sp, tp = Path(states_path), Path(tracks_path)
    states = pd.read_parquet(sp) if sp.exists() else None
    tracks = pd.read_parquet(tp) if tp.exists() else None
    return states, tracks


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
        risk_cols = ["scenario_id", "risk_score"]
        for c in ["min_ttc_s", "max_closing_speed_mps", "num_ttc_below_3s", "num_ttc_below_1_5s"]:
            if c in risk_metrics.columns:
                risk_cols.append(c)
        df = df.merge(risk_metrics[risk_cols], on="scenario_id", how="left")

    if comfort_metrics is not None and "comfort_score" in comfort_metrics.columns:
        comfort_cols = ["scenario_id", "comfort_score"]
        # Add comfort detail columns if available
        for c in ["max_acceleration_mps2", "max_deceleration_mps2", "max_jerk_mps3"]:
            if c in comfort_metrics.columns:
                comfort_cols.append(c)
        df = df.merge(comfort_metrics[comfort_cols], on="scenario_id", how="left")

    return df


def compute_live_risk_score(row: pd.Series, ttc_warning: float, ttc_critical: float) -> dict:
    """Recompute risk score in the UI layer using current slider thresholds."""
    min_ttc = row.get("min_ttc_s", None)
    max_closing = row.get("max_closing_speed_mps", 0.0) or 0.0
    num_below_3s = row.get("num_ttc_below_3s", 0) or 0
    num_below_1_5s = row.get("num_ttc_below_1_5s", 0) or 0

    # TTC component: linear scale from warning → critical
    if pd.isna(min_ttc) or ttc_warning <= ttc_critical:
        ttc_component = 0.0
    else:
        ttc_component = max(0.0, min(1.0,
            (ttc_warning - min_ttc) / (ttc_warning - ttc_critical)
        ))

    closing_component = min(1.0, float(max_closing) / 15.0)
    breach_component = min(1.0, (float(num_below_3s) + 2.0 * float(num_below_1_5s)) / 20.0)

    risk_score = 0.5 * ttc_component + 0.3 * closing_component + 0.2 * breach_component
    return {
        "ttc_component": round(ttc_component, 3),
        "closing_component": round(closing_component, 3),
        "breach_component": round(breach_component, 3),
        "risk_score": round(risk_score, 3),
    }


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
    ttc_warning: float,
    ttc_critical: float,
    interaction_distance: int,
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

    # Display active assumptions
    st.markdown("**Validation assumptions currently applied:**")
    st.info(
        f"- TTC warning threshold: **{ttc_warning:.1f} seconds**\n"
        f"- TTC critical threshold: **{ttc_critical:.1f} seconds**\n"
        f"- Close interaction distance: **{interaction_distance} meters**\n\n"
        "Waymo provides raw trajectories. Risk and complexity scores are derived metrics "
        "computed using the assumptions above."
    )


# ============================================================
# SECTION 2 — TOP SCENARIOS TABLE
# ============================================================

def render_top_scenarios_table(merged: pd.DataFrame):
    st.header("2 — Top Scenarios")

    sort_col = risk_col(merged)

    # Full ordered column list — include only columns that exist
    preferred = [
        "scenario_id",
        "risk_score",
        "min_ttc_s",
        "max_closing_speed_mps",
        "num_ttc_below_3s",
        "num_ttc_below_1_5s",
        "scenario_interest_score",
        "comfort_score",
        "num_tracks",
    ]
    display_cols = [c for c in preferred if c in merged.columns]

    sorted_df = merged.sort_values(sort_col, ascending=False)
    st.dataframe(sorted_df[display_cols], use_container_width=True, hide_index=True)

    if not has_risk_score(merged):
        st.caption(
            "⚠️ Risk proxy = `scenario_interest_score` "
            "(true risk metrics not yet available)"
        )
    else:
        st.caption(
            "Sorted by `risk_score` descending · "
            "TTC = Time-to-Collision (seconds) · "
            "closing speed in m/s · "
            "breaches = timesteps below TTC threshold"
        )


# ============================================================
# SECTION 3 — RISK OVERVIEW
# ============================================================

def render_risk_overview(merged: pd.DataFrame, ttc_warning: float, ttc_critical: float):
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

    if has_risk_score(merged):
        st.warning(
            "**Important — how to read risk_score:**\n\n"
            "TTC thresholds classify individual interaction moments (safe / warning / critical). "
            "`risk_score` is a **scenario-level summary** that combines three independent dimensions:\n\n"
            "- **Time margin** → min TTC across all actor-timestep pairs\n"
            "- **Severity** → maximum closing speed\n"
            "- **Exposure** → count of timesteps where TTC was below threshold\n\n"
            "A scenario with `risk_score = 1.0` does not simply mean \"critical TTC\". "
            "It means the combined risk signals (TTC + closing speed + breaches) reached the maximum level "
            "under the current formula.\n\n"
            f"Current thresholds: warning < **{ttc_warning:.1f}s**, critical < **{ttc_critical:.1f}s**. "
            "Adjust in the sidebar to change interpretation."
        )


# ============================================================
# SECTION 4 — COMPLEXITY OVERVIEW
# ============================================================

def render_complexity_overview(merged: pd.DataFrame, interaction_distance: int):
    st.header("4 — Complexity Overview")

    col_a, col_b = st.columns(2)

    # A. Close interactions distribution (adjusted by slider)
    with col_a:
        if "num_close_interactions" in merged.columns:
            # Adjust interaction count based on slider distance
            adjustment_factor = interaction_distance / 5.0  # Original threshold was 5m
            merged["num_close_interactions_adjusted"] = merged["num_close_interactions"] * adjustment_factor
            
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(
                merged["num_close_interactions_adjusted"].dropna(),
                bins=15,
                color="#3a7ca5",
                edgecolor="white",
                alpha=0.85,
            )
            ax.set_xlabel(f"Close Interactions (< {interaction_distance}m)")
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
        f"More close interactions (within {interaction_distance}m) generally means more interaction complexity. "
        "More tracks generally means denser scenarios. "
        f"Interaction count adjusted from original 5m threshold to {interaction_distance}m threshold."
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
        st.markdown(
            "Collision risk is modelled across **three independent dimensions**:\n\n"
            "| Dimension | Metric | Meaning |\n"
            "|-----------|--------|---------|\n"
            "| **Time margin** | `min_ttc_s` | How close in time was the nearest approach? |\n"
            "| **Severity** | `max_closing_speed_mps` | How fast were objects converging? |\n"
            "| **Exposure** | `num_ttc_below_3s / num_ttc_below_1_5s` | How often did TTC breach thresholds? |\n\n"
            "Conceptual model: **Risk ≈ Time × Severity × Exposure**"
        )

        st.code("""\
risk_score = 0.5 * ttc_component + 0.3 * closing_component + 0.2 * breach_component

# TTC component — linear between warning and critical thresholds (set in sidebar)
if ttc_warning <= ttc_critical:
    ttc_component = 0.0
else:
    ttc_component = max(0.0, min(1.0,
        (ttc_warning - min_ttc_s) / (ttc_warning - ttc_critical)
    ))
# TTC >= warning  → component = 0  (safe)
# TTC <= critical → component = 1  (critical)
# between them    → linear interpolation

closing_component = min(1.0, max_closing_speed_mps / 15.0)
breach_component  = min(1.0, (num_ttc_below_3s + 2 * num_ttc_below_1_5s) / 20.0)
""", language="python")

        st.markdown(
            "**Main Assumptions:**\n"
            "- TTC is only computed when closing speed > 0 (objects approaching)\n"
            "- Closing speeds > 15 m/s (≈34 mph) saturate the severity component\n"
            "- The breach normalizer (20) assumes up to 10 warning events or 10 critical events in a scenario\n"
            "- The **stored** `risk_score` uses fixed thresholds (3 s / 1.5 s); "
            "the **live** score in Scenario Review uses sidebar sliders"
        )

    # Complexity Score Logic
    with st.expander("🔵 Complexity Score Logic"):
        st.markdown(
            "Complexity proxies how demanding the driving environment is for the SDC."
        )

        st.code("""\
scenario_interest_score = (
    0.30 * min(1.0, 10.0 / (min_sdc_distance_m + 0.1))   # proximity
  + 0.25 * min(1.0, num_close_interactions / 50.0)        # interaction count
  + 0.20 * min(1.0, sdc_max_speed_mps / 25.0)            # SDC speed
  + 0.15 * min(1.0, num_unique_close_actors / 10.0)       # actor diversity
  + 0.10 * min(1.0, sdc_distance_traveled_m / 200.0)      # coverage
)
""", language="python")

        st.markdown(
            "**Main Assumptions:**\n"
            "- 'Close' interaction baseline = 5 m (adjustable via sidebar)\n"
            "- More unique actors nearby = higher complexity\n"
            "- Components normalized to [0, 1]\n"
            "- Currently used as proxy; a dedicated complexity score is planned"
        )

    # Comfort Score Logic
    with st.expander("🟢 Comfort Score Logic"):
        st.markdown(
            "Comfort measures ride abruptness from SDC motion only. "
            "**Higher score = less comfortable.**"
        )

        st.code("""\
comfort_score = 0.25*accel_component + 0.30*decel_component + 0.30*jerk_component + 0.15*heading_component

accel_component   = min(1.0, max_acceleration_mps2 / 4.0)
decel_component   = min(1.0, max_deceleration_mps2 / 4.0)
jerk_component    = min(1.0, max_jerk_mps3 / 10.0)
heading_component = min(1.0, max_heading_rate_radps / 0.8)

# acceleration[t] = (speed[t] - speed[t-1]) / dt      dt = 0.1 s
# jerk[t]         = (acceleration[t] - acceleration[t-1]) / dt
# heading_rate[t] = wrapped_angle_diff(heading[t], heading[t-1]) / dt
""", language="python")

        st.markdown(
            "**Main Assumptions:**\n"
            "- Accel/decel > 4 m/s² is uncomfortable\n"
            "- Jerk > 10 m/s³ is uncomfortable\n"
            "- Heading rate > 0.8 rad/s is uncomfortable\n"
            "- dt = 0.1 s (Waymo 10 Hz scenario rate)\n"
            "- SDC motion only — other actors excluded"
        )

    # Metric Glossary
    with st.expander("📖 Metric Glossary"):
        st.markdown(
            "| Metric | Definition |\n"
            "|--------|------------|\n"
            "| `scenario_id` | Unique identifier for a Waymo scenario (hex string) |\n"
            "| `min_ttc_s` | Minimum Time-to-Collision observed across all actor/timestep pairs (seconds) |\n"
            "| `max_closing_speed_mps` | Maximum closing speed across all approaching actor/timestep pairs (m/s) |\n"
            "| `num_ttc_below_3s` | Number of actor/timestep pairs where TTC < 3 seconds |\n"
            "| `num_ttc_below_1_5s` | Number of actor/timestep pairs where TTC < 1.5 seconds |\n"
            "| `sdc_max_speed_mps` | Maximum speed of the SDC (self-driving car) during the scenario |\n"
            "| `min_sdc_distance_m` | Minimum distance between SDC and any other actor during the scenario |\n"
            "| `num_tracks` | Total number of actor tracks in the scenario |\n"
            "| `comfort_score` | Composite ride-comfort abruptness score [0, 1]; higher = less comfortable |\n"
            "| `risk_score` | Composite safety risk score [0, 1]; higher = more risk |\n"
            "| `scenario_interest_score` | Composite complexity proxy [0, 1]; higher = more interaction density |"
        )


# ============================================================
# SECTION 9 — SCENARIO REVIEW (ENGINEER MODE)
# ============================================================

def render_scenario_review(
    merged: pd.DataFrame,
    risk_metrics: pd.DataFrame | None,
    comfort_metrics: pd.DataFrame | None,
    ttc_warning: float,
    ttc_critical: float,
    interaction_distance: int,
    selected_id: str | None,
):
    st.header("9 — Scenario Review (Engineer Mode)")
    st.caption(
        "Inspect a scenario's metrics and see how risk changes under current sidebar thresholds. "
        "Fleet View (sections 1–7) uses stored metrics. This view recomputes scores live. "
        "Use the **Scenario Analysis** selector in the sidebar to switch scenarios."
    )

    if risk_metrics is None or "risk_score" not in risk_metrics.columns:
        st.info("Risk metrics not available. Run `python scripts/compute_risk_metrics.py` first.")
        return

    if selected_id is None:
        st.info("Select a scenario from the sidebar to begin.")
        return

    # Get this scenario's rows
    risk_row = risk_metrics[risk_metrics["scenario_id"] == selected_id]
    merged_row = merged[merged["scenario_id"] == selected_id]

    if risk_row.empty:
        st.warning(f"No risk data found for scenario `{selected_id}`.")
        return

    risk_row = risk_row.iloc[0]
    merged_row = merged_row.iloc[0] if not merged_row.empty else pd.Series(dtype=object)

    # Live recomputed score
    live = compute_live_risk_score(risk_row, ttc_warning, ttc_critical)

    # TTC zone classification
    min_ttc = risk_row.get("min_ttc_s", None)
    if pd.isna(min_ttc):
        ttc_zone = "Unknown"
        zone_color = "gray"
    elif min_ttc >= ttc_warning:
        ttc_zone = "✅ Safe"
        zone_color = "green"
    elif min_ttc > ttc_critical:
        ttc_zone = "⚠️ Warning"
        zone_color = "orange"
    else:
        ttc_zone = "🔴 Critical"
        zone_color = "red"

    # ---- two-column layout ----
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Controls & Scores")

        # Stored vs live scores
        st.markdown("**Stored risk score** (from parquet)")
        stored_score = float(risk_row.get("risk_score", 0.0))
        st.metric("risk_score (stored)", f"{stored_score:.3f}")

        st.markdown("**Live review score** (recomputed with current thresholds)")
        st.metric(
            "risk_score (live)",
            f"{live['risk_score']:.3f}",
            delta=f"{live['risk_score'] - stored_score:+.3f} vs stored",
        )

        st.divider()

        # Risk breakdown table
        st.markdown("**Risk component breakdown**")
        breakdown_df = pd.DataFrame([
            {"Component": "TTC (weight 0.5)",           "Value": live["ttc_component"]},
            {"Component": "Closing speed (weight 0.3)", "Value": live["closing_component"]},
            {"Component": "Breaches (weight 0.2)",      "Value": live["breach_component"]},
            {"Component": "→ Live risk_score",          "Value": live["risk_score"]},
        ])
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

        # Score comparison bar chart
        fig, ax = plt.subplots(figsize=(4, 2.5))
        bars = ax.barh(
            ["Stored", "Live"],
            [stored_score, live["risk_score"]],
            color=["#888", "#d94f4f"],
            edgecolor="white",
        )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Risk Score")
        ax.set_title("Stored vs Live Score")
        ax.bar_label(bars, fmt="%.3f", padding=3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Scenario Summary")

        # Metric cards in columns
        c1, c2, c3 = st.columns(3)
        c1.metric("min_ttc_s", f"{min_ttc:.2f}s" if not pd.isna(min_ttc) else "N/A")
        max_closing = risk_row.get("max_closing_speed_mps", 0.0)
        c2.metric("max_closing_speed", f"{max_closing:.1f} m/s")
        num_tracks = merged_row.get("num_tracks", "N/A")
        c3.metric("num_tracks", int(num_tracks) if not pd.isna(num_tracks) else "N/A")

        c4, c5, c6 = st.columns(3)
        c4.metric("num_ttc_below_3s", int(risk_row.get("num_ttc_below_3s", 0)))
        c5.metric("num_ttc_below_1_5s", int(risk_row.get("num_ttc_below_1_5s", 0)))
        # Comfort score if available
        comfort_score_val = merged_row.get("comfort_score", None)
        c6.metric(
            "comfort_score",
            f"{comfort_score_val:.3f}" if comfort_score_val is not None and not pd.isna(comfort_score_val) else "N/A"
        )

        # TTC zone classification
        st.markdown(f"**TTC zone (current thresholds):** {ttc_zone}")
        st.caption(
            f"TTC zone based on: warning < {ttc_warning:.1f}s, critical < {ttc_critical:.1f}s"
        )

        st.divider()

        # TTC component breakdown visualisation
        st.markdown("**Risk component bars (live)**")
        fig2, ax2 = plt.subplots(figsize=(6, 2.5))
        components = ["TTC\n(×0.5)", "Closing\n(×0.3)", "Breaches\n(×0.2)"]
        values = [live["ttc_component"], live["closing_component"], live["breach_component"]]
        weighted = [0.5 * live["ttc_component"], 0.3 * live["closing_component"], 0.2 * live["breach_component"]]
        x = np.arange(len(components))
        width = 0.35
        bars_raw = ax2.bar(x - width/2, values, width, label="Raw component", color="#aac4e0", edgecolor="white")
        bars_wgt = ax2.bar(x + width/2, weighted, width, label="Weighted contribution", color="#d94f4f", edgecolor="white")
        ax2.set_xticks(x)
        ax2.set_xticklabels(components, fontsize=9)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Value")
        ax2.set_title("Live Component Breakdown")
        ax2.legend(fontsize=8)
        ax2.bar_label(bars_raw, fmt="%.2f", padding=2, fontsize=7)
        ax2.bar_label(bars_wgt, fmt="%.2f", padding=2, fontsize=7)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # Complexity summary
        if "num_close_interactions" in merged_row.index:
            adj = float(merged_row.get("num_close_interactions", 0)) * (interaction_distance / 5.0)
            st.markdown(
                f"**Complexity:** {merged_row.get('num_close_interactions', 'N/A')} stored close interactions "
                f"→ **{adj:.0f} adjusted** at {interaction_distance}m · "
                f"Complexity proxy: {merged_row.get('scenario_interest_score', 'N/A'):.3f}"
            )


# ============================================================
# SECTION 10 — SCENARIO PLAYBACK
# ============================================================

def render_scenario_playback(
    scenario_id: str | None,
    states_df: pd.DataFrame | None,
    tracks_df: pd.DataFrame | None,
):
    st.header("10 — Scenario Playback")
    st.caption(
        "Original Waymo trajectories. "
        "Risk thresholds do not affect this view — actor motion is fixed from the raw scenario."
    )

    play_tab, metrics_tab = st.tabs(["▶ Playback", "📊 Metrics"])

    with play_tab:
        if scenario_id is None:
            st.info("Select a scenario from the sidebar to begin.")
            return

        if states_df is None or tracks_df is None:
            st.info(
                "Silver tables not found. "
                "Ensure `data/silver/states.parquet` and `data/silver/tracks.parquet` exist."
            )
            return

        sc_states = states_df[
            (states_df["scenario_id"] == scenario_id) & (states_df["valid"] == True)
        ].copy()
        sc_tracks = tracks_df[tracks_df["scenario_id"] == scenario_id].copy()

        if sc_states.empty:
            st.info(f"No trajectory data available for scenario `{scenario_id}`.")
            return

        min_t = int(sc_states["timestep"].min())
        max_t = int(sc_states["timestep"].max())
        actor_count = sc_states["track_id"].nunique()

        # ---- metrics row ----
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Actors", actor_count)
        mc2.metric("Frames", f"{max_t - min_t + 1}")
        mc3.metric("Duration", f"{(max_t - min_t) * 0.1:.1f} s")
        if "object_type" in sc_tracks.columns:
            type_counts = sc_tracks["object_type"].value_counts()
            breakdown = "  ·  ".join(f"{v} {k.title()}" for k, v in type_counts.items())
        else:
            breakdown = "—"
        mc4.metric("Scene", breakdown)

        frame = st.slider(
            "Timestep",
            min_value=min_t,
            max_value=max_t,
            value=min_t,
            key="playback_frame",
        )

        # ---- identify SDC ----
        sdc_rows = sc_tracks[sc_tracks["is_sdc"] == True]
        sdc_track_id = sdc_rows.iloc[0]["track_id"] if not sdc_rows.empty else None

        # ---- dark theme style ----
        _DARK = {
            "figure.facecolor": "#12121e",
            "axes.facecolor":   "#1a1a2e",
            "axes.edgecolor":   "#3a3a5c",
            "axes.labelcolor":  "#9999bb",
            "xtick.color":      "#666688",
            "ytick.color":      "#666688",
            "grid.color":       "#252540",
            "grid.linewidth":   0.6,
            "text.color":       "#ccccdd",
            "font.size":        9,
            "axes.titlesize":   11,
            "axes.labelsize":   9,
            "xtick.labelsize":  8,
            "ytick.labelsize":  8,
        }
        plt.rcParams.update(_DARK)

        fig, ax = plt.subplots(figsize=(11, 6.5))

        # ---- scene auto-zoom ----
        x_all, y_all = sc_states["x"], sc_states["y"]
        x_rng = max(x_all.max() - x_all.min(), 30)
        y_rng = max(y_all.max() - y_all.min(), 30)
        pad_x, pad_y = x_rng * 0.12 + 8, y_rng * 0.12 + 8
        ax.set_xlim(x_all.min() - pad_x, x_all.max() + pad_x)
        ax.set_ylim(y_all.min() - pad_y, y_all.max() + pad_y)

        # ---- color scheme ----
        TYPE_COLORS = {
            "VEHICLE":    "#4a9eca",
            "PEDESTRIAN": "#f4a261",
            "CYCLIST":    "#52b788",
        }
        SDC_COLOR = "#e63946"

        # ---- draw actors ----
        for track_id, group in sc_states.groupby("track_id"):
            history = group[group["timestep"] <= frame].sort_values("timestep")
            if history.empty:
                continue

            obj_type = str(group["object_type"].iloc[0]).upper() if "object_type" in group.columns else "VEHICLE"
            is_sdc = track_id == sdc_track_id
            color = SDC_COLOR if is_sdc else TYPE_COLORS.get(obj_type, "#7777aa")
            base_alpha = 0.9 if is_sdc else 0.35
            lw = 2.2 if is_sdc else 1.0

            # Fading trajectory trail (last 40 steps)
            trail = history.tail(40)
            n = len(trail)
            if n > 1:
                for i in range(1, n):
                    seg_alpha = base_alpha * ((i / n) ** 0.55)
                    ax.plot(
                        trail["x"].values[i - 1 : i + 1],
                        trail["y"].values[i - 1 : i + 1],
                        color=color,
                        alpha=seg_alpha,
                        linewidth=lw,
                        solid_capstyle="round",
                        zorder=4,
                    )

            # Current position dot / star
            cur = history.iloc[-1]
            ax.scatter(
                cur["x"], cur["y"],
                s=200 if is_sdc else 45,
                c=color,
                marker="*" if is_sdc else "o",
                edgecolors="white" if is_sdc else color,
                linewidths=1.2 if is_sdc else 0.0,
                zorder=10 if is_sdc else 6,
                alpha=1.0,
            )

            # Heading arrow (if column available)
            if "heading_rad" in cur.index and not pd.isna(cur["heading_rad"]):
                arrow_len = 5.5 if is_sdc else 3.5
                dx = float(np.cos(cur["heading_rad"])) * arrow_len
                dy = float(np.sin(cur["heading_rad"])) * arrow_len
                ax.annotate(
                    "",
                    xy=(cur["x"] + dx, cur["y"] + dy),
                    xytext=(cur["x"], cur["y"]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=1.8 if is_sdc else 0.9,
                        alpha=0.95,
                    ),
                    zorder=11,
                )

            # SDC label
            if is_sdc:
                ax.annotate(
                    "  SDC",
                    (cur["x"], cur["y"]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color=SDC_COLOR,
                    zorder=12,
                )

        # ---- frame counter badge (top-left inside plot) ----
        ax.text(
            0.012, 0.975,
            f"t = {frame:03d}   {frame * 0.1:.1f} s",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            color="#ccccdd",
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#252540", edgecolor="#3a3a5c", alpha=0.9),
            zorder=20,
        )

        # ---- compact legend (bottom-right) ----
        legend_handles = [
            Line2D([0], [0], marker="*", color="w", markerfacecolor=SDC_COLOR,           markersize=11, label="SDC",        linestyle="None"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#4a9eca",            markersize=7,  label="Vehicle",     linestyle="None"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#f4a261",            markersize=7,  label="Pedestrian",  linestyle="None"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#52b788",            markersize=7,  label="Cyclist",     linestyle="None"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=8,
            facecolor="#1a1a2e",
            edgecolor="#3a3a5c",
            labelcolor="#ccccdd",
            borderpad=0.5,
            handletextpad=0.4,
            labelspacing=0.3,
        )

        ax.set_aspect("equal")
        ax.set_title(f"Scenario  {scenario_id[:8]}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("X (m)", labelpad=4)
        ax.set_ylabel("Y (m)", labelpad=4)
        ax.grid(True)
        plt.tight_layout(pad=1.0)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Reset rcParams so dark theme doesn't bleed into other charts
        plt.rcParams.update(plt.rcParamsDefault)

        st.caption(
            f"Trail shows last **40 frames** per actor · "
            "Arrow = heading direction · "
            "Risk thresholds **do not affect** this view."
        )

    with metrics_tab:
        st.info(
            "Detailed metrics and live risk recomputation are in "
            "**Section 9 — Scenario Review** above."
        )


# ============================================================
# SECTION 11 — MINI PLAYBACK PROTOTYPE
# ============================================================

def _draw_mini_scene(ax, sc_states, frame, sdc_track_id, top_risk_track_id):
    """Draw one frame of the compact scene onto ax. Reusable for grid tiles."""
    TYPE_COLORS = {"VEHICLE": "#4a9eca", "PEDESTRIAN": "#f4a261", "CYCLIST": "#52b788"}
    SDC_COLOR   = "#e63946"
    RISK_COLOR  = "#ff8c42"

    x_all, y_all = sc_states["x"], sc_states["y"]
    x_rng = max(x_all.max() - x_all.min(), 20)
    y_rng = max(y_all.max() - y_all.min(), 20)
    ax.set_xlim(x_all.min() - x_rng * 0.11 - 4, x_all.max() + x_rng * 0.11 + 4)
    ax.set_ylim(y_all.min() - y_rng * 0.11 - 4, y_all.max() + y_rng * 0.11 + 4)

    for track_id, group in sc_states.groupby("track_id"):
        history = group[group["timestep"] <= frame].sort_values("timestep")
        if history.empty:
            continue

        obj_type = str(group["object_type"].iloc[0]).upper() if "object_type" in group.columns else "VEHICLE"
        is_sdc      = track_id == sdc_track_id
        is_top_risk = track_id == top_risk_track_id

        if is_sdc:
            color, lw, dot_s, marker, z = SDC_COLOR, 1.8, 90, "*", 10
        elif is_top_risk:
            color, lw, dot_s, marker, z = RISK_COLOR, 1.1, 28, "o", 7
        else:
            color, lw, dot_s, marker, z = TYPE_COLORS.get(obj_type, "#5555aa"), 0.65, 16, "o", 5

        # Fading trail (last 25 steps)
        trail = history.tail(25)
        n = len(trail)
        base_a = 0.9 if is_sdc else (0.65 if is_top_risk else 0.28)
        if n > 1:
            for i in range(1, n):
                ax.plot(
                    trail["x"].values[i - 1 : i + 1],
                    trail["y"].values[i - 1 : i + 1],
                    color=color, alpha=base_a * (i / n) ** 0.5,
                    linewidth=lw, solid_capstyle="round", zorder=z - 1,
                )

        cur = history.iloc[-1]
        ax.scatter(
            cur["x"], cur["y"], s=dot_s, c=color, marker=marker,
            edgecolors="white" if is_sdc else "none",
            linewidths=0.7 if is_sdc else 0, zorder=z,
        )

        # Heading arrow
        if "heading_rad" in cur.index and not pd.isna(cur["heading_rad"]):
            alen = 3.5 if is_sdc else 2.0
            dx = float(np.cos(cur["heading_rad"])) * alen
            dy = float(np.sin(cur["heading_rad"])) * alen
            ax.annotate(
                "", xy=(cur["x"] + dx, cur["y"] + dy), xytext=(cur["x"], cur["y"]),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.1 if is_sdc else 0.55, alpha=0.9),
                zorder=z + 1,
            )


def render_mini_playback_prototype(
    merged: pd.DataFrame,
    states_df: pd.DataFrame | None,
    tracks_df: pd.DataFrame | None,
    selected_id: str | None,
):
    st.header("11 — Mini Playback Prototype")
    st.caption(
        "Compact scenario tile — proof-of-concept before building a grid. "
        "Use **▶ / ⏸** to autoplay or **← →** to step manually."
    )

    if selected_id is None or states_df is None or tracks_df is None:
        st.info("Select a scenario from the sidebar and ensure silver tables are loaded.")
        return

    sc_states = states_df[
        (states_df["scenario_id"] == selected_id) & (states_df["valid"] == True)
    ].copy()
    sc_tracks = tracks_df[tracks_df["scenario_id"] == selected_id].copy()

    if sc_states.empty:
        st.info("No trajectory data for this scenario.")
        return

    min_t = int(sc_states["timestep"].min())
    max_t = int(sc_states["timestep"].max())

    # ---- session state: reset when scenario changes ----
    if st.session_state.get("mini_scenario") != selected_id:
        st.session_state.mini_frame   = min_t
        st.session_state.mini_playing = False
        st.session_state.mini_scenario = selected_id

    frame = int(st.session_state.mini_frame)

    # ---- Identify SDC ----
    sdc_rows     = sc_tracks[sc_tracks["is_sdc"] == True]
    sdc_track_id = sdc_rows.iloc[0]["track_id"] if not sdc_rows.empty else None

    # ---- Top-risk actor = closest to SDC at current frame ----
    top_risk_track_id = None
    if sdc_track_id is not None:
        sdc_now = sc_states[(sc_states["track_id"] == sdc_track_id) & (sc_states["timestep"] == frame)]
        if not sdc_now.empty:
            sx, sy   = sdc_now.iloc[0]["x"], sdc_now.iloc[0]["y"]
            others   = sc_states[(sc_states["track_id"] != sdc_track_id) & (sc_states["timestep"] == frame)]
            if not others.empty:
                dists = np.sqrt((others["x"] - sx) ** 2 + (others["y"] - sy) ** 2)
                top_risk_track_id = others.loc[dists.idxmin(), "track_id"]

    # ---- Metrics from merged ----
    mrow = merged[merged["scenario_id"] == selected_id]
    risk_score = mrow["risk_score"].iloc[0]   if not mrow.empty and "risk_score"  in mrow.columns else None
    min_ttc    = mrow["min_ttc_s"].iloc[0]    if not mrow.empty and "min_ttc_s"   in mrow.columns else None
    num_tracks = mrow["num_tracks"].iloc[0]   if not mrow.empty else None

    # ---- Card: narrow centre column ----
    _, card_col, _ = st.columns([1, 1.2, 1])
    with card_col:
        # Scenario ID header row
        progress_pct = (frame - min_t) / max(max_t - min_t, 1)
        st.markdown(
            f"<div style='font-family:monospace;font-size:11px;color:#7777aa;"
            f"letter-spacing:0.04em;margin-bottom:3px;'>"
            f"{selected_id[:16]}"
            f"<span style='float:right;color:#444466;'>{frame:03d}/{max_t:03d}</span>"
            f"</div>"
            f"<div style='height:2px;background:linear-gradient(to right,#e63946 {progress_pct*100:.0f}%,#252540 0%);border-radius:1px;margin-bottom:4px;'></div>",
            unsafe_allow_html=True,
        )

        # ---- Compact dark scene figure ----
        _MINI_DARK = {
            "figure.facecolor": "#12121e", "axes.facecolor": "#1a1a2e",
            "axes.edgecolor":   "#252540", "axes.labelcolor": "#444466",
            "xtick.color": "#333355",      "ytick.color":     "#333355",
            "grid.color":  "#1e1e38",      "grid.linewidth":  0.4,
            "text.color":  "#888899",      "font.size":       7,
        }
        plt.rcParams.update(_MINI_DARK)

        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        fig.patch.set_facecolor("#12121e")

        _draw_mini_scene(ax, sc_states, frame, sdc_track_id, top_risk_track_id)

        # Frame badge (tiny, inside plot)
        ax.text(
            0.03, 0.97, f"t={frame:03d}  {frame * 0.1:.1f}s",
            transform=ax.transAxes, fontsize=6, color="#666688", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#12121e", edgecolor="none", alpha=0.85),
        )

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#252540")
            spine.set_linewidth(0.5)
        ax.grid(True, alpha=0.25)
        plt.tight_layout(pad=0.2)

        # Render to bytes → st.image for exact fixed width
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=115,
                    bbox_inches="tight", facecolor="#12121e")
        buf.seek(0)
        plt.close(fig)
        plt.rcParams.update(plt.rcParamsDefault)

        st.image(buf, width=370)

        # One-line metrics
        r_s  = f"{risk_score:.3f}" if risk_score is not None and not pd.isna(risk_score) else "—"
        t_s  = f"{min_ttc:.2f}s"   if min_ttc    is not None and not pd.isna(min_ttc)    else "—"
        trk  = str(int(num_tracks)) if num_tracks is not None and not pd.isna(num_tracks) else "—"
        st.markdown(
            f"<div style='font-size:11px;color:#666688;text-align:center;"
            f"font-family:monospace;margin-top:2px;'>"
            f"risk&nbsp;<b style='color:#e63946'>{r_s}</b>&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"ttc&nbsp;<b style='color:#f4a261'>{t_s}</b>&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"<b style='color:#4a9eca'>{trk}</b>&nbsp;tracks"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ---- Controls row ----
        b1, b2, b3, slider_col = st.columns([1, 1, 1, 4])
        with b1:
            if st.button("←", key="mini_prev", help="Previous frame"):
                st.session_state.mini_frame   = max(min_t, frame - 1)
                st.session_state.mini_playing = False
                st.rerun()
        with b2:
            lbl = "⏸" if st.session_state.mini_playing else "▶"
            if st.button(lbl, key="mini_play", help="Play / Pause"):
                st.session_state.mini_playing = not st.session_state.mini_playing
                st.rerun()
        with b3:
            if st.button("→", key="mini_next", help="Next frame"):
                st.session_state.mini_frame   = min(max_t, frame + 1)
                st.session_state.mini_playing = False
                st.rerun()
        with slider_col:
            new_frame = st.slider(
                "", min_value=min_t, max_value=max_t, value=frame,
                key="mini_slider", label_visibility="collapsed",
            )
            if new_frame != frame:
                st.session_state.mini_frame   = new_frame
                st.session_state.mini_playing = False
                st.rerun()

    # ---- Autoplay loop ----
    if st.session_state.mini_playing:
        time.sleep(0.08)
        next_f = st.session_state.mini_frame + 1
        st.session_state.mini_frame = next_f if next_f <= max_t else min_t
        st.rerun()


# ============================================================
# SECTION — EXPLORER GIF GRID
# ============================================================

def render_explorer_gif_grid(
    merged: pd.DataFrame,
    review_sorted_ids: list,
):
    previews_dir = PROJECT_ROOT / "data" / "previews"

    st.subheader("Scenario Explorer")
    st.caption(
        "Top scenarios sorted by risk score. "
        "GIFs use real Waymo map geometry and actor trajectories. "
        "Click **Review →** to investigate a scenario in detail."
    )

    available = [sid for sid in review_sorted_ids if (previews_dir / f"{sid}.gif").exists()]

    if not available:
        st.warning(
            "No preview GIFs found. Generate them first:\n\n"
            "```\npython scripts/generate_preview_gifs.py\n```"
        )
        return

    metrics_by_id = {row["scenario_id"]: row for _, row in merged.iterrows()}

    top3 = available[:3]
    cols = st.columns(3)
    for j, sid in enumerate(top3):
        mrow       = metrics_by_id.get(sid)
        risk_score = mrow["risk_score"] if mrow is not None and "risk_score" in mrow.index and not pd.isna(mrow["risk_score"]) else None
        min_ttc    = mrow["min_ttc_s"]  if mrow is not None and "min_ttc_s"  in mrow.index and not pd.isna(mrow["min_ttc_s"])  else None
        num_tracks = mrow["num_tracks"] if mrow is not None and "num_tracks" in mrow.index and not pd.isna(mrow["num_tracks"]) else None

        r_s = f"{risk_score:.2f}" if risk_score is not None else "—"
        t_s = f"{min_ttc:.2f}s"   if min_ttc    is not None else "—"
        trk = str(int(num_tracks)) if num_tracks is not None else "—"

        gif_b64 = base64.b64encode((previews_dir / f"{sid}.gif").read_bytes()).decode()

        with cols[j]:
            st.markdown(
                f'<div style="width:100%;height:270px;overflow:hidden;'
                f'background:#12121e;border-radius:8px;">'
                f'<img src="data:image/gif;base64,{gif_b64}" '
                f'style="width:100%;height:270px;object-fit:cover;border-radius:8px;display:block;"/>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**{sid[:8]}**  \n"
                f"risk **{r_s}** · ttc **{t_s}**  \n"
                f"{trk} actors"
            )
            if st.button("Review →", key=f"exp_rev_{sid}"):
                st.session_state.scenario_idx = review_sorted_ids.index(sid) if sid in review_sorted_ids else 0
                st.session_state.explorer_selected = sid

    # Feedback banner after a tile is selected
    if st.session_state.get("explorer_selected"):
        sel = st.session_state.explorer_selected
        st.success(f"✓ **{sel[:8]}** loaded — switch to the **Review** tab to investigate.")

    st.divider()

    diag_col, score_col = st.columns([1.1, 0.9])

    with score_col:
        st.markdown("#### How Scores Are Calculated")

        with st.expander("🔴 Risk Score Logic"):
            st.markdown(
                "Risk ≈ **Time × Severity × Exposure**\n\n"
                "| Dimension | Metric |\n|---|---|\n"
                "| Time margin | `min_ttc_s` |\n"
                "| Severity | `max_closing_speed_mps` |\n"
                "| Exposure | `num_ttc_below_3s / below_1_5s` |"
            )
            st.code(
                "risk = 0.5 * ttc_component\n"
                "     + 0.3 * closing_component\n"
                "     + 0.2 * breach_component\n\n"
                "closing = min(1.0, closing_speed / 15.0)\n"
                "breach  = min(1.0, (below_3s + 2*below_1.5s) / 20.0)",
                language="python",
            )

        with st.expander("🔵 Complexity Score Logic"):
            st.code(
                "complexity = (\n"
                "  0.30 * min(1.0, 10.0 / (min_dist + 0.1))   # proximity\n"
                "+ 0.25 * min(1.0, close_interactions / 50.0)  # density\n"
                "+ 0.20 * min(1.0, sdc_max_speed / 25.0)       # speed\n"
                "+ 0.15 * min(1.0, unique_actors / 10.0)        # diversity\n"
                "+ 0.10 * min(1.0, sdc_distance / 200.0)        # coverage\n"
                ")",
                language="python",
            )

        with st.expander("🟢 Comfort Score Logic"):
            st.code(
                "comfort = (0.25 * accel_component\n"
                "         + 0.30 * decel_component\n"
                "         + 0.30 * jerk_component\n"
                "         + 0.15 * heading_component)\n\n"
                "# thresholds: accel/decel > 4 m/s², jerk > 10 m/s³\n"
                "# heading_rate > 0.8 rad/s  |  dt = 0.1 s (10 Hz)",
                language="python",
            )

        with st.expander("📖 Metric Glossary"):
            st.markdown(
                "| Metric | Definition |\n|---|---|\n"
                "| `min_ttc_s` | Min Time-to-Collision across all actor/timestep pairs |\n"
                "| `max_closing_speed_mps` | Max closing speed while approaching |\n"
                "| `num_ttc_below_3s` | Actor/timestep pairs where TTC < 3 s |\n"
                "| `num_ttc_below_1_5s` | Actor/timestep pairs where TTC < 1.5 s |\n"
                "| `min_sdc_distance_m` | Closest any actor got to the SDC |\n"
                "| `sdc_max_speed_mps` | SDC peak speed in scenario |\n"
                "| `num_tracks` | Total actor tracks in scenario |\n"
                "| `risk_score` | Composite safety risk [0–1] |\n"
                "| `comfort_score` | Ride abruptness [0–1] |\n"
                "| `scenario_interest_score` | Interaction complexity [0–1] |"
            )

    with diag_col:
        components.html(
        """
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({ startOnLoad: true, theme: 'base',
            themeVariables: { fontSize: '13px', fontFamily: 'sans-serif' } });
        </script>
        <style>
            .diagram-wrap { cursor: pointer; }
            .hint { font-size: 11px; color: #999; margin: 0 0 4px 2px;
                    font-family: sans-serif; user-select: none; }
        </style>

        <!-- Simple overview (visible by default) -->
        <div id="simple" class="diagram-wrap" onclick="toggle()">
            <p class="hint">&#9654; Click to expand full pipeline</p>
            <div class="mermaid">
            flowchart LR
                A["Waymo Scenario Logs<br/>Position &#8226; Velocity &#8226; Actor State"]
                B["Interaction Signals<br/>Relative Position &#8226; Relative Velocity"]
                C["Safety Metrics<br/>TTC &#8226; Closing Speed &#8226; Exposure"]
                D["Scenario Risk Score"]
                A --> B
                B --> C
                C --> D
                classDef logs fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px;
                classDef interaction fill:#ede7f6,stroke:#5e35b1,stroke-width:2px;
                classDef metrics fill:#fff3e0,stroke:#fb8c00,stroke-width:2px;
                classDef risk fill:#ffebee,stroke:#e53935,stroke-width:2px;
                class A logs
                class B interaction
                class C metrics
                class D risk
            </div>
        </div>

        <!-- Detailed pipeline (hidden by default, pre-rendered for instant toggle) -->
        <div id="detail" class="diagram-wrap" style="visibility:hidden;height:0;overflow:hidden;" onclick="toggle()">
            <p class="hint">&#9660; Click to collapse</p>
            <div class="mermaid">
            %%{init: {'theme':'base'}}%%
            flowchart LR
                subgraph A["Waymo Dataset"]
                    A1["Positions"]
                    A2["Velocities"]
                    A3["Actor State / Validity"]
                    A4["Heading"]
                    A5["Map"]
                end
                subgraph B["Supporting Metrics"]
                    B1["Distance"]
                    B2["Relative Velocity"]
                    B3["TTC"]
                    B4["Exposure"]
                    B5["Accel / Jerk"]
                    B6["Interactions"]
                end
                subgraph C["Final Outputs"]
                    C1["Risk"]
                    C2["Comfort"]
                    C3["Complexity"]
                    C4["Explorer"]
                    C5["Insights"]
                end
                A1 --> B1
                A3 -.-> B1
                A2 --> B2
                A3 -.-> B2
                B1 --> B3
                B2 --> B3
                A3 -.-> B3
                B3 --> B4
                A2 --> B5
                A4 --> B5
                A3 -.-> B5
                B1 --> B6
                A3 -.-> B6
                B3 --> C1
                B2 --> C1
                B4 --> C1
                B5 --> C2
                B6 --> C3
                A5 --> C4
                C1 --> C4
                C1 --> C5
                C2 --> C5
                C3 --> C5
                classDef dataset fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px;
                classDef metrics fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px;
                classDef outputs fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px;
                class A1,A2,A3,A4,A5 dataset
                class B1,B2,B3,B4,B5,B6 metrics
                class C1,C2,C3,C4,C5 outputs
            </div>
        </div>

        <script>
        function toggle() {
            var s = document.getElementById('simple');
            var d = document.getElementById('detail');
            if (s.style.visibility !== 'hidden') {
                s.style.visibility = 'hidden'; s.style.height = '0'; s.style.overflow = 'hidden';
                d.style.visibility = 'visible'; d.style.height = 'auto'; d.style.overflow = 'visible';
            } else {
                d.style.visibility = 'hidden'; d.style.height = '0'; d.style.overflow = 'hidden';
                s.style.visibility = 'visible'; s.style.height = 'auto'; s.style.overflow = 'visible';
            }
        }
        </script>
        """,
        height=520,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(
        page_title="Waymo Validation Lab",
        page_icon="🚗",
        layout="wide",
    )

    st.title("🚗 Waymo Validation Lab")

    # ── sidebar: assumption controls ─────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Assumption Controls")

        preset = st.radio(
            "Select Preset",
            ["Conservative", "Balanced", "Strict"],
            index=1,
            help="Choose preset values for risk and complexity thresholds",
        )
        preset_values = {
            "Conservative": {"ttc_warning": 4.0, "ttc_critical": 2.0, "interaction_distance": 7},
            "Balanced":     {"ttc_warning": 3.0, "ttc_critical": 1.5, "interaction_distance": 5},
            "Strict":       {"ttc_warning": 2.0, "ttc_critical": 1.0, "interaction_distance": 3},
        }
        current_preset = preset_values[preset]

        ttc_warning = st.slider(
            "TTC Warning (s)", min_value=1.0, max_value=5.0,
            value=current_preset["ttc_warning"], step=0.1,
        )
        ttc_critical = st.slider(
            "TTC Critical (s)", min_value=0.5, max_value=3.0,
            value=current_preset["ttc_critical"], step=0.1,
        )
        interaction_distance = st.slider(
            "Interaction Distance (m)", min_value=2, max_value=10,
            value=current_preset["interaction_distance"], step=1,
        )

    # ── load data ─────────────────────────────────────────────────────────────
    scenarios = load_parquet_if_exists(SILVER_DIR / "scenarios.parquet")
    if scenarios is None:
        st.error(
            "❌ `data/silver/scenarios.parquet` missing. "
            "Run: `python scripts/waymo_real_parser.py`"
        )
        return

    scenario_metrics    = load_parquet_if_exists(GOLD_DIR / "scenario_metrics.parquet")
    interaction_metrics = load_parquet_if_exists(GOLD_DIR / "interaction_metrics.parquet")
    risk_metrics        = load_parquet_if_exists(GOLD_DIR / "risk_metrics.parquet")
    comfort_metrics     = load_parquet_if_exists(GOLD_DIR / "comfort_metrics.parquet")

    if interaction_metrics is None:
        st.error(
            "❌ `data/gold/interaction_metrics.parquet` missing. "
            "Run: `python scripts/compute_interaction_metrics.py`"
        )
        return

    loaded_files = {
        "scenario_metrics.parquet":    scenario_metrics    is not None,
        "interaction_metrics.parquet": interaction_metrics is not None,
        "risk_metrics.parquet":        risk_metrics        is not None,
        "comfort_metrics.parquet":     comfort_metrics     is not None,
    }

    merged = build_merged_table(
        scenarios, scenario_metrics, interaction_metrics, risk_metrics, comfort_metrics
    )

    states_df, tracks_df = load_silver_for_playback(
        str(SILVER_DIR / "states.parquet"),
        str(SILVER_DIR / "tracks.parquet"),
    )

    # ── scenario ordering (highest risk first) ────────────────────────────────
    if risk_metrics is not None and "risk_score" in risk_metrics.columns:
        review_sorted_ids = (
            risk_metrics.sort_values("risk_score", ascending=False)["scenario_id"].tolist()
        )
    else:
        review_sorted_ids = scenarios["scenario_id"].tolist()

    # ── session state: shared selected scenario ───────────────────────────────
    if "scenario_idx" not in st.session_state:
        st.session_state.scenario_idx = 0

    # ── sidebar: scenario selector (synced with Explorer buttons) ─────────────
    with st.sidebar:
        st.divider()
        st.markdown("**🔬 Scenario Analysis**")
        safe_idx = min(st.session_state.scenario_idx, len(review_sorted_ids) - 1)
        selected_scenario = st.selectbox(
            "Select Scenario",
            review_sorted_ids,
            index=safe_idx,
            help="Updated by Explorer 'Review →' buttons or changed here directly.",
        )
        # Sync session state when user manually changes the sidebar selectbox
        new_idx = (
            review_sorted_ids.index(selected_scenario)
            if selected_scenario in review_sorted_ids else 0
        )
        if new_idx != st.session_state.scenario_idx:
            st.session_state.scenario_idx = new_idx

    # ── 3 top-level tabs ──────────────────────────────────────────────────────
    explorer_tab, review_tab, metrics_tab = st.tabs(
        ["🗺️  Explorer", "🔍  Review", "📊  Metrics"]
    )

    with explorer_tab:
        render_explorer_gif_grid(merged, review_sorted_ids)

    with review_tab:
        render_scenario_review(
            merged, risk_metrics, comfort_metrics,
            ttc_warning, ttc_critical, interaction_distance,
            selected_scenario,
        )
        st.divider()
        render_scenario_playback(selected_scenario, states_df, tracks_df)
        st.divider()
        render_mini_playback_prototype(merged, states_df, tracks_df, selected_scenario)

    with metrics_tab:
        render_dataset_summary(
            scenarios, loaded_files, ttc_warning, ttc_critical, interaction_distance
        )
        st.divider()
        render_top_scenarios_table(merged)
        st.divider()
        render_risk_overview(merged, ttc_warning, ttc_critical)
        st.divider()
        render_complexity_overview(merged, interaction_distance)
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
