#!/usr/bin/env python3
"""
Waymo Validation Lab — Public / Portfolio Version

Scenario safety scoring dashboard built on real Waymo Open Dataset logs.
Showcases risk, complexity, and comfort metrics across 10 real AV scenarios.

Run with: streamlit run scripts/app.py
"""

import io
import time
import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import plotly.graph_objects as go

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
    return "interaction_percentile" if "interaction_percentile" in df.columns else ("risk_score" if has_risk_score(df) else "scenario_interest_score")


def risk_label(df: pd.DataFrame) -> str:
    if "risk_score" in df.columns and df["risk_score"].notna().any():
        return "Interaction Score"
    return "Interaction Risk Proxy (scenario_interest_score)"


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
# SECTION — METRIC SUMMARY CARDS
# ============================================================

def _mini_hist(ax, values, color):
    ax.hist(values, bins=10, color=color, alpha=0.85, edgecolor="none")
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7, colors="#555555")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.5)


def _render_metric_cards(merged: pd.DataFrame):
    cards = [
        {"title": "Interaction Score", "col": "risk_score",              "color": "#e53935", "label": "Avg Interaction Score",       "fmt": "{:.3f}", "xlabel": "Interaction Score"},
        {"title": "Complexity",        "col": "scenario_interest_score", "color": "#1e88e5", "label": "Avg Complexity Score",        "fmt": "{:.3f}", "xlabel": "Scenario Complexity Score"},
        {"title": "Comfort",           "col": "comfort_score",           "color": "#43a047", "label": "Avg Comfort Score",           "fmt": "{:.3f}", "xlabel": "Comfort / Smoothness Score"},
    ]
    cols = st.columns(3)
    for col_ui, card in zip(cols, cards):
        series = merged[card["col"]].dropna() if card["col"] in merged.columns else pd.Series([], dtype=float)
        avg = series.mean() if not series.empty else None
        val_str = card["fmt"].format(avg) if avg is not None else "—"
        with col_ui:
            st.markdown(
                f'<div class="av-card">'
                f'<div class="av-card-title">{card["title"]}</div>'
                f'<div class="av-card-value" style="color:{card["color"]}">{val_str}</div>'
                f'<div class="av-card-sub">{card["label"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if not series.empty:
                fig, ax = plt.subplots(figsize=(3, 1.5))
                fig.patch.set_facecolor("#ffffff")
                _mini_hist(ax, series.values, card["color"])
                ax.set_xlabel(card["xlabel"], fontsize=8, color="#555555")
                ax.set_ylabel("Count", fontsize=8, color="#555555")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


# ============================================================
# SECTION — EXPLORER GIF GRID
# ============================================================

@st.cache_data(show_spinner=False)
def _load_gif_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def render_explorer_gif_grid(
    merged: pd.DataFrame,
    review_sorted_ids: list,
):
    previews_dir = PROJECT_ROOT / "data" / "previews"

    st.subheader("Scenario Explorer")
    st.caption("Top 3 scenarios by interaction criticality. GIFs use real Waymo map geometry.")

    available = [sid for sid in review_sorted_ids if (previews_dir / f"{sid}.gif").exists()]

    if not available:
        st.warning("No preview GIFs found.\n\n```\npython scripts/generate_preview_gifs.py\n```")
        return

    metrics_by_id = {row["scenario_id"]: row for _, row in merged.iterrows()}

    # ── GIF grid ──────────────────────────────────────────────
    cols = st.columns(3)
    for j, sid in enumerate(available[:3]):
        mrow       = metrics_by_id.get(sid)
        risk_score = mrow["risk_score"] if mrow is not None and "risk_score" in mrow.index and not pd.isna(mrow["risk_score"]) else None
        min_ttc    = mrow["min_ttc_s"]  if mrow is not None and "min_ttc_s"  in mrow.index and not pd.isna(mrow["min_ttc_s"])  else None
        num_tracks = mrow["num_tracks"] if mrow is not None and "num_tracks" in mrow.index and not pd.isna(mrow["num_tracks"]) else None

        r_s = f"{risk_score:.2f}" if risk_score is not None else "—"
        t_s = f"{min_ttc:.2f}s"   if min_ttc    is not None else "—"
        trk = str(int(num_tracks)) if num_tracks is not None else "—"
        gif_b64 = _load_gif_b64(str(previews_dir / f"{sid}.gif"))

        with cols[j]:
            st.markdown(
                f'<div style="width:100%;height:270px;overflow:hidden;background:#12121e;border-radius:8px;">'
                f'<img src="data:image/gif;base64,{gif_b64}" '
                f'style="width:100%;height:270px;object-fit:cover;border-radius:8px;display:block;"/>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{sid[:8]}**  \nrisk **{r_s}** · ttc **{t_s}**  \n{trk} actors")

    # ── Metric summary cards ───────────────────────────────────
    st.divider()
    _render_metric_cards(merged)

    # ── Interactive Plotly scatter ─────────────────────────────
    st.divider()
    st.markdown("#### Interaction Score vs. Complexity")
    st.caption("Higher interaction scores indicate tighter time margins, higher closing speeds, or repeated close interactions.")

    plot_df = merged.dropna(subset=["risk_score", "scenario_interest_score"]).copy()
    if not plot_df.empty:
        # ── colour buckets (percentile-based, rename for UI) ──
        pct = plot_df["risk_score"].rank(pct=True)
        bucket_label = [
            "Review now"  if p >= 0.95 else
            "Review next" if p >= 0.85 else
            "Monitor"
            for p in pct
        ]
        bucket_color = {
            "Review now":  "#e63946",
            "Review next": "#f4a261",
            "Monitor":     "#4a9eca",
        }
        plot_df = plot_df.copy()
        plot_df["_bucket"] = bucket_label

        # ── build optional tooltip columns safely ──
        for col in ["comfort_score", "min_ttc_s", "max_closing_speed_mps",
                    "num_tracks", "num_ttc_below_3s"]:
            if col not in plot_df.columns:
                plot_df[col] = float("nan")

        fig_pl = go.Figure()

        for bucket in ["Monitor", "Review next", "Review now"]:
            sub = plot_df[plot_df["_bucket"] == bucket]
            if sub.empty:
                continue

            def _fmt(v, fmt="{:.3f}"):
                return fmt.format(v) if pd.notna(v) else "—"

            custom = sub[["scenario_id", "risk_score", "scenario_interest_score",
                          "comfort_score", "min_ttc_s", "max_closing_speed_mps",
                          "num_tracks", "num_ttc_below_3s"]].values

            fig_pl.add_trace(go.Scatter(
                x=sub["scenario_interest_score"],
                y=sub["risk_score"],
                mode="markers",
                name=bucket,
                marker=dict(
                    color=bucket_color[bucket],
                    size=7,
                    opacity=0.85,
                    line=dict(width=0.6, color="rgba(255,255,255,0.35)"),
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Interaction Score: %{customdata[1]:.3f}<br>"
                    "Complexity Score:  %{customdata[2]:.3f}<br>"
                    "Comfort Score:     %{customdata[3]:.3f}<br>"
                    "Min TTC:           %{customdata[4]:.2f} s<br>"
                    "Max Closing Speed: %{customdata[5]:.1f} m/s<br>"
                    "Actors:            %{customdata[6]:.0f}<br>"
                    "<extra></extra>"
                ),
            ))

        # quadrant reference lines
        fig_pl.add_hline(y=0.6, line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"))
        fig_pl.add_vline(x=0.6, line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"))

        fig_pl.update_layout(
            height=420,
            margin=dict(l=50, r=20, t=20, b=50),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            xaxis=dict(
                title="Complexity Score",
                range=[-0.02, 1.05],
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
                tickfont=dict(size=11, color="#aaaaaa"),
                title_font=dict(size=12, color="#aaaaaa"),
            ),
            yaxis=dict(
                title="Interaction Score",
                range=[-0.02, 1.05],
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
                tickfont=dict(size=11, color="#aaaaaa"),
                title_font=dict(size=12, color="#aaaaaa"),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.01,
                xanchor="right", x=1,
                font=dict(size=11, color="#aaaaaa"),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
            hoverlabel=dict(
                bgcolor="#1a1f2e",
                bordercolor="#333d55",
                font=dict(size=12, color="#e2e8f0", family="monospace"),
            ),
        )

        st.plotly_chart(
            fig_pl,
            use_container_width=True,
        )

        st.caption("⬆ Top-right = priority review zone")
    else:
        st.info("Not enough data to render Interaction Score vs. Complexity chart.")

    # ── Methodology (collapsed by default) ────────────────────
    st.divider()
    with st.expander("Methodology (derived metrics)"):
        st.markdown(
            "Interaction metrics are derived from actor trajectories contained in the Waymo Open Dataset.  \n"
            "These derived signals summarize relative motion, proximity, and temporal margins between actors."
        )

        diagrams_dir = PROJECT_ROOT / "data" / "diagrams"
        img_simple = diagrams_dir / "pipeline_simple.png"
        img_full   = diagrams_dir / "pipeline_full.png"

        if "pipeline_expanded" not in st.session_state:
            st.session_state.pipeline_expanded = False

        if st.session_state.pipeline_expanded:
            if img_full.exists():
                st.image(str(img_full), use_container_width=True)
            if st.button("▲ Collapse pipeline", key="pipeline_collapse"):
                st.session_state.pipeline_expanded = False
                st.rerun()
        else:
            if img_simple.exists():
                st.image(str(img_simple), use_container_width=True)
            if st.button("▼ Expand full pipeline", key="pipeline_expand"):
                st.session_state.pipeline_expanded = True
                st.rerun()


# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(
        page_title="Waymo Validation Lab",
        page_icon="",
        layout="wide",
    )

    # ── global CSS ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* ── page background ── */
        .stApp { background-color: #ffffff; }
        section[data-testid="stSidebar"] { display: none; }

        /* ── hide default Streamlit header decoration ── */
        header[data-testid="stHeader"] { background: transparent; }

        /* ── tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid #d1d9e6;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #8899bb;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 8px 20px;
            border-radius: 4px 4px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background: #ffffff !important;
            color: #1a2540 !important;
            border-top: 2px solid #3b82f6;
        }

        /* ── metric cards ── */
        .av-card {
            background: #ffffff;
            border: 1px solid #d1d9e6;
            border-radius: 8px;
            padding: 18px 20px 12px 20px;
            margin-bottom: 8px;
        }
        .av-card-title {
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #8899bb;
            margin-bottom: 4px;
        }
        .av-card-value {
            font-family: 'JetBrains Mono', 'Fira Mono', 'Courier New', monospace;
            font-size: 1.9rem;
            font-weight: 700;
            color: #1a2540;
            line-height: 1.1;
        }
        .av-card-sub {
            font-size: 0.72rem;
            color: #8899bb;
            margin-top: 4px;
        }

        /* ── footer ── */
        .av-footer {
            margin-top: 48px;
            padding-top: 16px;
            border-top: 1px solid #d1d9e6;
            font-size: 0.65rem;
            color: #8899bb;
            letter-spacing: 0.04em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── page header ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="padding: 48px 0 40px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 48px;">
            <div style="font-size:2.4rem; font-weight:700; color:#1a2540; line-height:1.1; letter-spacing:-0.01em;">
                AV Validation Lab
            </div>
            <div style="font-size:1.1rem; color:#4b5563; margin-top:10px; font-weight:400;">
                Evaluation metrics derived from the Waymo Open Dataset
            </div>
            <div style="font-size:0.8rem; color:#6b7280; font-family:monospace; margin-top:8px;">
                250 scenarios &nbsp;·&nbsp; 18,151 actor tracks &nbsp;·&nbsp; Motion Dataset v1.3.1 &nbsp;·&nbsp; Validation Split
            </div>
            <div style="font-size:0.8rem; margin-top:6px;">
                <a href="https://waymo.com/open/" style="color:#3b82f6; text-decoration:none;">Dataset</a>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                <a href="https://github.com/rafaelmaranon/waymo-validation-lab" style="color:#3b82f6; text-decoration:none;">GitHub</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fixed scoring parameters for the public app
    ttc_warning          = 3.0
    ttc_critical         = 1.5
    interaction_distance = 5

    # ── load data ──────────────────────────────────────────────────────────────────────────
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

    merged = build_merged_table(
        scenarios, scenario_metrics, interaction_metrics, risk_metrics, comfort_metrics
    )

    # ── derived UI-facing metric: interaction percentile ──────────────────────────────────
    if "risk_score" in merged.columns and merged["risk_score"].notna().any():
        merged["interaction_percentile"] = merged["risk_score"].rank(pct=True)
    else:
        merged["interaction_percentile"] = pd.Series([np.nan] * len(merged), index=merged.index)

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

    render_explorer_gif_grid(merged, review_sorted_ids)

    st.markdown(
        '<div class="av-footer">'
        'Data: <a href="https://waymo.com/open/" style="color:#3b82f6; text-decoration:none;">Waymo Open Motion Dataset (v1.3.1)</a>'
        ' &nbsp;·&nbsp; Validation Split &nbsp;·&nbsp; Evaluation metrics derived from actor trajectories'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
