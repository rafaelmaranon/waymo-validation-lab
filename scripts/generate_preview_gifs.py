#!/usr/bin/env python3
"""
Generate compact preview GIFs for the top scenarios.

Uses real Waymo protobuf data:
  - map_features (lane centerlines, road edges, crosswalks)
  - actor trajectories (x, y, heading, object_type, is_sdc)
  - risk metrics overlay (risk_score, min_ttc_s)

Output:  data/previews/{scenario_id}.gif

Run offline before starting the dashboard:
    python scripts/generate_preview_gifs.py
"""

import io
import sys
import struct
from pathlib import Path
from typing import Iterator, Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "proto"))

try:
    from waymo_open_dataset.protos import scenario_pb2
except ImportError as e:
    print(f"❌ Waymo protobuf import failed: {e}")
    print("   Ensure proto/waymo_open_dataset/protos/scenario_pb2.py exists.")
    sys.exit(1)

# ── paths ────────────────────────────────────────────────────────────────────
TFRECORD_DIR = Path.home() / "datasets" / "waymo" / "raw"
SILVER_DIR   = project_root / "data" / "silver"
GOLD_DIR     = project_root / "data" / "gold"
PREVIEWS_DIR = project_root / "data" / "previews"

# ── render config ─────────────────────────────────────────────────────────────
GIF_PX       = 320          # square output in pixels
GIF_DPI      = 100
FIGSIZE      = GIF_PX / GIF_DPI   # inches
TRAIL_FRAMES = 18
FPS          = 10           # frames per second in output GIF
FRAME_STEP   = 1            # take every Nth timestep (1 = all)
GIF_COLORS   = 96           # palette size for quantization

# ── visual constants ──────────────────────────────────────────────────────────
TYPE_COLORS = {"VEHICLE": "#4a9eca", "PEDESTRIAN": "#f4a261", "CYCLIST": "#52b788"}
SDC_COLOR   = "#e63946"
RISK_COLOR  = "#ff8c42"

_DARK_RCPARAMS = {
    "figure.facecolor": "#12121e",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#252540",
    "grid.color":       "#1e1e38",
    "grid.linewidth":   0.4,
    "font.size":        7,
    "axes.titlesize":   7,
}


# ── TFRecord reader ───────────────────────────────────────────────────────────

def read_tfrecord(path: Path) -> Iterator[bytes]:
    """Yield raw record bytes from a TFRecord file (no TF dependency)."""
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) != 8:
                break
            length = struct.unpack("<Q", header)[0]
            f.read(4)                      # masked crc of length
            data = f.read(length)
            if len(data) != length:
                break
            f.read(4)                      # masked crc of data
            yield data


# ── map geometry extraction ───────────────────────────────────────────────────

def extract_map_geometry(
    scenario,
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    Return (lane_lines, edge_lines, crosswalk_polys).
    Each element is a (xs: list, ys: list) tuple.
    """
    lanes: List[Tuple] = []
    edges: List[Tuple] = []
    crosswalks: List[Tuple] = []

    for mf in scenario.map_features:
        feat = mf.WhichOneof("feature_data")
        if feat == "lane":
            pts = mf.lane.polyline
            if len(pts) >= 2:
                lanes.append(([p.x for p in pts], [p.y for p in pts]))
        elif feat == "road_edge":
            pts = mf.road_edge.polyline
            if len(pts) >= 2:
                edges.append(([p.x for p in pts], [p.y for p in pts]))
        elif feat == "crosswalk":
            pts = mf.crosswalk.polygon
            if len(pts) >= 3:
                xs = [p.x for p in pts] + [pts[0].x]
                ys = [p.y for p in pts] + [pts[0].y]
                crosswalks.append((xs, ys))

    return lanes, edges, crosswalks


# ── per-frame actor renderer ──────────────────────────────────────────────────

def _draw_actors(
    ax,
    sc_states: pd.DataFrame,
    frame: int,
    sdc_track_id: Optional[str],
    risk_track_id: Optional[str],
) -> None:
    for track_id, group in sc_states.groupby("track_id"):
        history = group[group["timestep"] <= frame].sort_values("timestep")
        if history.empty:
            continue

        obj_type   = str(group["object_type"].iloc[0]).upper()
        is_sdc     = track_id == sdc_track_id
        is_risk    = track_id == risk_track_id

        if is_sdc:
            color, lw, dot_s, marker, z, base_a = SDC_COLOR,  1.6, 85, "*", 10, 0.90
        elif is_risk:
            color, lw, dot_s, marker, z, base_a = RISK_COLOR, 1.0, 24, "o",  7, 0.70
        else:
            color = TYPE_COLORS.get(obj_type, "#5555aa")
            lw, dot_s, marker, z, base_a = 0.55, 13, "o", 5, 0.26

        # Fading trajectory trail
        trail = history.tail(TRAIL_FRAMES)
        n = len(trail)
        if n > 1:
            for i in range(1, n):
                ax.plot(
                    trail["x"].values[i - 1 : i + 1],
                    trail["y"].values[i - 1 : i + 1],
                    color=color,
                    alpha=base_a * (i / n) ** 0.5,
                    linewidth=lw,
                    solid_capstyle="round",
                    zorder=z - 1,
                )

        cur = history.iloc[-1]
        ax.scatter(
            cur["x"], cur["y"],
            s=dot_s, c=color, marker=marker,
            edgecolors="white" if is_sdc else "none",
            linewidths=0.6 if is_sdc else 0,
            zorder=z,
        )

        # Heading arrow
        if "heading" in cur.index and not pd.isna(cur["heading"]):
            alen = 3.2 if is_sdc else 2.0
            dx = float(np.cos(cur["heading"])) * alen
            dy = float(np.sin(cur["heading"])) * alen
            ax.annotate(
                "",
                xy=(cur["x"] + dx, cur["y"] + dy),
                xytext=(cur["x"], cur["y"]),
                arrowprops=dict(
                    arrowstyle="-|>", color=color,
                    lw=1.0 if is_sdc else 0.45, alpha=0.88,
                ),
                zorder=z + 1,
            )


# ── single-scenario GIF generator ────────────────────────────────────────────

def generate_gif(
    scenario_proto,
    sc_states: pd.DataFrame,
    sc_tracks: pd.DataFrame,
    risk_row: Optional[Any],
    output_path: Path,
) -> None:
    scenario_id = scenario_proto.scenario_id

    # SDC identification
    sdc_rows     = sc_tracks[sc_tracks["is_sdc"] == True]
    sdc_track_id = sdc_rows.iloc[0]["track_id"] if not sdc_rows.empty else None

    # Closest-risk actor from risk_metrics
    risk_track_id = None
    if risk_row is not None:
        crid = getattr(risk_row, "closest_risk_actor_track_id", None)
        if crid and not pd.isna(crid):
            risk_track_id = str(crid)

    # Overlay metrics
    risk_score = getattr(risk_row, "risk_score", None) if risk_row is not None else None
    min_ttc    = getattr(risk_row, "min_ttc_s",  None) if risk_row is not None else None
    num_tracks = len(sc_tracks)

    # Map geometry (static for all frames)
    lanes, edges, crosswalks = extract_map_geometry(scenario_proto)

    # Scene bounding box
    valid = sc_states[sc_states["valid"] == True]
    if valid.empty:
        valid = sc_states
    x_all, y_all = valid["x"], valid["y"]
    x_rng = max(float(x_all.max() - x_all.min()), 30.0)
    y_rng = max(float(y_all.max() - y_all.min()), 30.0)
    pad_x = x_rng * 0.12 + 5
    pad_y = y_rng * 0.12 + 5
    xlim = (float(x_all.min()) - pad_x, float(x_all.max()) + pad_x)
    ylim = (float(y_all.min()) - pad_y, float(y_all.max()) + pad_y)

    # Overlay text (static part)
    r_str   = f"risk {risk_score:.2f}" if risk_score is not None and not pd.isna(risk_score) else ""
    ttc_str = f"ttc  {min_ttc:.1f}s"   if min_ttc    is not None and not pd.isna(min_ttc)    else ""
    id_str  = scenario_id[:8]

    # Timeline
    timesteps = sorted(sc_states["timestep"].unique())
    if FRAME_STEP > 1:
        timesteps = timesteps[::FRAME_STEP]

    plt.rcParams.update(_DARK_RCPARAMS)

    gif_frames: List[Image.Image] = []

    for frame in timesteps:
        fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE))
        fig.patch.set_facecolor("#12121e")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#252540")
            spine.set_linewidth(0.4)
        ax.grid(True, alpha=0.18)

        # ── static map ────────────────────────────────────────────────────────
        for xs, ys in lanes:
            ax.plot(xs, ys, color="#2a2a5a", lw=0.7, alpha=0.55,
                    zorder=1, solid_capstyle="round")
        for xs, ys in edges:
            ax.plot(xs, ys, color="#38385e", lw=0.9, alpha=0.42,
                    zorder=2, solid_capstyle="butt")
        for xs, ys in crosswalks:
            ax.fill(xs[:-1], ys[:-1], color="#3a3a60", alpha=0.18, zorder=2)
            ax.plot(xs, ys, color="#4a4a70", lw=0.5, alpha=0.30, zorder=2)

        # ── dynamic actors ────────────────────────────────────────────────────
        _draw_actors(ax, sc_states, frame, sdc_track_id, risk_track_id)

        # ── overlay badge ─────────────────────────────────────────────────────
        t_sec = frame * 0.1
        lines = [id_str]
        if r_str:
            lines.append(r_str)
        if ttc_str:
            lines.append(ttc_str)
        lines.append(f"{t_sec:.1f}s  ·  {num_tracks}trk")

        ax.text(
            0.025, 0.975,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=5.5, color="#9999bb", va="top",
            linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#0d0d1a",
                      edgecolor="none", alpha=0.88),
            zorder=20,
        )

        plt.tight_layout(pad=0.12)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=GIF_DPI,
                    bbox_inches="tight", facecolor="#12121e")
        buf.seek(0)
        img = Image.open(buf).copy().convert("RGB")
        gif_frames.append(img)
        plt.close(fig)
        buf.close()

    plt.rcParams.update(plt.rcParamsDefault)

    if not gif_frames:
        print(f"  ⚠  no frames rendered for {scenario_id}")
        return

    # Palette-quantize each frame → smaller GIF
    quantized = [
        f.quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT)
        for f in gif_frames
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        output_path,
        save_all=True,
        append_images=quantized[1:],
        optimize=True,
        duration=int(1000 / FPS),
        loop=0,
    )
    size_kb = output_path.stat().st_size / 1024
    print(f"  ✓  {output_path.name:<40s}  {len(gif_frames):3d} frames  {size_kb:5.0f} KB")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("WAYMO VALIDATION LAB — GENERATE PREVIEW GIFs")
    print("=" * 65)

    # Load silver / gold tables
    states_df = pd.read_parquet(SILVER_DIR / "states.parquet")
    tracks_df = pd.read_parquet(SILVER_DIR / "tracks.parquet")

    risk_path = GOLD_DIR / "risk_metrics.parquet"
    risk_df   = pd.read_parquet(risk_path) if risk_path.exists() else None

    # Scenario order: highest risk first
    if risk_df is not None:
        ordered_ids: List[str] = (
            risk_df.sort_values("risk_score", ascending=False)["scenario_id"].tolist()
        )
    else:
        ordered_ids = list(states_df["scenario_id"].unique())

    print(f"Scenarios to preview : {len(ordered_ids)}")

    # Find TFRecord
    tfrecord_files = list(TFRECORD_DIR.glob("*.tfrecord*"))
    if not tfrecord_files:
        print(f"❌ No TFRecord files found in {TFRECORD_DIR}")
        sys.exit(1)
    tfrecord_path = tfrecord_files[0]
    print(f"TFRecord             : {tfrecord_path.name}")

    # Index risk rows by scenario_id for fast lookup
    risk_index: Dict[str, Any] = {}
    if risk_df is not None:
        for _, row in risk_df.iterrows():
            risk_index[row["scenario_id"]] = row

    # Load all scenario protos into a dict keyed by scenario_id
    print("\nLoading scenario protos ...")
    scenario_protos: Dict[str, Any] = {}
    for raw in read_tfrecord(tfrecord_path):
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw)
        scenario_protos[sc.scenario_id] = sc
        if len(scenario_protos) >= 20:   # safety cap
            break
    print(f"Loaded {len(scenario_protos)} protos\n")

    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0
    for scenario_id in ordered_ids:
        if scenario_id not in scenario_protos:
            print(f"  skip {scenario_id[:8]} — proto not found in TFRecord")
            continue

        sc_states = states_df[
            (states_df["scenario_id"] == scenario_id)
        ].copy()
        sc_tracks = tracks_df[tracks_df["scenario_id"] == scenario_id].copy()

        if sc_states.empty:
            print(f"  skip {scenario_id[:8]} — no states")
            continue

        output_path = PREVIEWS_DIR / f"{scenario_id}.gif"
        print(f"  [{generated + 1:2d}] {scenario_id[:8]} ...", end="  ", flush=True)

        try:
            generate_gif(
                scenario_protos[scenario_id],
                sc_states,
                sc_tracks,
                risk_index.get(scenario_id),
                output_path,
            )
            generated += 1
        except Exception as exc:
            import traceback
            print(f"\n       ❌ failed: {exc}")
            traceback.print_exc()

    print(f"\n✅  Generated {generated} GIFs  →  {PREVIEWS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
