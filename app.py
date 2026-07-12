"""
QUASAR — AI-Powered Quantum Circuit Reliability Assessment
============================================================
Single-file Streamlit application implementing QUASAR_UI_UX_Specification.md.

Run:
    streamlit run app.py

Expected project layout (this file is designed to sit at the repo root,
alongside the existing, already-finished backend):

    app.py                      <- this file
    src/
        parser.py, analyzer.py, feature_extractor.py, noise_simulator.py,
        inference.py, explainability.py, recommendation_engine.py
    models/                     <- artifacts written by preprocess.py / train_model.py
        model_metrics.pkl, <winner>.pkl, preprocessing/...

DEMO MODE
---------
If qiskit, the `src` package, or trained model artifacts under `models/` are
not present, the app runs in DEMO MODE: circuit parsing falls back to a small
regex-based structural reader, and prediction / SHAP / LIME / recommendations
are replaced with clearly-labeled synthetic (but deterministic, seeded on the
circuit's own content) results. This lets the frontend be previewed and
deployed to a Hugging Face Space before training has finished, and never
silently pretends a synthetic number is a real model output.

This file only orchestrates existing backend modules — it does not
reimplement analysis, simulation, prediction, or explanation logic.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import math
import re
import tempfile
import time
from pathlib import Path
from turtle import color
from typing import Any, Callable

import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QUASAR — Quantum Circuit Reliability",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Optional heavy / backend imports — everything degrades gracefully
# ---------------------------------------------------------------------------
HAVE_QISKIT = True
try:
    from qiskit import QuantumCircuit  # noqa: F401
except Exception:
    HAVE_QISKIT = False

HAVE_SRC = True
try:
    from src.parser import load_qasm_file, QasmParsingError
    from src.analyzer import analyze_circuit
    from src.feature_extractor import extract_features
    from src.inference import run_inference, InferenceConfig, InferenceError
    from src.explainability import (
        explain_model,
        explain_local_circuit,
        ExplainConfig,
        ExplainabilityError,
    )
    from src.recommendation_engine import (
        generate_recommendations,
        RecommendationConfig,
        RecommendationEngineError,
    )
except Exception:
    HAVE_SRC = False

MODELS_DIR = Path("models")
MODEL_TRAINED = (MODELS_DIR / "model_metrics.pkl").exists()
BACKEND_READY = HAVE_QISKIT and HAVE_SRC
DEMO_MODE = not (BACKEND_READY and MODEL_TRAINED)

# Plotly is used for every interactive chart (gauge, gate-distribution strip,
# timing bar). Kept optional too, with a text-table fallback, since a Space
# should never hard-crash on a missing chart dependency.
HAVE_PLOTLY = True
try:
    import plotly.graph_objects as go
except Exception:
    HAVE_PLOTLY = False


# =============================================================================
# 1. DESIGN TOKENS + GLOBAL CSS  (matches QUASAR_UI_UX_Specification.md §1)
# =============================================================================

TOKENS = {
    "bg_primary": "#1A0F2E",
    "bg_secondary": "#231541",
    "bg_tertiary": "#2D1B52",
    "accent_primary": "#6C3EFF",
    "accent_secondary": "#35D8FF",
    "text_primary": "#F7F5FC",
    "text_secondary": "#B6ADCF",
    "text_muted": "#786F98",
    "border_subtle": "rgba(255,255,255,0.08)",
    "border_active": "rgba(108,62,255,0.45)",
}

GRADIENT = "linear-gradient(135deg, #6C3EFF 0%, #4C7FFF 50%, #35D8FF 100%)"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Audiowide&family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=Manrope:wght@600;700&display=swap');

        :root {{
            --bg-primary: {TOKENS['bg_primary']};
            --bg-secondary: {TOKENS['bg_secondary']};
            --bg-tertiary: {TOKENS['bg_tertiary']};
            --accent-primary: {TOKENS['accent_primary']};
            --accent-secondary: {TOKENS['accent_secondary']};
            --text-primary: {TOKENS['text_primary']};
            --text-secondary: {TOKENS['text_secondary']};
            --text-muted: {TOKENS['text_muted']};
            --border-subtle: {TOKENS['border_subtle']};
            --border-active: {TOKENS['border_active']};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        html, body {{
            background: var(--bg-primary);
        }}

        .stApp {{
            background: transparent;
            color: var(--text-primary);
        }}

        :root {{
            --qbloch-rx: 20deg;
            --qbloch-ry: -30deg;
        }}

        /* Fixed, full-viewport home for the animated quantum background.
           z-index:-1 + position:fixed keeps it painted behind every normal
           in-flow element on the page regardless of where in Streamlit's
           DOM tree it actually gets inserted. */
        #quasar-bg-layer {{
            position: fixed;
            inset: 0;
            z-index: -1;
            pointer-events: none;
            overflow: hidden;
            background: var(--bg-primary);
        }}
        #quasar-bg-layer svg {{
            width: 100%;
            height: 100%;
            display: block;
        }}

        /* Cursor-reactive 3D Bloch sphere (see render_bloch_sphere_hero).
           --qbloch-rx/--qbloch-ry are updated in real time by a tiny JS
           listener (render_mouse_tracker) attached to the parent document's
           mousemove -- pure CSS can't read absolute cursor position, so
           this is the one place a few lines of JS are used. Everything
           else here is CSS 3D transforms reacting to those two variables.
           --qbloch-rx is the polar angle theta (0deg at |0>, 180deg at
           |1>) and --qbloch-ry is the azimuthal angle phi (-180..180deg)
           -- see render_mouse_tracker's docstring for why the two must be
           combined in exactly `rotateY(phi) rotateX(theta)` order for the
           vector to be able to sweep the ENTIRE sphere rather than just
           one meridian plane. */
        #qbloch-wrap {{
            position: fixed;
            top: 10%;
            right: 3%;
            width: 380px;
            height: 380px;
            z-index: -1;
            pointer-events: none;
            perspective: 1000px;
            opacity: 0.68;
        }}
        #qbloch-scene {{
            position: relative;
            width: 100%;
            height: 100%;
            transform-style: preserve-3d;
            transform: rotateX(-14deg) rotateY(-20deg);
            filter: drop-shadow(0 0 28px rgba(108,62,255,0.20));
        }}
        .qbloch-ring, .qbloch-core, .qbloch-shell {{
            position: absolute;
            top: 50%;
            left: 50%;
            width: 300px;
            height: 300px;
            margin: -150px 0 0 -150px;
            border-radius: 50%;
            transform-style: preserve-3d;
        }}
        /* Solid, shaded globe -- static, never rotates -- gives the whole
           widget an actual "sphere" silhouette instead of reading as bare
           wireframe rings floating in space. */
        .qbloch-shell {{
            background:
                radial-gradient(circle at 32% 26%, rgba(255,255,255,0.20), transparent 22%),
                radial-gradient(circle at 38% 32%, rgba(108,62,255,0.30), rgba(53,216,255,0.06) 60%, rgba(12,8,32,0.05) 78%);
            box-shadow:
                inset -18px -18px 46px rgba(0,0,0,0.35),
                inset 10px 10px 30px rgba(108,62,255,0.18),
                0 0 40px rgba(53,216,255,0.10);
        }}
        .qbloch-ring {{
            border: 1px solid rgba(108,62,255,0.30);
        }}
        .qbloch-ring-equator {{ transform: rotateX(90deg);}}
        .qbloch-ring-meridian-a {{ transform: rotateY(0deg); }}
        .qbloch-ring-meridian-b {{ transform: rotateY(45deg); }}
        .qbloch-ring-meridian-c {{ transform: rotateY(90deg); border-color: rgba(53,216,255,0.42); }}
        .qbloch-ring-meridian-d {{ transform: rotateY(135deg); }}
        .qbloch-core {{
            border: 1px solid rgba(182,173,207,0.12);
        }}
        .qbloch-axis {{
            position: absolute;
            top: 50%;
            left: 50%;
            width: 1px;
            height: 300px;
            margin-left: -0.5px;
            margin-top: -150px;
            background: linear-gradient(rgba(182,173,207,0.30), transparent 42%, transparent 58%, rgba(182,173,207,0.30));
        }}
        #qbloch-vector {{
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            transform: rotateY(var(--qbloch-ry, -30deg)) rotateX(var(--qbloch-rx, 20deg));
            transition: transform 120ms ease-out;
        }}
        .qbloch-vector-line {{
            position: absolute;
            bottom: 0;
            left: -2px;
            width: 4px;
            height: 150px;
            border-radius: 999px;
            background: linear-gradient(to top, rgba(108,62,255,0.25), rgba(53,216,255,0.95));
            box-shadow: 0 0 8px rgba(53,216,255,0.35);
            transform-origin: bottom center;
        }}
        .qbloch-vector-tip {{
            position: absolute;
            top: -158px;
            left: -7px;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #35D8FF;
            box-shadow: 0 0 10px rgba(53,216,255,0.6);
        }}
        .qbloch-label {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform-style: preserve-3d;
            /* existing font/color rules stay on the base .qbloch-label class */
        }}
        .qbloch-label-0     {{ transform: translate3d(0, -168px, 0); }}
        .qbloch-label-1     {{ transform: translate3d(0, 168px, 0); }}
        .qbloch-label-plus  {{ transform: translate3d(168px, 0, 0); }}
        .qbloch-label-minus {{ transform: translate3d(-168px, 0, 0); }}
        .qbloch-label-plus-i {{transform: translate3d(0, 0, 168px) translate(-48px, 32px);}}
        .qbloch-label-minus-i {{transform: translate3d(0, 0, -168px) translate(48px, -32px);}}

        @keyframes qbg-drift {{
            0%, 100% {{ transform: translate(0, 0); }}
            50% {{ transform: translate(var(--dx, 20px), var(--dy, 20px)); }}
        }}
        @keyframes qbg-wave {{
            0%, 100% {{ transform: translateX(0); }}
            50% {{ transform: translateX(var(--wx, 30px)); }}
        }}
        @keyframes qbg-fall {{
            0% {{ transform: translateY(-140px); opacity: 0; }}
            8% {{ opacity: var(--peak, 0.8); }}
            85% {{ opacity: var(--peak, 0.8); }}
            100% {{ transform: translateY(980px); opacity: 0; }}
        }}
        @keyframes qbg-fall-sway {{
            0% {{ transform: translateY(-60px) translateX(0) rotate(0deg); opacity: 0; }}
            8% {{ opacity: var(--peak, 0.7); }}
            50% {{ transform: translateY(460px) translateX(var(--sway, 20px)) rotate(var(--rot, 8deg)); }}
            85% {{ opacity: var(--peak, 0.7); }}
            100% {{ transform: translateY(980px) translateX(calc(var(--sway, 20px) * -0.5)) rotate(calc(var(--rot, 8deg) * -1)); opacity: 0; }}
        }}
        @keyframes qbg-spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        @keyframes qbg-pulse-glow {{
            0%, 100% {{ opacity: 0.5; }}
            50% {{ opacity: 1; }}
        }}

        .qbg-drift {{ animation: qbg-drift ease-in-out infinite; }}
        .qbg-wave {{ animation: qbg-wave ease-in-out infinite; }}
        .qbg-fall {{ animation: qbg-fall linear infinite; }}
        .qbg-fall-sway {{ animation: qbg-fall-sway ease-in-out infinite; }}
        .qbg-spin {{ animation: qbg-spin linear infinite; transform-box: fill-box; transform-origin: center; }}
        .qbg-pulse-glow {{ animation: qbg-pulse-glow ease-in-out infinite; }}

        /* Kill Streamlit's default top padding so our own spacing scale governs */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1200px;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-subtle);
        }}
        section[data-testid="stSidebar"] * {{
            color: var(--text-secondary);
        }}

        /* Headings */
        h1, h2, h3 {{
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }}

        /* Buttons -> gradient pill */
        /* Buttons -> gradient pill */
        .stButton > button,
        .stDownloadButton > button {{
            background: linear-gradient(135deg, #6C3EFF 0%, #4C7FFF 50%, #35D8FF 100%);
            color: #FFFFFF !important;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            transition: transform 160ms ease, filter 160ms ease;
        }}

        /* Force the button label to stay white */
        .stButton > button *,
        .stDownloadButton > button * {{
            color: #FFFFFF !important;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: scale(1.02);
            filter: brightness(1.08);
        }}
        .stButton > button:active {{
            transform: scale(0.98);
        }}

        /* Tabs -> underlined text, not boxed pills */
        button[data-baseweb="tab"] {{
            background: transparent !important;
            color: var(--text-secondary) !important;
            font-weight: 500;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--text-primary) !important;
            border-bottom: 2px solid var(--accent-secondary) !important;
        }}

        /* Generic card shell */
        .quasar-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 16px;
            transition: border-color 180ms ease;
        }}
        .quasar-card:hover {{ border-color: var(--border-active); }}

        .quasar-card-tertiary {{
            background: var(--bg-tertiary);
            border-radius: 16px;
            padding: 16px;
        }}

        /* Metric card */
        .quasar-metric-label {{
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }}
        .quasar-metric-value {{
            font-family: 'Manrope', sans-serif;
            font-weight: 700;
            font-size: 36px;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
            line-height: 1.1;
        }}
        .quasar-metric-caption {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        /* Chips / badges */
        .quasar-chip {{
            display: inline-block;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            border-radius: 999px;
            padding: 4px 12px;
            font-size: 12px;
            margin: 2px 4px 2px 0;
        }}

        /* Empty / skeleton / error states */
        .quasar-empty {{
            text-align: center;
            padding: 48px 24px;
            color: var(--text-muted);
            border: 1px dashed var(--border-subtle);
            border-radius: 16px;
        }}
        .quasar-empty-glyph {{ font-size: 28px; margin-bottom: 8px; opacity: 0.6; }}

        .quasar-skeleton {{
            background: linear-gradient(90deg, var(--bg-secondary) 25%, var(--bg-tertiary) 50%, var(--bg-secondary) 75%);
            background-size: 200% 100%;
            animation: quasar-shimmer 1.4s infinite;
            border-radius: 16px;
            height: 96px;
            margin-bottom: 16px;
        }}
        @keyframes quasar-shimmer {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .quasar-skeleton {{ animation: none; opacity: 0.4; }}
        }}

        /* Live backend progress log (see render_explainability's terminal
           box) -- a rolling window of the most recent progress_callback
           messages, styled like a small terminal so it reads as "the
           backend is actively working" rather than a frozen spinner. */
        .qterm-box {{
            background: #0A0714;
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 14px 16px;
            margin: 10px 0 16px 0;
            font-family: 'Menlo', 'Consolas', monospace;
            font-size: 12.5px;
            line-height: 1.7;
            color: #8CF7C8;
            min-height: 148px;
            box-shadow: inset 0 0 24px rgba(108,62,255,0.06);
        }}
        .qterm-line {{
            white-space: pre-wrap;
            word-break: break-word;
            opacity: 0;
            animation: qterm-fade-in 220ms ease-out forwards;
        }}
        .qterm-line-prompt {{
            color: #35D8FF;
            margin-right: 6px;
        }}
        .qterm-dim {{
            color: var(--text-muted);
        }}
        .qterm-cursor {{
            display: inline-block;
            color: #35D8FF;
            animation: qterm-blink 1s step-end infinite;
        }}
        @keyframes qterm-fade-in {{
            from {{ opacity: 0; transform: translateY(2px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes qterm-blink {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .qterm-line {{ animation: none; opacity: 1; }}
            .qterm-cursor {{ animation: none; }}
        }}

        .quasar-error {{
            background: rgba(108,62,255,0.06);
            border: 1px solid var(--border-active);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }}
        .quasar-error-title {{ font-weight: 600; color: var(--text-primary); margin-bottom: 4px; }}
        .quasar-error-msg {{ color: var(--text-secondary); font-size: 14px; }}

        /* Upload dropzone */
        section[data-testid="stFileUploaderDropzone"] {{
            background: var(--bg-secondary) !important;
            border: 2px dashed var(--border-subtle) !important;
            border-radius: 16px !important;
        }}

        /* Gradient text utility */
        .quasar-gradient-text {{
            font-family: 'Orbitron', sans-serif;
            font-weight: 800;
            background: {GRADIENT};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        hr {{ border-color: var(--border-subtle); }}

        /* Pipeline stepper */
        .quasar-stepper {{ display: flex; align-items: center; margin: 8px 0 4px 0; }}
        .quasar-step {{ display: flex; flex-direction: column; align-items: center; flex: 1; }}
        .quasar-step-dot {{
            width: 16px; height: 16px; border-radius: 50%;
            border: 2px solid var(--text-muted);
            background: transparent;
        }}
        .quasar-step-dot.done {{ background: var(--accent-secondary); border-color: var(--accent-secondary); }}
        .quasar-step-dot.active {{
            border-color: var(--accent-primary);
            animation: quasar-pulse 2s infinite;
        }}
        .quasar-step-dot.error {{ background: transparent; border-color: var(--accent-primary); }}
        @keyframes quasar-pulse {{
            0%, 100% {{ opacity: 0.4; }} 50% {{ opacity: 1; }}
        }}
        .quasar-step-line {{ flex: 1; height: 2px; background: var(--border-subtle); }}
        .quasar-step-line.done {{ background: {GRADIENT}; }}
        .quasar-step-label {{ font-size: 11px; color: var(--text-secondary); margin-top: 6px; text-align: center; }}
        .quasar-step-time {{ font-size: 10px; color: var(--text-muted); }}

        .quasar-caption-row {{
            text-align: right; font-size: 12px; color: var(--text-muted); margin-top: 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Animated quantum background — SVG + CSS, with a few lines of JS solely to
# read the cursor position (pure CSS cannot do that). Layered motifs, all
# built from TOKENS so they always match the app's palette:
#   1. Probability-wave interference (soft blurred bands, bottom layer)
#   2. Circuit pulse grid (faint horizontal wires + traveling glow pulses)
#   3. Drifting qubit particle field (dots only — no connecting lines)
#   4. Rotating hexagon ring — a slow, faint echo of the QUASAR logo mark
# Generated once per session (seeded + cached in session_state) so it doesn't
# visually reset/jump on every Streamlit rerun. A separate, tiny JS snippet
# (render_mouse_tracker) drives the cursor-reactive 3D Bloch sphere
# (#qbloch-vector in the CSS above, rendered on the landing page by
# render_bloch_sphere_hero) by writing --qbloch-rx/--qbloch-ry custom
# properties onto the page.
# ---------------------------------------------------------------------------

_QBG_VIEW_W = 1600
_QBG_VIEW_H = 900


def _qbg_waves(rng: random.Random) -> str:
    """Slow, blurred violet/cyan bands — the ambient interference layer."""
    bands = []
    band_defs = [
        (160, "rgba(108,62,255,0.09)"),
        (520, "rgba(53,216,255,0.07)"),
    ]
    for y, color in band_defs:
        wx = rng.randint(60, 140)
        dur = rng.uniform(30, 48)
        delay = rng.uniform(0, 10)
        path = (
            f"M -100,{y} "
            f"C {_QBG_VIEW_W*0.25},{y - 70} {_QBG_VIEW_W*0.5},{y + 70} {_QBG_VIEW_W*0.75},{y} "
            f"S {_QBG_VIEW_W + 100},{y - 60} {_QBG_VIEW_W + 300},{y} "
            f"L {_QBG_VIEW_W + 300},{y + 260} L -100,{y + 260} Z"
        )
        bands.append(
            f'<path d="{path}" fill="{color}" filter="url(#qbg-blur)" '
            f'class="qbg-wave" style="--wx:{wx}px; animation-duration:{dur:.1f}s; '
            f'animation-delay:{delay:.1f}s;" />'
        )
    return "\n".join(bands)


def _qbg_circuit_grid(rng: random.Random) -> str:
    """Faint horizontal wires carrying a traveling glow pulse — no verticals,
    no diagonals, so this layer can never read as stray crossing lines."""
    parts = []
    rows = 5
    for i in range(rows):
        y = int((i + 1) * _QBG_VIEW_H / (rows + 1))
        parts.append(
            f'<path id="qbg-wire-{i}" d="M 0,{y} H {_QBG_VIEW_W}" '
            f'stroke="rgba(182,173,207,0.08)" stroke-width="1" fill="none" />'
        )
        dur = rng.uniform(7, 14)
        begin = rng.uniform(0, 9)
        parts.append(
            f'<circle r="3" fill="url(#qbg-pulse-grad)" filter="url(#qbg-glow)">'
            f'<animateMotion dur="{dur:.1f}s" begin="{begin:.1f}s" '
            f'repeatCount="indefinite">'
            f'<mpath href="#qbg-wire-{i}" /></animateMotion></circle>'
        )
    return "\n".join(parts)


def _qbg_particles(rng: random.Random, count: int = 22) -> str:
    """Drifting qubit dots. No connecting lines — purely ambient motion."""
    circles = []
    for _ in range(count):
        x = rng.uniform(0, _QBG_VIEW_W)
        y = rng.uniform(0, _QBG_VIEW_H)
        r = rng.uniform(1.4, 3.0)
        color = rng.choice(["#6C3EFF", "#35D8FF"])
        dx = rng.uniform(-40, 40)
        dy = rng.uniform(-30, 30)
        dur = rng.uniform(14, 26)
        delay = rng.uniform(0, 10)
        opacity = rng.uniform(0.3, 0.6)
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" '
            f'opacity="{opacity:.2f}" class="qbg-drift" '
            f'style="--dx:{dx:.1f}px; --dy:{dy:.1f}px; '
            f'animation-duration:{dur:.1f}s; animation-delay:{delay:.1f}s; '
            f'transform-box: fill-box; transform-origin: center;" />'
        )
    return "\n".join(circles)


def _qbg_hexagon_motif() -> str:
    """A large, faint, slowly-rotating hexagon ring in the top-right corner —
    a quiet nod to the QUASAR logo's isometric hex mark, for brand cohesion."""
    cx, cy, r = _QBG_VIEW_W - 220, 200, 260
    pts = []
    for i in range(6):
        angle = math.pi / 180 * (60 * i - 90)
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    points_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    inner_pts = []
    for i in range(6):
        angle = math.pi / 180 * (60 * i - 90 + 30)
        inner_pts.append((cx + (r * 0.6) * math.cos(angle), cy + (r * 0.6) * math.sin(angle)))
    inner_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in inner_pts)
    return (
        f'<g class="qbg-spin" style="animation-duration:120s; transform-origin:{cx}px {cy}px;">'
        f'<polygon points="{points_attr}" fill="none" stroke="url(#qbg-hex-grad)" '
        f'stroke-width="1.2" opacity="0.18" />'
        f'</g>'
        f'<g class="qbg-spin" style="animation-duration:90s; animation-direction:reverse; '
        f'transform-origin:{cx}px {cy}px;">'
        f'<polygon points="{inner_attr}" fill="none" stroke="url(#qbg-hex-grad)" '
        f'stroke-width="1" opacity="0.14" />'
        f'</g>'
    )


def _qbg_ket_rain(rng: random.Random, count: int = 32) -> str:
    """Falling quantum state symbols (|0>, |1>, |+>, |->, |+i>, |−i>, |ψ>)
    drifting down the viewport like rain — same purple/cyan palette, kept
    visible and slow so they read as ambient texture rather than UI text."""
    kets = ["|0⟩", "|1⟩", "|+⟩", "|−⟩", "|+i⟩", "|−i⟩", "|ψ⟩", "|φ⟩"]
    colors = ["rgba(108,62,255,", "rgba(53,216,255,", "rgba(182,173,207,"]
    parts = []
    for _ in range(count):
        x = rng.uniform(40, _QBG_VIEW_W - 40)
        ket = rng.choice(kets)
        color_base = rng.choice(colors)
        opacity = rng.uniform(0.28, 0.58)
        color = f"{color_base}{opacity:.2f})"
        font_size = rng.uniform(14, 22)
        dur = rng.uniform(16, 32)
        delay = rng.uniform(0, 18)
        sway = rng.uniform(-25, 25)
        rot = rng.uniform(-12, 12)
        parts.append(
            f'<text x="{x:.1f}" y="-20" fill="{color}" '
            f'font-family="monospace" font-size="{font_size:.1f}" '
            f'class="qbg-fall-sway" '
            f'style="--sway:{sway:.1f}px; --rot:{rot:.1f}deg; --peak:{opacity:.2f}; '
            f'animation-duration:{dur:.1f}s; animation-delay:{delay:.1f}s;" '
            f'text-anchor="middle">{ket}</text>'
        )
    return "\n".join(parts)


def render_quantum_background() -> None:
    """Inject the layered, animated quantum-themed background (once per session).

    Cached in `st.session_state` so the random layout is generated exactly
    once per browser session rather than re-randomizing (and visually
    jumping) on every Streamlit rerun.
    """
    if "quasar_bg_html" not in st.session_state:
        seed = random.randint(0, 1_000_000)
        rng = random.Random(seed)

        svg_body = "\n".join(
            [
                _qbg_waves(rng),
                _qbg_hexagon_motif(),
                _qbg_circuit_grid(rng),
                _qbg_ket_rain(rng),
                _qbg_particles(rng),
            ]
        )

        st.session_state["quasar_bg_html"] = f"""
        <div id="quasar-bg-layer">
            <svg viewBox="0 0 {_QBG_VIEW_W} {_QBG_VIEW_H}" preserveAspectRatio="xMidYMid slice"
                 xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <filter id="qbg-blur" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur stdDeviation="35" />
                    </filter>
                    <filter id="qbg-glow" x="-200%" y="-200%" width="500%" height="500%">
                        <feGaussianBlur stdDeviation="2.5" result="blur" />
                        <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                    <radialGradient id="qbg-pulse-grad">
                        <stop offset="0%" stop-color="#35D8FF" stop-opacity="1" />
                        <stop offset="100%" stop-color="#35D8FF" stop-opacity="0" />
                    </radialGradient>
                    <linearGradient id="qbg-hex-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#6C3EFF" />
                        <stop offset="100%" stop-color="#35D8FF" />
                    </linearGradient>
                </defs>
                {svg_body}
            </svg>
        </div>
        """

    st.markdown(st.session_state["quasar_bg_html"], unsafe_allow_html=True)


def render_mouse_tracker() -> None:
    """Wire up the cursor-reactive Bloch sphere (see #qbloch-vector CSS).

    Pure CSS cannot read the cursor's absolute page position, so this is
    the one deliberate bit of JS in the whole background system. It runs
    inside an invisible (0-height) components.html iframe -- Streamlit's
    st.markdown does not execute <script> tags at all, so the state
    vector would silently never move without this. The listener is
    attached to the PARENT document (same-origin, so this is allowed)
    and writes --qbloch-rx/--qbloch-ry custom properties (in degrees)
    onto its root element.

    The cursor position IS the calculation, not just a nudge: the
    vertical position of the cursor is mapped directly to the state's
    polar angle theta (0deg at the very top of the page -> the |0>
    pole, 180deg at the very bottom -> the |1> pole), and the horizontal
    position to the azimuthal angle phi (a full -180..180deg sweep
    across the page width). Because theta and phi together are exactly
    the two spherical coordinates that specify a point on a unit sphere,
    every reachable (theta, phi) pair traces out the entire sphere
    surface as the cursor visits the entire page -- not just a limited
    tilt range -- so the state vector can reach any point, including
    both poles and the far/back hemisphere.

    The Bloch sphere's state-vector element (see render_bloch_sphere_hero)
    rotates via `transform: rotateY(var(--qbloch-ry)) rotateX(var(--qbloch-rx))`.
    That composition order matters: the vector's rest position lies
    exactly on the rotateY axis, so a naive `rotateX(...) rotateY(...)`
    order (rotateY applied to the vector first) would have no visible
    effect from the Y-rotation at all, and the vector would only ever
    sweep back and forth in a single plane. Applying rotateX first tilts
    the vector off that axis, so the outer rotateY can then genuinely
    carry it all the way around -- this is what makes every point on the
    sphere reachable rather than just one meridian.
    """
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            if (doc.documentElement.dataset.quasarMouseTracker === "1") return;
            doc.documentElement.dataset.quasarMouseTracker = "1";
            doc.addEventListener('mousemove', function(e) {
                const nx = e.clientX / window.parent.innerWidth;   // 0 .. 1, left -> right
                const ny = e.clientY / window.parent.innerHeight;  // 0 .. 1, top -> bottom
                const theta = ny * 180;          // 0deg = |0> pole, 180deg = |1> pole
                const phi = (nx - 0.5) * 360;     // -180deg .. 180deg, full azimuthal sweep
                doc.documentElement.style.setProperty('--qbloch-rx', theta.toFixed(2) + 'deg');
                doc.documentElement.style.setProperty('--qbloch-ry', phi.toFixed(2) + 'deg');
            });
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def render_bloch_sphere_hero() -> None:
    """A cursor-reactive, pseudo-3D Bloch sphere rendered with pure CSS 3D
    transforms -- a shaded, static "shell" gives the widget an actual
    spherical silhouette (rather than bare floating rings), a wireframe
    of one equator + four meridians stays fixed for a stable sense of
    depth, and the state vector itself (the glowing line + dot running
    from the center to the surface) rotates live with the cursor, via
    the --qbloch-rx (theta) / --qbloch-ry (phi) properties
    render_mouse_tracker writes onto the document root -- see that
    function's docstring for how cursor position maps to a point on the
    sphere. Positioned as a fixed, low-opacity, non-interactive
    decorative element in the landing hero's empty right-hand space,
    using the same purple/cyan palette as the rest of the background."""
    st.markdown(
        """
        <div id="qbloch-wrap">
          <div id="qbloch-scene">
            <div class="qbloch-shell"></div>
            <div class="qbloch-core"></div>
            <div class="qbloch-ring qbloch-ring-equator"></div>
            <div class="qbloch-ring qbloch-ring-meridian-a"></div>
            <div class="qbloch-ring qbloch-ring-meridian-b"></div>
            <div class="qbloch-ring qbloch-ring-meridian-c"></div>
            <div class="qbloch-ring qbloch-ring-meridian-d"></div>
            <div class="qbloch-axis"></div>
            <div class="qbloch-label qbloch-label-0">|0⟩</div>
            <div class="qbloch-label qbloch-label-1">|1⟩</div>
            <div class="qbloch-label qbloch-label-plus">|+⟩</div>
            <div class="qbloch-label qbloch-label-minus">|−⟩</div>
            <div class="qbloch-label qbloch-label-plus-i">|+i⟩</div>
            <div class="qbloch-label qbloch-label-minus-i">|−i⟩</div>
            <div id="qbloch-vector">
              <div class="qbloch-vector-line"></div>
              <div class="qbloch-vector-tip"></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def logo_svg(size: int = 64) -> str:
    """Inline SVG approximating the provided QUASAR isometric logo mark."""
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="qgrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#6C3EFF"/>
          <stop offset="50%" stop-color="#4C7FFF"/>
          <stop offset="100%" stop-color="#35D8FF"/>
        </linearGradient>
      </defs>
      <polygon points="50,8 50,55 15,72 15,35" fill="none" stroke="url(#qgrad)" stroke-width="1.4" opacity="0.9"/>
      <polygon points="50,8 50,55 85,72 85,35" fill="none" stroke="url(#qgrad)" stroke-width="1.4" opacity="0.9"/>
      <polygon points="15,72 50,55 85,72 50,90" fill="none" stroke="url(#qgrad)" stroke-width="1.4" opacity="0.9"/>
    </svg>
    """

LOGO_PATH = Path("assets/quasar_logo.png")

@st.cache_data(show_spinner=False)
def logo_img_tag(width: int) -> str:
    """Base64-embeds the real QUASAR logo file so it works as a drop-in
    replacement for logo_svg() (still inline HTML, no extra request).
    Falls back to the placeholder SVG if the file isn't found."""
    if not LOGO_PATH.exists():
        return logo_svg(width)
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return (
        f'<img src="data:image/png;base64,{encoded}" width="{width}" '
        f'style="display:block; margin:0 auto;">'
    )

# =============================================================================
# 2. SAMPLE CIRCUITS  (cached; used by the Landing page gallery)
# =============================================================================

_SAMPLE_QASM: dict[str, str] = {
    "Bell State": """OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
""",
    "GHZ State": """OPENQASM 3;
include "stdgates.inc";
qubit[4] q;
bit[4] c;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
""",
    "Hadamard Fan-out": """OPENQASM 3;
include "stdgates.inc";
qubit[4] q;
bit[4] c;
h q[0];
h q[1];
h q[2];
h q[3];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
""",
    "Grover Search": """OPENQASM 3;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
h q[0];
h q[1];
h q[2];
x q[2];
h q[2];
ccx q[0], q[1], q[2];
h q[2];
x q[2];
h q[0];
h q[1];
h q[2];
x q[0];
x q[1];
x q[2];
h q[2];
ccx q[0], q[1], q[2];
h q[2];
x q[0];
x q[1];
x q[2];
h q[0];
h q[1];
h q[2];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
""",
    "QFT (5 qubit)": """OPENQASM 3;
include "stdgates.inc";
qubit[5] q;
bit[5] c;
h q[0];
cp(1.5707963267948966) q[1], q[0];
cp(0.7853981633974483) q[2], q[0];
cp(0.39269908169872414) q[3], q[0];
cp(0.19634954084936207) q[4], q[0];
h q[1];
cp(1.5707963267948966) q[2], q[1];
cp(0.7853981633974483) q[3], q[1];
cp(0.39269908169872414) q[4], q[1];
h q[2];
cp(1.5707963267948966) q[3], q[2];
cp(0.7853981633974483) q[4], q[2];
h q[3];
cp(1.5707963267948966) q[4], q[3];
h q[4];
swap q[0], q[4];
swap q[1], q[3];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
c[4] = measure q[4];
""",
    "Stress Test": """OPENQASM 3;
include "stdgates.inc";
qubit[10] q;
bit[10] c;
h q[0]; h q[1]; h q[2]; h q[3]; h q[4];
h q[5]; h q[6]; h q[7]; h q[8]; h q[9];
cx q[0], q[1]; cx q[2], q[3]; cx q[4], q[5]; cx q[6], q[7]; cx q[8], q[9];
rz(0.7853981633974483) q[0];
rz(1.5707963267948966) q[1];
ry(0.39269908169872414) q[2];
cx q[1], q[2]; cx q[3], q[4]; cx q[5], q[6]; cx q[7], q[8]; cx q[9], q[0];
swap q[0], q[3];
swap q[4], q[7];
ccx q[0], q[1], q[2];
ccx q[3], q[4], q[5];
cz q[6], q[7];
cx q[8], q[9];
h q[0]; h q[2]; h q[4]; h q[6]; h q[8];
c[0] = measure q[0]; c[1] = measure q[1]; c[2] = measure q[2];
c[3] = measure q[3]; c[4] = measure q[4]; c[5] = measure q[5];
c[6] = measure q[6]; c[7] = measure q[7]; c[8] = measure q[8]; c[9] = measure q[9];
""",
}

_SAMPLE_META = {
    "Bell State": ("2 qubits", "Canonical maximally-entangled pair"),
    "GHZ State": ("4 qubits", "Multi-qubit entanglement chain"),
    "Hadamard Fan-out": ("4 qubits", "Uniform superposition, no entanglement"),
    "Grover Search": ("3 qubits", "Amplitude amplification, 1 iteration"),
    "QFT (5 qubit)": ("5 qubits", "Quantum Fourier Transform"),
    "Stress Test": ("10 qubits", "Dense, mixed-gate large-scale circuit"),
}


@st.cache_data(show_spinner=False)
def get_sample_qasm(name: str) -> str:
    return _SAMPLE_QASM[name]


# =============================================================================
# 3. DEMO-MODE FALLBACKS (used only when BACKEND_READY / MODEL_TRAINED is False)
# =============================================================================

_GATE_RE = re.compile(
    r"^\s*(h|x|y|z|s|sdg|t|tdg|sx|rx|ry|rz|u1|u2|u3|cx|cz|cp|swap|ccx)\b", re.IGNORECASE
)
_QUBIT_DECL_RE = re.compile(r"qubit\[(\d+)\]")
_BIT_DECL_RE = re.compile(r"\bbit\[(\d+)\]")
_MEASURE_RE = re.compile(r"\bmeasure\b", re.IGNORECASE)

_ENTANGLING = {"cx", "cz", "cp", "swap", "ccx"}
_PARAMETERIZED = {"rx", "ry", "rz", "cp", "u1", "u2", "u3"}
_THREE_Q = {"ccx"}
_TWO_Q = {"cx", "cz", "cp", "swap"}


def demo_structural_reader(qasm_text: str) -> dict[str, Any]:
    """Lightweight regex-based structural reader used only in DEMO MODE, when
    qiskit is unavailable and the real parser/analyzer/feature_extractor
    modules cannot run. Never used when the real backend is present.
    """
    m = _QUBIT_DECL_RE.search(qasm_text)
    num_qubits = int(m.group(1)) if m else 0
    b = _BIT_DECL_RE.search(qasm_text)
    num_clbits = int(b.group(1)) if b else 0

    gate_counts: dict[str, int] = {}
    total_ops = 0
    for line in qasm_text.splitlines():
        gm = _GATE_RE.match(line)
        if gm:
            name = gm.group(1).lower()
            gate_counts[name] = gate_counts.get(name, 0) + 1
            total_ops += 1
    num_measurements = len(_MEASURE_RE.findall(qasm_text))
    total_ops += num_measurements

    single = sum(c for g, c in gate_counts.items() if g not in _ENTANGLING)
    two = sum(c for g, c in gate_counts.items() if g in _TWO_Q)
    three = sum(c for g, c in gate_counts.items() if g in _THREE_Q)
    parameterized = sum(c for g, c in gate_counts.items() if g in _PARAMETERIZED)
    entangling = sum(c for g, c in gate_counts.items() if g in _ENTANGLING)

    # Depth is only approximable without a real DAG; use a conservative
    # heuristic (gate lines / a rough parallelism factor) and label it as such.
    approx_depth = max(1, round(total_ops / max(1, num_qubits / 2)))

    return {
        "num_qubits": num_qubits,
        "num_clbits": num_clbits,
        "depth": approx_depth,
        "width": num_qubits + num_clbits,
        "total_operations": total_ops,
        "gate_counts": gate_counts,
        "gates_used": sorted(g for g in gate_counts if g != "measure"),
        "num_measurements": num_measurements,
        "single_qubit_gates": single,
        "two_qubit_gates": two,
        "three_qubit_gates": three,
        "parameterized_gates": parameterized,
        "entangling_gates": entangling,
        "_approx": True,
    }


def demo_prediction(seed_text: str, features: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, seeded synthetic prediction for DEMO MODE only."""
    rng = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest(), 16) % (2**32))
    entangling_ratio = features["entangling_gates"] / max(1, features["total_operations"])
    depth_penalty = min(1.0, features["depth"] / 80)
    base_high = max(0.05, 0.9 - entangling_ratio * 0.6 - depth_penalty * 0.5)
    noise = rng.uniform(-0.05, 0.05)
    p_high = max(0.02, min(0.95, base_high + noise))
    remaining = 1 - p_high
    p_medium = remaining * rng.uniform(0.4, 0.7)
    p_low = remaining - p_medium
    probs = {"LOW": p_low, "MEDIUM": p_medium, "HIGH": p_high}
    predicted = max(probs, key=probs.get)
    band = {"HIGH": (95, 100), "MEDIUM": (85, 95), "LOW": (0, 85)}[predicted]
    score = band[0] + (band[1] - band[0]) * rng.uniform(0.3, 0.9)
    return {
        "reliability_class": predicted,
        "confidence": probs[predicted],
        "class_probabilities": probs,
        "reliability_score_estimate": score,
        "model_name": "demo_synthetic",
        "unavailable_features": ["estimated_fidelity", "total_variation_distance"],
    }


def demo_recommendations(features: dict[str, Any]) -> list[dict[str, Any]]:
    recs = []
    if features["two_qubit_gates"] >= 5:
        recs.append({
            "title": "Reduce CX (CNOT) gate count",
            "detail": "This circuit uses a significant number of two-qubit gates, "
                      "typically the noisiest primitive on real hardware. Consider "
                      "commutation-based transpiler optimization to cancel or merge "
                      "adjacent pairs.",
            "category": "cx_gates",
            "shap_impact": 0.14,
            "triggering_features": ["two_qubit_gates", "entangling_gates"],
        })
    if features["depth"] >= 20:
        recs.append({
            "title": "Reduce circuit depth",
            "detail": f"This circuit has an (approximate) depth of {features['depth']} "
                      "layers. Deeper circuits accumulate more decoherence exposure. "
                      "Consider parallelizing independent single-qubit operations.",
            "category": "depth",
            "shap_impact": 0.09,
            "triggering_features": ["depth"],
        })
    if features["entangling_gates"] >= 8:
        recs.append({
            "title": "Reduce entangling gate count",
            "detail": "Every entangling operation carries substantially higher error "
                      "than a single-qubit gate. Review whether every entangling "
                      "operation is structurally necessary.",
            "category": "entangling_gates",
            "shap_impact": 0.07,
            "triggering_features": ["entangling_gates"],
        })
    recs.sort(key=lambda r: r["shap_impact"], reverse=True)
    return recs[:5]


# =============================================================================
# 4. SESSION STATE
# =============================================================================

STAGES = ["Upload", "Parse", "Analyze", "Features", "Predict", "Explain", "Recommend"]


def init_session_state() -> None:
    defaults = {
        "route": "landing",
        "circuit_name": None,
        "qasm_text": None,
        "stage_index": -1,        # -1 = nothing loaded yet
        "stage_error": None,      # (stage_name, message) if a stage failed
        "stage_timings": {},      # stage_name -> seconds
        "analysis": None,
        "features": None,
        "prediction": None,
        "explainability": None,
        "local_shap": None,
        "recommendations": None,
        "explain_requested": False,
        "explain_running": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_pipeline() -> None:
    for key in [
        "circuit_name", "qasm_text", "stage_index", "stage_error", "stage_timings",
        "analysis", "features", "prediction", "explainability", "local_shap", "recommendations",
        "explain_requested", "explain_running",
    ]:
        st.session_state[key] = None
    st.session_state["stage_index"] = -1
    st.session_state["stage_timings"] = {}
    st.session_state["explain_requested"] = False
    st.session_state["explain_running"] = False


def time_stage(name: str, fn: Callable[[], Any]) -> Any:
    start = time.perf_counter()
    result = fn()
    st.session_state["stage_timings"][name] = time.perf_counter() - start
    return result


# =============================================================================
# 5. REUSABLE COMPONENTS
# =============================================================================

def metric_card(label: str, value: str, caption: str = "") -> str:
    cap = f'<div class="quasar-metric-caption">{caption}</div>' if caption else ""
    return f"""
    <div class="quasar-card">
        <div class="quasar-metric-label">{label}</div>
        <div class="quasar-metric-value">{value}</div>
        {cap}
    </div>
    """


def metric_grid(items: list[tuple[str, str, str]], columns: int = 4) -> None:
    cols = st.columns(columns)
    for i, (label, value, caption) in enumerate(items):
        with cols[i % columns]:
            st.markdown(metric_card(label, value, caption), unsafe_allow_html=True)


def empty_state(glyph: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="quasar-empty">
            <div class="quasar-empty-glyph">{glyph}</div>
            <div style="color: var(--text-primary); font-weight: 600; margin-bottom: 4px;">{title}</div>
            <div style="font-size: 13px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def loading_skeleton(n_cards: int = 4) -> None:
    cols = st.columns(min(n_cards, 4))
    for i in range(n_cards):
        with cols[i % len(cols)]:
            st.markdown('<div class="quasar-skeleton"></div>', unsafe_allow_html=True)


def error_panel(title: str, message: str, exc_details: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="quasar-error">
            <div class="quasar-error-title">⚠ {title}</div>
            <div class="quasar-error-msg">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if exc_details:
        with st.expander("Technical details"):
            st.code(exc_details)


EXPLAIN_STEPS = [
    "Loading model artifacts",
    "Computing SHAP values",
    "Generating SHAP plots",
    "Computing LIME explanations",
    "Writing explanation report",
    "Computing local SHAP for circuit",
    "Building recommendations",
]


def gate_distribution_strip(gate_counts: dict[str, int]) -> None:
    """Segmented horizontal bar — reused for gate distribution, SHAP ranking bars,
    and recommendation impact meters (one chart idiom, several contexts)."""
    if not gate_counts:
        st.caption("No gate data available.")
        return
    if not HAVE_PLOTLY:
        st.table({"gate": list(gate_counts.keys()), "count": list(gate_counts.values())})
        return
    items = sorted(gate_counts.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]
    n = len(names)
    colors = [
        f"rgba({int(108 + (35 - 108) * i / max(1, n - 1))}, "
        f"{int(62 + (216 - 62) * i / max(1, n - 1))}, "
        f"{int(255 + (255 - 255) * i / max(1, n - 1))}, 0.9)"
        for i in range(n)
    ]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors))
    fig.update_layout(
        autosize=True,
        margin=dict(l=8, r=8, t=8, b=8),
        height=max(160, 28 * n),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TOKENS["text_secondary"], family="Inter"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def reliability_gauge(score: float, label: str) -> None:
    if not HAVE_PLOTLY:
        st.metric("Reliability Score", f"{score:.1f} / 100", label)
        return
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "", "font": {"size": 48, "family": "Manrope"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TOKENS["text_muted"]},
                "bar": {"color": TOKENS["accent_secondary"]},
                "bgcolor": TOKENS["bg_tertiary"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 100], "color": TOKENS["bg_tertiary"]},
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        autosize=True,
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TOKENS["text_primary"]),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown(
        f"<div style='text-align:center; font-family: Space Grotesk; font-weight:600; "
        f"font-size:22px;' class='quasar-gradient-text'>{label}</div>",
        unsafe_allow_html=True,
    )


def probability_bars(probabilities: dict[str, float]) -> None:
    for cls in ["LOW", "MEDIUM", "HIGH"]:
        p = probabilities.get(cls, 0.0)
        st.markdown(
            f"""
            <div style="margin-bottom:8px;">
              <div style="display:flex; justify-content:space-between; font-size:13px;
                          color:var(--text-secondary); margin-bottom:3px;">
                <span>{cls}</span><span>{p*100:.1f}%</span>
              </div>
              <div style="background:var(--bg-tertiary); border-radius:8px; height:8px; overflow:hidden;">
                <div style="width:{max(2, p*100):.1f}%; height:100%; background:{GRADIENT};"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def pipeline_stepper() -> None:
    idx = st.session_state["stage_index"]
    err = st.session_state["stage_error"]
    timings = st.session_state["stage_timings"]

    html_parts = ['<div class="quasar-stepper">']
    for i, name in enumerate(STAGES):
        if err and STAGES.index(err[0]) == i:
            dot_class = "error"
        elif i <= idx:
            dot_class = "done"
        elif i == idx + 1 and not err:
            dot_class = "active"
        else:
            dot_class = ""

        line_class = "done" if i < idx else ""
        t = timings.get(name)
        time_label = f'<div class="quasar-step-time">{t:.2f}s</div>' if t is not None else ""
        html_parts.append(
            f'<div class="quasar-step">'
            f'<div class="quasar-step-dot {dot_class}"></div>'
            f'<div class="quasar-step-label">{name}</div>{time_label}</div>'
        )
        if i < len(STAGES) - 1:
            html_parts.append(f'<div class="quasar-step-line {line_class}" style="margin-top:8px;"></div>')
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)

    if timings:
        total = sum(timings.values())
        breakdown = " · ".join(f"{k} {v:.2f}s" for k, v in timings.items())
        suffix = " so far…" if idx < len(STAGES) - 1 and not err else ""
        st.markdown(
            f'<div class="quasar-caption-row">Total elapsed: {total:.2f}s{suffix} '
            f'({breakdown})</div>',
            unsafe_allow_html=True,
        )


def render_figure_card(fig: Any) -> None:
    """Render a matplotlib figure inside a `.quasar-card-tertiary` card.

    Embeds the figure as a base64 PNG inside a SINGLE st.markdown call.
    (Splitting an opening <div> and closing </div> across two separate
    st.markdown/st.pyplot calls does NOT nest them — Streamlit renders
    each call as its own independent DOM node — so the opening div was
    rendering as its own empty, padded, colored box floating above the
    plot. This is the fix.)
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=TOKENS["bg_tertiary"], dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    st.markdown(
        f'<div class="quasar-card-tertiary" style="text-align:center;">'
        f'<img src="data:image/png;base64,{b64}" style="max-width:100%; border-radius:8px;" />'
        f"</div>",
        unsafe_allow_html=True,
    )
    plt.close(fig)


def render_image_card(path: str | Path, caption: str | None = None) -> None:
    """Render an on-disk image inside a `.quasar-card-tertiary` card.

    Same single-markdown-call fix as `render_figure_card`, for images
    already saved to disk (e.g. SHAP/LIME plots) rather than a live
    matplotlib figure.
    """
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    caption_html = (
        f'<div style="text-align:center; color:var(--text-muted); '
        f'font-size:12px; margin-top:8px;">{caption}</div>'
        if caption
        else ""
    )
    st.markdown(
        f'<div class="quasar-card-tertiary">'
        f'<img src="data:image/png;base64,{b64}" style="width:100%; border-radius:8px;" />'
        f"{caption_html}</div>",
        unsafe_allow_html=True,
    )


def circuit_diagram_card(qasm_text: str) -> None:
    st.markdown("**Circuit Diagram**")
    if HAVE_QISKIT:
        try:
            from qiskit.qasm3 import loads as qasm3_loads

            qc = qasm3_loads(qasm_text)
            if qc.num_qubits > 14 or qc.depth() > 80:
                empty_state("⤢", "Diagram too large to render inline",
                            "Download the circuit to view it — rendering this many "
                            "qubits/layers inline would be illegible.")
                return
            fig = qc.draw(output="mpl", style={"backgroundcolor": TOKENS["bg_tertiary"]})
            render_figure_card(fig)
            return
        except Exception as exc:
            error_panel(
                "Diagram unavailable",
                "The circuit parsed, but the diagram could not be rendered.",
                str(exc),
            )
            return
    # Fallback ASCII-ish gate list for demo mode without qiskit
    gate_lines = [l.strip() for l in qasm_text.splitlines() if _GATE_RE.match(l)]
    st.code("\n".join(gate_lines[:40]) + ("\n…" if len(gate_lines) > 40 else ""), language="text")
    st.caption("Text view — install qiskit for a rendered circuit diagram.")


# =============================================================================
# 6. PIPELINE ORCHESTRATION
# =============================================================================

def run_pipeline(circuit_name: str, qasm_text: str) -> None:
    """Runs Parse -> Analyze -> Features -> Predict. Explain/Recommend are
    deferred (see §10.11 of the spec) until the user opens Explainability."""
    st.session_state["circuit_name"] = circuit_name
    st.session_state["qasm_text"] = qasm_text
    st.session_state["stage_index"] = 0  # Upload done
    st.session_state["stage_error"] = None

    tmp_path = None
    try:
        if BACKEND_READY:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".qasm", delete=False
            ) as tmp:
                tmp.write(qasm_text)
                tmp_path = Path(tmp.name)
            circuit = time_stage("Parse", lambda: load_qasm_file(tmp_path))
            st.session_state["stage_index"] = 1

            analysis = time_stage("Analyze", lambda: analyze_circuit(circuit))
            st.session_state["analysis"] = analysis
            st.session_state["stage_index"] = 2

            features = time_stage("Features", lambda: extract_features(circuit))
            st.session_state["features"] = features
            st.session_state["stage_index"] = 3

            if MODEL_TRAINED:
                result = time_stage(
                    "Predict",
                    lambda: run_inference(tmp_path, InferenceConfig()),
                )
                st.session_state["prediction"] = {
                    "reliability_class": result.reliability_class,
                    "confidence": result.confidence,
                    "class_probabilities": result.class_probabilities,
                    "reliability_score_estimate": result.reliability_score_estimate,
                    "model_name": result.model_name,
                    "unavailable_features": result.unavailable_features,
                }
            else:
                merged = {**analysis, **features}
                st.session_state["prediction"] = time_stage(
                    "Predict", lambda: demo_prediction(qasm_text, merged)
                )
            st.session_state["stage_index"] = 4
        else:
            structural = time_stage("Parse", lambda: demo_structural_reader(qasm_text))
            st.session_state["stage_index"] = 1
            st.session_state["analysis"] = structural
            st.session_state["stage_index"] = 2
            st.session_state["features"] = structural
            st.session_state["stage_index"] = 3
            st.session_state["prediction"] = time_stage(
                "Predict", lambda: demo_prediction(qasm_text, structural)
            )
            st.session_state["stage_index"] = 4
    except Exception as exc:  # noqa: BLE001 — surfaced via the stepper's error state
        failed_stage = STAGES[st.session_state["stage_index"] + 1] \
            if st.session_state["stage_index"] + 1 < len(STAGES) else STAGES[-1]
        st.session_state["stage_error"] = (failed_stage, str(exc))
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


@st.cache_resource(show_spinner=False)
def _cached_explainability(_models_dir: str):
    """Cached so SHAP/LIME only ever compute once per model version per
    session. Both parameters are prefixed with an underscore so Streamlit
    excludes them from the cache key -- a callback isn't meaningfully
    hashable/comparable across reruns anyway, and this remains, as
    before, effectively a single-entry cache for the session's lifetime.
    This cached function deliberately does not accept or call Streamlit
    progress callbacks. Streamlit cannot replay cached UI writes into
    layout blocks created outside the cached function."""
    return explain_model(ExplainConfig(), progress_callback=None)


def _uncached_explainability(shap_progress_callback=None):
    """Non-cached version that accepts a SHAP progress callback for
    displaying live progress in the Streamlit dashboard."""
    return explain_model(
        ExplainConfig(),
        progress_callback=None,
        shap_progress_callback=shap_progress_callback,
    )


def run_explain_and_recommend(
    qasm_text: str, progress_callback: Callable[[str], None] | None = None,
    shap_progress_callback: Callable[[str, float], None] | None = None,
) -> None:
    features = st.session_state["features"] or {}
    if BACKEND_READY and MODEL_TRAINED:
        tmp_path = None
        try:
            if progress_callback:
                progress_callback("Preparing cached global SHAP + LIME model explanations…")
            # Use uncached version if we have a SHAP progress callback
            if shap_progress_callback:
                summary = time_stage(
                    "Explain",
                    lambda: _uncached_explainability(shap_progress_callback),
                )
            else:
                summary = time_stage(
                    "Explain",
                    lambda: _cached_explainability(str(MODELS_DIR)),
                )
            st.session_state["explainability"] = summary
            st.session_state["stage_index"] = 5

            if progress_callback:
                progress_callback("Global SHAP + LIME ready. Computing local SHAP for this circuit…")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False) as tmp:
                tmp.write(qasm_text)
                tmp_path = Path(tmp.name)
            st.session_state["local_shap"] = time_stage(
                "Local SHAP",
                lambda: explain_local_circuit(
                    tmp_path,
                    ExplainConfig(shap_sample_size=1),
                    progress_callback=progress_callback,
                ),
            )
            if progress_callback:
                progress_callback("Building recommendations from SHAP + circuit structure…")
            result = time_stage(
                "Recommend",
                lambda: generate_recommendations(
                    tmp_path, RecommendationConfig(), explainability_summary=summary
                ),
            )
            st.session_state["recommendations"] = result.recommendations
            st.session_state["stage_index"] = 6
        except Exception as exc:  # noqa: BLE001
            st.session_state["stage_error"] = ("Explain", str(exc))
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
    else:
        if progress_callback:
            progress_callback("DEMO MODE: no trained backend — generating synthetic explanations…")
        st.session_state["explainability"] = {"shap_method": "demo_synthetic"}
        st.session_state["local_shap"] = None
        st.session_state["stage_index"] = 5
        st.session_state["recommendations"] = time_stage(
            "Recommend", lambda: demo_recommendations(features)
        )
        st.session_state["stage_index"] = 6
    st.session_state["explain_requested"] = True


# =============================================================================
# 7. PAGE: LANDING
# =============================================================================

def render_landing() -> None:
    render_bloch_sphere_hero()
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    cols = st.columns([1, 3])
    with cols[0]:
        st.markdown(
            f"<div style='text-align:left; padding-top:4px;'>{logo_img_tag(300)}</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            "<h1 class='quasar-gradient-text' style='font-family: \"Orbitron\", sans-serif; font-weight:800; text-align:left; font-size:56px; margin:8px 0 0 0; line-height:1.1;'>QUASAR</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:left; color:var(--text-secondary); font-size:17px; "
            "margin:4px 0 0 0;'>"
            "AI-Powered Quantum Circuit Reliability Assessment</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:left; color:var(--text-muted); font-size:14px; "
            "max-width:560px; margin:12px 0 0 0px;'>"
            "Predict, explain, and optimize the reliability of OpenQASM 3 circuits "
            "before they ever touch real quantum hardware.</p>",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("Start Analysis  →", width="stretch"):
            st.session_state["route"] = "analysis"
            st.rerun()

    if DEMO_MODE:
        st.markdown(
            "<p style='text-align:center; color:var(--text-muted); font-size:12px;'>"
            "Running in DEMO MODE — showing synthetic results until a trained backend "
            "is connected.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:24px;'></div><hr>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:var(--text-muted); font-size:12px; "
        "letter-spacing:0.08em; text-transform:uppercase; margin-top:32px;'>"
        "Or start from a known circuit</p>",
        unsafe_allow_html=True,
    )

    names = list(_SAMPLE_QASM.keys())
    cols = st.columns(3)
    for i, name in enumerate(names):
        qubits, desc = _SAMPLE_META[name]
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="quasar-card">
                    <div style="font-weight:600; font-size:16px;">{name}</div>
                    <div style="color:var(--text-muted); font-size:12px; margin:4px 0;">{qubits}</div>
                    <div style="color:var(--text-secondary); font-size:13px; min-height:36px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Analyze {name} →", key=f"sample_{name}", width="stretch"):
                run_pipeline(name, get_sample_qasm(name))
                st.session_state["route"] = "analysis"
                st.rerun()

    render_research_metrics_strip()


def render_research_metrics_strip() -> None:
    st.markdown("<hr style='margin-top:24px;'>", unsafe_allow_html=True)
    metadata_path = MODELS_DIR / "training_metadata.json"
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            metrics = meta.get("metrics_by_model", {}).get(meta.get("winning_model", ""), {})
            circuits = "50,000+"
            accuracy = f"{metrics.get('accuracy', 0) * 100:.1f}%" if metrics else "—"
            f1 = f"{metrics.get('f1_macro', 0):.2f}" if metrics else "—"
        except Exception:
            circuits, accuracy, f1 = "50,000+", "—", "—"
    else:
        circuits, accuracy, f1 = "50,000+", "—", "—"

    cols = st.columns(3)
    stats = [
        (circuits, "circuits generated"),
        (accuracy, "test accuracy"),
        (f1, "macro F1"),
    ]
    for col, (value, label) in zip(cols, stats):
        with col:
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<div class='quasar-metric-value' style='font-size:26px;'>{value}</div>"
                f"<div class='quasar-metric-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )
    st.markdown(
        "<p style='text-align:center; color:var(--text-muted); font-size:12px; "
        "margin-top:12px;'>Explainable via SHAP (TreeExplainer) + LIME</p>",
        unsafe_allow_html=True,
    )


# =============================================================================
# 8. PAGE: ANALYSIS
# =============================================================================

def render_analysis() -> None:
    st.markdown("## Analysis")

    if st.session_state["qasm_text"] is None:
        render_upload_panel()
        empty_state("⬚", "No circuit loaded yet", "Upload a .qasm file or pick a sample to begin.")
        return

    # Collapsed source summary bar
    name = st.session_state["circuit_name"]
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(
            f"<div class='quasar-card' style='padding:14px 20px; display:flex; "
            f"align-items:center;'>📄&nbsp; <b>{name}</b>"
            f"&nbsp;·&nbsp;<span style='color:var(--text-muted)'>loaded</span></div>",
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("Change ↺", width="stretch"):
            reset_pipeline()
            st.rerun()

    pipeline_stepper()

    err = st.session_state["stage_error"]
    if err:
        error_panel(
            f"{err[0]} failed",
            "The pipeline could not complete this stage. You can try another file, "
            "or inspect the technical details below.",
            err[1],
        )

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    render_circuit_visualization()
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    render_circuit_overview()
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    render_feature_extraction()
    st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)
    render_prediction_hero()
    st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
    explain_placeholders = render_explainability()
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    render_recommendations()
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    render_downloads()

    if explain_placeholders is not None:
        # The rest of the page (recommendations, downloads — including
        # their now-disabled/locked buttons) has already rendered above
        # in this same pass, so nothing on screen disappears while this
        # runs; only the progress bar/status text below actually update
        # live as the computation proceeds.
        shap_bar, status_text = explain_placeholders

        def _log(message: str) -> None:
            status_text.caption(message)

        def _shap_progress(desc: str, pct: float) -> None:
            display_desc = desc if desc else "Computing SHAP values"
            pct_capped = min(max(pct, 0.0), 1.0)
            shap_bar.progress(pct_capped, text=f"{display_desc} — {pct_capped*100:.0f}%")

        run_explain_and_recommend(
            st.session_state["qasm_text"],
            progress_callback=_log,
            shap_progress_callback=_shap_progress,
        )

        if st.session_state["stage_error"] and st.session_state["stage_error"][0] == "Explain":
            status_text.error(f"Failed: {st.session_state['stage_error'][1]}")
        else:
            shap_bar.progress(1.0, text="Done")
            status_text.caption("Rendering SHAP + LIME below…")
        time.sleep(0.5)
        shap_bar.empty()
        status_text.empty()
        st.session_state["explain_running"] = False
        st.rerun()


def render_upload_panel() -> None:
    uploaded = st.file_uploader("Upload OpenQASM 3 circuit", type=["qasm"], label_visibility="collapsed")
    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="ignore")
        run_pipeline(uploaded.name, text)
        st.rerun()


def render_circuit_visualization() -> None:
    st.markdown("### Circuit Visualization")
    if st.session_state["stage_index"] < 1 and not st.session_state["stage_error"]:
        loading_skeleton(1)
        return
    if st.session_state["stage_error"] and st.session_state["stage_error"][0] in ("Parse",):
        error_panel("Circuit could not be parsed", st.session_state["stage_error"][1])
        return
    circuit_diagram_card(st.session_state["qasm_text"])


def render_circuit_overview() -> None:
    st.markdown("### Circuit Overview")
    a = st.session_state["analysis"]
    if a is None:
        if st.session_state["stage_error"]:
            error_panel("Circuit analysis unavailable", st.session_state["stage_error"][1])
        else:
            loading_skeleton(4)
        return
    approx_note = " (approximate)" if a.get("_approx") else ""
    metric_grid(
        [
            ("Qubits", str(a["num_qubits"]), ""),
            ("Classical Bits", str(a["num_clbits"]), ""),
            ("Depth", str(a["depth"]), approx_note.strip()),
            ("Width", str(a["width"]), ""),
        ]
    )
    metric_grid(
        [
            ("Operation Count", str(a["total_operations"]), ""),
            ("Measurements", str(a["num_measurements"]), ""),
            ("Unique Gate Types", str(len(a.get("gates_used", []))), ""),
        ],
        columns=3,
    )
    gate_distribution_strip(a.get("gate_counts", {}))


def render_feature_extraction() -> None:
    st.markdown("### Machine Learning Features")
    f = st.session_state["features"]
    if f is None:
        if st.session_state["stage_error"]:
            error_panel("Feature extraction unavailable", st.session_state["stage_error"][1])
        else:
            loading_skeleton(4)
        return
    metric_grid(
        [
            ("Single-Qubit Gates", str(f["single_qubit_gates"]), ""),
            ("Two-Qubit Gates", str(f["two_qubit_gates"]), ""),
            ("Three-Qubit Gates", str(f["three_qubit_gates"]), ""),
            ("Parameterized Gates", str(f["parameterized_gates"]), ""),
        ]
    )
    metric_grid(
        [
            ("Entangling Gates", str(f["entangling_gates"]), ""),
            ("Measurement Gates", str(f.get("measurement_gates", f.get("num_measurements", 0))), ""),
        ],
        columns=2,
    )


def render_prediction_hero() -> None:
    st.markdown(
        "<h2 style='text-align:center;'>Predicted Reliability</h2>",
        unsafe_allow_html=True,
    )
    pred = st.session_state["prediction"]
    if pred is None:
        if st.session_state["stage_error"]:
            error_panel("Prediction unavailable", st.session_state["stage_error"][1])
            reliability_gauge(0, "—")
        else:
            loading_skeleton(1)
        return

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        reliability_gauge(
            pred["reliability_score_estimate"], pred["reliability_class"]
        )
        st.markdown(
            f"<p style='text-align:center; color:var(--text-secondary);'>"
            f"Confidence: {pred['confidence']*100:.1f}% &nbsp;·&nbsp; "
            f"Model: {pred['model_name']}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='max-width:480px; margin:0 auto;'>", unsafe_allow_html=True)
        probability_bars(pred["class_probabilities"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center; color:var(--text-muted); font-size:12px; "
            "margin-top:12px;'>ⓘ Reliability Score is a heuristic estimate derived "
            "from class probabilities, not a direct regression prediction.</p>",
            unsafe_allow_html=True,
        )
        if pred.get("unavailable_features"):
            st.markdown(
                "<p style='text-align:center; color:var(--text-muted); font-size:12px;'>"
                "Predicted from structural features only — noise-simulation metrics "
                "unavailable for this run.</p>",
                unsafe_allow_html=True,
            )


def render_explainability() -> tuple[Any, Any] | None:
    """Render the Explainability section.

    Returns a `(shap_bar, status_text)` placeholder pair when a SHAP+LIME
    computation still needs to run for this script pass, or `None`
    otherwise. The caller (`render_analysis`) is responsible for actually
    invoking `run_explain_and_recommend` using those placeholders — but
    only *after* it has finished rendering the rest of the page (the
    Recommendations and Downloads sections) in this same pass, so every
    button on the page stays visible (locked, not vanished) for the full
    duration of the computation instead of disappearing until it's done.
    """
    st.markdown("### Explainability")
    if st.session_state["prediction"] is None:
        empty_state("◔", "Explainability", "Run a prediction first to unlock SHAP and LIME.")
        return None

    if not st.session_state["explain_requested"]:
        if st.button("Compute SHAP + LIME explanations", key="explain_btn"):
            st.session_state["explain_requested"] = True
            st.session_state["explain_running"] = True
            st.rerun()
        empty_state("◔", "Not yet computed", "Explainability is deferred until requested — "
                                             "it is the most expensive pipeline stage.")
        return None

    if st.session_state.get("explain_running"):
        st.button(
            "Compute SHAP + LIME explanations",
            key="explain_btn_running",
            disabled=True,
            width="stretch",
        )
        shap_bar = st.progress(0, text="Computing SHAP values…")
        status_text = st.empty()
        status_text.caption("Starting SHAP + LIME computation…")
        return shap_bar, status_text

    if st.session_state["stage_error"] and st.session_state["stage_error"][0] == "Explain":
        error_panel("Explainability unavailable", st.session_state["stage_error"][1])
        return None

    st.button(
        "Compute SHAP + LIME explanations",
        key="explain_btn_done",
        disabled=True,
        width="stretch",
    )

    tab_shap, tab_lime = st.tabs(["SHAP", "LIME"])
    summary = st.session_state["explainability"]

    with tab_shap:
        if DEMO_MODE:
            st.caption("Method: demo_synthetic (connect a trained backend for real SHAP values)")
            demo_ranking = {
                "depth": 0.18, "two_qubit_gates": 0.15, "entangling_gates": 0.13,
                "gate_cx": 0.10, "total_operations": 0.07,
            }
            gate_distribution_strip(demo_ranking)
        else:
            st.caption(f"Method: {summary.shap_method}")
            st.markdown(
                "<p style='color:var(--text-muted); font-size:12.5px; margin-top:-4px;'>"
                "The local SHAP table below is recomputed for the current circuit. "
                "The ranking and beeswarm plots after it are the trained model's "
                "<em>global</em> behavior, cached once per model for speed.</p>",
                unsafe_allow_html=True,
            )
            local_shap = st.session_state.get("local_shap")
            if local_shap is not None:
                st.markdown("**Local SHAP for this circuit**")
                st.caption(
                    f"Predicted class: {local_shap.predicted_class} "
                    f"({local_shap.confidence:.1%} confidence) · Method: {local_shap.shap_method}"
                )
                positives = [item for item in local_shap.contributions if item.shap_value >= 0]
                negatives = [item for item in local_shap.contributions if item.shap_value < 0]
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Supports prediction**")
                    if positives:
                        for item in positives:
                            st.write(
                                f"✓ {item.feature} = {item.value:.3g} "
                                f"&nbsp;&nbsp;+{item.shap_value:.4f}"
                            )
                    else:
                        st.write("No positive local SHAP contributors in the top set.")
                with c2:
                    st.markdown("**Opposes prediction**")
                    if negatives:
                        for item in negatives:
                            st.write(
                                f"✕ {item.feature} = {item.value:.3g} "
                                f"&nbsp;&nbsp;{item.shap_value:.4f}"
                            )
                    else:
                        st.write("No negative local SHAP contributors in the top set.")
                if local_shap.unavailable_features:
                    st.caption(
                        "Noise-derived feature placeholders used: "
                        + ", ".join(local_shap.unavailable_features)
                    )
                st.markdown("<hr style='margin:18px 0;'>", unsafe_allow_html=True)
            else:
                empty_state("◔", "Local SHAP unavailable", "No circuit-specific SHAP result was stored.")
            ranking = dict(summary.global_feature_ranking)
            gate_distribution_strip(ranking)
            cols = st.columns(min(3, len(summary.plot_paths)) or 1)
            beeswarm_plots = {k: v for k, v in summary.plot_paths.items() if "beeswarm" in k}
            for i, (name, path) in enumerate(beeswarm_plots.items()):
                with cols[i % len(cols)]:
                    render_image_card(path, caption=name)

    with tab_lime:
        if DEMO_MODE:
            st.caption("Synthetic local explanation (demo mode)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Supports prediction**")
                st.write("✓ depth ≤ 30 &nbsp;&nbsp;+0.041")
                st.write("✓ two_qubit_gates ≤ 5 &nbsp;&nbsp;+0.033")
            with c2:
                st.markdown("**Opposes prediction**")
                st.write("✕ entangling_gates > 12 &nbsp;&nbsp;−0.022")
        else:
            matching = next(
                (ex for ex in summary.lime_examples
                 if ex.predicted_class == st.session_state["prediction"]["reliability_class"]),
                None,
            )
            if matching is None:
                empty_state("◔", "No LIME example available",
                            "No representative circuit was found for this predicted class.")
            else:
                c1, c2 = st.columns(2)
                positives = [c for c in matching.contributions if c[1] > 0]
                negatives = [c for c in matching.contributions if c[1] <= 0]
                with c1:
                    st.markdown("**Supports prediction**")
                    for feat, w in positives:
                        st.write(f"✓ {feat} &nbsp;&nbsp;+{w:.3f}")
                with c2:
                    st.markdown("**Opposes prediction**")
                    for feat, w in negatives:
                        st.write(f"✕ {feat} &nbsp;&nbsp;{w:.3f}")
                if matching.plot_path:
                    render_image_card(matching.plot_path)


def render_recommendations() -> None:
    st.markdown("### Recommendations")
    if not st.session_state["explain_requested"]:
        empty_state("✓", "Not yet available", "Compute explainability above to unlock recommendations.")
        return
    if st.session_state["stage_error"] and st.session_state["stage_error"][0] == "Recommend":
        error_panel("Recommendations unavailable", st.session_state["stage_error"][1])
        return

    recs = st.session_state["recommendations"] or []
    if not recs:
        st.markdown(
            "<div class='quasar-empty'><div class='quasar-empty-glyph'>✓</div>"
            "<div style='color:var(--text-primary); font-weight:600;'>"
            "No optimization recommendations.</div>"
            "<div>The circuit already appears structurally efficient.</div></div>",
            unsafe_allow_html=True,
        )
        return

    for i, rec in enumerate(recs, start=1):
        title = rec["title"] if isinstance(rec, dict) else rec.title
        detail = rec["detail"] if isinstance(rec, dict) else rec.detail
        impact = rec["shap_impact"] if isinstance(rec, dict) else rec.shap_impact
        category = rec["category"] if isinstance(rec, dict) else rec.category
        with st.container():
            st.markdown(
                f"""
                <div class="quasar-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-weight:600; font-size:16px;">{i}. {title}</div>
                        <span class="quasar-chip">{category}</span>
                    </div>
                    <div style="color:var(--text-muted); font-size:12px; margin:6px 0;">
                        Impact: {impact:.3f}
                    </div>
                    <div style="color:var(--text-secondary); font-size:14px;">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_downloads() -> None:
    st.markdown("### Downloads")
    is_running = st.session_state.get("explain_running", False)
    cols = st.columns(3)
    features_csv = ""
    if st.session_state["features"]:
        rows = "\n".join(f"{k},{v}" for k, v in st.session_state["features"].items() if not isinstance(v, dict))
        features_csv = "feature,value\n" + rows
    with cols[0]:
        st.download_button(
            "⬇ Features CSV", features_csv or "no data",
            file_name="features.csv", width="stretch",
            disabled=(not features_csv or is_running),
        )
    with cols[1]:
        json_blob = json.dumps(
            {
                "circuit_name": st.session_state["circuit_name"],
                "analysis": st.session_state["analysis"],
                "prediction": st.session_state["prediction"],
                "recommendations": st.session_state["recommendations"],
            },
            default=str,
            indent=2,
        )
        st.download_button(
            "⬇ JSON Results", json_blob, file_name="quasar_results.json",
            width="stretch",
            disabled=is_running,
        )
    with cols[2]:
        explainability_obj = st.session_state["explainability"]
        report_path = getattr(explainability_obj, "report_path", None)
        report_text = None
        if report_path is not None:
            try:
                report_text = Path(report_path).read_text()
            except OSError:
                report_text = None

        if report_text:
            report_content = report_text
        elif DEMO_MODE:
            report_content = (
                "Explanation report is only produced by the real backend. "
                "This Space is currently running in DEMO MODE (no trained "
                "model found under models/), so SHAP/LIME explanations are "
                "synthetic placeholders rather than a real report."
            )
        else:
            report_content = (
                "No explanation report has been generated yet for this "
                "session. Click 'Compute Explainability' on the "
                "Explainability tab first, then come back to download it."
            )
        st.download_button(
            "⬇ Explanation Report",
            report_content,
            file_name="explanation_report.txt",
            width="stretch",
            disabled=is_running,
        )


# =============================================================================
# 9. PAGE: ABOUT
# =============================================================================

def render_about() -> None:
    st.markdown("## About QUASAR")
    st.markdown(
        "QUASAR predicts the reliability of OpenQASM 3 quantum circuits before "
        "execution, using a trained ensemble classifier over structural circuit "
        "features — then explains every prediction with SHAP and LIME, and "
        "recommends concrete structural improvements."
    )

    st.markdown("#### Pipeline")
    pipeline_labels = [
        "Circuit Gen", "Parser", "Analyzer", "Features", "Noise Sim",
        "Dataset Gen", "Preprocess", "Train", "Evaluate", "Explain", "Infer", "Recommend",
    ]
    st.markdown(
        "<div class='quasar-stepper'>" + "".join(
            f"<div class='quasar-step'><div class='quasar-step-dot done'></div>"
            f"<div class='quasar-step-label'>{lbl}</div></div>"
            + ("<div class='quasar-step-line done' style='margin-top:8px;'></div>" if i < len(pipeline_labels) - 1 else "")
            for i, lbl in enumerate(pipeline_labels)
        ) + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Dataset & Model")
    render_research_metrics_strip()

    st.markdown("#### Explainability Methods")
    st.markdown(
        "SHAP (`TreeExplainer`, with an automatic model-agnostic fallback) and "
        "LIME (local tabular surrogate models) — see `explainability.py`."
    )

    st.markdown("#### Technology Stack")
    st.markdown("".join(
        f"<span class='quasar-chip'>{chip}</span>"
        for chip in ["Qiskit", "Qiskit Aer", "scikit-learn", "SHAP", "LIME", "Streamlit", "Plotly"]), unsafe_allow_html=True,)


# =============================================================================
# 10. SIDEBAR + ROUTER
# =============================================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:8px;'>"
            f"{logo_img_tag(200)}<span style='font-family:Space Grotesk; font-weight:600; ",
            #f"font-size:16px; color:var(--text-primary);'>QUASAR</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        if st.button("Analysis", width="stretch"):
            st.session_state["route"] = "analysis"
            st.rerun()
        if st.button("About", width="stretch"):
            st.session_state["route"] = "about"
            st.rerun()

        st.markdown("<div style='height:120px;'></div><hr>", unsafe_allow_html=True)
        st.markdown(
            "[GitHub](https://github.com) &nbsp;·&nbsp; [Docs](https://docs.example.com)"
        )
        st.caption("v1.0.0" + ("  ·  DEMO MODE" if DEMO_MODE else ""))


def main() -> None:
    init_session_state()
    inject_css()
    render_quantum_background()
    render_mouse_tracker()

    if st.session_state["route"] == "landing":
        render_landing()
        return

    render_sidebar()
    if st.session_state["route"] == "analysis":
        render_analysis()
    else:
        render_about()


if __name__ == "__main__":
    main()
