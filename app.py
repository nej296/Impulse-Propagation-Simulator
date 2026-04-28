"""
Hodgkin-Huxley Neuron Simulator — Interactive Streamlit App
"""

import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hodgkin_huxley import run_simulation, run_propagation, get_i_inj, V_REST, h_inf

st.set_page_config(
    page_title="Hodgkin-Huxley Neuron Simulator",
    page_icon="🧠",
    layout="wide",
)

THEME = {
    "bg": "#05070d",
    "panel": "#0c1220",
    "panel_alt": "#111a2d",
    "ink": "#eef6ff",
    "muted": "#8fa1ba",
    "line": "#24324f",
    "cyan": "#29f0ff",
    "blue": "#69a7ff",
    "amber": "#ffb84d",
    "rose": "#ff5c8a",
    "green": "#7dffb2",
}


def inject_global_css():
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,600;6..72,700&display=swap');

            :root {{
                --bg: {THEME["bg"]};
                --panel: {THEME["panel"]};
                --panel-alt: {THEME["panel_alt"]};
                --ink: {THEME["ink"]};
                --muted: {THEME["muted"]};
                --line: {THEME["line"]};
                --cyan: {THEME["cyan"]};
                --blue: {THEME["blue"]};
                --amber: {THEME["amber"]};
                --rose: {THEME["rose"]};
                --green: {THEME["green"]};
            }}

            .stApp {{
                color: var(--ink);
                background:
                    radial-gradient(circle at 15% 12%, rgba(41, 240, 255, 0.16), transparent 28rem),
                    radial-gradient(circle at 86% 4%, rgba(255, 92, 138, 0.14), transparent 24rem),
                    linear-gradient(135deg, #05070d 0%, #071120 48%, #03040a 100%);
            }}

            .stApp::before {{
                content: "";
                position: fixed;
                inset: 0;
                pointer-events: none;
                background-image:
                    linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
                background-size: 46px 46px;
                mask-image: linear-gradient(to bottom, rgba(0,0,0,0.8), transparent 75%);
            }}

            .block-container {{
                max-width: 1420px;
                padding-top: 0.75rem;
                padding-bottom: 3rem;
            }}

            [data-testid="stHeader"] {{
                display: none;
            }}

            [data-testid="stSidebar"] {{
                background:
                    linear-gradient(180deg, rgba(12, 18, 32, 0.98), rgba(5, 7, 13, 0.98)),
                    radial-gradient(circle at top, rgba(41, 240, 255, 0.18), transparent 18rem);
                border-right: 1px solid rgba(105, 167, 255, 0.22);
            }}

            [data-testid="stSidebar"] * {{
                font-family: "IBM Plex Mono", monospace;
            }}

            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {{
                color: var(--ink);
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }}

            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p {{
                color: var(--muted);
            }}

            .stNumberInput input,
            .stSelectbox div[data-baseweb="select"] > div,
            .stSlider [data-baseweb="slider"] {{
                border-color: rgba(105, 167, 255, 0.28) !important;
            }}

            [data-testid="stMarkdownContainer"] p {{
                margin-bottom: 0.25rem;
            }}

            .param-section {{
                margin: 0.8rem 0 0.35rem;
                padding-top: 0.7rem;
                border-top: 1px solid rgba(105, 167, 255, 0.18);
                color: var(--cyan);
                font-family: "IBM Plex Mono", monospace;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }}

            .param-note {{
                margin: -0.2rem 0 0.55rem;
                color: var(--muted);
                font-family: "IBM Plex Mono", monospace;
                font-size: 0.68rem;
                line-height: 1.35;
            }}

            .stButton > button {{
                border: 1px solid rgba(41, 240, 255, 0.42);
                border-radius: 999px;
                color: var(--ink);
                background: linear-gradient(135deg, rgba(41, 240, 255, 0.18), rgba(105, 167, 255, 0.12));
                box-shadow: 0 0 24px rgba(41, 240, 255, 0.08);
                font-family: "IBM Plex Mono", monospace;
                font-weight: 700;
                letter-spacing: 0.03em;
            }}

            .stButton > button:hover {{
                border-color: var(--cyan);
                box-shadow: 0 0 34px rgba(41, 240, 255, 0.22);
                transform: translateY(-1px);
            }}

            .stPlotlyChart {{
                border: 1px solid rgba(105, 167, 255, 0.24);
                border-radius: 28px;
                padding: 0.55rem;
                background: linear-gradient(180deg, rgba(12, 18, 32, 0.88), rgba(8, 12, 22, 0.94));
                box-shadow: 0 22px 80px rgba(0, 0, 0, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.04);
                overflow: hidden;
            }}

            div[data-testid="stExpander"] {{
                border: 1px solid rgba(105, 167, 255, 0.22);
                border-radius: 22px;
                background: rgba(12, 18, 32, 0.72);
                overflow: hidden;
            }}

            .hero {{
                position: relative;
                border: 0;
                border-radius: 26px;
                padding: 0.95rem 1.45rem 1rem;
                margin-bottom: 0.55rem;
                overflow: hidden;
                background:
                    linear-gradient(135deg, rgba(17, 26, 45, 0.92), rgba(5, 7, 13, 0.84)),
                    radial-gradient(circle at 90% 20%, rgba(255, 184, 77, 0.22), transparent 18rem);
                box-shadow: 0 28px 100px rgba(0, 0, 0, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.06);
            }}

            .hero h1 {{
                max-width: 760px;
                margin: 0 0 0.45rem;
                color: var(--ink);
                font-family: "Newsreader", Georgia, serif;
                font-size: clamp(2rem, 3.3vw, 3.35rem);
                line-height: 0.95;
                letter-spacing: -0.045em;
            }}

            .hero p {{
                max-width: 760px;
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.45;
                margin-bottom: 0;
            }}

            .axon-stage {{
                position: relative;
                border: 1px solid rgba(105, 167, 255, 0.2);
                border-radius: 24px;
                margin: 0 0 0.65rem;
                padding: 0.45rem;
                background:
                    linear-gradient(180deg, rgba(12, 18, 32, 0.76), rgba(5, 7, 13, 0.92)),
                    radial-gradient(circle at 18% 50%, rgba(41, 240, 255, 0.14), transparent 16rem);
                box-shadow: 0 20px 70px rgba(0, 0, 0, 0.32), inset 0 1px 0 rgba(255, 255, 255, 0.04);
                overflow: hidden;
            }}

            .axon-stage::before {{
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(90deg, rgba(41, 240, 255, 0.08), transparent 28%, rgba(255, 184, 77, 0.07));
                pointer-events: none;
            }}

            .axon-stage svg {{
                position: relative;
                display: block;
                width: 100%;
                height: auto;
            }}

            .axon-label {{
                position: absolute;
                right: 1.1rem;
                bottom: 0.9rem;
                color: var(--muted);
                font-family: "IBM Plex Mono", monospace;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}

            .hero-strip {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.65rem;
                margin-top: 1.4rem;
            }}

            .chip {{
                border: 1px solid rgba(143, 161, 186, 0.2);
                border-radius: 999px;
                padding: 0.55rem 0.75rem;
                color: var(--ink);
                background: rgba(255, 255, 255, 0.045);
                font-family: "IBM Plex Mono", monospace;
                font-size: 0.78rem;
            }}

            .status-card,
            .metric-card {{
                border: 1px solid rgba(105, 167, 255, 0.2);
                border-radius: 24px;
                padding: 1rem 1.1rem;
                background: linear-gradient(180deg, rgba(17, 26, 45, 0.78), rgba(8, 12, 22, 0.84));
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
            }}

            .status-card {{
                margin: 1rem 0;
                border-left: 4px solid var(--cyan);
            }}

            .status-card.success {{ border-left-color: var(--green); }}
            .status-card.warning {{ border-left-color: var(--amber); }}
            .status-card.idle {{ border-left-color: var(--cyan); }}

            .status-kicker,
            .metric-label {{
                color: var(--muted);
                font-family: "IBM Plex Mono", monospace;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.09em;
                text-transform: uppercase;
            }}

            .status-title {{
                margin-top: 0.25rem;
                color: var(--ink);
                font-size: 1.05rem;
                font-weight: 700;
            }}

            .metric-value {{
                margin-top: 0.35rem;
                color: var(--ink);
                font-family: "IBM Plex Mono", monospace;
                font-size: 1.45rem;
                font-weight: 700;
            }}

            .metric-value.accent {{ color: var(--cyan); }}
            .metric-value.hot {{ color: var(--rose); }}
            .metric-value.warm {{ color: var(--amber); }}

            .sidebar-panel {{
                border: 1px solid rgba(105, 167, 255, 0.22);
                border-radius: 22px;
                padding: 1rem;
                margin: 0.6rem 0 1rem;
                background: rgba(255, 255, 255, 0.035);
            }}

            .sidebar-title {{
                color: var(--ink);
                font-family: "Newsreader", Georgia, serif;
                font-size: 1.9rem;
                line-height: 1;
                letter-spacing: -0.04em;
            }}

            .sidebar-copy {{
                color: var(--muted);
                font-size: 0.78rem;
                line-height: 1.5;
                margin-top: 0.55rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_neuron_animation(
    slot,
    progress=0.0,
    is_running=False,
    hold_pulse=False,
    node_active=None,
    node_passive=None,
    node_refrac=None,
):
    """Axon schematic. If node_active / node_passive (0–1 per recording site) are
    provided, draws local Na⁺ regeneration (amber) vs upstream axial coupling (blue)
    at cable sample points; node_refrac (0–1) adds a rose ring for Na⁺ inactivation /
    refractoriness. Otherwise falls back to a traveling pulse."""
    progress = float(np.clip(progress, 0.0, 1.0))
    signal_offset = -progress * 930
    signal_opacity = 1.0 if (is_running or hold_pulse) else 0.42
    cyan = THEME["cyan"]
    blue = THEME["blue"]
    amber = THEME["amber"]
    rose = THEME["rose"]
    ink = THEME["ink"]
    muted = THEME["muted"]
    line = THEME["line"]
    panel = THEME["panel"]
    panel_alt = THEME["panel_alt"]

    segment_nodes = ""
    traveling = ""
    if node_active is not None and node_passive is not None and len(node_active) == len(node_passive):
        n_nodes = len(node_active)
        for j in range(n_nodes):
            s = j / max(1, n_nodes - 1)
            nx = 174.0 + 854.0 * s
            ny = 118.0 - 4.0 * float(np.sin(np.pi * s))
            a = float(np.clip(node_active[j], 0.0, 1.0))
            p = float(np.clip(node_passive[j], 0.0, 1.0))
            r_pass = 8.0 + 24.0 * p
            r_act = 5.0 + 14.0 * a
            op_pass = 0.10 + 0.45 * p
            op_act = 0.30 + 0.70 * a
            r_fr = 0.0
            if node_refrac is not None and j < len(node_refrac):
                r_fr = float(np.clip(node_refrac[j], 0.0, 1.0))
            refractory_ring = ""
            if r_fr > 0.05:
                refractory_ring = (
                    f'<circle r="{r_act + 7.0:.1f}" fill="none" stroke="{rose}" '
                    f'stroke-width="{1.2 + 4.0 * r_fr:.1f}" '
                    f'opacity="{0.22 + 0.58 * r_fr:.2f}" />'
                )
            segment_nodes += f"""
<g transform="translate({nx:.1f},{ny:.1f})">
<circle r="{r_pass:.1f}" fill="{blue}" opacity="{op_pass:.2f}" />
<circle r="{r_act:.1f}" fill="{amber}" opacity="{op_act:.2f}" filter="url(#pulse-glow)" />
{refractory_ring}
</g>"""
    else:
        traveling = f"""<g filter="url(#pulse-glow)" opacity="{signal_opacity:.2f}">
<path pathLength="1000" d="M174 118 C270 104 344 108 430 118 S612 133 730 118 S900 102 1028 118" fill="none" stroke="{amber}" stroke-width="20" stroke-linecap="round" stroke-dasharray="46 954" stroke-dashoffset="{signal_offset:.1f}" />
<path pathLength="1000" d="M174 118 C270 104 344 108 430 118 S612 133 730 118 S900 102 1028 118" fill="none" stroke="{amber}" stroke-opacity="0.22" stroke-width="38" stroke-linecap="round" stroke-dasharray="70 930" stroke-dashoffset="{signal_offset:.1f}" />
</g>"""

    html = f"""<div class="axon-stage">
<svg viewBox="0 0 1200 230" role="img" aria-label="Animated neuron: passive axial spread and active Na regeneration along the axon">
<defs>
<filter id="pulse-glow" x="-80%" y="-80%" width="260%" height="260%">
<feGaussianBlur stdDeviation="8" result="blur" />
<feMerge>
<feMergeNode in="blur" />
<feMergeNode in="SourceGraphic" />
</feMerge>
</filter>
<linearGradient id="axon-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
<stop offset="0%" stop-color="{cyan}" stop-opacity="0.95" />
<stop offset="58%" stop-color="{blue}" stop-opacity="0.75" />
<stop offset="100%" stop-color="{cyan}" stop-opacity="0.82" />
</linearGradient>
<radialGradient id="soma-gradient" cx="45%" cy="38%" r="68%">
<stop offset="0%" stop-color="#ffffff" stop-opacity="0.95" />
<stop offset="38%" stop-color="{cyan}" stop-opacity="0.95" />
<stop offset="100%" stop-color="{blue}" stop-opacity="0.48" />
</radialGradient>
</defs>
<circle cx="118" cy="118" r="58" fill="url(#soma-gradient)" opacity="0.78" />
<circle cx="118" cy="118" r="30" fill="{panel}" stroke="{ink}" stroke-opacity="0.5" stroke-width="2" />
<circle cx="118" cy="118" r="73" fill="none" stroke="{cyan}" stroke-opacity="0.18" stroke-width="2" />
<path id="axon-core" pathLength="1000" d="M174 118 C270 104 344 108 430 118 S612 133 730 118 S900 102 1028 118" fill="none" stroke="{line}" stroke-width="44" stroke-linecap="round" opacity="0.92" />
<path pathLength="1000" d="M174 118 C270 104 344 108 430 118 S612 133 730 118 S900 102 1028 118" fill="none" stroke="url(#axon-gradient)" stroke-width="28" stroke-linecap="round" opacity="0.9" />
<path pathLength="1000" d="M174 118 C270 104 344 108 430 118 S612 133 730 118 S900 102 1028 118" fill="none" stroke="{cyan}" stroke-width="5" stroke-linecap="round" stroke-dasharray="9 19" opacity="0.28" />
<g opacity="0.96">
<circle cx="1048" cy="118" r="38" fill="{line}" opacity="0.96" />
<circle cx="1048" cy="118" r="29" fill="{panel_alt}" stroke="{cyan}" stroke-opacity="0.74" stroke-width="5" />
<circle cx="1048" cy="118" r="11" fill="{blue}" opacity="0.58" />
</g>
{segment_nodes}
{traveling}
<text x="182" y="199" fill="{muted}" font-family="IBM Plex Mono, monospace" font-size="13">soma</text>
<text x="1010" y="201" fill="{muted}" font-family="IBM Plex Mono, monospace" font-size="13">axon terminal</text>
</svg>
</div>"""
    slot.markdown(html, unsafe_allow_html=True)


inject_global_css()

params_col, main_col = st.columns([0.26, 0.74], gap="medium")

# ── Parameters ────────────────────────────────────────────────────────────────
with params_col:
    with st.container(border=True):
        st.markdown('<div class="param-section">Parameters</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="param-note">Tune the current clamp and display options.</div>',
            unsafe_allow_html=True,
        )

        i_inj = st.slider(
            "Current density (µA/cm²)",
            min_value=-20.0,
            max_value=50.0,
            value=0.0,
            step=0.1,
        )
        stim_type = st.selectbox("Protocol", ["Single pulse", "Pulse train", "Step current"])
        stim_type_val = {"Single pulse": "pulse", "Pulse train": "train", "Step current": "step"}[stim_type]

        timing_cols = st.columns(2)
        with timing_cols[0]:
            stim_on = st.number_input("Onset", min_value=0.0, value=5.0, step=0.5, format="%.1f")
        with timing_cols[1]:
            stim_dur = st.number_input("Duration", min_value=0.1, value=1.0, step=0.1, format="%.1f")

        train_freq, train_n = 50, 5
        if stim_type_val == "train":
            train_cols = st.columns(2)
            with train_cols[0]:
                train_freq = st.number_input("Hz", min_value=1, value=50)
            with train_cols[1]:
                train_n = st.number_input("Pulses", min_value=1, value=5)

        win_ms = st.slider("Window (ms)", min_value=20, max_value=300, value=100, step=10)
        anim_speed = st.slider("Speed", 1, 10, 5)
        propagation = st.checkbox("Cable mode", value=False)
        stacked_vm = st.checkbox(
            "Stacked V_m (textbook)",
            value=False,
            disabled=not propagation,
            help="Subplot per recording site—passive depolarization and delayed spike line up like lecture slides.",
        )
        show_refs = st.checkbox("References", value=False)

        for k, v in (
            ("anim_active", False),
            ("anim_paused", False),
            ("anim_end", 0),
            ("anim_n_pts", 0),
            ("anim_step", 1),
            ("anim_stride", 1),
            ("anim_delay_s", 0.04),
            ("anim_sig", ""),
        ):
            st.session_state.setdefault(k, v)

        anim_sig = "|".join(
            str(x)
            for x in (
                i_inj,
                stim_on,
                stim_dur,
                stim_type_val,
                train_freq,
                train_n,
                win_ms,
                anim_speed,
                propagation,
                stacked_vm,
                show_refs,
            )
        )
        if st.session_state["anim_sig"] != anim_sig:
            st.session_state["anim_active"] = False
            st.session_state["anim_paused"] = False
            st.session_state["anim_end"] = 0
            st.session_state["anim_sig"] = anim_sig

        action_cols = st.columns(3)
        with action_cols[0]:
            run_btn = st.button("Run", width="stretch", type="primary")
        with action_cols[1]:
            pause_btn = st.button(
                "Resume" if st.session_state["anim_paused"] else "Pause",
                width="stretch",
                disabled=not st.session_state["anim_active"],
            )
        with action_cols[2]:
            if st.button("Reset", width="stretch"):
                st.session_state.clear()
                st.rerun()

        if run_btn:
            st.session_state["anim_active"] = True
            st.session_state["anim_paused"] = False
            st.session_state["anim_end"] = 0
            st.session_state["anim_n_pts"] = 0

        if pause_btn and st.session_state["anim_active"]:
            st.session_state["anim_paused"] = not st.session_state["anim_paused"]

with main_col:
    st.markdown(
        """
        <section class="hero">
            <h1>Impulse Propagation Simulator</h1>
            <p>
                Explore voltage-dependent Na⁺ and K⁺ currents (Chapter 3), threshold and regeneration,
                and—in cable mode—how passive spread along the axon recruits the next segment’s active
                response so the impulse stays full-sized.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    neuron_slot = st.empty()

# ── Simulation — auto-reruns on every parameter change ───────────────────────
@st.cache_data
def cached_sim(i_inj, stim_on, stim_dur, stim_type_val, train_freq, train_n, win_ms):
    return run_simulation(
        i_inj=i_inj, stim_on=stim_on, stim_dur=stim_dur,
        stim_type=stim_type_val, train_freq=train_freq,
        train_n=train_n, win_ms=win_ms,
    )

@st.cache_data
def cached_propagation(i_inj, stim_on, stim_dur, stim_type_val, train_freq, train_n, win_ms):
    return run_propagation(
        i_inj=i_inj, stim_on=stim_on, stim_dur=stim_dur,
        stim_type=stim_type_val, train_freq=train_freq,
        train_n=train_n, win_ms=win_ms,
    )

t, V, m, h, n, iNa, iK, iL, eNa, eK, eL = cached_sim(
    i_inj, stim_on, stim_dur, stim_type_val, train_freq, train_n, win_ms
)

I_stim = np.array([
    get_i_inj(ti, i_inj, stim_on, stim_dur, stim_type_val, train_freq, train_n)
    for ti in t
])
stim_mask = I_stim != 0

# ── Helpers ───────────────────────────────────────────────────────────────────
def _stim_intervals(mask, t):
    starts, ends, in_stim = [], [], False
    for i, active in enumerate(mask):
        if active and not in_stim:
            starts.append(t[i]); in_stim = True
        elif not active and in_stim:
            ends.append(t[i]); in_stim = False
    if in_stim:
        ends.append(t[-1])
    return starts, ends

def _xtick_step(win_ms):
    return 5 if win_ms <= 50 else 10 if win_ms <= 200 else 25

# ── Colour palette for propagation traces ─────────────────────────────────────
_PROP_COLORS = [THEME["cyan"], THEME["amber"], THEME["green"], THEME["rose"], THEME["blue"]]

# ── Plotly figure (single compartment) ───────────────────────────────────────
def build_figure(t_s, V_s, iNa_s, iK_s, iL_s, mask_s, eNa, eK, win_ms, show_refs):
    voltage_fig = go.Figure()
    current_fig = go.Figure()
    _add_stim_shading(voltage_fig, mask_s, t_s)
    _add_stim_shading(current_fig, mask_s, t_s)

    voltage_fig.add_trace(go.Scatter(
        x=t_s, y=V_s, mode="lines", name="V_m",
        line=dict(color=THEME["cyan"], width=3),
        hovertemplate="<b>t = %{x:.2f} ms</b><br>V_m = %{y:.2f} mV<extra></extra>",
    ))

    if show_refs:
        for y_val, color, label in [
            (eNa,    THEME["rose"],  f"E_Na ({eNa:.0f} mV)"),
            (eK,     THEME["blue"],  f"E_K ({eK:.0f} mV)"),
            (V_REST, THEME["muted"], f"Rest ({V_REST} mV)"),
            (-55,    THEME["amber"], "Threshold (−55 mV)"),
        ]:
            voltage_fig.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1,
                                  opacity=0.6, annotation_text=label,
                                  annotation_position="right",
                                  annotation_font=dict(size=8, color=color))

    # Chapter 3: early inward Na⁺ vs delayed outward K⁺ (HH ionic currents)
    current_fig.add_trace(go.Scatter(
        x=t_s, y=-iNa_s, mode="lines", name="−I_Na",
        line=dict(color=THEME["rose"], width=2.4),
        hovertemplate="<b>t = %{x:.2f} ms</b><br>−I_Na (inward) = %{y:.2f} µA/cm²<extra></extra>",
    ))
    current_fig.add_trace(go.Scatter(
        x=t_s, y=iK_s, mode="lines", name="I_K",
        line=dict(color=THEME["blue"], width=2.2),
        hovertemplate="<b>t = %{x:.2f} ms</b><br>I_K = %{y:.2f} µA/cm²<extra></extra>",
    ))
    current_fig.add_trace(go.Scatter(
        x=t_s, y=iL_s, mode="lines", name="I_L",
        line=dict(color=THEME["green"], width=1.4, dash="dot"),
        hovertemplate="<b>t = %{x:.2f} ms</b><br>I_L = %{y:.2f} µA/cm²<extra></extra>",
    ))

    current_fig.add_hline(y=0, line_color=THEME["line"], line_width=1)

    _apply_layout(voltage_fig, win_ms, "Voltage", "Vm (mV)")
    _apply_layout(current_fig, win_ms, "Ionic current", "µA/cm²")
    voltage_fig.update_yaxes(range=[-100, 80])
    return voltage_fig, current_fig


# ── Plotly figure (propagation — voltage + currents) ─────────────────────────
def build_propagation_voltage_stack(t_s, V_traces_s, positions_um, mask_s, eNa, eK, win_ms, show_refs):
    n = len(V_traces_s)
    titles = tuple(f"V_m @ {p} µm" for p in positions_um)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.034,
        subplot_titles=titles,
    )
    for i, (V_s, pos_um) in enumerate(zip(V_traces_s, positions_um)):
        row = i + 1
        for s, e in zip(*_stim_intervals(mask_s, t_s)):
            fig.add_vrect(
                x0=s, x1=e, fillcolor=THEME["amber"], opacity=0.14,
                layer="below", line_width=0, row=row, col=1,
            )
        c = _PROP_COLORS[i]
        fig.add_trace(
            go.Scatter(
                x=t_s, y=V_s, mode="lines",
                line=dict(color=c, width=2.1),
                hovertemplate=(
                    f"<b>{pos_um} µm</b><br>t = %{{x:.2f}} ms<br>"
                    "V_m = %{y:.2f} mV<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row, col=1,
        )
        if show_refs:
            for y_val, color in [
                (eNa, THEME["rose"]),
                (eK, THEME["blue"]),
                (V_REST, THEME["muted"]),
                (-55, THEME["amber"]),
            ]:
                fig.add_hline(
                    y=y_val, line_dash="dash", line_color=color, line_width=1,
                    opacity=0.52, row=row, col=1,
                )

    tick_step = _xtick_step(win_ms)
    xticks = list(np.arange(0, win_ms + tick_step, tick_step))
    row_h = min(124, max(88, 540 // n))
    fig.update_layout(
        plot_bgcolor="rgba(5, 7, 13, 0)",
        paper_bgcolor="rgba(5, 7, 13, 0)",
        hovermode="x unified",
        height=72 + n * row_h,
        margin=dict(l=72, r=26, t=52, b=44),
        title=dict(text="<b>Voltage (stacked)</b>", x=0.5, xanchor="center"),
        font=dict(color=THEME["ink"], size=10, family="IBM Plex Mono, monospace"),
        title_font=dict(color=THEME["ink"]),
        showlegend=False,
        hoverlabel=dict(
            bgcolor=THEME["panel"],
            bordercolor=THEME["cyan"],
            font=dict(color=THEME["ink"], family="IBM Plex Mono, monospace"),
        ),
    )
    fig.update_annotations(font_size=10, font_color=THEME["muted"])
    for r in range(1, n + 1):
        fig.update_yaxes(
            range=[-100, 80], row=r, col=1,
            showgrid=True, gridcolor="rgba(105, 167, 255, 0.08)",
            showline=True, linecolor=THEME["line"], tickcolor=THEME["line"],
            ticks="outside", mirror=False,
        )
        fig.update_xaxes(
            range=[0, win_ms], tickvals=xticks,
            showgrid=True, gridcolor="rgba(105, 167, 255, 0.08)",
            showline=True, linecolor=THEME["line"], tickcolor=THEME["line"],
            ticks="outside", row=r, col=1,
        )
    for r in range(1, n):
        fig.update_xaxes(showticklabels=False, row=r, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=n, col=1)
    fig.add_annotation(
        text="Vm (mV)",
        xref="paper",
        yref="paper",
        x=-0.12,
        y=0.5,
        textangle=-90,
        showarrow=False,
        font=dict(color=THEME["ink"], size=12, family="IBM Plex Mono, monospace"),
    )
    return fig


def build_propagation_figure(
    t_s,
    V_traces_s,
    iNa_traces_s,
    i_pas_traces_s,
    positions_um,
    mask_s,
    eNa,
    eK,
    win_ms,
    show_refs,
    stacked_voltage=False,
):
    if stacked_voltage:
        voltage_fig = build_propagation_voltage_stack(
            t_s, V_traces_s, positions_um, mask_s, eNa, eK, win_ms, show_refs
        )
    else:
        voltage_fig = go.Figure()
        _add_stim_shading(voltage_fig, mask_s, t_s)
        for i, (V_s, pos_um) in enumerate(zip(V_traces_s, positions_um)):
            label = f"{pos_um} µm"
            voltage_fig.add_trace(go.Scatter(
                x=t_s, y=V_s, mode="lines", name=label,
                line=dict(color=_PROP_COLORS[i], width=2),
                hovertemplate=f"<b>{label}</b><br>t = %{{x:.2f}} ms<br>V_m = %{{y:.2f}} mV<extra></extra>",
            ))

        if show_refs:
            for y_val, color, label in [
                (eNa,    THEME["rose"],  f"E_Na ({eNa:.0f} mV)"),
                (eK,     THEME["blue"],  f"E_K ({eK:.0f} mV)"),
                (V_REST, THEME["muted"], f"Rest ({V_REST} mV)"),
                (-55,    THEME["amber"], "Threshold (−55 mV)"),
            ]:
                voltage_fig.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1,
                                      opacity=0.6, annotation_text=label,
                                      annotation_position="right",
                                      annotation_font=dict(size=8, color=color))

        _apply_layout(voltage_fig, win_ms, "Voltage", "Vm (mV)")
        voltage_fig.update_yaxes(range=[-100, 80])

    current_fig = go.Figure()
    _add_stim_shading(current_fig, mask_s, t_s)

    # Solid = local inward Na⁺ (regenerative / active); dotted = upstream axial coupling (passive spread)
    for i, (iNa_loc, i_pas, pos_um) in enumerate(
        zip(iNa_traces_s, i_pas_traces_s, positions_um)
    ):
        c = _PROP_COLORS[i]
        loc_lbl = f"{pos_um} µm"
        current_fig.add_trace(go.Scatter(
            x=t_s, y=-iNa_loc, mode="lines",
            line=dict(color=c, width=2.2),
            hovertemplate=(
                f"<b>{loc_lbl} · Na⁺ active (−I_Na)</b><br>"
                "t = %{x:.2f} ms<br>%{y:.2f} µA/cm²<extra></extra>"
            ),
        ))
        current_fig.add_trace(go.Scatter(
            x=t_s, y=i_pas, mode="lines",
            line=dict(color=c, width=2.0, dash="dot"),
            hovertemplate=(
                f"<b>{loc_lbl} · passive axial</b><br>"
                "t = %{x:.2f} ms<br>%{y:.2f} µA/cm²<extra></extra>"
            ),
        ))
    current_fig.add_hline(y=0, line_color=THEME["line"], line_width=1)

    _apply_layout(current_fig, win_ms, "Active vs passive current", "µA/cm²")
    return voltage_fig, current_fig


def _add_stim_shading(fig, mask_s, t_s):
    for s, e in zip(*_stim_intervals(mask_s, t_s)):
        fig.add_vrect(x0=s, x1=e, fillcolor=THEME["amber"], opacity=0.16,
                      layer="below", line_width=0)


def _apply_layout(fig, win_ms, title, y_title):
    tick_step = _xtick_step(win_ms)
    xticks    = list(np.arange(0, win_ms + tick_step, tick_step))
    fig.update_layout(
        plot_bgcolor="rgba(5, 7, 13, 0)",
        paper_bgcolor="rgba(5, 7, 13, 0)",
        hovermode="x unified", height=360,
        margin=dict(l=72, r=38, t=48, b=60),
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center"),
        font=dict(color=THEME["ink"], size=11, family="IBM Plex Mono, monospace"),
        title_font=dict(color=THEME["ink"]),
        showlegend=False,
        hoverlabel=dict(
            bgcolor=THEME["panel"],
            bordercolor=THEME["cyan"],
            font=dict(color=THEME["ink"], family="IBM Plex Mono, monospace"),
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(105, 167, 255, 0.08)",
                     showline=True, linecolor=THEME["line"],
                     tickcolor=THEME["line"], ticks="outside", tickvals=xticks,
                     title_text="Time (ms)",
                     range=[0, win_ms], mirror=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(105, 167, 255, 0.08)",
                     showline=True, linecolor=THEME["line"],
                     tickcolor=THEME["line"], ticks="outside", mirror=False,
                     title_text="")
    fig.add_annotation(
        text=y_title,
        xref="paper",
        yref="paper",
        x=-0.16,
        y=0.5,
        textangle=-90,
        showarrow=False,
        font=dict(color=THEME["ink"], size=12, family="IBM Plex Mono, monospace"),
    )


def _propagation_animation_nodes(iNa_traces, i_pas_traces, t_idx):
    """Normalize recorded traces for axon-node highlights (active vs passive)."""
    act_stack = np.stack([-np.asarray(tr, dtype=float) for tr in iNa_traces], axis=0)
    pas_stack = np.maximum(0.0, np.stack([np.asarray(tr, dtype=float) for tr in i_pas_traces], axis=0))
    act_scale = float(np.percentile(act_stack, 99)) if act_stack.size else 0.0
    act_scale = max(act_scale, 1e-6)
    pas_scale = float(np.percentile(pas_stack, 99)) if pas_stack.size else 0.0
    pas_scale = max(pas_scale, 1e-6)
    node_act, node_pas = [], []
    for j in range(len(iNa_traces)):
        node_act.append(float(np.clip(-iNa_traces[j][t_idx] / act_scale, 0.0, 1.0)))
        node_pas.append(float(np.clip(max(0.0, i_pas_traces[j][t_idx]) / pas_scale, 0.0, 1.0)))
    return node_act, node_pas


def _propagation_refrac_nodes(h_traces, t_idx):
    """Refractory cue from Na⁺ inactivation h (Chapter 3: limits re-excitation)."""
    h_rest = max(float(h_inf(V_REST)), 1e-6)
    return [
        float(np.clip(1.0 - float(tr[t_idx]) / h_rest, 0.0, 1.0))
        for tr in h_traces
    ]


def _advance_sweep_animation(n_pts, anim_speed):
    """Step the sweep once per rerun; supports pause via session_state."""
    ss = st.session_state
    step = max(1, n_pts // 200)
    stride = step * max(1, anim_speed // 3)
    delay = 0.04 / (anim_speed / 5)

    cfg_changed = (
        ss["anim_n_pts"] != n_pts
        or ss["anim_step"] != step
        or abs(ss["anim_delay_s"] - delay) > 1e-9
    )
    if cfg_changed:
        ss["anim_n_pts"] = n_pts
        ss["anim_step"] = step
        ss["anim_stride"] = stride
        ss["anim_delay_s"] = delay
        if ss["anim_active"] and ss["anim_end"] == 0:
            ss["anim_end"] = min(stride, n_pts)
        else:
            ss["anim_end"] = min(ss["anim_end"], n_pts)
        if not ss["anim_active"]:
            return

    if not ss["anim_active"] or ss["anim_paused"]:
        return

    end = min(ss["anim_end"], n_pts)
    if end >= n_pts:
        ss["anim_active"] = False
        return

    next_end = min(end + stride, n_pts)
    ss["anim_end"] = next_end
    time.sleep(delay)
    st.rerun()


# ── Render ────────────────────────────────────────────────────────────────────
with main_col:
    voltage_col, current_col = st.columns(2)
    voltage_slot = voltage_col.empty()
    current_slot = current_col.empty()

if propagation:
    with st.spinner("Running cable model…"):
        t_p, V_traces, positions_um, iNa_tr_p, i_pas_tr_p, h_tr_p, eNa_p, eK_p, _eLp = cached_propagation(
            i_inj, stim_on, stim_dur, stim_type_val, train_freq, train_n, win_ms
        )

    I_stim_p = np.array([
        get_i_inj(ti, i_inj, stim_on, stim_dur, stim_type_val, train_freq, train_n)
        for ti in t_p
    ])
    mask_p = I_stim_p != 0

    n_pts = len(t_p)
    _advance_sweep_animation(n_pts, anim_speed)

    voltage_fig, current_fig = build_propagation_figure(
        t_p, V_traces, iNa_tr_p, i_pas_tr_p, positions_um, mask_p,
        eNa_p, eK_p, win_ms, show_refs, stacked_voltage=stacked_vm,
    )
    voltage_slot.plotly_chart(voltage_fig, width="stretch")
    current_slot.plotly_chart(current_fig, width="stretch")

    prog = st.session_state["anim_end"] / n_pts if n_pts else 0.0
    active = st.session_state["anim_active"]
    paused = st.session_state["anim_paused"]
    t_idx = min(max(st.session_state["anim_end"], 0), max(0, n_pts - 1))
    na_nodes, p_nodes = _propagation_animation_nodes(iNa_tr_p, i_pas_tr_p, t_idx)
    refrac_nodes = _propagation_refrac_nodes(h_tr_p, t_idx)
    render_neuron_animation(
        neuron_slot,
        progress=prog,
        is_running=active and not paused,
        hold_pulse=active and paused,
        node_active=na_nodes,
        node_passive=p_nodes,
        node_refrac=refrac_nodes,
    )

else:
    n_pts = len(t)
    _advance_sweep_animation(n_pts, anim_speed)

    voltage_fig, current_fig = build_figure(t, V, iNa, iK, iL, stim_mask, eNa, eK, win_ms, show_refs)
    voltage_slot.plotly_chart(voltage_fig, width="stretch")
    current_slot.plotly_chart(current_fig, width="stretch")

    prog = st.session_state["anim_end"] / n_pts if n_pts else 0.0
    active = st.session_state["anim_active"]
    paused = st.session_state["anim_paused"]
    render_neuron_animation(
        neuron_slot,
        progress=prog,
        is_running=active and not paused,
        hold_pulse=active and paused,
    )
