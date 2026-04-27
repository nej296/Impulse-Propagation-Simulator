"""Advanced Hodgkin-Huxley Neuron Simulator — Streamlit dashboard."""

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

from hodgkin_huxley import (
    G_K,
    G_L,
    G_NA,
    V_REST,
    get_i_inj,
    run_propagation,
    run_simulation,
)

st.set_page_config(
    page_title="Impulse Propagation Studio",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 1.0rem;}
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 0.96rem;
        }
        .app-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            letter-spacing: -0.01em;
        }
        .subtitle {color: #4a5568; margin-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">⚡ Impulse Propagation Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Scientifically grounded Hodgkin–Huxley exploration with real-time electrophysiology views.</div>',
    unsafe_allow_html=True,
)

if "playing" not in st.session_state:
    st.session_state.playing = False

# Sidebar
with st.sidebar:
    st.header("Experiment setup")
    i_inj = st.slider("Injected current density (µA/cm²)", -20.0, 100.0, 10.0, 0.5)
    stim_type = st.selectbox("Stimulus waveform", ["Single pulse", "Pulse train", "Step current"])
    stim_type_val = {"Single pulse": "pulse", "Pulse train": "train", "Step current": "step"}[stim_type]

    c1, c2 = st.columns(2)
    with c1:
        stim_on = st.number_input("Onset (ms)", min_value=0.0, value=5.0, step=0.5)
    with c2:
        stim_dur = st.number_input("Duration (ms)", min_value=0.1, value=1.0, step=0.1)

    train_freq, train_n = 60, 5
    if stim_type_val == "train":
        c3, c4 = st.columns(2)
        with c3:
            train_freq = st.number_input("Train frequency (Hz)", min_value=1, value=60)
        with c4:
            train_n = st.number_input("Pulses", min_value=1, value=5)

    st.divider()
    st.subheader("Numerics")
    win_ms = st.slider("Simulation window (ms)", 20, 200, 80, 10)
    dt = st.select_slider("Time step dt (ms)", options=[0.005, 0.01, 0.02, 0.05], value=0.01)
    live_speed = st.slider("Playback speed", 1, 12, 6)
    show_refs = st.checkbox("Show reversal/reference lines", value=True)

    st.divider()
    propagation = st.toggle("Cable propagation view", value=False)
    n_comp = st.slider("Cable compartments", 10, 80, 40, 5, disabled=not propagation)
    g_coupling = st.slider("Axial coupling", 1.0, 12.0, 5.0, 0.5, disabled=not propagation)

    st.divider()
    play = st.button("▶ Play timeline", use_container_width=True, type="primary")
    if st.button("⏹ Stop", use_container_width=True):
        st.session_state.playing = False

    if play:
        st.session_state.playing = True


@st.cache_data(show_spinner=False)
def cached_simulation(i_inj, stim_on, stim_dur, stim_type_val, train_freq, train_n, win_ms, dt):
    return run_simulation(
        i_inj=i_inj,
        stim_on=stim_on,
        stim_dur=stim_dur,
        stim_type=stim_type_val,
        train_freq=train_freq,
        train_n=train_n,
        win_ms=win_ms,
        dt=dt,
        method="rk4",
    )


@st.cache_data(show_spinner=False)
def cached_propagation(i_inj, stim_on, stim_dur, stim_type_val, train_freq, train_n, win_ms, n_comp, g_coupling):
    return run_propagation(
        i_inj=i_inj,
        stim_on=stim_on,
        stim_dur=stim_dur,
        stim_type=stim_type_val,
        train_freq=train_freq,
        train_n=train_n,
        win_ms=win_ms,
        n_comp=n_comp,
        g_coupling=g_coupling,
    )


def compute_stimulus(t):
    return np.array(
        [get_i_inj(ti, i_inj, stim_on, stim_dur, stim_type_val, train_freq, train_n) for ti in t]
    )


def stimulus_intervals(mask, t):
    starts, ends = [], []
    active = False
    for i, m in enumerate(mask):
        if m and not active:
            starts.append(t[i])
            active = True
        elif not m and active:
            ends.append(t[i])
            active = False
    if active:
        ends.append(t[-1])
    return starts, ends


def add_stimulus_background(fig, t, mask, rows=(1, 2, 3)):
    starts, ends = stimulus_intervals(mask, t)
    for s, e in zip(starts, ends):
        for r in rows:
            fig.add_vrect(
                x0=s,
                x1=e,
                fillcolor="#FFD166",
                opacity=0.18,
                layer="below",
                line_width=0,
                row=r,
                col=1,
            )


def add_reference_lines(fig, eNa, eK, eL):
    if not show_refs:
        return
    refs = [
        (eNa, "#e63946", f"E_Na {eNa:.1f} mV"),
        (eK, "#1d4ed8", f"E_K {eK:.1f} mV"),
        (eL, "#64748b", f"E_L {eL:.1f} mV"),
        (V_REST, "#6b7280", f"Rest {V_REST} mV"),
        (-55, "#f59e0b", "Threshold -55 mV"),
    ]
    for y, c, txt in refs:
        fig.add_hline(
            y=y,
            line_dash="dot",
            line_color=c,
            line_width=1,
            opacity=0.65,
            row=1,
            col=1,
            annotation_text=txt,
            annotation_position="right",
            annotation_font=dict(size=9, color=c),
        )


def make_single_compartment_figure(t, V, iNa, iK, iL, Iinj, eNa, eK, eL):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.45, 0.3, 0.25],
        subplot_titles=("Membrane voltage", "Ionic current decomposition", "Input stimulus"),
    )

    mask = Iinj != 0
    add_stimulus_background(fig, t, mask)

    fig.add_trace(go.Scatter(x=t, y=V, mode="lines", name="V_m", line=dict(color="#111827", width=2.3)), row=1, col=1)
    add_reference_lines(fig, eNa, eK, eL)

    fig.add_trace(go.Scatter(x=t, y=iNa, mode="lines", name="I_Na", line=dict(color="#ef4444", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=iK, mode="lines", name="I_K", line=dict(color="#2563eb", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=iL, mode="lines", name="I_L", line=dict(color="#64748b", width=1.8)), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=t, y=iNa + iK + iL, mode="lines", name="I_total", line=dict(color="#111827", width=2.2, dash="dash")),
        row=2,
        col=1,
    )

    fig.add_trace(go.Scatter(x=t, y=Iinj, mode="lines", name="I_inj", line=dict(color="#7c3aed", width=2)), row=3, col=1)

    fig.update_layout(
        hovermode="x unified",
        height=830,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=35, t=80, b=35),
    )
    fig.update_xaxes(title="Time (ms)", row=3, col=1)
    fig.update_yaxes(title="Voltage (mV)", row=1, col=1)
    fig.update_yaxes(title="Current (µA/cm²)", row=2, col=1)
    fig.update_yaxes(title="Current (µA/cm²)", row=3, col=1)
    return fig


def make_propagation_figure(t, V_traces, positions_um, Iinj):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.72, 0.28],
        subplot_titles=("Action potential propagation along cable", "Injected current at x=0"),
    )

    palette = ["#0f172a", "#2563eb", "#0f766e", "#b45309", "#dc2626"]
    for i, (v, pos) in enumerate(zip(V_traces, positions_um)):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=v,
                mode="lines",
                line=dict(width=2.2, color=palette[i % len(palette)]),
                name=f"{pos} µm",
            ),
            row=1,
            col=1,
        )

    mask = Iinj != 0
    add_stimulus_background(fig, t, mask, rows=(1, 2))
    fig.add_trace(go.Scatter(x=t, y=Iinj, mode="lines", name="I_inj", line=dict(color="#7c3aed", width=2)), row=2, col=1)

    fig.update_layout(
        hovermode="x unified",
        height=700,
        margin=dict(l=45, r=25, t=70, b=35),
    )
    fig.update_xaxes(title="Time (ms)", row=2, col=1)
    fig.update_yaxes(title="Voltage (mV)", row=1, col=1)
    fig.update_yaxes(title="Current (µA/cm²)", row=2, col=1)
    return fig


run_data = cached_simulation(i_inj, stim_on, stim_dur, stim_type_val, train_freq, train_n, win_ms, dt)
t, V, m, h, n, iNa, iK, iL, eNa, eK, eL = run_data
Iinj = compute_stimulus(t)

ap_peaks, _ = find_peaks(V, height=0, distance=max(1, int(2 / dt)))
firing_rate_hz = len(ap_peaks) / (win_ms / 1000)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Peak voltage", f"{np.max(V):.1f} mV")
kpi2.metric("AP count", f"{len(ap_peaks)}")
kpi3.metric("Estimated firing rate", f"{firing_rate_hz:.1f} Hz")
kpi4.metric("Min voltage", f"{np.min(V):.1f} mV")

if len(ap_peaks) > 0:
    st.success("Suprathreshold response detected. Real-time traces and ionic breakdown are synchronized to a fixed dt grid.")
else:
    st.info("Subthreshold response. Increase amplitude or pulse duration to elicit spikes.")

tab1, tab2, tab3 = st.tabs(["📈 Electrophysiology", "🧬 Gates & Conductance", "🧵 Cable Propagation"])

with tab1:
    fig = make_single_compartment_figure(t, V, iNa, iK, iL, Iinj, eNa, eK, eL)
    slot = st.empty()

    if st.session_state.playing:
        n_pts = len(t)
        stride = max(1, n_pts // 220)
        for end in range(stride, n_pts + stride, stride * max(1, live_speed // 2)):
            if not st.session_state.playing:
                break
            j = min(end, n_pts)
            partial = make_single_compartment_figure(
                t[:j], V[:j], iNa[:j], iK[:j], iL[:j], Iinj[:j], eNa, eK, eL
            )
            slot.plotly_chart(partial, use_container_width=True)
            time.sleep(0.025 / (live_speed / 6))
        st.session_state.playing = False
    else:
        slot.plotly_chart(fig, use_container_width=True)

with tab2:
    gNa = G_NA * (m ** 3) * h
    gK = G_K * (n ** 4)

    fig_gates = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                              subplot_titles=("Gate variables", "Effective conductances"))
    fig_gates.add_trace(go.Scatter(x=t, y=m, mode="lines", name="m", line=dict(color="#ef4444", width=2)), row=1, col=1)
    fig_gates.add_trace(go.Scatter(x=t, y=h, mode="lines", name="h", line=dict(color="#14b8a6", width=2)), row=1, col=1)
    fig_gates.add_trace(go.Scatter(x=t, y=n, mode="lines", name="n", line=dict(color="#2563eb", width=2)), row=1, col=1)

    fig_gates.add_trace(go.Scatter(x=t, y=gNa, mode="lines", name="g_Na(t)", line=dict(color="#dc2626", width=2.1)), row=2, col=1)
    fig_gates.add_trace(go.Scatter(x=t, y=gK, mode="lines", name="g_K(t)", line=dict(color="#1d4ed8", width=2.1)), row=2, col=1)
    fig_gates.add_trace(go.Scatter(x=t, y=np.full_like(t, G_L), mode="lines", name="g_L", line=dict(color="#6b7280", width=1.8, dash="dot")), row=2, col=1)

    fig_gates.update_layout(height=700, hovermode="x unified", margin=dict(l=45, r=25, t=70, b=35))
    fig_gates.update_xaxes(title="Time (ms)", row=2, col=1)
    fig_gates.update_yaxes(title="Gate value", range=[0, 1.05], row=1, col=1)
    fig_gates.update_yaxes(title="Conductance (mS/cm²)", row=2, col=1)
    st.plotly_chart(fig_gates, use_container_width=True)

with tab3:
    if not propagation:
        st.info("Enable **Cable propagation view** in the sidebar to compute multi-compartment dynamics.")
    else:
        with st.spinner("Computing cable propagation..."):
            tp, traces, positions_um, eNa_p, eK_p, eL_p = cached_propagation(
                i_inj,
                stim_on,
                stim_dur,
                stim_type_val,
                train_freq,
                train_n,
                win_ms,
                n_comp,
                g_coupling,
            )

        Iinj_p = compute_stimulus(tp)
        fig_prop = make_propagation_figure(tp, traces, positions_um, Iinj_p)
        st.plotly_chart(fig_prop, use_container_width=True)

        peaks = [tp[np.argmax(v)] for v in traces]
        st.caption(
            "Propagation timing (peak arrival): "
            + ", ".join([f"{pos} µm → {tpk:.2f} ms" for pos, tpk in zip(positions_um, peaks)])
        )

with st.expander("Scientific notes"):
    st.markdown(
        """
- The simulator uses classic Hodgkin–Huxley channel kinetics with conductance-based ionic currents.
- Single-compartment traces use a fixed-step RK4 integrator so voltage and current samples are synchronized in time.
- Sign convention: inward ionic current is negative; outward current is positive.
- Conductance panel reports effective channel conductance: \(g_{Na}(t)=\bar{g}_{Na}m^3h\), \(g_K(t)=\bar{g}_Kn^4\).
"""
    )
