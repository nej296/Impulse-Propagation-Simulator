"""
Hodgkin-Huxley Neuron Simulator — Interactive Streamlit App
"""

import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from hodgkin_huxley import run_simulation, run_propagation, get_i_inj, V_REST

st.set_page_config(
    page_title="Hodgkin-Huxley Neuron Simulator",
    page_icon="🧠",
    layout="wide",
)

st.title("Hodgkin-Huxley Neuron Simulator")
st.caption("Computational model of squid axon — Hodgkin & Huxley 1952")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Current Clamp")

st.sidebar.subheader("Stimulus")
i_inj    = st.sidebar.number_input("I_inj (A)",     min_value=-20.0, max_value=50.0,
                                    value=0.0, step=0.1, format="%.1f")
stim_on  = st.sidebar.number_input("Onset (ms)",    min_value=0.0,  value=5.0,  step=0.5)
stim_dur = st.sidebar.number_input("Duration (ms)", min_value=0.1,  value=1.0,  step=0.1)
stim_type = st.sidebar.selectbox("Type", ["Single pulse", "Pulse train", "Step current"])
stim_type_val = {"Single pulse": "pulse", "Pulse train": "train", "Step current": "step"}[stim_type]

train_freq, train_n = 50, 5
if stim_type_val == "train":
    train_freq = st.sidebar.number_input("Freq (Hz)", min_value=1, value=50)
    train_n    = st.sidebar.number_input("Pulses",    min_value=1, value=5)

st.sidebar.subheader("Simulation")
win_ms      = st.sidebar.number_input("Window (ms)", min_value=20, value=100, step=10)
propagation = st.sidebar.checkbox("Show AP Propagation along axon", value=False)
anim_speed  = st.sidebar.slider("Animation speed", 1, 10, 5)
show_refs   = st.sidebar.checkbox("Show E_Na / E_K / threshold", value=False)

st.sidebar.info("Time step: 0.01 ms (RK45)")
run_btn = st.sidebar.button("▶  Run & Animate", use_container_width=True, type="primary")

if st.sidebar.button("Reset to Defaults"):
    st.session_state.clear()
    st.rerun()

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

ap_peaks, _ = find_peaks(V, height=0, distance=50)
ap_fired    = len(ap_peaks) > 0

# ── AP status banner ──────────────────────────────────────────────────────────
if i_inj == 0:
    st.info("Set I_inj and the graph updates instantly. Use **▶ Run & Animate** for real-time playback.")
elif ap_fired:
    if propagation:
        st.success(f"Action potential fired — propagating along axon. Peak: {V[ap_peaks[0]]:.1f} mV")
    else:
        st.success(f"Action potential fired — {len(ap_peaks)} AP(s). Peak: {V[ap_peaks[0]]:.1f} mV")
else:
    st.warning("Sub-threshold — no action potential. Increase I_inj or duration.")

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
_PROP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# ── Plotly figure (single compartment) ───────────────────────────────────────
def build_figure(t_s, V_s, iNa_s, iK_s, iL_s, mask_s, eNa, eK, win_ms, show_refs):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=("<b>Membrane Potential</b>", "<b>Membrane Current</b>"),
    )

    for s, e in zip(*_stim_intervals(mask_s, t_s)):
        for row in [1, 2]:
            fig.add_vrect(x0=s, x1=e, fillcolor="gold", opacity=0.20,
                          layer="below", line_width=0, row=row, col=1)

    fig.add_trace(go.Scatter(
        x=t_s, y=V_s, mode="lines", name="V_m",
        line=dict(color="black", width=2),
        hovertemplate="<b>t = %{x:.2f} ms</b><br>V_m = %{y:.2f} mV<extra></extra>",
    ), row=1, col=1)

    if show_refs:
        for y_val, color, label in [
            (eNa,    "red",    f"E_Na ({eNa:.0f} mV)"),
            (eK,     "#1565c0", f"E_K ({eK:.0f} mV)"),
            (V_REST, "gray",   f"Rest ({V_REST} mV)"),
            (-55,    "orange", "Threshold (−55 mV)"),
        ]:
            fig.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1,
                          opacity=0.6, row=1, col=1,
                          annotation_text=label, annotation_position="right",
                          annotation_font=dict(size=8, color=color))

    i_total_s = iNa_s + iK_s + iL_s
    fig.add_trace(go.Scatter(
        x=t_s, y=i_total_s, mode="lines", name="I_membrane",
        line=dict(color="black", width=2),
        hovertemplate="<b>t = %{x:.2f} ms</b><br>I_m = %{y:.2f} A<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(y=0, line_color="#aaaaaa", line_width=0.8, row=2, col=1)

    _apply_layout(fig, win_ms)
    fig.update_yaxes(title_text="V_m (mV)",       range=[-100, 80], row=1, col=1)
    fig.update_yaxes(title_text="I_membrane (A)",                    row=2, col=1)
    return fig


# ── Plotly figure (propagation — 5 traces) ───────────────────────────────────
def build_propagation_figure(t_s, V_traces_s, positions_um, mask_s, eNa, eK,
                              win_ms, show_refs):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=("<b>AP Propagation along Axon</b>",
                        "<b>Stimulus (comp. 0)</b>"),
        row_heights=[0.72, 0.28],
    )

    for s, e in zip(*_stim_intervals(mask_s, t_s)):
        for row in [1, 2]:
            fig.add_vrect(x0=s, x1=e, fillcolor="gold", opacity=0.20,
                          layer="below", line_width=0, row=row, col=1)

    for i, (V_s, pos_um) in enumerate(zip(V_traces_s, positions_um)):
        label = f"{pos_um} µm"
        fig.add_trace(go.Scatter(
            x=t_s, y=V_s, mode="lines", name=label,
            line=dict(color=_PROP_COLORS[i], width=2),
            hovertemplate=f"<b>{label}</b><br>t = %{{x:.2f}} ms<br>V_m = %{{y:.2f}} mV<extra></extra>",
        ), row=1, col=1)

    if show_refs:
        for y_val, color, label in [
            (eNa,    "red",    f"E_Na ({eNa:.0f} mV)"),
            (eK,     "#1565c0", f"E_K ({eK:.0f} mV)"),
            (V_REST, "gray",   f"Rest ({V_REST} mV)"),
            (-55,    "orange", "Threshold (−55 mV)"),
        ]:
            fig.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1,
                          opacity=0.6, row=1, col=1,
                          annotation_text=label, annotation_position="right",
                          annotation_font=dict(size=8, color=color))

    # Bottom panel: injected current pulse (comp 0)
    I_stim_s = np.array([
        get_i_inj(ti, i_inj, stim_on, stim_dur, stim_type_val, train_freq, train_n)
        for ti in t_s
    ])
    fig.add_trace(go.Scatter(
        x=t_s, y=I_stim_s, mode="lines", name="I_inj",
        line=dict(color="black", width=1.5),
        hovertemplate="<b>t = %{x:.2f} ms</b><br>I_inj = %{y:.2f} A<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="#aaaaaa", line_width=0.8, row=2, col=1)

    _apply_layout(fig, win_ms)
    fig.update_yaxes(title_text="V_m (mV)", range=[-100, 80], row=1, col=1)
    fig.update_yaxes(title_text="I_inj (A)",                   row=2, col=1)
    return fig


def _apply_layout(fig, win_ms):
    tick_step = _xtick_step(win_ms)
    xticks    = list(np.arange(0, win_ms + tick_step, tick_step))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified", height=520,
        margin=dict(l=70, r=110, t=55, b=50),
        font=dict(color="black", size=11),
        legend=dict(x=1.02, y=0.98, bgcolor="white",
                    bordercolor="#cccccc", borderwidth=1),
    )
    for row in [1, 2]:
        fig.update_xaxes(showgrid=False, showline=True, linecolor="black",
                         ticks="outside", tickvals=xticks,
                         range=[0, win_ms], mirror=False, row=row, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_yaxes(showgrid=False, showline=True, linecolor="black",
                     ticks="outside", mirror=False)


# ── Render ────────────────────────────────────────────────────────────────────
plot_slot = st.empty()

if propagation:
    with st.spinner("Running cable model…"):
        t_p, V_traces, positions_um, eNa_p, eK_p, _ = cached_propagation(
            i_inj, stim_on, stim_dur, stim_type_val, train_freq, train_n, win_ms
        )

    I_stim_p = np.array([
        get_i_inj(ti, i_inj, stim_on, stim_dur, stim_type_val, train_freq, train_n)
        for ti in t_p
    ])
    mask_p = I_stim_p != 0

    if run_btn:
        n_pts = len(t_p)
        step  = max(1, n_pts // 200)
        delay = 0.04 / (anim_speed / 5)
        for end in range(step, n_pts + step, step * max(1, anim_speed // 3)):
            end    = min(end, n_pts)
            slices = [V[:end] for V in V_traces]
            fig    = build_propagation_figure(
                t_p[:end], slices, positions_um, mask_p[:end],
                eNa_p, eK_p, win_ms, show_refs)
            plot_slot.plotly_chart(fig, use_container_width=True)
            time.sleep(delay)
    else:
        fig = build_propagation_figure(
            t_p, V_traces, positions_um, mask_p,
            eNa_p, eK_p, win_ms, show_refs)
        plot_slot.plotly_chart(fig, use_container_width=True)

else:
    if run_btn:
        n_pts = len(t)
        step  = max(1, n_pts // 200)
        delay = 0.04 / (anim_speed / 5)
        for end in range(step, n_pts + step, step * max(1, anim_speed // 3)):
            end = min(end, n_pts)
            fig = build_figure(t[:end], V[:end], iNa[:end], iK[:end], iL[:end],
                               stim_mask[:end], eNa, eK, win_ms, show_refs)
            plot_slot.plotly_chart(fig, use_container_width=True)
            time.sleep(delay)
    else:
        fig = build_figure(t, V, iNa, iK, iL, stim_mask, eNa, eK, win_ms, show_refs)
        plot_slot.plotly_chart(fig, use_container_width=True)

# ── Info ──────────────────────────────────────────────────────────────────────
with st.expander("How to read the plots"):
    st.markdown("""
**Membrane Potential** — Black trace shows V_m. Gold shading = stimulus active.

**Membrane Current** — Total current (I_Na + I_K + I_L). Inward = negative, outward = positive.
- Goes sharply negative during depolarisation (Na⁺ inward current dominates)
- Crosses zero at the AP peak, then goes positive as K⁺ outward current takes over during repolarisation
- Returns to zero at rest

**Hover** anywhere on either chart to see exact (time, value) coordinates.

**Finding threshold:** Start at 0 A and increase I_inj in 0.1 steps.
The status bar turns green when an AP fires. Typical threshold for a 1 ms pulse: **6–8 A**.
    """)
