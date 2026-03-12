"""
Hodgkin-Huxley Neuron Simulator — Interactive Streamlit App
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from hodgkin_huxley import (
    run_simulation,
    V_REST,
)

st.set_page_config(
    page_title="Hodgkin-Huxley Neuron Simulator",
    page_icon="🧠",
    layout="wide",
)

st.title("Hodgkin-Huxley Neuron Simulator")
st.caption(
    "Computational model of squid axon — membrane potential and ionic currents from the 1952 equations"
)

st.markdown("---")

# Sidebar: Current Clamp controls
st.sidebar.header("Current Clamp")

st.sidebar.subheader("Stimulus")
i_inj = st.sidebar.slider(
    "I_inj (µA/cm²)",
    min_value=-20.0,
    max_value=30.0,
    value=0.0,
    step=0.5,
)
stim_on = st.sidebar.number_input("Onset (ms)", min_value=0.0, value=5.0, step=0.5)
stim_dur = st.sidebar.number_input("Duration (ms)", min_value=0.1, value=1.0, step=0.1)
stim_type = st.sidebar.selectbox(
    "Type",
    ["Single pulse", "Pulse train", "Step current"],
    format_func=lambda x: x,
)
if stim_type == "Single pulse":
    stim_type_val = "pulse"
elif stim_type == "Pulse train":
    stim_type_val = "train"
else:
    stim_type_val = "step"

train_freq = 50
train_n = 5
if stim_type_val == "train":
    train_freq = st.sidebar.number_input("Freq (Hz)", min_value=1, value=50)
    train_n = st.sidebar.number_input("Pulses", min_value=1, value=5)

st.sidebar.subheader("Simulation")
win_ms = st.sidebar.number_input("Window (ms)", min_value=20, value=100, step=10)
speed_note = st.sidebar.info("Time step: 0.01 ms (RK45)")

if st.sidebar.button("Reset to Defaults"):
    st.session_state.clear()
    st.rerun()

# Run simulation (auto-runs when inputs change)
t, V, I_total, eNa, eK, eL = run_simulation(
    i_inj=i_inj,
    stim_on=stim_on,
    stim_dur=stim_dur,
    stim_type=stim_type_val,
    train_freq=train_freq,
    train_n=train_n,
    win_ms=win_ms,
)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
fig.patch.set_facecolor("#0e1117")
ax1.set_facecolor("#0e1117")
ax2.set_facecolor("#0e1117")

ax1.plot(t, V, "g-", lw=2, label="V_m")
ax1.axhline(eNa, color="r", ls="--", alpha=0.5, label="E_Na")
ax1.axhline(eK, color="b", ls="--", alpha=0.5, label="E_K")
ax1.axhline(V_REST, color="gray", ls="--", alpha=0.5, label="Rest")
ax1.axhline(-40, color="orange", ls="--", alpha=0.5, label="Threshold")
ax1.set_ylabel("V_m (mV)")
ax1.set_ylim(-100, 80)
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.tick_params(colors="white")
ax1.xaxis.label.set_color("white")
ax1.yaxis.label.set_color("white")
ax1.spines["bottom"].set_color("white")
ax1.spines["top"].set_color("white")
ax1.spines["left"].set_color("white")
ax1.spines["right"].set_color("white")

ax2.plot(t, I_total, "b-", lw=2)
ax2.axhline(0, color="gray", ls="-", alpha=0.3)
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("I_total (µA/cm²)")
ax2.grid(True, alpha=0.3)
ax2.tick_params(colors="white")
ax2.xaxis.label.set_color("white")
ax2.yaxis.label.set_color("white")
ax2.spines["bottom"].set_color("white")
ax2.spines["top"].set_color("white")
ax2.spines["left"].set_color("white")
ax2.spines["right"].set_color("white")

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Tips
with st.expander("Tips"):
    st.markdown(
        """
    - **Single pulse (1 ms):** Use ~20 µA/cm² for one action potential.
    - **Step current:** Use 7–10 µA/cm² sustained for 50+ ms to see repetitive firing.
    - **AP shape:** Peak ~+30 to +40 mV, undershoot ~-75 to -80 mV (physiologically accurate).
    """
    )
