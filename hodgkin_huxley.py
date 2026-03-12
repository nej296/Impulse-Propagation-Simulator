"""
Hodgkin-Huxley Neuron Simulator
Computational model of squid axon — membrane potential and ionic currents from the 1952 equations.
"""

import numpy as np
from scipy.integrate import solve_ivp

# Constants
R = 8.314462618  # J/(mol·K)
F = 96485.33212  # C/mol
T = 310  # K (37°C)
V_REST = -65

# Default parameters (hardcoded per spec)
NA_OUT, NA_IN = 145, 12
K_OUT, K_IN = 4, 155
G_NA, G_K, G_L = 120, 36, 0.3
C_M = 1.0


def nernst(z, out, inn):
    """Nernst potential in mV."""
    return (R * T / (z * F)) * np.log(out / inn) * 1000


def _safe_exp(x, limit=700):
    """Avoid overflow in exp."""
    return np.exp(np.clip(x, -limit, limit))


def alpha_m(V):
    x = V + 35
    if np.abs(x) < 1e-6:
        return 1.0
    return 0.1 * x / (1 - _safe_exp(-x / 10))


def beta_m(V):
    return 4 * _safe_exp(-(V + 60) / 18)


def alpha_h(V):
    return 0.07 * _safe_exp(-(V + 60) / 20)


def beta_h(V):
    return 1 / (1 + _safe_exp(-(V + 30) / 10))


def alpha_n(V):
    x = V + 55
    if np.abs(x) < 1e-6:
        return 0.1
    return 0.01 * x / (1 - _safe_exp(-x / 10))


def beta_n(V):
    return 0.125 * _safe_exp(-(V + 60) / 80)


def m_inf(V):
    return alpha_m(V) / (alpha_m(V) + beta_m(V))


def h_inf(V):
    return alpha_h(V) / (alpha_h(V) + beta_h(V))


def n_inf(V):
    return alpha_n(V) / (alpha_n(V) + beta_n(V))


def compute_eL(eNa, eK):
    """Compute E_L so that I_total = 0 at V_rest. I_Na + I_K + I_L = 0 => E_L = V_rest + (I_Na + I_K)/g_L."""
    m = m_inf(V_REST)
    h = h_inf(V_REST)
    n = n_inf(V_REST)
    iNa = G_NA * m**3 * h * (V_REST - eNa)
    iK = G_K * n**4 * (V_REST - eK)
    return V_REST + (iNa + iK) / G_L


def get_i_inj(t, i_inj, stim_on, stim_dur, stim_type, train_freq, train_n):
    """Injected current at time t (µA/cm²)."""
    if stim_type == "step":
        return i_inj if t >= stim_on else 0
    if stim_type == "pulse":
        return i_inj if (stim_on <= t < stim_on + stim_dur) else 0
    if stim_type == "train":
        period = 1000 / train_freq
        for i in range(train_n):
            t0 = stim_on + i * period
            if t0 <= t < t0 + stim_dur:
                return i_inj
        return 0
    return 0


def hh_deriv(t, y, eNa, eK, eL, i_inj, stim_on, stim_dur, stim_type, train_freq, train_n):
    """Hodgkin-Huxley ODE system: d/dt [V, m, h, n]."""
    V, m, h, n = y
    i_inj_t = get_i_inj(t, i_inj, stim_on, stim_dur, stim_type, train_freq, train_n)

    iNa = G_NA * m**3 * h * (V - eNa)
    iK = G_K * n**4 * (V - eK)
    iL = G_L * (V - eL)
    iTotal = iNa + iK + iL

    dV = (i_inj_t - iTotal) / C_M
    dm = alpha_m(V) * (1 - m) - beta_m(V) * m
    dh = alpha_h(V) * (1 - h) - beta_h(V) * h
    dn = alpha_n(V) * (1 - n) - beta_n(V) * n

    return [dV, dm, dh, dn]


def run_simulation(
    i_inj=0,
    stim_on=5,
    stim_dur=1,
    stim_type="pulse",
    train_freq=50,
    train_n=5,
    win_ms=100,
):
    """Run HH simulation and return t, V, I_total."""
    eNa = nernst(1, NA_OUT, NA_IN)
    eK = nernst(1, K_OUT, K_IN)
    eL = compute_eL(eNa, eK)

    y0 = [V_REST, m_inf(V_REST), h_inf(V_REST), n_inf(V_REST)]

    def deriv(t, y):
        return hh_deriv(
            t, y, eNa, eK, eL, i_inj, stim_on, stim_dur, stim_type, train_freq, train_n
        )

    sol = solve_ivp(
        deriv,
        [0, win_ms],
        y0,
        method="RK45",
        t_eval=np.arange(0, win_ms, 0.01),
        rtol=1e-8,
        atol=1e-10,
    )

    t = sol.t
    V = sol.y[0]
    m, h, n = sol.y[1], sol.y[2], sol.y[3]

    iNa = G_NA * m**3 * h * (V - eNa)
    iK = G_K * n**4 * (V - eK)
    iL = G_L * (V - eL)
    I_total = iNa + iK + iL

    return t, V, I_total, eNa, eK, eL


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example: single pulse
    t, V, I_total, eNa, eK, eL = run_simulation(
        i_inj=20, stim_on=5, stim_dur=1, stim_type="pulse", win_ms=50
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax1.plot(t, V, "g-", lw=2, label="V_m")
    ax1.axhline(eNa, color="r", ls="--", alpha=0.5, label="E_Na")
    ax1.axhline(eK, color="b", ls="--", alpha=0.5, label="E_K")
    ax1.axhline(V_REST, color="gray", ls="--", alpha=0.5, label="Rest")
    ax1.axhline(-40, color="orange", ls="--", alpha=0.5, label="Threshold")
    ax1.set_ylabel("V_m (mV)")
    ax1.set_ylim(-100, 80)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, I_total, "b-", lw=2)
    ax2.axhline(0, color="gray", ls="-", alpha=0.3)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("I_total (µA/cm²)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hh_example.png", dpi=150)
    plt.show()
