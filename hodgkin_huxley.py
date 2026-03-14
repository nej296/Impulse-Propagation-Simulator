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
# K_OUT = 9 mM gives E_K ≈ -76 mV, matching the original HH undershoot of ~-75 mV.
# Using K_OUT = 4 mM (standard mammalian CSF) gives E_K ≈ -98 mV which causes the
# undershoot to chase far too deep. The original HH squid axon concentrations were
# chosen to give E_K ~ -77 mV; K_OUT = 9 mM with mammalian K_IN = 155 mM replicates
# that equilibrium at body temperature.
K_OUT, K_IN = 9, 155
G_NA, G_K, G_L = 120, 36, 0.3
C_M = 1.0


def nernst(z, out, inn):
    """Nernst potential in mV."""
    return (R * T / (z * F)) * np.log(out / inn) * 1000


# ── Vectorised gate functions (arrays of V) used by cable model ───────────────
def _am_v(V):
    x = V + 40
    return np.where(np.abs(x) < 1e-6, 1.0,
                    0.1 * x / (1 - np.exp(np.clip(-x / 10, -700, 700))))

def _bm_v(V): return 4.0  * np.exp(np.clip(-(V + 65) / 18, -700, 700))
def _ah_v(V): return 0.07 * np.exp(np.clip(-(V + 65) / 20, -700, 700))
def _bh_v(V): return 1.0  / (1.0 + np.exp(np.clip(-(V + 35) / 10, -700, 700)))

def _an_v(V):
    x = V + 55
    return np.where(np.abs(x) < 1e-6, 0.1,
                    0.01 * x / (1 - np.exp(np.clip(-x / 10, -700, 700))))

def _bn_v(V): return 0.125 * np.exp(np.clip(-(V + 65) / 80, -700, 700))


def _safe_exp(x, limit=700):
    """Avoid overflow in exp."""
    return np.exp(np.clip(x, -limit, limit))


def alpha_m(V):
    # Standard HH at V_rest = -65 mV: singularity at V = -40 mV
    x = V + 40
    if np.abs(x) < 1e-6:
        return 1.0
    return 0.1 * x / (1 - _safe_exp(-x / 10))


def beta_m(V):
    return 4 * _safe_exp(-(V + 65) / 18)


def alpha_h(V):
    return 0.07 * _safe_exp(-(V + 65) / 20)


def beta_h(V):
    return 1 / (1 + _safe_exp(-(V + 35) / 10))


def alpha_n(V):
    x = V + 55
    if np.abs(x) < 1e-6:
        return 0.1
    return 0.01 * x / (1 - _safe_exp(-x / 10))


def beta_n(V):
    return 0.125 * _safe_exp(-(V + 65) / 80)


def m_inf(V):
    return alpha_m(V) / (alpha_m(V) + beta_m(V))


def h_inf(V):
    return alpha_h(V) / (alpha_h(V) + beta_h(V))


def n_inf(V):
    return alpha_n(V) / (alpha_n(V) + beta_n(V))


def compute_eL(eNa, eK):
    """Compute E_L so that I_total = 0 at V_rest."""
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
    """Run HH simulation and return t, V, m, h, n, iNa, iK, iL, eNa, eK, eL."""
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

    return t, V, m, h, n, iNa, iK, iL, eNa, eK, eL


def run_propagation(
    i_inj=10,
    stim_on=5,
    stim_dur=1,
    stim_type="pulse",
    train_freq=50,
    train_n=5,
    win_ms=100,
    n_comp=40,
    dx_um=100,
    g_coupling=5.0,
):
    """Multi-compartment cable model.

    n_comp compartments of length dx_um µm each, coupled axially with g_coupling
    (mS/cm²). Stimulus injected at compartment 0 only. Returns V traces at 5
    evenly spaced positions.
    """
    eNa = nernst(1, NA_OUT, NA_IN)
    eK  = nernst(1, K_OUT, K_IN)
    eL  = compute_eL(eNa, eK)

    m0 = m_inf(V_REST)
    h0 = h_inf(V_REST)
    n0 = n_inf(V_REST)
    y0 = np.tile([V_REST, m0, h0, n0], n_comp).astype(float)

    record_idx = [0, n_comp // 4, n_comp // 2, 3 * n_comp // 4, n_comp - 1]

    def deriv(t, y):
        y2d = y.reshape(n_comp, 4)
        V = y2d[:, 0]
        m = y2d[:, 1]
        h = y2d[:, 2]
        n = y2d[:, 3]

        iNa  = G_NA * m**3 * h * (V - eNa)
        iK   = G_K  * n**4     * (V - eK)
        iL_  = G_L             * (V - eL)
        i_ion = iNa + iK + iL_

        # Axial coupling (no-flux boundaries)
        i_ax = np.zeros(n_comp)
        i_ax[:-1] += g_coupling * (V[1:]  - V[:-1])
        i_ax[1:]  += g_coupling * (V[:-1] - V[1:])

        i_ext = np.zeros(n_comp)
        i_ext[0] = get_i_inj(t, i_inj, stim_on, stim_dur, stim_type, train_freq, train_n)

        dV = (i_ax - i_ion + i_ext) / C_M
        am = _am_v(V); bm = _bm_v(V)
        ah = _ah_v(V); bh = _bh_v(V)
        an = _an_v(V); bn = _bn_v(V)
        dm = am * (1 - m) - bm * m
        dh = ah * (1 - h) - bh * h
        dn = an * (1 - n) - bn * n

        return np.stack([dV, dm, dh, dn], axis=1).ravel()

    sol = solve_ivp(
        deriv,
        [0, win_ms],
        y0,
        method="RK45",
        t_eval=np.arange(0, win_ms, 0.05),
        rtol=1e-5,
        atol=1e-7,
        max_step=0.1,
    )

    t   = sol.t
    y3d = sol.y.T.reshape(len(t), n_comp, 4)

    V_traces     = [y3d[:, i, 0] for i in record_idx]
    positions_um = [i * dx_um    for i in record_idx]

    return t, V_traces, positions_um, eNa, eK, eL


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t, V, m, h, n, iNa, iK, iL, eNa, eK, eL = run_simulation(
        i_inj=10, stim_on=5, stim_dur=1, stim_type="pulse", win_ms=50
    )

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    axes[0].plot(t, V, "g-", lw=2, label="V_m")
    axes[0].axhline(eNa, color="r", ls="--", alpha=0.5, label=f"E_Na ({eNa:.0f} mV)")
    axes[0].axhline(eK, color="b", ls="--", alpha=0.5, label=f"E_K ({eK:.0f} mV)")
    axes[0].axhline(V_REST, color="gray", ls="--", alpha=0.5, label="Rest")
    axes[0].axhline(-55, color="orange", ls="--", alpha=0.5, label="Threshold")
    axes[0].set_ylabel("V_m (mV)")
    axes[0].set_ylim(-100, 80)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, iNa, color="tomato", lw=2, label="I_Na")
    axes[1].plot(t, iK, color="steelblue", lw=2, label="I_K")
    axes[1].plot(t, iL, color="mediumseagreen", lw=1.5, label="I_L")
    axes[1].axhline(0, color="gray", ls="-", alpha=0.3)
    axes[1].set_ylabel("Current (µA/cm²)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, m, color="gold", lw=2, label="m (Na act.)")
    axes[2].plot(t, h, color="cyan", lw=2, label="h (Na inact.)")
    axes[2].plot(t, n, color="violet", lw=2, label="n (K act.)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Gate probability")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hh_example.png", dpi=150)
    plt.show()
