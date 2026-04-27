# Impulse Propagation Simulator

Interactive Hodgkin-Huxley neuron simulator — computational model of squid axon membrane dynamics using the 1952 Hodgkin-Huxley equations.

## Features

- **Modern dashboard UI** — tabbed electrophysiology, gating/conductance, and cable-propagation views
- **Scientifically aligned units** — injected and membrane currents are shown as current density (µA/cm²)
- **Time-synchronized traces** — fixed-step RK4 integration for single-compartment voltage/current alignment
- **Current clamp protocols** — single pulse, pulse train, or sustained step
- **Propagation mode** — multi-compartment cable model with peak-arrival timing readout

## Installation

```bash
git clone https://github.com/nej296/Impulse-Propagation-Simulator.git
cd Impulse-Propagation-Simulator
pip install -r requirements.txt
```

## Run the interactive app

```bash
streamlit run app.py
```

## Run the core module (standalone)

```python
from hodgkin_huxley import run_simulation

t, V, m, h, n, iNa, iK, iL, eNa, eK, eL = run_simulation(
    i_inj=20, stim_on=5, stim_dur=1, stim_type="pulse", win_ms=50
)
```

Or from the command line:

```bash
python hodgkin_huxley.py
```

This runs an example and saves `hh_example.png`.

## Parameters (hardcoded)

- **Conductances:** g_Na = 120 mS/cm², g_K = 36 mS/cm², g_L = 0.3 mS/cm², C_m = 1 µF/cm²
- **Ion concentrations:** [Na⁺]_out = 145 mM, [Na⁺]_in = 12 mM, [K⁺]_out = 9 mM, [K⁺]_in = 155 mM
- E_Na and E_K are computed from Nernst; E_L is set so I_total = 0 at rest

## Author

Nicholas Johnson

## License

MIT
