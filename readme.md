# Higher-Order Statistical Structure from Nonlinear Dynamics

[![Status](https://img.shields.io/badge/Status-In%20Development-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-blue)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-informational)]()

> **Hernández, D., Zamora-López, G., Laureys, S., Hindriks, R., Gomez, F., & Tewarie, P.K.B.**
> *"Higher-order statistical structure emerges from nonlinear dynamics without explicit higher-order coupling."*
> (2025, preprint)

---

## Overview

This repository contains the full simulation code for a minimal three-node
Wilson–Cowan network study that asks a fundamental question in computational
neuroscience and complex-systems theory:

> **Do higher-order statistical dependencies in neural signals require
> explicit higher-order synaptic interactions, or can they emerge generically
> from nonlinear second-order dynamics?**

We compare four coupling architectures — two second-order (additive and diffusive)
and two explicit third-order (additive and diffusive) — and quantify
higher-order statistical structure using **Total Correlation minus Dual Total
Correlation (TC − DTC)** and **third-order cumulants (κ₁₂₃)**. We vary
three control parameters: external drive *P*, coupling strength *K*, and the
sigmoid slope *a_E* (local nonlinear gain).

### Main finding

In the oscillatory (post-Hopf) regime, the second-order additive model produces
TC − DTC values that are on average **16-fold** above the fixed-point
baseline, reaching magnitudes comparable to the explicit third-order model.
Increasing *a_E* amplifies TC − DTC up to **28-fold**, independently of
coupling order. This confirms that **proximity to a Hopf bifurcation and
local nonlinear gain — not coupling architecture — are the primary drivers of
higher-order statistical structure**.

---

## Repository Structure

```
Criticality_HOI/
├── Scripts/                          # Python source modules (importable package)
│   ├── __init__.py
│   ├── parameters.py                 # Heterogeneous per-node parameter sets
│   ├── parameters_random.py          # Homogeneous global parameters & sweep grids
│   ├── dynamics.py                   # Wilson–Cowan RHS for all 4 coupling types
│   ├── simulation.py                 # RK4 integrator + stochastic wrappers
│   ├── metrics.py                    # TC, DTC, cumulants, skewness, kurtosis
│   ├── oscillation_detection.py      # Poincaré-section limit-cycle detector
│   └── visualization.py              # Matplotlib + Plotly plotting utilities
│
├── Notebook/                         # Jupyter notebooks for all experiments
│   ├── W_C_2nd_Order.ipynb            # Second-order additive: (P, K) sweep
│   ├── W_C_2nd_Order_Diffusive.ipynb  # Second-order diffusive: (P, K) sweep
│   ├── W_C_3rd_Order.ipynb            # Third-order additive: (P, K) sweep
│   ├── W_C_3rd_Order_Diffusive.ipynb  # Third-order diffusive: (P, K) sweep
│   ├── W_C_2nd_Order_Lines.ipynb      # Second-order additive: metric curves vs K
│   ├── W_C_2nd_Order_Diffusive_Lines.ipynb
│   ├── W_C_3rd_Order_Lines.ipynb
│   ├── W_C_3rd_Order_Diffusive_Lines.ipynb
│   ├── W_C_2nd_Order_a+k.ipynb        # Second-order additive: a_E–K sweep
│   ├── W_C_3rd_Order_a+k.ipynb        # Third-order additive: a_E–K sweep
│   ├── Results_comparisons.ipynb     # Load saved results, generate paper figures
│   └── Test.ipynb                    # Quick exploration / sanity checks
│
├── results/                          # Pre-computed metric arrays and paper figures
│   ├── metrics_Pairwise_noise.npz
│   ├── metrics_Pairwise_diffusive_noise.npz
│   ├── metrics_Coupling_noise.npz
│   ├── metrics_Coupling_diffusive_noise.npz
│   ├── metrics_Pairwise_noise_a.npz
│   ├── metrics_Coupling_noise_a.npz
│   ├── fig1_PK_additive.pdf
│   ├── fig2_PK_diffusive.pdf
│   ├── fig3_aK_additive.pdf
│   └── paper_stats.json
│
├── data/                             # Simulation inputs
│   ├── patients/
│   └── patients_criticality/
│
├── Images/                           # Reference images
├── archive/                          # Legacy notebooks and scripts (not maintained)
│   ├── Notebooks_BackUps/
│   ├── Notebooks_BackUps_2/
│   └── Scripts_BackUps/
│
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

---

## Model Description

### Wilson–Cowan Network

We study a minimal network of **N = 3** coupled Wilson–Cowan nodes with
all-to-all connectivity. Each node *i* contains an excitatory variable
*Eᵢ(t)* and an inhibitory variable *Iᵢ(t)*.

**Inhibitory population (all architectures):**

$$\tau_I^{(i)} \dot{I}_i = -I_i + S\!\left(c_{IE}^{(i)} E_i - c_{II}^{(i)} I_i + Q^{(i)}\right)$$

**Excitatory population — four coupling architectures:**

| Architecture | Label | Coupling term |
|---|---|---|
| Second-order additive | PA | K Σⱼ Mᵢⱼ Eⱼ |
| Second-order diffusive | PD | K Σⱼ Mᵢⱼ (Eⱼ − Eᵢ) |
| Third-order additive | TA | K₃ Σⱼₖ Tᵢⱼₖ Eⱼ Eₖ |
| Third-order diffusive | TD | K₃ Σⱼₖ Tᵢⱼₖ (Eⱼ−Eᵢ)(Eₖ−Eᵢ) |

The logistic sigmoid transfer function is S(x; a) = 1 / (1 + exp(−ax)).

### Higher-Order Statistical Measures

- **TC (Total Correlation):** measures the total redundancy across all nodes.
- **DTC (Dual Total Correlation):** measures the total synergy.
- **TC − DTC (O-information):** positive → net redundancy; negative → net synergy.
- **κ₁₂₃ (Third-order cumulant):** normalized co-skewness of the joint excitatory activity.

All measures are computed under a Gaussian approximation using the empirical
covariance matrix of the post-transient excitatory time series.

### Stochastic Integration

The network is integrated with a 4th-order Runge–Kutta (RK4) scheme. Small
additive Gaussian noise (σ_E = 0.01, Euler–Maruyama) is applied to the
excitatory population to compute the covariance matrix at each (P, K) point.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Criticality_HOI.git
cd Criticality_HOI
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Reproducing paper results

All experiments are self-contained Jupyter notebooks. The recommended order
for reproducing the paper figures is:

| Step | Notebook | Output file |
|------|----------|-------------|
| 1 | `W_C_2nd_Order.ipynb` | `results/metrics_Pairwise_noise.npz` |
| 2 | `W_C_2nd_Order_Diffusive.ipynb` | `results/metrics_Pairwise_diffusive_noise.npz` |
| 3 | `W_C_3rd_Order.ipynb` | `results/metrics_Coupling_noise.npz` |
| 4 | `W_C_3rd_Order_Diffusive.ipynb` | `results/metrics_Coupling_diffusive_noise.npz` |
| 5 | `W_C_2nd_Order_a+k.ipynb` | `results/metrics_Pairwise_noise_a.npz` |
| 6 | `W_C_3rd_Order_a+k.ipynb` | `results/metrics_Coupling_noise_a.npz` |
| 7 | `Results_comparisons.ipynb` | `results/fig1_PK_additive.pdf`, etc. |

> **Note:** Pre-computed `.npz` files are already included in `results/`,
> so you can jump directly to step 7 to reproduce the figures without
> re-running the full parameter sweeps.

### Quick start (single simulation)

```python
import sys
sys.path.append('path/to/Criticality_HOI/')  # add repo root to path

from Scripts.parameters_random import *
from Scripts.simulation import simulate_wc_stochastic
from Scripts.metrics import TC_gauss, DTC_gauss

import numpy as np

# Initial condition
state0 = 0.1 * np.random.randn(2 * N_nodes)

# Run stochastic simulation (second-order additive, P=5, K=0.4)
t, states = simulate_wc_stochastic(state0, P=5.0, K=0.4, M=M, T=20, dt=dt)

# Discard 2 s transient, extract excitatory time series
E = states[int(2.0 / dt):, :N_nodes]

# Compute TC and DTC under Gaussian approximation
Sigma = np.cov(E.T)
tc  = TC_gauss(Sigma)
dtc = DTC_gauss(Sigma)
print(f"TC = {tc:.4f} | DTC = {dtc:.4f} | TC−DTC = {tc - dtc:.4f} nats")
```

---

## Parameter Space

The phase diagrams cover the following grids:

| Parameter | Range | Steps | Description |
|-----------|-------|-------|-------------|
| External drive P | 1.0 → 10.0 | 20 | Controls proximity to Hopf bifurcation |
| Coupling strength K | 0.0 → 1.0 | 20 | Second-order inter-node coupling |
| Coupling strength K₃ | 0.0 → 1.0 | 20 | Third-order inter-node coupling |
| Sigmoid slope a_E | 0.5 → 3.0 | 20 | Local nonlinear gain |

Each grid point: T = 10 s, dt = 10⁻³ s, σ_E = 0.01, 2 s transient discarded.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical arrays and linear algebra |
| `scipy` | Covariance estimation, interpolation |
| `matplotlib` | Static figures |
| `plotly` | Interactive contour and line plots |
| `kaleido` | Plotly static image export |
| `jupyter` | Notebook environment |

---

## Authors

| Name | Affiliation |
|------|-------------|
| **Diego Hernández** | CERVO Brain Research Center, Université Laval, Canada; Dept. of Systems and Computing Engineering, Universidad Nacional de Colombia, Colombia |
| **Gorka Zamora-López** | Dept. of Complex Systems, Institute of Computer Science, Czech Academy of Sciences, Czech Republic |
| **Steven Laureys** | CERVO Brain Research Center, Université Laval, Canada; Coma Science Group, GIGA-Consciousness, Université de Liège, Belgium |
| **Rikkert Hindriks** | Dept. of Mathematics, VU University, Amsterdam, Netherlands |
| **Francisco Gomez** | Dept. of Mathematics, Universidad Nacional de Colombia, Colombia; CERVO Brain Research Center, Université Laval, Canada |
| **Prejaas K.B. Tewarie** | CERVO Brain Research Center, Université Laval, Canada; Sir Peter Mansfield Imaging Centre, University of Nottingham, United Kingdom |

---

## Contact

- **Diego Hernández** — dieahernandezcas@unal.edu.co
- **Prejaas K.B. Tewarie** — Prejaas.KBTewarie@cervo.ulaval.ca

---

## License

Released under the **MIT License**. See `LICENSE` for details.

---

## References

1. Wilson, H.R. & Cowan, J.D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. *Biophysical Journal*, 12(1), 1–24.
2. Battiston, F. et al. (2021). The physics of higher-order interactions in complex systems. *Nature Physics*, 17, 1093–1098.
3. Bick, C. et al. (2023). What are higher-order networks? *SIAM Review*, 65(3), 686–731.
4. Hindriks, R. et al. (2024). Higher-order functional connectivity analysis of resting-state fMRI using multivariate cumulants. *Human Brain Mapping*, 45(5), e26663.
5. Varley, T.F. et al. (2023). Partial entropy decomposition reveals higher-order information structures in human brain activity. *PNAS*, 120(30), e2300888120.
6. Stramaglia, S. et al. (2021). Quantifying dynamical high-order interdependencies from the O-information. *Physical Review Research*, 3, 033090.
7. Rosas, F.E. et al. (2022). Disentangling high-order mechanisms and high-order behaviours in complex systems. *Nature Physics*, 18, 476–477.
