# Optimal Market Making — Imperial College London
### Market Microstructure Coursework

Implementation and extension of Guéant's (2017) optimal market-making framework, combining analytical ODE solvers, Monte Carlo simulation, empirical backtesting on LOBSTER data, and a Reinforcement Learning extension with Hawkes processes.

---

## Overview

This project reproduces and extends the theoretical results of:

- **Guéant, O. (2017)**. *Optimal market making*. Applied Mathematical Finance, 24(2), 112–154.
- **Lalor, D. & Swishchuk, A. (2025)**. *Market making with Hawkes processes* (arXiv:2502.17417).

The framework models a market maker who continuously posts bid and ask limit orders to maximize a CARA utility over a finite horizon [0, T]. The core mathematical object is a Hamilton-Jacobi-Bellman (HJB) equation reduced to a system of ODEs, solved numerically and compared to closed-form approximations.

---

## Project Structure

```
project/
├── market_making/
│   ├── core/
│   │   ├── intensity.py       # Intensity model Λ(δ) = A·exp(−k·δ), Hamiltonian
│   │   ├── hawkes.py          # Multivariate Hawkes process with softplus nonlinearity
│   │   ├── closed_form.py     # Closed-form quote approximation (Guéant §4)
│   │   ├── solver_1d.py       # Single-asset ODE solver (Model A & B, Newton/tridiagonal)
│   │   └── solver_2d.py       # Two-asset ODE solver with sparse Newton iteration
│   ├── simulation/
│   │   ├── simulator.py       # Monte Carlo simulator (1D & 2D)
│   │   └── backtest.py        # Backtest engine (optimal, naive, closed-form strategies)
│   ├── params/
│   │   ├── assets.py          # Calibrated parameters: IG & HY CDX (Guéant §6)
│   │   └── scenarios.py       # Named experiment scenarios
│   └── data/
│       └── calibrate.py       # Fit Λ(δ) = A·exp(−k·δ) to trade data (LOBSTER)
│
├── notebooks/
│   ├── 01_single_asset.ipynb          # Optimal quotes for IG & HY; reproduces Guéant §6
│   ├── 02_closed_form.ipynb           # ODE-exact vs closed-form approximation quality
│   ├── 03_model_a_vs_b.ipynb          # Model A (ξ=γ) vs Model B (ξ=0) comparison
│   ├── 04_multi_asset.ipynb           # Two-asset optimal market making (ρ=0.9)
│   ├── 05_monte_carlo.ipynb           # Monte Carlo validation & strategy comparison
│   ├── 06_compare_hawkes_vs_poisson.ipynb  # Hawkes vs Poisson fill processes
│   ├── 07_paper_extensions.ipynb      # Non-exponential intensities, Riccati, d>2 assets
│   ├── 08_lobster_backtest.ipynb      # Empirical backtest on LOBSTER equity data
│   ├── 09_terminal_penalty.ipynb      # Terminal inventory penalty sensitivity
│   └── 10_Extended_Guéant_with_RL.ipynb   # Hawkes + SAC reinforcement learning agent
│
└── tests/                             # Extension modules
    ├── adverse_selection.py           # Market making under price drift (informed flow)
    ├── fee_analysis.py                # Maker fee impact & break-even analysis
    ├── penalty.py                     # Terminal penalty: quadratic & linear
    └── rl_agent.py                    # SAC agent for Hawkes-extended problem
```

---

## Mathematical Framework

### Intensity Model
Order arrival rate as a function of distance-to-mid δ:

```
Λ(δ) = A · exp(−k · δ)
```

### HJB Equation (reduced ODE)
The value function θ_n(t) satisfies:

```
∂_t θ_n + ½γσ²(nΔ)² − H_ξ(p_bid) − H_ξ(p_ask) = 0
```

with terminal condition `θ_n(T) = −ℓ(|nΔ|)` (optional liquidation penalty).

### Models
- **Model A** (`ξ = γ`): risk on both mark-to-market and inventory. Reduces to a linear tridiagonal system via `v_n = exp(−k·θ_n/Δ)`.
- **Model B** (`ξ = 0`): inventory penalty only. Solved with Newton iteration on implicit Euler.

### Hawkes Extension
Self-exciting order flow:

```
λ_i(t) = φ(λ̄_i + Σ_j α_ij · y_j(t))
```

where `φ(x) = ln(1 + exp(x))` (softplus) and `y_j(t)` is the excitation state decaying at rate `β_ij`.

### Calibrated Parameters (Guéant §6)

| Asset | σ ($/√s) | A (1/s) | k (1/$) | Δ ($) | Q |
|-------|-----------|---------|---------|-------|---|
| IG CDX | 5.83e−6 | 9.10e−4 | 1.79e4 | 50M | 4 |
| HY CDX | 2.15e−5 | 1.06e−3 | 5.47e3 | 10M | 4 |

Risk aversion: γ = 6×10⁻⁵, correlation: ρ = 0.9, horizon: T = 7200 s.

---

## Notebooks Guide

| Notebook | Description |
|----------|-------------|
| `01_single_asset` | Reproduces Guéant §6 figures: optimal bid/ask quotes, inventory skew, time-to-maturity effects for IG and HY CDX |
| `02_closed_form` | Assesses the quality of the parabolic closed-form approximation vs full ODE, especially near T |
| `03_model_a_vs_b` | Compares quoting strategies under Model A (ξ=γ) and Model B (ξ=0); spread and skew analysis |
| `04_multi_asset` | Extends to two correlated assets (IG + HY, ρ=0.9); 2D inventory grid ODE |
| `05_monte_carlo` | Validates ODE solution via Monte Carlo: P&L distributions, inventory control, optimal vs naive |
| `06_hawkes_vs_poisson` | Demonstrates order-flow clustering under Hawkes processes vs Poisson baseline |
| `07_paper_extensions` | Non-exponential intensity models, Riccati-based approximation, scaling to d>2 assets |
| `08_lobster_backtest` | Empirical backtest using LOBSTER Level-2 equity data; calibrates parameters from real LOB |
| `09_terminal_penalty` | Sensitivity analysis of terminal inventory penalty ℓ(q); linear vs quadratic; short-T regime |
| `10_rl_extension` | SAC reinforcement learning agent on Hawkes-extended environment; comparison vs ODE & closed-form |

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Running Notebooks

```bash
cd project
jupyter lab
```

### Package Installation (editable)

```bash
cd project
pip install -e .
```

---

## Key Results

- Full numerical reproduction of Guéant (2017) optimal quotes for IG and HY CDX markets
- Closed-form approximation is highly accurate away from the terminal time T
- Model A and B produce qualitatively similar skew but differ in spread magnitude
- Monte Carlo validation confirms inventory mean-reversion under optimal strategy
- LOBSTER empirical backtest shows significantly higher Sharpe vs naive fixed-spread strategy
- Hawkes processes exhibit order-flow clustering that improves fill rate estimation
- SAC agent achieves competitive performance vs ODE solution in the Hawkes-extended setting

---

## References

1. Guéant, O. (2017). *Optimal market making*. Applied Mathematical Finance, 24(2), 112–154.
2. Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217–224.
3. Lalor, D. & Swishchuk, A. (2025). *Market making with Hawkes processes*. arXiv:2502.17417.
4. Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013). *Dealing with the inventory risk*. Mathematics and Financial Economics, 7(4), 477–507.

---

## Course Information

**Module**: Market Microstructure
**Institution**: Imperial College London
**Term**: Term 2
