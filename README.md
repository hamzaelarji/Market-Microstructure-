# Optimal Market Making

**Course**: MATH 70125 — Market Microstructure (Prof. Mathieu Rosenbaum)  
**Institution**: Imperial College London  
**Authors**: Hamza Boukraichi, Diane Murzi, Chedi Mnif

---

## Abstract

This project replicates and extends the optimal market-making framework of Guéant (2017). A market maker continuously posts bid and ask limit orders so as to maximise CARA utility over a finite horizon. The core mathematical object is a Hamilton–Jacobi–Bellman equation reduced to a system of coupled ODEs, solved numerically and compared to closed-form approximations. Extensions include multi-asset generalisation, Monte Carlo validation, empirical backtesting on LOBSTER equity data, Hawkes self-exciting fill dynamics, and a Soft Actor-Critic reinforcement learning agent.

**Primary reference**: Guéant, O. (2017). *Optimal market making*. Applied Mathematical Finance, 24(2), 112–154.

---

## Project Structure

All source modules and notebooks reside at the project root.

### Core Modules

| Module | Description |
|--------|-------------|
| `intensity.py` | Intensity model Λ(δ) = A·exp(−kδ) and Hamiltonian computation |
| `solver_1d.py` | Single-asset ODE solver (Model A linearisation, Model B Newton iteration) |
| `solver_2d.py` | Two-asset ODE solver with sparse Newton iteration on the 2D inventory grid |
| `closed_form.py` | Closed-form quote approximation (Guéant §4) |
| `hawkes.py` | Multivariate Hawkes process with softplus non-linearity |
| `calibrate.py` | MLE calibration of Λ(δ) = A·exp(−kδ) from trade/LOB data |
| `backtest.py` | Backtest engine: optimal, naïve, and closed-form strategies |
| `assets.py` | Calibrated parameters for IG and HY CDX indices (Guéant §6) |
| `scenarios.py` | Named experiment scenarios for numerical studies |

### Notebooks

| # | Notebook | Guéant sections | Description |
|---|----------|-----------------|-------------|
| 01 | `01_single_asset` | §3, §6.1–6.2 | Reproduces optimal bid/ask quotes for IG and HY CDX. Figures 1–5 (IG) and 10–14 (HY). |
| 02 | `02_closed_form` | §4 | ODE-exact vs closed-form approximation quality. Shows accuracy degrades near terminal time T and improves at lower volatility. |
| 03 | `03_model_a_vs_b` | §3 | Model A (ξ = γ, CARA) vs Model B (ξ = 0, mean-variance): spread magnitude, inventory skew, and strategy comparison. |
| 04 | `04_multi_asset` | §5 | Two correlated assets (IG + HY, ρ = 0.9) on a 2D inventory grid. |
| 05 | `05_monte_carlo` | Extension | Monte Carlo validation of ODE-derived strategy. P&L distributions, inventory mean-reversion, Sharpe ratio comparison. |
| 06 | `06_compare_hawkes_vs_poisson` | Extension | Hawkes vs Poisson fill processes: order-flow clustering, excitation dynamics, impact on quote placement. |
| 07 | `07_paper_extensions` | §4–5 | Non-exponential intensities, Riccati-based approximation, scaling beyond d = 2 assets. |
| 08 | `08_lobster_backtest` | Extension | Empirical backtest on LOBSTER L1 equity data (AAPL, SPY, TSLA). MLE calibration from real LOB, cross-asset Sharpe comparison. |
| 09 | `09_terminal_penalty` | §2–3 | Terminal inventory penalty sensitivity: linear vs quadratic ℓ(q), short-horizon regime analysis. |
| 10 | `10_rl_market_making` | Extension | Soft Actor-Critic agent on a Hawkes-extended environment. Benchmarked against ODE and closed-form strategies. |

---

## Installation

### Requirements

Python ≥ 3.10. Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `numpy`, `scipy`, `matplotlib`, `plotly`, `pandas`, `torch` (for NB10).

### Running

Notebooks are provided both as `.ipynb` and as equivalent `.py` scripts (Jupytext percent format). To run interactively:

```bash
jupyter lab
```

Or execute a script directly:

```bash
python 01_single_asset.py
```

### Data

- **Guéant §6 parameters**: built-in via `assets.py` (no external data required for NB01–07, 09).
- **LOBSTER data** (NB08): requires L1 message/orderbook files from [LOBSTER](https://lobsterdata.com/). Place files under a `data/` directory; see notebook header for expected filenames.
- **Hawkes / RL** (NB06, 10): synthetic data generated at runtime.

---

## References

1. Guéant, O. (2017). *Optimal market making*. Applied Mathematical Finance, 24(2), 112–154.
2. Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217–224.
3. Lalor, D. & Swishchuk, A. (2025). *Market making with Hawkes processes*. arXiv:2502.17417.
4. Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013). *Dealing with the inventory risk*. Mathematics and Financial Economics, 7(4), 477–507.