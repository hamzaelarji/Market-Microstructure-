"""Parameters from Section 6 of Optimal Market Making (Guéant 2017).

Model A : ξ = γ  (CARA utility → risk aversion on execution + price risk)
Model B : ξ = 0   (mean‐variance → only penalise inventory)

In both models, ξΔ = ξ · Δ is the product entering the Hamiltonian.
"""

IG = dict(
    sigma=5.83e-6,   # volatility of the upfront rate ($/√s)
    A=9.10e-4,       # base arrival rate of orders (1/s)
    k=1.79e4,        # decay of intensity with distance to mid (1/$)
    Delta=50e6,      # notional per trade ($)
    Q=4,             # max inventory in lots (integer)
)

HY = dict(
    sigma=2.15e-5,   # ($/√s)
    A=1.06e-3,       # (1/s)
    k=5.47e3,        # (1/$)
    Delta=10e6,      # ($)
    Q=4,             # lots (integer)
)

GAMMA = 6e-5         # risk‐aversion parameter (1/$)
RHO = 0.9            # correlation between IG and HY
T = 7200             # trading horizon (s)
