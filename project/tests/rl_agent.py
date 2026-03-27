"""Tabular Q-learning agent for market making.

Architecture (following Falces Marin et al. 2022, PLOS ONE):
  • State:  (inventory_bucket, time_bucket)
  • Action: γ_level ∈ {γ_low, γ_mid, γ_high}
  • Quotes: computed via closed-form approximation with chosen γ
  • Reward: ΔMtM − λ_penalty · |inventory|

The agent learns a state-dependent risk aversion γ(t, n),
then quotes are set by the analytical Guéant formulas with that γ.
This is more principled than learning quotes directly because
the CF formulas encode the correct structural relationships.
"""

import numpy as np
from market_making.core.closed_form import approx_quotes
from market_making.core.intensity import fill_prob


# ═══════════════════════════════════════════════════════════════
#  Environment
# ═══════════════════════════════════════════════════════════════

class MarketMakingEnv:
    """Gym-like environment for MM with configurable fill dynamics.

    fill_mode: "poisson" | "hawkes"
    """

    def __init__(self, params, T=3600.0, N_t=3600, fill_mode="poisson",
                 hawkes_cfg=None, inv_penalty_lambda=0.01, seed=None):
        self.params = params
        self.sigma = params["sigma"]
        self.A = params["A"]
        self.k = params["k"]
        self.Delta = params["Delta"]
        self.Q = int(params["Q"])
        self.T = T
        self.N_t = N_t
        self.dt = T / N_t
        self.fill_mode = fill_mode
        self.inv_penalty = inv_penalty_lambda
        self.rng = np.random.default_rng(seed)

        # Hawkes state
        self.hawkes_cfg = hawkes_cfg or {"beta": 10.0, "alpha_self": 2.0, "alpha_cross": 0.5}
        self.y_bid = 0.0
        self.y_ask = 0.0

        self.reset()

    def reset(self):
        self.S = 0.0
        self.X = 0.0
        self.n = 0
        self.t_idx = 0
        self.y_bid = 0.0
        self.y_ask = 0.0
        return self._state()

    def _state(self):
        return (self.n, self.t_idx)

    def step(self, gamma_action):
        """Execute one time step with the given γ.

        Returns (next_state, reward, done, info).
        """
        n_old = self.n
        mtm_old = self.X + self.n * self.Delta * self.S

        # Compute quotes with the agent's chosen γ
        xi = gamma_action
        try:
            db, da = approx_quotes(np.array([self.n]), self.params, gamma_action, xi=xi)
            db, da = float(db[0]), float(da[0])
        except Exception:
            db, da = 1.0 / self.k, 1.0 / self.k

        db = max(db, 1e-8)
        da = max(da, 1e-8)

        # Price update
        dW = self.rng.standard_normal()
        self.S += self.sigma * np.sqrt(self.dt) * dW

        # Fill simulation
        lam_b = self.A * np.exp(-self.k * db)
        lam_a = self.A * np.exp(-self.k * da)

        if self.fill_mode == "hawkes":
            from market_making.core.hawkes import softplus
            lam_b = float(softplus(lam_b + self.y_bid))
            lam_a = float(softplus(lam_a + self.y_ask))
            beta = self.hawkes_cfg["beta"]
            self.y_bid *= np.exp(-beta * self.dt)
            self.y_ask *= np.exp(-beta * self.dt)

        bid_fill = self.rng.random() < fill_prob(lam_b, self.dt)
        ask_fill = self.rng.random() < fill_prob(lam_a, self.dt)

        if bid_fill and self.n < self.Q:
            self.X -= (self.S - db) * self.Delta
            self.n += 1
        if ask_fill and self.n > -self.Q:
            self.X += (self.S + da) * self.Delta
            self.n -= 1

        if self.fill_mode == "hawkes":
            a_s = self.hawkes_cfg["alpha_self"]
            a_c = self.hawkes_cfg["alpha_cross"]
            if bid_fill:
                self.y_bid += a_s
                self.y_ask += a_c
            if ask_fill:
                self.y_ask += a_s
                self.y_bid += a_c

        self.t_idx += 1
        mtm_new = self.X + self.n * self.Delta * self.S

        # Reward: MtM change minus inventory penalty
        reward = (mtm_new - mtm_old) - self.inv_penalty * abs(self.n) * self.Delta * self.sigma * np.sqrt(self.dt)

        done = self.t_idx >= self.N_t
        return self._state(), float(reward), done, {"bid_fill": bid_fill, "ask_fill": ask_fill}


# ═══════════════════════════════════════════════════════════════
#  Q-Learning Agent
# ═══════════════════════════════════════════════════════════════

class QLearningAgent:
    """Tabular Q-learning agent.

    State is discretised: (inv_bucket, time_bucket).
    Actions are γ levels.
    """

    def __init__(self, Q_max=4, n_time_buckets=10, gamma_levels=None,
                 lr=0.1, gamma_discount=0.99, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=0.995):
        self.Q_max = Q_max
        self.n_inv = 2 * Q_max + 1
        self.n_time = n_time_buckets
        self.gamma_levels = gamma_levels or [0.001, 0.01, 0.05]
        self.n_actions = len(self.gamma_levels)

        # Q-table: (inv_bucket, time_bucket, action) → value
        self.q_table = np.zeros((self.n_inv, self.n_time, self.n_actions))

        self.lr = lr
        self.gamma_discount = gamma_discount
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def _discretise(self, state, N_t):
        n, t_idx = state
        inv_bucket = int(np.clip(n + self.Q_max, 0, self.n_inv - 1))
        time_bucket = int(np.clip(t_idx * self.n_time / max(N_t, 1), 0, self.n_time - 1))
        return inv_bucket, time_bucket

    def choose_action(self, state, N_t):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        ib, tb = self._discretise(state, N_t)
        return int(np.argmax(self.q_table[ib, tb]))

    def update(self, state, action, reward, next_state, done, N_t):
        ib, tb = self._discretise(state, N_t)
        nib, ntb = self._discretise(next_state, N_t)

        if done:
            target = reward
        else:
            target = reward + self.gamma_discount * np.max(self.q_table[nib, ntb])

        self.q_table[ib, tb, action] += self.lr * (target - self.q_table[ib, tb, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_policy_map(self):
        """Return the greedy policy as a 2D array (inv × time) → best γ index."""
        return np.argmax(self.q_table, axis=2)

    def get_gamma_map(self):
        """Return the greedy γ values as a 2D array (inv × time)."""
        policy = self.get_policy_map()
        return np.array(self.gamma_levels)[policy]


# ═══════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════

def train_agent(params, n_episodes=500, T=3600.0, N_t=360,
                fill_mode="poisson", hawkes_cfg=None,
                gamma_levels=None, seed=42):
    """Train Q-learning agent and return trained agent + episode rewards.

    Uses coarse time steps (N_t=360 → 10s per step) for training speed.
    """
    Q_max = int(params["Q"])
    agent = QLearningAgent(Q_max=Q_max, gamma_levels=gamma_levels or [0.002, 0.01, 0.05])

    episode_rewards = []
    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        env = MarketMakingEnv(params, T=T, N_t=N_t, fill_mode=fill_mode,
                              hawkes_cfg=hawkes_cfg, seed=int(rng.integers(1e9)))
        state = env.reset()
        total_reward = 0.0

        while True:
            action_idx = agent.choose_action(state, N_t)
            gamma_val = agent.gamma_levels[action_idx]
            next_state, reward, done, _ = env.step(gamma_val)
            agent.update(state, action_idx, reward, next_state, done, N_t)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

    return agent, np.array(episode_rewards)


# ═══════════════════════════════════════════════════════════════
#  Benchmark
# ═══════════════════════════════════════════════════════════════

def benchmark_agent(agent, params, gamma_fixed, N_test=500, T=3600.0, N_t=360,
                    fill_mode="poisson", hawkes_cfg=None, seed=123):
    """Benchmark RL agent vs fixed-γ strategies.

    Returns dict of {strategy_name: {"pnl": array, "sharpe": float, ...}}.
    """
    results = {}
    rng = np.random.default_rng(seed)

    for name, get_gamma in [
        ("RL Agent", lambda state, N_t_: agent.gamma_levels[agent.choose_action(state, N_t_)]),
        ("ODE Optimal", lambda state, N_t_: gamma_fixed),
        ("Naive", None),
    ]:
        pnls = []
        for _ in range(N_test):
            env = MarketMakingEnv(params, T=T, N_t=N_t, fill_mode=fill_mode,
                                  hawkes_cfg=hawkes_cfg, seed=int(rng.integers(1e9)))
            state = env.reset()

            if name == "Naive":
                # Symmetric spread, no inventory management
                gamma_naive = gamma_fixed
                xi = gamma_naive
                half_spread_val = approx_quotes(np.array([0]), params, gamma_naive, xi=xi)
                half_s = float(half_spread_val[0][0])

            total_reward = 0.0
            while True:
                if name == "Naive":
                    gamma_val = gamma_fixed
                else:
                    gamma_val = get_gamma(state, N_t)

                next_state, reward, done, _ = env.step(gamma_val)
                state = next_state
                total_reward += reward
                if done:
                    break

            pnls.append(env.X + env.n * env.Delta * env.S)

        pnls = np.array(pnls)
        std = float(np.std(pnls))
        results[name] = {
            "pnl": pnls,
            "mean_pnl": float(np.mean(pnls)),
            "std_pnl": std,
            "sharpe": float(np.mean(pnls) / max(std, 1e-12)),
        }

    return results
