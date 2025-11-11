import numpy as np
from scipy.special import logit, expit as sigmoid
import torch
from scipy.stats import truncnorm
from scipy.optimize import brentq

def volatility_distribution(lambda_param=5):
    while True:
        nu = np.random.exponential(1./lambda_param)
        if nu < 0.4:
            break
    return nu

def false_positive_rate_distribution(lambda_param=5):
    while True:
        false_positive_rate = np.random.exponential(1./lambda_param) + 0.1
        if false_positive_rate < 0.4:
            break
    return false_positive_rate

def false_positive_rate(llrmax=None, return_stimulus_range=False):
    if llrmax is None:
        llrmax = 0.8 + np.random.rand() * (7 - 0.8)
    nlevel = 50      # Number of discrete stimulus levels
    # Generate stimulus value distribution
    level_list = np.arange(-nlevel, nlevel+1)/nlevel
    llr_list = level_list * llrmax
    p_gen = 1/(1 + np.exp(-llr_list))
    p_gen = p_gen/p_gen.sum()
    stimulus_range = np.round(np.arange(-1.0, 1.01, 0.01), 2)

    if return_stimulus_range:
        return stimulus_range, p_gen, llrmax
    else:
        return p_gen, llrmax

def gaussian_false_positive_rate(mu=None, false_positive_feedback=None, return_stimulus_range=False):
    """
    Truncated N(mu, sigma^2) on [-1, 1].
    Chooses sigma so that P(X < 0 | X ∈ [-1,1]) = false_positive_feedback.

    Returns either the grid and the (discrete) normalized probabilities over that grid,
    or just the probabilities.
    """
    if mu is None:
        mu = (np.random.rand() * 0.9 + 0.1) * 100
    if false_positive_feedback is None:
        false_positive_feedback = false_positive_rate_distribution()

    lo, hi = -1.0, 1.0
    p = float(false_positive_feedback)

    # Solve for sigma > 0 such that CDF_trunc(0) = p
    def cdf_at_zero_minus_p(sigma):
        a = (lo - mu) / sigma
        b = (hi - mu) / sigma
        return truncnorm.cdf(0.0, a, b, loc=mu, scale=sigma) - p

    # Find a bracket [s_lo, s_hi] with opposite signs, then root-find
    s_lo, s_hi = 1e-6, 1.0
    f_lo = cdf_at_zero_minus_p(s_lo)
    f_hi = cdf_at_zero_minus_p(s_hi)
    # expand until sign change or up to a large cap
    cap = 1e6
    while f_lo * f_hi > 0 and s_hi < cap:
        s_hi *= 2.0
        f_hi = cdf_at_zero_minus_p(s_hi)

    if f_lo * f_hi > 0:
        raise RuntimeError("Could not bracket a solution for sigma; "
                           "the requested (mu, false_positive_feedback) may be infeasible.")

    sigma = brentq(cdf_at_zero_minus_p, s_lo, s_hi)

    # Build discrete probabilities on a fixed grid and normalize (for a simple PMF approximation)
    stimulus_range = np.round(np.arange(-1.0, 1.01, 0.01), 2)
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    pdf_vals = truncnorm.pdf(stimulus_range, a, b, loc=mu, scale=sigma)

    p_gen = pdf_vals / pdf_vals.sum()

    if return_stimulus_range:
        return stimulus_range, p_gen, mu, false_positive_feedback
    else:
        return p_gen, mu, false_positive_feedback
    
class SwitchingBandit:
    def __init__(self, n_arms=2, n_trials=100, nb_tasks=100):
        self.n_arms = n_arms
        self.n_trials = n_trials
        self.trial = None
        self.reset(nb_tasks)

    def _generate_task_schedule(self, nb_tasks=100):
        # 1. Generate the drifting switch probability 'nu'
        self.nu = np.zeros([nb_tasks, self.n_trials])
        # 2. Generate the sequence of correct arms
        self.correct_arms = np.zeros([nb_tasks, self.n_trials], dtype=int)
        self.idx_arm0 = np.zeros([nb_tasks, self.n_trials], dtype=int)
        self.idx_arm1 = np.zeros([nb_tasks, self.n_trials], dtype=int)
        self.feedback_arm0 = np.zeros([nb_tasks, self.n_trials], dtype=float)
        self.feedback_arm1 = np.zeros([nb_tasks, self.n_trials], dtype=float)
        self.proba_emission_arm0 = torch.zeros([nb_tasks, self.n_trials], dtype=float)
        self.proba_emission_arm1 = torch.zeros([nb_tasks, self.n_trials], dtype=float)
        self.p_gen = np.zeros([nb_tasks, 201])
        self.mus = np.zeros([nb_tasks])
        self.false_positive_feedback = np.zeros([nb_tasks])
        stimulus_range = np.round(np.arange(-100, 101, 1), 2)
    
        for i in range(nb_tasks):
            self.nu[i] = volatility_distribution()
            self.correct_arms[i, 0] = np.random.randint(self.n_arms)

            for t in range(1, self.n_trials):
                # Update correct arm
                if np.random.rand() < self.nu[i, t-1]:  # Switch occurs
                    other_arms = [arm for arm in range(self.n_arms) if arm != self.correct_arms[i, t-1]]
                    self.correct_arms[i, t] = np.random.choice(other_arms)
                else:  # No switch
                    self.correct_arms[i, t] = self.correct_arms[i, t-1]

            self.p_gen[i], self.mus[i], self.false_positive_feedback[i] = gaussian_false_positive_rate()
            self.idx_arm0[i] = np.random.choice(np.arange(len(stimulus_range)), p=self.p_gen[i], size=[self.n_trials], replace=True)
            self.feedback_arm0[i] = stimulus_range[self.idx_arm0[i]]
            self.feedback_arm0[i][self.correct_arms[i].astype(bool)] *= -1
            self.feedback_arm1[i] = -self.feedback_arm0[i]

            self.idx_arm0[i][self.correct_arms[i].astype(bool)] = len(stimulus_range) - self.idx_arm0[i][self.correct_arms[i].astype(bool)] - 1
            self.idx_arm1[i] = len(stimulus_range) - self.idx_arm0[i] - 1

            self.proba_emission_arm0[i] = torch.from_numpy(self.p_gen[i][self.idx_arm0[i]])
            self.proba_emission_arm1[i] = torch.from_numpy(self.p_gen[i][self.idx_arm1[i]])


    def reset(self, nb_tasks):
        self.n_tasks = nb_tasks
        self._generate_task_schedule(nb_tasks)
        self.trial = 0

    def pullArm(self, arm_index):
        
        assert np.all(np.isin(arm_index, [0, 1]))

        reward = (arm_index == 0) * self.feedback_arm0[:, self.trial] + (arm_index == 1) * self.feedback_arm1[:, self.trial]        

        self.trial += 1
        
        return reward


if __name__ == "__main__":
    env = SwitchingBandit(n_trials=200)
    env.reset(100)
    print(env.nu)