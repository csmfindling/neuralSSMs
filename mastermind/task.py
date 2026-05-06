import numpy as np
from scipy.special import logit, expit as sigmoid
import torch
from scipy.stats import truncnorm
from scipy.optimize import brentq
import itertools

def truncated_exponential_logpdf(x, lambda_param, ub):
    """
    Compute the logpdf of an exponential distribution with rate lambda_param,
    truncated to [0, ub].

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the logpdf. Must be in [0, ub].
    lambda_param : float
        Rate parameter (lambda > 0) of the exponential.
    ub : float
        Upper bound of the truncation (must be > 0).

    Returns
    -------
    logpdf : array_like
        The log-probability density at x.
    """
    x = np.asarray(x)
    # Set logpdf to -np.inf where x is outside [0, ub]
    logpdf = np.full_like(x, -np.inf, dtype=np.float64)
    if lambda_param <= 0 or (ub is not None and ub <= 0):
        raise ValueError("lambda_param and ub must be positive.")
    # Only compute logpdf for valid x in [0, ub]
    if ub is not None:
        Z = 1 - np.exp(-lambda_param * ub)  # normalization constant
        mask = (x >= 0) & (x <= ub)
    else:
        mask = (x >= 0)
        Z = 1
    logpdf[mask] = np.log(lambda_param) - lambda_param * x[mask] - np.log(Z)
    return logpdf

def sample_truncated_exponential(size=1, lambda_param=1.0, ub=1.0, u=None):
    """
    Sample from an exponential distribution with rate lambda_param,
    truncated to [0, ub].

    Parameters
    ----------
    size : int or tuple of ints
        Number of samples or shape of the returned samples.
    lambda_param : float
        Rate parameter (lambda > 0).
    ub : float
        Upper bound (must be > 0).

    Returns
    -------
    samples : np.ndarray
        Samples from the truncated exponential distribution.
    """
    if lambda_param <= 0 or (ub is not None and ub <= 0):
        raise ValueError("lambda_param and ub must be positive.")
    size = size if isinstance(size, tuple) else (size,)
    if ub is None:
        return np.random.exponential(1./lambda_param, size)
    # Inverse transform sampling for truncated exponential
    if u is None:
        u = np.random.uniform(0, 1, size)
    Z = 1 - np.exp(-lambda_param * ub)
    samples = -np.log(1 - u * Z) / lambda_param
    return samples


def false_positive_rate_distribution(lambda_param=2):
    while True:
        false_positive_rate = np.random.exponential(1./lambda_param)
        if false_positive_rate < 0.4:
            break
    return false_positive_rate

def volatility_distribution(lambda_param=10):
    while True:
        nu = np.random.exponential(1./lambda_param)
        if nu < 0.2:
            break
    return nu


class Mastermind:
    def __init__(self, n_arms=4, n_trials=100):
        self.n_arms = n_arms
        self.n_trials = n_trials
        self.n_symbols = self.n_arms
        self.K = np.math.factorial(n_arms)
        self.state_space_mapping = np.array(list(itertools.permutations(np.arange(n_arms), n_arms)))

    def _generate_task_schedule(self, nb_tasks=100, nus=None, ffbs=None, correct_combinations=None):
        self.n_tasks = nb_tasks
        # 1. Generate the drifting switch probability 'nu' and false positive rate 'ffb'
        self.nu = np.zeros([nb_tasks, self.n_trials])
        self.ffb = np.zeros([nb_tasks, self.n_trials])        
        # 2. Generate the sequence of correct combinations
        self.correct_combination = np.zeros([nb_tasks, self.n_trials], dtype=int)
        self.agent_type = 'nSSM'
        # 3. Generate the sequence of stimuli
        self.stimulus = np.random.randint(0, self.n_symbols, size=(nb_tasks, self.n_trials))
        
        for i in range(nb_tasks):
            self.nu[i] = nus[i] if nus is not None else volatility_distribution()
            self.ffb[i] = ffbs[i] if ffbs is not None else false_positive_rate_distribution()

            if correct_combinations is None:
                self.correct_combination[i, 0] = np.random.randint(self.K)            
                for t in range(1, self.n_trials):
                    # Update correct arm
                    if np.random.rand() < self.nu[i, t-1]:  # Switch occurs
                        other_combinations = [combination for combination in range(self.K) if combination != self.correct_combination[i, t-1]]
                        self.correct_combination[i, t] = np.random.choice(other_combinations)
                    else:  # No switch
                        self.correct_combination[i, t] = self.correct_combination[i, t-1]
            else:
                self.correct_combination = correct_combinations
            
        self.correct_mapping = self.state_space_mapping[self.correct_combination]
        self.correct_action = np.stack([np.array([self.correct_mapping[i, j, self.stimulus[i, j]] for j in range(self.n_trials)]) for i in range(nb_tasks)], axis=0)
        self.feedback_when_correct = (np.random.rand(nb_tasks, self.n_trials) > self.ffb) * 1


if __name__ == "__main__":
    env = Mastermind(n_trials=200)
    env._generate_task_schedule(100)
    print(env.nu)