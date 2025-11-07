import numpy as np
from scipy.special import logit, expit as sigmoid
import torch

def volatility_distribution(lambda_param=10):
    while True:
        nu = np.random.exponential(1./lambda_param)
        if nu < 0.4:
            break
    return nu

def false_positive_rate(llrmax=None, return_stimulus_range=False):
    if llrmax is None:
        llrmax = 0.8 + np.random.rand() * (7 - 0.8)
    nlevel = 50      # Number of discrete stimulus levels
    # Generate stimulus value distribution
    level_list = np.arange(-nlevel, nlevel+1)/nlevel
    llr_list = level_list * llrmax
    p_gen = 1/(1 + np.exp(-llr_list))
    p_gen = p_gen/p_gen.sum()
    stimulus_range = np.round(np.arange(-1.0, 1.02, 0.02), 2)

    if return_stimulus_range:
        return stimulus_range, p_gen
    else:
        return p_gen

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
        self.feedback_arm0 = np.zeros([nb_tasks, self.n_trials], dtype=float)
        self.feedback_arm1 = np.zeros([nb_tasks, self.n_trials], dtype=float)
        self.p_gen = np.zeros([nb_tasks, 101])
        stimulus_range = np.round(np.arange(-1.0, 1.02, 0.02), 2)
    
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
                self.p_gen[i] = false_positive_rate()
                self.idx_arm0[i] = np.random.choice(np.arange(len(stimulus_range)), p=self.p_gen, size=[self.n_trials], replace=True)
                self.feedback_arm0[i] = stimulus_range[self.idx_arm0[i]]
                self.feedback_arm0[i][self.correct_arms[i].astype(bool)] *= -1
                self.feedback_arm1[i] = -self.feedback_arm0[i]


        nlevel = 50      # Number of discrete stimulus levels
        llrmax = 0.8 + np.random.rand() * (7 - 0.8)
        # Generate stimulus value distribution
        level_list = np.arange(-nlevel, nlevel+1)/nlevel
        
        llr_list = level_list * llrmax
        self.p_gen = 1/(1 + np.exp(-llr_list))
        self.p_gen = self.p_gen/self.p_gen.sum()
        
        self.idx_arm0 = np.random.choice(np.arange(len(stimulus_range)), p=self.p_gen, size=[nb_tasks, self.n_trials], replace=True)

        self.feedback_arm0 = stimulus_range[self.idx_arm0.astype(int)]
        self.feedback_arm0[self.correct_arms.astype(bool)] *= -1
        self.feedback_arm1 = -self.feedback_arm0

        self.idx_arm0[self.correct_arms.astype(bool)] = len(stimulus_range) - self.idx_arm0[self.correct_arms.astype(bool)] - 1
        self.idx_arm1 = len(stimulus_range) - self.idx_arm0 - 1

        self.proba_emission_arm0 = torch.from_numpy(self.p_gen[self.idx_arm0])
        self.proba_emission_arm1 = torch.from_numpy(self.p_gen[self.idx_arm1])

        import ipdb; ipdb.set_trace()

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