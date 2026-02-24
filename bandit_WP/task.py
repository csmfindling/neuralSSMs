import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from bandit.task import volatility_distribution, gaussian_false_positive_rate, mu_distribution


class probabilistic_task:
    def __init__(self):
        self.probas = None
        self.greedy = None
        self.context = None
        self.probabilistic_rewards = None
        self.num_trials = None

    def generate_test_task(self, num_tasks=100, num_trials=200, num_steps=5, cues=np.arange(0, 4), probas=None, variable_length=False, taus=None, mus=None, ffs=None, nus=None, KO_WP=False):
        self.agent_type = 'nSSM'
                
        self.num_steps = num_steps
        self.num_tasks = num_tasks
        self.num_trials = num_trials

        self.correct_arms = np.zeros([num_tasks, num_trials], dtype=int)
        self.nu = np.zeros([num_tasks, num_trials])

        # sample emission probabilities        
        if probas is None:
            if taus is None:
                taus = np.random.choice([0.0, 0.01, 0.03, 0.06, 0.1], size=num_tasks)
            self.probas = np.zeros((num_tasks, num_trials, 4))
        else:
            self.probas = probas

        self.idx_arm0 = np.zeros([num_tasks, num_trials], dtype=int)
        self.idx_arm1 = np.zeros([num_tasks, num_trials], dtype=int)
        self.feedback_arm0 = np.zeros([num_tasks, num_trials], dtype=float)
        self.feedback_arm1 = np.zeros([num_tasks, num_trials], dtype=float)
        self.proba_emission_arm0 = torch.zeros([num_tasks, num_trials], dtype=float)
        self.proba_emission_arm1 = torch.zeros([num_tasks, num_trials], dtype=float)
        self.p_gen = np.zeros([num_tasks, 201])
        self.mus = np.zeros([num_tasks])
        self.sigmas = np.zeros([num_tasks])
        self.false_positive_feedback = np.zeros([num_tasks])
        stimulus_range = np.round(np.arange(-1.0, 1.01, 0.01), 2)

        # sample correct arms
        for i in range(num_tasks):
            self.nu[i] = nus[i] if nus is not None else volatility_distribution()
            self.correct_arms[i, 0] = np.random.randint(2)

            for t in range(1, num_trials):
                # Update correct arm
                if np.random.rand() < self.nu[i, t-1]:
                    other_arms = [arm for arm in range(2) if arm != self.correct_arms[i, t-1]]
                    self.correct_arms[i, t] = np.random.choice(other_arms)
                else:  # No switch
                    self.correct_arms[i, t] = self.correct_arms[i, t-1]

            if probas is None:
                if taus[i] > 0:
                    while True:
                        candidate_switches = np.array([np.random.geometric(taus[i]) for _ in range(100)]).cumsum()
                        if candidate_switches[-1] > num_trials:
                            break
                    switches = candidate_switches[candidate_switches < num_trials]
                    if len(switches) > 0:
                        if switches[-1] != num_trials:
                            switches = np.concatenate((switches, [num_trials]))
                        uniq_probas = np.array([np.random.permutation(np.arange(2, 10, 2) * 0.1) for _ in range(len(switches))])
                        nb_trials_per_block = np.concatenate((switches[:1], switches[1:] - switches[:-1]))
                        self.probas[i] = np.vstack([np.repeat(uniq_probas[k][None], nb_trials_per_block[k], axis=0) for k in range(len(uniq_probas))])
                if taus[i] == 0 or len(switches) == 0:
                    uniq_probas = np.random.permutation(np.arange(2, 10, 2) * 0.1)
                    self.probas[i] = np.repeat(uniq_probas[None], num_trials, axis=0)
                    
            self.p_gen[i], self.mus[i], self.sigmas[i], self.false_positive_feedback[i] = gaussian_false_positive_rate(
                false_positive_feedback=ffs[i] if ffs is not None else None,
                mu=mus[i] if mus is not None else None
            )
            self.idx_arm0[i] = np.random.choice(np.arange(len(stimulus_range)), p=self.p_gen[i], size=[num_trials], replace=True)
            self.feedback_arm0[i] = stimulus_range[self.idx_arm0[i]]
            self.feedback_arm0[i][self.correct_arms[i].astype(bool)] *= -1
            self.feedback_arm1[i] = -self.feedback_arm0[i]

            self.idx_arm0[i][self.correct_arms[i].astype(bool)] = len(stimulus_range) - self.idx_arm0[i][self.correct_arms[i].astype(bool)] - 1
            self.idx_arm1[i] = len(stimulus_range) - self.idx_arm0[i] - 1

            self.proba_emission_arm0[i] = torch.from_numpy(self.p_gen[i][self.idx_arm0[i]])
            self.proba_emission_arm1[i] = torch.from_numpy(self.p_gen[i][self.idx_arm1[i]])

        if not KO_WP:
            probas_reshaped = np.reshape(self.probas, (num_tasks * num_trials, 4))
            num_tasks_reshaped = num_tasks * num_trials        

            H = self.correct_arms.reshape(num_tasks * num_trials)
            cues_seq = np.concatenate(
                [np.random.choice(cues, size=(1, int(num_steps)), p=(p if h else (1-p)) / sum(p if h else (1-p))) for (h, p) in zip(H, probas_reshaped)]
            )
            
            if variable_length and num_steps > 5:
                nb_steps_increment = 5
                num_steps_mask = num_steps // nb_steps_increment
                one_hot_array = OneHotEncoder(categories=[np.arange(num_steps_mask * nb_steps_increment + 1)], sparse_output=False).fit_transform(
                    np.random.randint(num_steps_mask + 1, size=num_tasks_reshaped)[:, None] * nb_steps_increment
                )
                is_masked = one_hot_array.cumsum(axis=1)
            elif variable_length:
                one_hot_array = OneHotEncoder(max_categories=num_steps - 1, sparse_output=False).fit_transform(
                    np.random.randint(num_steps - 1, size=num_tasks_reshaped)[:, None]
                )
                is_masked = np.hstack((np.zeros(num_tasks_reshaped)[:, None], one_hot_array.cumsum(axis=1)))                
            else:
                is_masked = np.zeros((num_tasks_reshaped, num_steps))
            
            cues_masked = torch.from_numpy(cues_seq * (is_masked == 0) - 1 * (is_masked == 1))
            cues_masked_reshaped = torch.reshape(cues_masked, (num_tasks, num_trials, num_steps))

            self.correct_weather = torch.from_numpy(self.correct_arms)
            self.probabilistic_rewards = torch.from_numpy(self.feedback_arm0[None])
            self.context = cues_masked_reshaped


if __name__ == "__main__":
    self = probabilistic_task()
    self.generate_test_task(num_tasks=2, num_trials=200, num_steps=3)
    print(self.probas)
    print(self.probabilistic_rewards)
    print(self.context)
    print(self.greedy)

    cw = self.correct_weather.squeeze()
    pr = (self.probabilistic_rewards.squeeze() > 0).float()
