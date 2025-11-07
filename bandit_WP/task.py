import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch


class probabilistic_task:
    def __init__(self):
        self.probas = None
        self.greedy = None
        self.context = None
        self.probabilistic_rewards = None
        self.num_trials = None

    def generate_test_task(self, num_tasks=100, num_trials=200, num_steps=5, cues=np.arange(0, 4), probas=None, variable_length=False, tau=None):
        if probas is None:
            if tau is None:
                taus = np.random.choice([0.0, 0.01, 0.03, 0.06, 0.1], size=num_tasks)
            else:
                taus = np.repeat(tau, num_tasks)
            self.probas = np.zeros((num_tasks, num_trials, 4))
            for i in range(num_tasks):
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
        else:
            self.probas = probas

        num_tasks, num_trials, _ = self.probas.shape    
        probas_reshaped = np.reshape(self.probas, (num_tasks * num_trials, 4))
        num_tasks_reshaped = num_tasks * num_trials

        self.correct_arms = np.zeros([num_tasks, num_trials], dtype=int)
        nu = np.zeros([num_tasks, num_trials])

        for i in range(num_tasks):
            nu[i] = np.random.choice([0.1])
            self.correct_arms[i, 0] = np.random.randint(2)

            for t in range(1, num_trials):
                # Update correct arm
                if np.random.rand() < nu[i, t-1]:
                    other_arms = [arm for arm in range(2) if arm != self.correct_arms[i, t-1]]
                    self.correct_arms[i, t] = np.random.choice(other_arms)
                else:  # No switch
                    self.correct_arms[i, t] = self.correct_arms[i, t-1]
                        
        H = self.correct_arms.reshape(num_tasks * num_trials)
        cues_seq = np.concatenate(
            [np.random.choice(cues, size=(1, int(num_steps)), p=(p if h else (1-p)) / sum(p if h else (1-p))) for (h, p) in zip(H, probas_reshaped)]
        )
        
        if variable_length:
            if num_steps > 0:
                num_steps_mask = num_steps // 4
                one_hot_array = OneHotEncoder(categories=[np.arange(num_steps_mask * 4 + 1)], sparse_output=False).fit_transform(
                    np.random.randint(num_steps_mask + 1, size=num_tasks_reshaped)[:, None] * 4
                )
                is_masked = one_hot_array.cumsum(axis=1)
            else:
                is_masked = np.ones((num_tasks_reshaped, num_steps))
        else:
            is_masked = np.zeros((num_tasks_reshaped, num_steps))
        
        #rewards = 2. * np.array([~H, H]) - 1
        llrmax = 1.7954  # Maximum log likelihood ratio
        nlevel = 50      # Number of discrete stimulus levels

        # Generate stimulus value distribution
        level_list = np.arange(-nlevel, nlevel+1)/nlevel
        llr_list = level_list * llrmax
        self.p_gen = 1/(1 + np.exp(-llr_list))
        self.p_gen = self.p_gen/self.p_gen.sum()
        stimulus_range = np.round(np.arange(-1.0, 1.02, 0.02), 2)
        
        self.idx_arm0 = np.random.choice(np.arange(len(stimulus_range)), p=self.p_gen, size=[num_tasks, num_trials], replace=True)
        self.idx_arm0[self.correct_arms.astype(bool)] = len(stimulus_range) - self.idx_arm0[self.correct_arms.astype(bool)] - 1
        self.idx_arm1 = len(stimulus_range) - self.idx_arm0 - 1
        self.feedback_arm0 = stimulus_range[self.idx_arm0.astype(int)]
        self.feedback_arm1 = -self.feedback_arm0

        self.proba_emission_arm0 = torch.from_numpy(self.p_gen[self.idx_arm0])
        self.proba_emission_arm1 = torch.from_numpy(self.p_gen[self.idx_arm1])

        rewards_reshaped = np.vstack([self.feedback_arm0[None], self.feedback_arm1[None]])
        
        cues_masked = torch.from_numpy(cues_seq * (is_masked == 0) - 1 * (is_masked == 1))

        cues_masked_reshaped = torch.reshape(cues_masked, (num_tasks, num_trials, num_steps))
        nu_reshaped = torch.reshape(torch.from_numpy(nu), (num_tasks, num_trials))

        self.correct_weather = torch.from_numpy(self.correct_arms)
        self.probabilistic_rewards = torch.from_numpy(rewards_reshaped)
        self.context = cues_masked_reshaped
        self.num_trials = num_trials
        self.num_steps = num_steps
        self.num_tasks = num_tasks
    
        return rewards_reshaped, cues_masked_reshaped, nu_reshaped

if __name__ == "__main__":
    self = probabilistic_task()
    self.generate_test_task(num_tasks=2, num_trials=200, num_steps=3)
    print(self.probas)
    print(self.probabilistic_rewards)
    print(self.context)
    print(self.greedy)

    cw = self.correct_weather.squeeze()
    pr = (self.probabilistic_rewards.squeeze() > 0).float()
