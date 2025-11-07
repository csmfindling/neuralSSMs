import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch


def generate_test_task(num_steps=5, cues=np.arange(0, 4), probas=None, variable_length=False):
    num_tasks, num_trials, _ = probas.shape    
    probas_reshaped = np.reshape(probas, (num_tasks * num_trials, 4))
    num_tasks_reshaped = num_tasks * num_trials
    H = np.random.rand(num_tasks_reshaped) < 0.5
    cues_seq = np.concatenate(
        [np.random.choice(cues, size=(1, int(num_steps)), p=(p if h else (1-p)) / sum(p if h else (1-p))) for (h, p) in zip(H, probas_reshaped)]
    )
    
    if variable_length:
        one_hot_array = OneHotEncoder(max_categories=num_steps - 1, sparse_output=False).fit_transform(
            np.random.randint(num_steps - 1, size=num_tasks_reshaped)[:, None]
        )
        is_masked = np.hstack((np.zeros(num_tasks_reshaped)[:, None], one_hot_array.cumsum(axis=1)))
    else:
        is_masked = np.zeros((num_tasks_reshaped, num_steps))

    logodd1 = (np.log(probas_reshaped[np.arange(num_tasks_reshaped)[:, None], cues_seq] / (1 - probas_reshaped[np.arange(len(cues_seq))[:, None], cues_seq])) * (is_masked == 0)).sum(axis=1)
    H1 = np.exp(logodd1) / (1 + np.exp(logodd1))
    #random_numb = np.random.rand(num_tasks_reshaped)
    rewards = 2. * np.array([~H, H]) - 1
    greedy = torch.from_numpy((H1 > 0.5) * 1)
    cues_masked = torch.from_numpy(cues_seq * (is_masked == 0) - 1 * (is_masked == 1))

    cues_masked_reshaped = torch.reshape(cues_masked, (num_tasks, num_trials, num_steps))
    rewards_reshaped = torch.reshape(torch.from_numpy(rewards), (2, num_tasks, num_trials))
    greedy_reshaped = torch.reshape(greedy, (num_tasks, num_trials))

    return rewards_reshaped, cues_masked_reshaped, greedy_reshaped

class probabilistic_task:
    def __init__(self):
        self.probas = None
        self.greedy = None
        self.context = None
        self.probabilistic_rewards = None
        self.num_trials = None

    # resets the tasks
    def reset(self, num_tasks=100, num_trials=200, num_steps=3, reset_probas=True, variable_length=True, tau=None):
        if not reset_probas and self.probas is None:
            raise ValueError("Probas not set")
        if reset_probas:
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
        noisy_rew, context, self.greedy = generate_test_task(num_steps=num_steps, cues=np.arange(0, 4), probas=self.probas, variable_length=variable_length)  # if Astar
        self.probabilistic_rewards = noisy_rew
        self.context = context
        self.num_trials = num_trials
        self.num_steps = num_steps
        self.num_tasks = num_tasks

    # return the rewards the `num_tasks` tasks given actions `actions`
    def pullArm(self, actions):
        return self.probabilistic_rewards[np.array(actions, dtype=np.int), range(len(actions))]


if __name__ == "__main__":
    env = probabilistic_task()
    env.reset(num_tasks=100, num_trials=200, num_steps=3, reset_probas=True)
    print(env.probas)
    print(env.probabilistic_rewards)
    print(env.context)
    print(env.greedy)