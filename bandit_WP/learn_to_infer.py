import numpy as np
import torch
import sys
from task import probabilistic_task
from torch.utils import tensorboard
from torch.distributions import Categorical
import os
from torch.nn.functional import logsigmoid
import glob
import re
from func_utils import compute_emission, compute_volatility, compute_association

class Worker(torch.nn.Module):
    def __init__(
            self, game, model_path, model_name, num_units=32, episode_count_max=5e4
        ):
        super().__init__()
        self.model_path = model_path
        self.episode_count_max = episode_count_max
        self.model_name = model_name + "_episodeNbMax_{0}_numUnits_{1}".format(
            int(self.episode_count_max), num_units
        )
        self.env = game
        self.nb_units = num_units
        # association RNN
        self.gru_association = torch.nn.GRU(input_size=5, hidden_size=self.nb_units, batch_first=True, bias=True)
        self.W_output_association = torch.nn.Parameter(torch.zeros(self.nb_units, 4))
        # transition RNN
        self.gru_transition = torch.nn.GRU(input_size=1, hidden_size=self.nb_units, batch_first=True, bias=True)
        self.W_output_transition = torch.nn.Parameter(torch.zeros(self.nb_units, 1))
        # emission RNN
        self.gru_emission = torch.nn.GRU(input_size=3, hidden_size=self.nb_units, batch_first=True, bias=True)
        self.W_output_emission = torch.nn.Parameter(torch.zeros(self.nb_units, 101))

    def evaluate(self, rnn_state_association=None, rnn_state_transition=None, rnn_state_emission=None, update_state=True, KO_transition=False, KO_WP=False):
        """
        Evaluates the model by running forward passes and optionally returning rewards
        
        Args:
            num_trials: Number of parallel trials to run
            num_steps: Number of steps per sequence
        """        
        if not update_state and (rnn_state_association is None or rnn_state_transition is None or rnn_state_emission is None):
            raise ValueError("RNN states must be provided if update_state is False")

        # process contexts
        contexts = self.env.context.to(torch.int64)
        nb_tasks = self.env.num_tasks

        # Initialize RNN state
        rnn_state_association = rnn_state_association if rnn_state_association is not None else torch.zeros(1, self.env.num_tasks, self.nb_units)
        rnn_state_transition = rnn_state_transition if rnn_state_transition is not None else torch.zeros(1, self.env.num_tasks, self.nb_units)
        rnn_state_emission = rnn_state_emission if rnn_state_emission is not None else torch.zeros(1, self.env.num_tasks, self.nb_units)

        # pre-compute parameters
        all_probas_association = torch.zeros([self.env.num_tasks, self.env.num_trials, 4])
        all_probas_transition = torch.zeros([self.env.num_tasks, self.env.num_trials])
        all_probas_emission = torch.zeros([self.env.num_tasks, self.env.num_trials, 101])        

        # pre-compute log alphas and log predicts
        log_alphas = torch.ones([self.env.num_tasks, 2]) * np.log(0.5)

        # pre-compute all actions and outcomes
        all_actions = torch.zeros([self.env.num_tasks, self.env.num_trials])
        all_outcomes = torch.zeros([self.env.num_tasks, self.env.num_trials])

        for i_trial in range(self.env.num_trials):
            # compute volatility parameters
            if KO_transition:
                logvol, log_1_minus_vol = compute_volatility(
                    torch.zeros_like(rnn_state_transition), torch.zeros_like(self.W_output_transition)
                )
            else:
                logvol, log_1_minus_vol = compute_volatility(rnn_state_transition, self.W_output_transition)

            # compute log predicts for transition
            logpredict_transition = torch.stack([
                torch.logaddexp(log_alphas[:, 0] + log_1_minus_vol, log_alphas[:, 1] + logvol),
                torch.logaddexp(log_alphas[:, 1] + log_1_minus_vol, log_alphas[:, 0] + logvol)
            ]).squeeze().T # p(z_t , y_{1:(t-1)}) = \sum_s p(z_t | z_{t-1}=s) • p(z_{t-1}=s, y_{1:(t-1)})
            
            # renormalize log predicts
            if not KO_WP:
                logits_association = compute_association(rnn_state_association, self.W_output_association)
                logpredict_1 = logsigmoid(logits_association)
                logpredict_0 = logsigmoid(-logits_association)

                # compute logpredicts
                logpredict_association = torch.stack([
                    (logpredict_0[np.arange(self.env.num_tasks)[:, None], contexts[:, i_trial]] * (contexts[:, i_trial] != -1)).sum(axis=-1),
                    (logpredict_1[np.arange(self.env.num_tasks)[:, None], contexts[:, i_trial]] * (contexts[:, i_trial] != -1)).sum(axis=-1)
                ]).squeeze().T # p(c_t | z_t)
            
            logpredict_total = logpredict_transition if KO_WP else logpredict_association + logpredict_transition
            # p(z_t, c_t, y_{1:(t-1)}) = p(c_t | z_t) • p(z_t , y_{1:(t-1)})

            if logpredict_total.isnan().any() or rnn_state_association.isnan().any() or self.W_output_association.isnan().any() or self.W_output_transition.isnan().any() or self.W_output_emission.isnan().any():
                import ipdb; ipdb.set_trace()
                
            selected_action = (
                (logpredict_total[:, 0] == logpredict_total[:, 1]) * torch.randint(high=2, size=(nb_tasks,)) + 
                (logpredict_total[:, 0] != logpredict_total[:, 1]) * logpredict_total.argmax(dim=1)
            )

            all_actions[:, i_trial] = selected_action
            all_outcomes[:, i_trial] = self.env.probabilistic_rewards[selected_action, np.arange(self.env.num_tasks), i_trial]

            # compute emission probabilities
            p_gen = compute_emission(rnn_state_emission, self.W_output_emission)
            proba_emission_arm0 = p_gen[:, torch.arange(self.env.num_tasks), self.env.idx_arm0[:, i_trial]]
            proba_emission_arm1 = p_gen[:, torch.arange(self.env.num_tasks), self.env.idx_arm1[:, i_trial]]
            emission_probs = torch.stack([proba_emission_arm0, proba_emission_arm1]).squeeze().T

            log_alphas = logpredict_total + emission_probs.log() # p(z_t, c_t, y_{1:t}) = p(y_t | z_t) • p(z_t, c_t, y_{1:(t-1)})

            # Update RNN state
            if update_state:
                # update emission RNN state
                pfiltering = (logpredict_total - torch.logsumexp(logpredict_total, dim=-1, keepdims=True))
                selected_pfiltering = (
                    (pfiltering[:, 0] == pfiltering[:, 1]) * torch.randint(high=2, size=(self.env.num_tasks,)) + 
                    (pfiltering[:, 0] != pfiltering[:, 1]) * pfiltering.argmax(dim=1)
                )            
                input_state = torch.vstack(
                    (
                        pfiltering[torch.arange(self.env.num_tasks), selected_pfiltering].unsqueeze(-1).T.exp(),
                        torch.from_numpy(self.env.feedback_arm0[:, i_trial]).unsqueeze(0) * (1 - 2 * selected_pfiltering),
                    )
                ).float().detach()
                _, rnn_state_emission = self.gru_emission(input_state.T.unsqueeze(1), rnn_state_emission)

                # update association RNN state
                p_gen = compute_emission(rnn_state_emission, self.W_output_emission)
                outcomes = (2 * torch.tensor([p_gen[:, i_k, :k].sum() for i_k, k in enumerate(self.env.idx_arm0[:, i_trial])]) - 1)
                input_state = torch.vstack(
                    (                        
                        torch.vstack([outcomes]),
                        torch.vstack([(contexts[:, i_trial] == k).sum(dim=1) for k in range(4)])
                    )
                ).float()
                _, rnn_state_association = self.gru_association(input_state.T.unsqueeze(1), rnn_state_association)

                # update transition RNN state
                input_state = (
                    emission_probs[torch.arange(self.env.n_tasks), selected_action].log() -  emission_probs[torch.arange(self.env.n_tasks), 1 - selected_action].log()
                ).float().detach()
                _, rnn_state_transition = self.gru_transition(input_state.unsqueeze(-1).unsqueeze(-1), rnn_state_transition)

            # compute parameters
            all_probas_association[:, i_trial] = torch.sigmoid(compute_association(rnn_state_association, self.W_output_association)).squeeze()
            all_probas_transition[:, i_trial] = compute_volatility(rnn_state_transition, self.W_output_transition, _return_exp=True).detach()
            all_probas_emission[:, i_trial] = compute_emission(rnn_state_emission, self.W_output_emission).detach()

        return {
            'outcomes': all_outcomes,
            'selected_actions': all_actions,
            'greedy': self.env.greedy,
            'probas_association': all_probas_association,
            'probas_transition': all_probas_transition,
            'probas_emission': all_probas_emission,
            "probas": self.env.probas,
        }

    def load_model(self, nb_episodes=None):
        shortened_model_name = '_'.join(self.model_name.split('_')[:3]) + '_'
        nb_episodes = nb_episodes if nb_episodes is not None else self.episode_count_max

        # association model
        id_WP = int(shortened_model_name.split('id')[-1].split('_')[0])
        shortened_model_name = "fullRNN_id{0}_".format(id_WP)
        model_pattern = re.compile(f"{shortened_model_name}")
        model_files = glob.glob(f"/Users/csmfindling/Documents/Postdoc-Geneva/neuralHMMs/code/WP/results/source/saved_models/WP_GRU*odeNbMax_{int(nb_episodes)}*_trainWithEmission_True*")
        path_emission_WP = next((file for file in model_files if model_pattern.search(file)), None)
        emission_model_WP = torch.load(path_emission_WP + "/model-" + str(int(2e4)) + ".pth")

        # transition and emission model
        id_bandit = int(shortened_model_name.split('id')[-1].split('_')[0])
        shortened_model_name = "fullRNN_id{0}_".format(id_bandit)
        model_pattern = re.compile(f"{shortened_model_name}")
        model_files = glob.glob(f"/Users/csmfindling/Documents/Postdoc-Geneva/neuralHMMs/code/bandit/results/source/saved_models/banditGRU*odeNbMax_{int(nb_episodes)}_*logodds*")
        path_transition = next((file for file in model_files if model_pattern.search(file)), None)
        transition_emission_model = torch.load(path_transition + "/model-" + str(int(5e4)) + ".pth")        

        for (key, value) in self.named_parameters():
            if "association" in key:
                value.data = emission_model_WP[key.replace('_association', '')].data
            elif "transition" in key:
                value.data = transition_emission_model[key.replace('_transition', '')].data
            elif "emission" in key:
                value.data = transition_emission_model[key.replace('_emission', '')].data
            else:
                raise ValueError(f"Key {key} not found in association, transition or emission model")

if __name__ == "__main__":
    try:
        index = int(sys.argv[1])
    except:
        index = 2

    np.random.seed(index)
    torch.manual_seed(index)

    self = Worker(
        probabilistic_task(),
        "results/source/saved_models",
        "rnn_pf_id{0}".format(index),
        optimizer="Adam",
        init_type="xavier",
        episode_count_max=5e4,
        num_units=32,
    )

    self.load_model()

    np.random.seed(2)
    _, _, _ = self.env.generate_test_task(num_tasks=1000, num_trials=500, num_steps=13, tau=0.01, variable_length=True)
    result = self.evaluate()
    print("Without KO: ", (result['rewarded'].mean() + 1) / 2, (result['actions'].numpy() == self.env.correct_arms).mean())
    result = self.evaluate(KO_transition=True)
    print("KO transition: ", (result['rewarded'].mean() + 1) / 2, (result['actions'].numpy() == self.env.correct_arms).mean())
    result = self.evaluate(KO_WP=True)
    print("KO WP: ", (result['rewarded'].mean() + 1) / 2, (result['actions'].numpy() == self.env.correct_arms).mean())