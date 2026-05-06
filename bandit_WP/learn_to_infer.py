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
            self, game, model_path, model_name, num_units=32, episode_count_max=5e4, train_from_scratch=False, entropy_reg=None
        ):
        super().__init__()
        self.model_path = model_path
        self.episode_count_max = episode_count_max
        self.model_name = model_name + "_episodeNbMax_{0}_numUnits_{1}_trainFromScratch_{2}".format(
            int(self.episode_count_max), num_units, train_from_scratch
        )
        self.entropy_reg = entropy_reg
        if self.entropy_reg is not None:
            self.model_name = self.model_name + "_policyReg_{0}".format(str(entropy_reg).replace(".", "_"))
        self.env = game
        self.nb_units = num_units        
        # association RNN
        self.gru_association = torch.nn.GRU(input_size=5, hidden_size=self.nb_units, batch_first=True, bias=True)
        self.W_output_association = torch.nn.Parameter(torch.zeros(self.nb_units, 4))
        # transition RNN
        self.gru_transition = torch.nn.GRU(input_size=1, hidden_size=self.nb_units, batch_first=True, bias=True)
        self.W_output_transition = torch.nn.Parameter(torch.zeros(self.nb_units, 1))
        self.initial_rnn_transition = torch.ones(1, 1, self.nb_units) * -1
        # emission RNN
        self.gru_emission = torch.nn.GRU(input_size=2, hidden_size=self.nb_units, batch_first=True, bias=True)
        self.W_output_emission = torch.nn.Parameter(torch.zeros(self.nb_units, 201))
        self.initial_rnn_emission = torch.zeros(1, 1, self.nb_units)

        self.train_from_scratch = train_from_scratch
        if self.train_from_scratch:
            self.summary_writer = tensorboard.SummaryWriter("results/source/trainings_fullRNN/" + str(self.model_name))
            with torch.no_grad():
                for name, param in self.gru_association.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param)
                torch.nn.init.xavier_uniform_(self.W_output_association)
                for name, param in self.gru_transition.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param)
                torch.nn.init.xavier_uniform_(self.W_output_transition)
                for name, param in self.gru_emission.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param)
                torch.nn.init.xavier_uniform_(self.W_output_emission)
            self.optimizer = torch.optim.RMSprop(
                list(self.gru_association.parameters()) + list(self.gru_transition.parameters()) + list(self.gru_emission.parameters()) + 
                [self.W_output_association, self.W_output_transition, self.W_output_emission], lr=1e-3
            )

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
        rnn_state_transition = rnn_state_transition if rnn_state_transition is not None else self.initial_rnn_transition.tile(1, nb_tasks, 1)
        rnn_state_emission = rnn_state_emission if rnn_state_emission is not None else self.initial_rnn_emission.tile(1, nb_tasks, 1)

        # pre-compute parameters
        all_probas_association = torch.zeros([self.env.num_tasks, self.env.num_trials, 4])
        all_probas_transition = torch.zeros([self.env.num_tasks, self.env.num_trials])
        all_probas_emission = torch.zeros([self.env.num_tasks, self.env.num_trials, 201])        
        all_logpredict_total = torch.zeros([self.env.num_tasks, self.env.num_trials, 2])

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
            #import ipdb; ipdb.set_trace()

            if logpredict_total.isnan().any() or rnn_state_association.isnan().any() or self.W_output_association.isnan().any() or self.W_output_transition.isnan().any() or self.W_output_emission.isnan().any():
                import ipdb; ipdb.set_trace()
                
            selected_action = (
                (logpredict_total[:, 0] == logpredict_total[:, 1]) * torch.randint(high=2, size=(nb_tasks,)) + 
                (logpredict_total[:, 0] != logpredict_total[:, 1]) * logpredict_total.argmax(dim=1)
            )

            all_actions[:, i_trial] = selected_action
            all_outcomes[:, i_trial] = torch.from_numpy(self.env.feedback_arm0[:, i_trial]) * (1 - 2 * selected_action)

            # compute emission probabilities
            p_gen = compute_emission(rnn_state_emission, self.W_output_emission)
            proba_emission_arm0 = p_gen[:, torch.arange(self.env.num_tasks), self.env.idx_arm0[:, i_trial]]
            proba_emission_arm1 = p_gen[:, torch.arange(self.env.num_tasks), self.env.idx_arm1[:, i_trial]]
            emission_probs = torch.stack([proba_emission_arm0, proba_emission_arm1]).squeeze().T

            if self.train_from_scratch:
                log_alphas = logpredict_total + emission_probs.log() # p(z_t, c_t, y_{1:t}) = p(y_t | z_t) • p(z_t, c_t, y_{1:(t-1)})
            else:
                log_alphas = logpredict_transition + emission_probs.log() # p(z_t, c_t, y_{1:t}) = p(y_t | z_t) • p(z_t, c_t, y_{1:(t-1)})

            # Update RNN state
            if update_state:
                logpredict_pfiltering = logpredict_transition if not KO_transition else logpredict_association
                pfiltering = (logpredict_pfiltering - torch.logsumexp(logpredict_pfiltering, dim=-1, keepdims=True))
                selected_pfiltering = ( 
                    (pfiltering[:, 0] == pfiltering[:, 1]) * torch.randint(high=2, size=(nb_tasks,)) + 
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
                if not KO_WP:
                    p_gen = compute_emission(rnn_state_emission, self.W_output_emission)
                    outcomes = torch.tanh(torch.tensor([-p_gen[:, i_k, k].log() + p_gen[:, i_k, -k].log() for i_k, k in enumerate(self.env.idx_arm0[:, i_trial])])).detach()            
                    input_state = torch.vstack(
                        (                        
                            torch.vstack([outcomes]),
                            torch.vstack([(contexts[:, i_trial] == k).sum(dim=1) for k in range(4)])
                        )
                    ).float()
                    _, rnn_state_association = self.gru_association(input_state.T.unsqueeze(1), rnn_state_association)
        
                # update transition RNN state
                if not KO_transition:
                    input_state = (
                        emission_probs[torch.arange(self.env.num_tasks), selected_pfiltering].log() -  emission_probs[torch.arange(self.env.num_tasks), 1 - selected_pfiltering].log()
                    ).float().detach()
                    _, rnn_state_transition = self.gru_transition(input_state.unsqueeze(-1).unsqueeze(-1), rnn_state_transition)

            # compute parameters
            all_probas_association[:, i_trial] = torch.sigmoid(compute_association(rnn_state_association, self.W_output_association)).squeeze()
            all_probas_transition[:, i_trial] = compute_volatility(rnn_state_transition, self.W_output_transition, _return_exp=True).detach()
            all_probas_emission[:, i_trial] = compute_emission(rnn_state_emission, self.W_output_emission)
            all_logpredict_total[:, i_trial] = logpredict_total

        return {
            'outcomes': all_outcomes,
            'selected_actions': all_actions,
            'greedy': self.env.greedy,
            'probas_association': all_probas_association,
            'probas_transition': all_probas_transition,
            'probas_emission': all_probas_emission,
            "probas": self.env.probas,
            "logalphas": log_alphas,
            "logpredicts_total": all_logpredict_total,
        }

    def load_model(self):
        if self.train_from_scratch:
            model_dir = f"{self.model_path}/{self.model_name}"
            model_file = f"{model_dir}/model-{int(self.episode_count_max)}.pth"
            self.load_state_dict(torch.load(model_file))
            print("loaded model")
        else:
            id_model = int(self.model_name.split('_')[3][2:])

            # association model        
            model_WP = f"/Users/csmfindling/Documents/Postdoc-Geneva/neuralHMMs/code/WP/results/source/saved_models/WP_GRU_agent{id_model - 1}_init_xavier_optim_Adam_episodeNbMax_50000_numUnits_32_trainWithEmission_False_trainInCatTaskFromScratch_False"
            emission_model_WP = torch.load(model_WP + "/model-50000.pth")

            # transition and emission model
            model_bandit = f"/Users/csmfindling/Documents/Postdoc-Geneva/neuralHMMs/code/bandit/results/source/saved_models/banditGRU_newinit_val_0_beta2_id{id_model}_init_xavier_optim_Adam_episodeNbMax_50000_numUnits_32_rnnType_GRU_inputType_logodds"
            transition_emission_model = torch.load(model_bandit + "/model-50000.pth")

            for (key, value) in self.named_parameters():
                if "association" in key:
                    value.data = emission_model_WP[key].data
                elif "transition" in key:
                    value.data = transition_emission_model[key].data
                elif "emission" in key:
                    value.data = transition_emission_model[key].data #transition_emission_model[key].data when trainWithEmission_True
                else:
                    raise ValueError(f"Key {key} not found in association, transition or emission model")


    def train(self, num_trials=500, num_steps=5):
        """
        Main training/evaluation loop
        
        Args:
            num_trials: Number of parallel trials to run
            num_steps: Number of steps per sequence
        """
        if not self.train_from_scratch:
            raise ValueError("Train from scratch must be True to train")
        
        episode_count = 0
        self.episode_rewards = []
        while episode_count <= self.episode_count_max:  # stopping criterion moved to loop condition
            # reset environment
            self.env.generate_test_task(num_tasks=10, num_trials=num_trials, num_steps=num_steps, variable_length=True)

            # evaluate model
            result = self.evaluate()
            marginal_loss = -torch.logsumexp(result["logalphas"], dim=-1).sum()
            slope_loss = torch.relu(result['probas_emission'][:, :, :100].sum(axis=-1) - 0.5).sum() * 1e6
            if self.entropy_reg is not None:
                logpredicts = result['logpredicts_total']
                logpredicts_norm = logpredicts - torch.logsumexp(logpredicts, dim=-1, keepdims=True)
                entropy_loss = -torch.sum(torch.exp(logpredicts_norm) * logpredicts_norm, dim=-1).sum()
                total_loss = marginal_loss + slope_loss - self.entropy_reg * entropy_loss
            else:
                total_loss = marginal_loss + slope_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
                
            correct = (result['selected_actions'] == self.env.correct_weather).float().mean()
            probas = self.env.probas.squeeze()
            categorical_probs = result['probas_association']
                
            self.episode_rewards.append(correct)

            # Periodic evaluation and logging
            if episode_count % 10 == 0 and episode_count != 0:
                
                # Save model checkpoint
                if episode_count % 500 == 0:
                    model_dir = f"{self.model_path}/{self.model_name}"
                    os.makedirs(model_dir, exist_ok=True)
                    model_file = f"{model_dir}/model-{episode_count}.pth"
                    torch.save(self.state_dict(), model_file)
                    print("Saved Model")
                
                # Log metrics
                mean_reward = np.mean(self.episode_rewards[-10:])
                
                self.summary_writer.add_scalar(
                    "Train/Reward_train_A", 
                    float(mean_reward), 
                    episode_count
                )

                if self.train_from_scratch and self.entropy_reg is not None:
                    self.summary_writer.add_scalar(
                        "Train/Entropy_Loss",
                        float(entropy_loss.detach().numpy()),
                        episode_count
                    )

                self.summary_writer.add_scalar(
                    "Train/NegLogLikelihood_Loss",
                    float(marginal_loss.detach().numpy()) / num_trials,
                    episode_count
                )

                self.summary_writer.add_scalar(
                    "Train/act_vs_infer_Proba",
                    abs(np.abs(probas - categorical_probs.detach().numpy()).mean()),
                    episode_count
                )

                self.summary_writer.add_scalar(
                    "Train/W_output_association_norm",
                    float(self.W_output_association.norm().detach().cpu().numpy()),
                    episode_count
                )

                self.summary_writer.flush()
            
            episode_count += 1


if __name__ == "__main__":
    try:
        index = int(sys.argv[1])
    except:
        index = 1

    entropy_regs = [0.1, 0.3, 0.5, 1, 2, 4]

    index_agent = index % 30
    index_reg = index // 30

    np.random.seed(index_agent)
    torch.manual_seed(index_agent)

    self = Worker(
        probabilistic_task(),
        "results/source/saved_models",
        "bandit_WP_GRU_agent{0}".format(index_agent),
        train_from_scratch=True,
        entropy_reg=entropy_regs[index_reg]
    )
    self.train()

    self.load_model()

    np.random.seed(2)
    self.env.generate_test_task(num_tasks=100, num_trials=500, num_steps=13, variable_length=True, nus=[0.2] * 100)
    result = self.evaluate()
    print("Without KO: ", (result['outcomes'].mean() + 1) / 2, (result['selected_actions'].numpy() == self.env.correct_arms).mean())
    result = self.evaluate(KO_transition=True)
    print("KO transition: ", (result['outcomes'].mean() + 1) / 2, (result['selected_actions'].numpy() == self.env.correct_arms).mean())
    result = self.evaluate(KO_WP=True)
    print("KO WP: ", (result['outcomes'].mean() + 1) / 2, (result['selected_actions'].numpy() == self.env.correct_arms).mean())