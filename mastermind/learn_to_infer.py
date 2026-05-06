import numpy as np
import torch
import sys
from task import Mastermind
from torch.utils import tensorboard
import os
from func_utils import compute_volatility, compute_emission_mastermind


class Worker(torch.nn.Module):
    def __init__(
            self, game, model_path, model_name, num_units=32, init_type="xavier", optimizer="Adam", episode_count_max=5e4,
            rnn_type="GRU", input_type='logodds'
        ):
        assert input_type in ['reward', 'logodds']
        super().__init__()
        self.model_path = model_path
        self.env = game
        self.episode_rewards = []
        self.init_type = init_type
        self.episode_count_max = episode_count_max
        self.input_type = input_type
        self.model_name = model_name + "_init_{0}_optim_{1}_episodeNbMax_{2}_numUnits_{3}_rnnType_{4}_inputType_{5}".format(
            self.init_type, optimizer, int(self.episode_count_max), num_units, rnn_type, input_type
        )
        self.summary_writer = tensorboard.SummaryWriter("results/source/trainings_fullRNN/" + str(self.model_name))
        self.nb_units = num_units

        self.gru_transition = torch.nn.GRU(input_size=1, hidden_size=self.nb_units, batch_first=True, bias=True)
        self.W_output_transition = torch.nn.Parameter(torch.zeros(self.nb_units, 1))
        self.initial_rnn_transition = torch.nn.Parameter(torch.ones(1, 1, self.nb_units) * -1)

        # emission RNN
        self.W_output_emission = torch.nn.Parameter(torch.zeros(self.nb_units, 1))
        self.gru_emission = torch.nn.GRU(input_size=2, hidden_size=self.nb_units, batch_first=True, bias=True)
        self.initial_rnn_emission = torch.nn.Parameter(torch.zeros(1, 1, self.nb_units))

        with torch.no_grad():
            # transition RNN
            for name, param in self.gru_transition.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
            self.W_output_transition[:] = 0 # make this a positive number so that volatility at time 0 is null

            # emission RNN
            for name, param in self.gru_emission.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
            torch.nn.init.xavier_uniform_(self.W_output_emission)
        
        self.optimizer = torch.optim.RMSprop(
            [self.W_output_transition, self.W_output_emission] +  list(self.gru_transition.parameters()) + list(self.gru_emission.parameters()), 
            lr=1e-3
        )
    
    def evaluate(self, rnn_state_transition=None, rnn_state_emission=None, update_state=True, use_ground_truth=False):
        """
        Evaluates the model by running forward passes and optionally returning rewards
        
        Args:
            num_trials: Number of parallel trials to run
            num_steps: Number of steps per sequence
        """
        nb_tasks = self.env.n_tasks
        # Initialize RNN state
        rnn_state_transition = rnn_state_transition if rnn_state_transition is not None else self.initial_rnn_transition.tile(1, nb_tasks, 1)
        rnn_state_emission = rnn_state_emission if rnn_state_emission is not None else self.initial_rnn_emission.tile(1, nb_tasks, 1)
        log_alphas = torch.ones([self.env.n_tasks, self.env.K]) * np.log(1./self.env.K)

        # pre-compute parameters
        params_transition = torch.zeros([self.env.n_tasks, self.env.n_trials])
        params_emission = torch.zeros([self.env.n_tasks, self.env.n_trials])
        log_predict_probs_all_trials = torch.ones([self.env.n_tasks, self.env.n_trials, self.env.K]) * np.log(1./self.env.K)
        all_actions = torch.zeros([self.env.n_tasks, self.env.n_trials])
        all_rewards = torch.zeros([self.env.n_tasks, self.env.n_trials])
        all_selected_combinations = torch.zeros([self.env.n_tasks, self.env.n_trials])

        # evaluate for each trial
        for i_trial in range(self.env.n_trials):
                    
            # compute volatility parameters
            if use_ground_truth:
                vol = torch.tensor(self.env.nu[:, i_trial])
                logvol, log_1_minus_vol = vol.log(), torch.log1p(-vol)
            else:
                logvol, log_1_minus_vol = compute_volatility(rnn_state_transition, self.W_output_transition, _return_exp=False)
            
            log_predict_probs = torch.stack([
                torch.logaddexp(
                    log_alphas[:, k] + log_1_minus_vol, 
                    torch.logsumexp(log_alphas[:, [j for j in range(self.env.K) if j != k]], dim=-1) + logvol - torch.log(torch.tensor(self.env.K - 1))
                    )
                for k in range(self.env.K)
            ], axis=-1) # p(z_t , y_{1:(t-1)}) = \sum_s p(z_t | z_{t-1}=s) • p(z_{t-1}=s, y_{1:(t-1)})

            if log_predict_probs.isnan().any() or rnn_state_transition.isnan().any() or self.W_output_transition.isnan().any():
                import ipdb; ipdb.set_trace()
            
            # select action
            selected_combination = log_predict_probs.argmax(dim=1)
            selected_mapping = self.env.state_space_mapping[selected_combination.cpu().numpy()]  # Fixed: convert tensor to numpy for indexing
            selected_action = selected_mapping[np.arange(self.env.n_tasks), self.env.stimulus[:, i_trial]]

            # pull arm and get reward
            reward = torch.from_numpy(
                (selected_action == self.env.correct_action[:, i_trial]) * self.env.feedback_when_correct[:, i_trial] +
                (selected_action != self.env.correct_action[:, i_trial]) * (1 - self.env.feedback_when_correct[:, i_trial])
            )

            # actions predicted by each combination
            predicted_actions = np.stack([self.env.state_space_mapping[i, self.env.stimulus[:, i_trial]] for i in range(self.env.K)], axis=0)

            # compute emission probabilities
            bernoulli_rate = compute_emission_mastermind(rnn_state_emission, self.W_output_emission)
            
            emission_probs = (
                torch.from_numpy(predicted_actions == selected_action[None]) * (bernoulli_rate * reward + (1 - bernoulli_rate) * (1 - reward)) +
                torch.from_numpy(predicted_actions != selected_action[None]) * (bernoulli_rate * (1 - reward) + (1 - bernoulli_rate) * reward)
            ).T

            # compute log alphas
            log_alphas = log_predict_probs + emission_probs.log() # p(z_t , y_{1:t}) = p(z_t , y_{1:(t-1)}) • p(y_t | z_t)
            log_predict_probs_all_trials[:, i_trial] = log_predict_probs

            # Update RNN states
            if update_state:
                # update transition RNN state
                input_state = (
                    emission_probs[torch.arange(self.env.n_tasks), selected_combination].log()
                ).float().detach()
                _, rnn_state_transition = self.gru_transition(input_state.unsqueeze(-1).unsqueeze(-1), rnn_state_transition)

                # update emission RNN state
                pfiltering = (log_predict_probs - torch.logsumexp(log_predict_probs, dim=-1, keepdims=True))

                input_state = torch.vstack(
                    (
                        pfiltering[torch.arange(self.env.n_tasks), selected_combination].unsqueeze(-1).T.exp(),
                        reward,
                    )
                ).float().detach()
                _, rnn_state_emission = self.gru_emission(input_state.T.unsqueeze(1), rnn_state_emission)

            # compute parameters            
            params_transition[:, i_trial] = compute_volatility(rnn_state_transition, self.W_output_transition, _return_exp=True)
            params_emission[:, i_trial] = compute_emission_mastermind(rnn_state_emission, self.W_output_emission)
            all_actions[:, i_trial] = torch.from_numpy(selected_action)
            all_rewards[:, i_trial] = reward
            all_selected_combinations[:, i_trial] = selected_combination
                
        return {
            'rnn_state_transition': rnn_state_transition,
            'rnn_state_emission': rnn_state_emission,
            'params_transition': params_transition,
            'params_emission': params_emission,
            'log_alphas': log_alphas,
            'actions': all_actions,
            'rewards': all_rewards,
            "log_predict_probs_all_trials": log_predict_probs_all_trials,
        }

    def load_model(self, nb_episodes=None):
        nb_episodes = nb_episodes if nb_episodes is not None else self.episode_count_max
        model_dir = f"{self.model_path}/{self.model_name}"
        model_file = f"{model_dir}/model-{int(nb_episodes)}.pth"
        self.load_state_dict(torch.load(model_file))

    def train(self, nb_tasks=200):
        """
        Main training/evaluation loop
        
        Args:
            num_trials: Number of parallel trials to run
            num_steps: Number of steps per sequence
        """
        episode_count = 0
        while episode_count <= self.episode_count_max:  # stopping criterion moved to loop condition
            # reset environment
            self.env._generate_task_schedule(nb_tasks=nb_tasks)

            # evaluate model
            result = self.evaluate()
            
            marginal_loss = -torch.logsumexp(result["log_alphas"], dim=1).mean()
            total_loss = marginal_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            correct = result['rewards'].float().mean()

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

                self.summary_writer.add_scalar(
                    "Train/marginal_loss",
                    float(marginal_loss.detach().numpy()),
                    episode_count
                )

                self.summary_writer.add_scalar(
                    "Train/NegLogLikelihood_Loss",
                    float(total_loss.detach().numpy()),
                    episode_count
                )

                self.summary_writer.flush()
            
            episode_count += 1


if __name__ == "__main__":
    try:
        index = int(sys.argv[1])
    except:
        index = 22

    #index = 1
    from learn_to_infer import Worker
    from task import Mastermind

    self = Worker(
        Mastermind(),
        "results/source/saved_models",
        "mastermindGRU_id{0}".format(index),
    )
    #self.train()

    self.load_model()
    nb_tasks = 1000
    nus = [0] * 500 + [0.08] * 500
    ffbs = [0.2] * 1000
    self.env._generate_task_schedule(nb_tasks=nb_tasks, nus=nus, ffbs=ffbs)

    result = self.evaluate(use_ground_truth=False)
    print(result['params_transition'][:, -1][:500].mean())
    print(result['params_transition'][:, -1][500:].mean())
    #
