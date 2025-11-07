import numpy as np
import torch
import sys
from task import SwitchingBandit
from torch.utils import tensorboard
import os
from func_utils import compute_volatility, compute_emission

def get_slope(data):
    """
    Calculates the slope of a linear regression with an intercept, but only returns the slope.

    Args:
        data (torch.Tensor): A 3D tensor of shape (batch, time, features).

    Returns:
        torch.Tensor: A 2D tensor of shape (batch, time) containing the slopes.
    """
    n_features = data.shape[-1]
    x = torch.arange(n_features, device=data.device, dtype=data.dtype)
    sum_x = torch.sum(x)
    sum_y = torch.sum(data, dim=-1)
    sum_x_squared = torch.sum(x ** 2)    
    x_reshaped = x.view(1, 1, -1)
    sum_xy = torch.sum(data * x_reshaped, dim=-1)
    denominator = n_features * sum_x_squared - sum_x ** 2
    numerator = n_features * sum_xy - sum_x * sum_y

    return numerator / denominator

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

        # transition RNN
        if rnn_type == "RNN":
            self.gru_transition = torch.nn.RNN(input_size=1, hidden_size=self.nb_units, batch_first=True, bias=True)
        else:
            self.gru_transition = torch.nn.GRU(input_size=1, hidden_size=self.nb_units, batch_first=True, bias=True)
        self.W_output_transition = torch.nn.Parameter(torch.zeros(self.nb_units, 1))
        self.initial_rnn_transition = torch.nn.Parameter(torch.zeros(1, 1, self.nb_units))

        # emission RNN
        self.W_output_emission = torch.nn.Parameter(torch.zeros(self.nb_units, 101))
        self.gru_emission = torch.nn.GRU(input_size=3, hidden_size=self.nb_units, batch_first=True, bias=True)
        self.initial_rnn_emission = torch.nn.Parameter(torch.zeros(1, 1, self.nb_units))

        with torch.no_grad():
            # transition RNN
            for name, param in self.gru_transition.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
            torch.nn.init.xavier_uniform_(self.W_output_transition)

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

        # pre-compute parameters
        params_transition = torch.zeros([self.env.n_tasks, self.env.n_trials])
        params_emission = torch.zeros([self.env.n_tasks, self.env.n_trials, 101])
        log_alphas = torch.ones([self.env.n_tasks, self.env.n_arms]) * np.log(0.5)
        all_actions = torch.zeros([self.env.n_tasks, self.env.n_trials])
        all_rewards = torch.zeros([self.env.n_tasks, self.env.n_trials])

        # evaluate for each trial
        for i_trial in range(self.env.n_trials):
            # compute volatility parameters
            if use_ground_truth:
                vol = torch.clamp(torch.tensor([self.env.nu[:, i_trial]])[None, None].float(), 1e-7, 1 - 1e-7)
                logvol, log_1_minus_vol = vol.log(), torch.log1p(-vol)
            else:
                logvol, log_1_minus_vol = compute_volatility(rnn_state_transition, self.W_output_transition, _return_exp=False)

            log_predict_probs = torch.stack([
                torch.logaddexp(log_alphas[:, 0] + log_1_minus_vol, log_alphas[:, 1] + logvol),
                torch.logaddexp(log_alphas[:, 1] + log_1_minus_vol, log_alphas[:, 0] + logvol)
            ]).squeeze().T

            if log_predict_probs.isnan().any() or rnn_state_transition.isnan().any() or self.W_output_transition.isnan().any():
                import ipdb; ipdb.set_trace()

            # select action
            selected_action = (
                (log_predict_probs[:, 0] == log_predict_probs[:, 1]) * torch.randint(high=2, size=(nb_tasks,)) + 
                (log_predict_probs[:, 0] != log_predict_probs[:, 1]) * log_predict_probs.argmax(dim=1)
            )

            # pull arm and get reward
            reward = self.env.pullArm(selected_action)

            # compute emission probabilities
            p_gen = compute_emission(rnn_state_emission, self.W_output_emission)
            proba_emission_arm0 = p_gen[:, torch.arange(self.env.n_tasks), self.env.idx_arm0[:, i_trial]]
            proba_emission_arm1 = p_gen[:, torch.arange(self.env.n_tasks), self.env.idx_arm1[:, i_trial]]
            emission_probs = torch.stack([proba_emission_arm0, proba_emission_arm1]).squeeze().T

            # compute log alphas
            log_alphas = emission_probs.log() + log_predict_probs

            # Update RNN states
            if update_state:
                # update transition RNN state
                input_state = (
                    emission_probs[torch.arange(self.env.n_tasks), selected_action].log() -  emission_probs[torch.arange(self.env.n_tasks), 1 - selected_action].log()
                ).float().detach()
                _, rnn_state_transition = self.gru_transition(input_state.unsqueeze(-1).unsqueeze(-1), rnn_state_transition)

                # update emission RNN state
                log_pfiltering = log_alphas - torch.logsumexp(log_alphas, dim=-1, keepdims=True)
                input_state = torch.vstack(
                    (
                        log_pfiltering.T,
                        torch.from_numpy(self.env.feedback_arm0[:, i_trial]).unsqueeze(0),
                    )
                ).float().detach()
                _, rnn_state_emission = self.gru_emission(input_state.T.unsqueeze(1), rnn_state_emission)

            # compute parameters            
            params_transition[:, i_trial] = compute_volatility(rnn_state_transition, self.W_output_transition, _return_exp=True)
            params_emission[:, i_trial] = compute_emission(rnn_state_emission, self.W_output_emission)
            all_actions[:, i_trial] = selected_action
            all_rewards[:, i_trial] = reward
                
        return {
            'rnn_state_transition': rnn_state_transition,
            'rnn_state_emission': rnn_state_emission,
            'params_transition': params_transition,
            'params_emission': params_emission,
            'log_alphas': log_alphas,
            'actions': all_actions,
            'rewards': all_rewards,
        }

    def load_model(self, nb_episodes=None):
        nb_episodes = nb_episodes if nb_episodes is not None else self.episode_count_max
        model_dir = f"{self.model_path}/{self.model_name}"
        model_file = f"{model_dir}/model-{int(nb_episodes)}.pth"
        self.load_state_dict(torch.load(model_file))

    def train(self, num_trials=100):
        """
        Main training/evaluation loop
        
        Args:
            num_trials: Number of parallel trials to run
            num_steps: Number of steps per sequence
        """
        episode_count = 0
        while episode_count <= self.episode_count_max:  # stopping criterion moved to loop condition
            # reset environment
            self.env.reset(nb_tasks=num_trials)

            # evaluate model
            result = self.evaluate()
            slopes = get_slope(result["params_emission"]).mean(axis=1)
            slope_loss = torch.relu(-slopes).sum() * 1e4
            marginal_loss = -torch.logsumexp(result["log_alphas"], dim=1).mean()
            total_loss = marginal_loss + slope_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            correct = (result['rewards'] == 1).float().mean()

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
                    "Train/Slope_Loss",
                    float(slope_loss.detach().numpy()),
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
        index = 12

    np.random.seed(index)
    torch.manual_seed(index)

    self = Worker(
        SwitchingBandit(n_trials=200),
        "results/source/saved_models",
        "banditGRU_id{0}".format(index),
    )
    self.train()
    self.load_model()
    self.env.reset(nb_tasks=10)
    result = self.evaluate(use_ground_truth=False)
