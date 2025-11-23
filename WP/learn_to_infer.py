import numpy as np
import torch
import sys
from task import probabilistic_task
from torch.utils import tensorboard
import os
from torch.nn.functional import logsigmoid
import glob
from func_utils import compute_emission, compute_association

class Worker(torch.nn.Module):
    def __init__(
            self, game, model_path, model_name, num_units=32, init_type="xavier", optimizer="Adam", episode_count_max=5e4, with_emission=False, train_with_emission=False
        ):
        super().__init__()
        if (not with_emission) and train_with_emission:
            raise ValueError("train_with_emission must be False if with_emission is False")        
        self.model_path = model_path
        self.env = game
        self.episode_rewards = []
        self.init_type = init_type
        self.episode_count_max = episode_count_max
        self.train_with_emission = train_with_emission
        self.model_name = model_name + "_init_{0}_optim_{1}_episodeNbMax_{2}_numUnits_{3}_trainWithEmission_{4}".format(
            self.init_type, optimizer, int(self.episode_count_max), num_units, self.train_with_emission
        )
        self.summary_writer = tensorboard.SummaryWriter("results/source/trainings_fullRNN/" + str(self.model_name))
        self.nb_units = num_units

        # association RNN
        self.W_output_association = torch.nn.Parameter(torch.zeros(self.nb_units, 4))
        self.gru_association = torch.nn.GRU(input_size=5, hidden_size=self.nb_units, batch_first=True, bias=True)

        # emission RNN
        self.with_emission = with_emission
        if with_emission or train_with_emission:
            # load emission model
            model_id = 121
            #model_id = int(model_name.split('id')[-1])

            # initialize emission RNN
            self.gru_emission = torch.nn.GRU(input_size=2, hidden_size=self.nb_units, batch_first=True, bias=True)
            self.W_output_emission = torch.nn.Parameter(torch.zeros(self.nb_units, 201))

            path_to_emission_networks = "../bandit/results/source/saved_models"
            name_of_emission_network = f"banditGRU_id{model_id}_init_xavier_optim_Adam_episodeNbMax_50000_numUnits_32_rnnType_GRU_inputType_logodds"
            files_to_load = sorted(glob.glob(path_to_emission_networks + "/" + name_of_emission_network + "/*"))            
            number_of_iterations_of_emission_model = np.max([int(f.split('-')[-1].split('.')[0]) for f in files_to_load])
            idx_of_emission_model = np.argmax([int(f.split('-')[-1].split('.')[0]) for f in files_to_load])
            print(f"loading the emission model with {number_of_iterations_of_emission_model} iterations")
            state_dict_emission = torch.load(files_to_load[idx_of_emission_model])
            for (key, val) in self.named_parameters():
                if 'emission' in key:
                    with torch.no_grad():
                        val[:] = state_dict_emission[key]
                    print(f"loaded {key}")
            print("loaded the emission model")

        # initialize optimizer
        with torch.no_grad():
            for name, param in self.gru_association.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
            torch.nn.init.xavier_uniform_(self.W_output_association)
        self.optimizer = torch.optim.RMSprop([self.W_output_association] + list(self.gru_association.parameters()), lr=1e-3 if not train_with_emission else 1e-4)

    def evaluate(self, rnn_state_association=None, rnn_state_emission=None, update_state=True, use_probabilitistic_reward=False):
        """
        Evaluates the model by running forward passes and optionally returning rewards
        
        Args:
            num_trials: Number of parallel trials to run
            num_steps: Number of steps per sequence
        
        4 possibilities:
        - correct weather are observed, no emission
        - probabilitistic rewards are observed, no emission
        - probabilitistic rewards are observed, with emission function but WP integration trained with emission function
        - probabilitistic rewards are observed, with emission function but WP integration trained with emission function
        """
        if (not update_state) and rnn_state_association is None:
            raise ValueError("RNN state association must be provided if update_state is False")
        
        # process contexts
        contexts = self.env.context.to(torch.int64)

        # Initialize RNN state and pre-compute parameters
        rnn_state_association = rnn_state_association if rnn_state_association is not None else torch.zeros(1, self.env.num_tasks, self.nb_units)

        # pre-compute parameters
        logalphas = torch.zeros([self.env.num_tasks, self.env.num_trials, 2])
        all_probas_association = torch.zeros([self.env.num_tasks, self.env.num_trials, 4])
        logpredicts = torch.zeros([self.env.num_tasks, self.env.num_trials, 2])

        # initialize emission RNN state if needed
        if self.with_emission:
            rnn_state_emission = rnn_state_emission if rnn_state_emission is not None else torch.zeros(1, self.env.num_tasks, self.nb_units)
            all_params_emission = torch.zeros([self.env.num_tasks, self.env.num_trials, 201])

        debug_pfiltering = []
        for i_trial in range(self.env.num_trials):
            # compute association logits
            logits_association = compute_association(rnn_state_association, self.W_output_association)
            logpredict_1 = logsigmoid(logits_association)
            logpredict_0 = logsigmoid(-logits_association)

            # compute logpredicts
            logpredict = np.log(0.5) + torch.stack([
                (logpredict_0[np.arange(self.env.num_tasks)[:, None], contexts[:, i_trial]] * (contexts[:, i_trial] != -1)).sum(axis=-1),
                (logpredict_1[np.arange(self.env.num_tasks)[:, None], contexts[:, i_trial]] * (contexts[:, i_trial] != -1)).sum(axis=-1)
            ]).squeeze().T # p(z_t, c_t) = p(z_t) • p(c_t | z_t)

            # compute emission probabilities if needed
            if self.with_emission and not use_probabilitistic_reward:
                p_gen = compute_emission(rnn_state_emission, self.W_output_emission)
                proba_emission_arm0 = p_gen[:, torch.arange(self.env.num_tasks), self.env.idx_arm0[:, i_trial]]
                proba_emission_arm1 = p_gen[:, torch.arange(self.env.num_tasks), self.env.idx_arm1[:, i_trial]]
                emission_probs = torch.stack([proba_emission_arm0, proba_emission_arm1]).squeeze().T
            else:
                if use_probabilitistic_reward:
                    emission_probs = torch.zeros([self.env.num_tasks, 2])
                else:
                    emission_probs = torch.stack([1 - self.env.correct_weather[:, i_trial], self.env.correct_weather[:, i_trial]]).T

            # compute logalphas and logpredicts
            logalphas[:, i_trial] = logpredict + emission_probs.log() # p(z_t, c_t, y_t) = p(z_t) • p(c_t | z_t) • p(y_t | z_t)
            logpredicts[:, i_trial] = logpredict # p(z_t, c_t) = p(z_t) • p(c_t | z_t)

            # update RNN state if needed
            if update_state:
                # update emission RNN state if needed
                if self.with_emission and (not use_probabilitistic_reward):
                    # update emission RNN state
                    pfiltering = (logalphas[:, i_trial] - torch.logsumexp(logalphas[:, i_trial], dim=-1, keepdims=True))
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
                if use_probabilitistic_reward:
                    outcomes = -self.env.probabilistic_rewards[0, :, i_trial]
                elif self.with_emission:
                    p_gen = compute_emission(rnn_state_emission, self.W_output_emission)
                    outcomes = torch.tanh(torch.tensor([-p_gen[:, i_k, k].log() + p_gen[:, i_k, -k].log() for i_k, k in enumerate(self.env.idx_arm0[:, i_trial])])).detach()
                else:
                    outcomes = 2 * self.env.correct_weather[:, i_trial] - 1

                input_state = torch.vstack(
                    (
                        torch.vstack([outcomes]), 
                        torch.vstack([(contexts[:, i_trial] == k).sum(dim=1) for k in range(4)])
                    )
                ).float()
                _, rnn_state_association = self.gru_association(input_state.T.unsqueeze(1), rnn_state_association)

            all_probas_association[:, i_trial] = torch.sigmoid(compute_association(rnn_state_association, self.W_output_association)).squeeze().detach()
            if self.with_emission:
                all_params_emission[:, i_trial] = compute_emission(rnn_state_emission, self.W_output_emission).detach()

        return {
            'rnn_state_association': rnn_state_association,
            'rnn_state_emission': rnn_state_emission if self.with_emission else None,
            'greedy': self.env.greedy,
            'logalphas': logalphas,
            'probas_association': all_probas_association,
            'params_emission': all_params_emission if self.with_emission else None,
            "true_association_probas": self.env.probas,
            "logpredicts": logpredicts,
        }

    def load_model(self, nb_episodes=None, trained_with_emission=None):
        if trained_with_emission is None:
            trained_with_emission = self.train_with_emission
        nb_episodes = nb_episodes if nb_episodes is not None else self.episode_count_max
        model_dir = f"{self.model_path}/{self.model_name}".replace('_debug', "").replace('_trainWithEmission_True', f"_trainWithEmission_{trained_with_emission}")
        model_file = f"{model_dir}/model-{int(nb_episodes)}.pth"
        state_dict = torch.load(model_file)
        print(f"loading model {model_file}")
        for (key, val) in self.named_parameters():
            if 'association' in key:
                with torch.no_grad():
                    val[:] = state_dict[key]
                print(f"loaded {key}")
            else:
                print(f"not loaded {key}")
        print(f"loaded model {model_file}")

    def train(self, num_trials=500, num_steps=3):
        """
        Main training/evaluation loop
        
        Args:
            num_trials: Number of parallel trials to run
            num_steps: Number of steps per sequence
        """
        episode_count = 0
        while episode_count <= self.episode_count_max:  # stopping criterion moved to loop condition
            # reset environment
            self.env.generate_test_task(num_tasks=10, num_trials=num_trials, num_steps=num_steps, probas=None, variable_length=True, tau=None)

            # evaluate model
            result = self.evaluate()
            marginal_loss = -torch.logsumexp(result["logalphas"], dim=-1).sum()

            self.optimizer.zero_grad()
            marginal_loss.backward()
            self.optimizer.step()
            
            probas = result['true_association_probas'].squeeze()
            categorical_probs = result['probas_association']
                
            correct = (result['logpredicts'].argmax(-1).detach() == self.env.correct_weather).float().mean()

            self.episode_rewards.append(correct)

            # Periodic evaluation and logging
            if episode_count % 10 == 0 and episode_count != 0:
                
                # Save model checkpoint
                if episode_count % 500 == 0 and False:
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
        index = 5

    np.random.seed(index)
    torch.manual_seed(index)

    self = Worker(
        probabilistic_task(),
        "results/source/saved_models",
        "WP_GRU_debug_id{0}".format(index),
        with_emission=True,
        #train_with_emission=True
    )
    
    self.load_model(trained_with_emission=False)
    self.train()
