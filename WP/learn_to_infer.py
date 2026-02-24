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
            self, game, model_path, model_name, num_units=32, init_type="xavier", optimizer="Adam", episode_count_max=5e4, w_emission=False, train_w_emission=False, train_in_cat_task_from_scratch=False, entropy_reg=None
        ):
        '''
        Args:
            game: The game object
            model_path: The path to the model
            model_name: The name of the model
            num_units: The number of units in the RNN
            init_type: The type of initialization
            optimizer: The optimizer
            episode_count_max: The maximum number of episodes
            w_emission: Whether to use an emission model. If True and unless indicated by train_in_cat_task_from_scratch, a pre-trained brandit emission model will be used to compute the emission probabilities.
            train_w_emission: Whether to train / finetune the emission and association models together given the presence of the bandit emission model.
            train_in_cat_task_from_scratch: Whether to train the association and emission models from scratch
            entropy_reg: The entropy regularization
        '''
        super().__init__()
        if train_w_emission and not w_emission:
            raise ValueError("you must load the emission model if you want to train the association and emission modules with the emission model")
        if train_in_cat_task_from_scratch:
            if train_w_emission is not True or w_emission is not True:
                raise ValueError("If train_in_cat_task_from_scratch is True, both train_w_emission and w_emission must be True.")
        self.model_path = model_path
        self.train_in_cat_task_from_scratch = train_in_cat_task_from_scratch
        self.env = game
        self.episode_rewards = []
        self.init_type = init_type
        self.episode_count_max = episode_count_max
        self.entropy_reg = entropy_reg
        self.train_w_emission = train_w_emission
        self.w_emission = w_emission
        self.model_name = model_name + "_init_{0}_optim_{1}_episodeNbMax_{2}_numUnits_{3}_trainWithEmission_{4}".format(
            self.init_type, optimizer, int(self.episode_count_max), num_units, self.train_w_emission
        )
        self.model_name = self.model_name + f"_trainInCatTaskFromScratch_{train_in_cat_task_from_scratch}"
        if entropy_reg is not None:
            self.model_name = self.model_name + "_policyReg_{0}".format(str(entropy_reg).replace(".", "_"))

        self.summary_writer = tensorboard.SummaryWriter("results/source/trainings_fullRNN/" + str(self.model_name))
        self.nb_units = num_units

        # association RNN
        self.W_output_association = torch.nn.Parameter(torch.zeros(self.nb_units, 4))
        self.gru_association = torch.nn.GRU(input_size=5, hidden_size=self.nb_units, batch_first=True, bias=True)
        # initialize association RNN
        with torch.no_grad():
            for name, param in self.gru_association.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
            torch.nn.init.xavier_uniform_(self.W_output_association)

        # emission RNN
        if train_in_cat_task_from_scratch:
            self.gru_emission = torch.nn.GRU(input_size=2, hidden_size=self.nb_units, batch_first=True, bias=True)
            self.W_output_emission = torch.nn.Parameter(torch.zeros(self.nb_units, 201))
            # initialize emission RNN
            with torch.no_grad():
                for name, param in self.gru_emission.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param)
                torch.nn.init.xavier_uniform_(self.W_output_emission)
            print('initialized the emission model')
        elif w_emission: # if we are loading a pretrained model, load the emission model
            # load emission model
            model_id = int(model_name.split('agent')[-1]) + 1

            # initialize emission RNN
            self.gru_emission = torch.nn.GRU(input_size=2, hidden_size=self.nb_units, batch_first=True, bias=True)
            self.W_output_emission = torch.nn.Parameter(torch.zeros(self.nb_units, 201))

            path_to_emission_networks = "../bandit/results/source/saved_models"
            name_of_emission_network = f"banditGRU_newinit_val_0_beta2_id{model_id}_init_xavier_optim_Adam_episodeNbMax_50000_numUnits_32_rnnType_GRU_inputType_logodds"
            files_to_load = sorted(glob.glob(path_to_emission_networks + "/" + name_of_emission_network + "/*"))            
            number_of_iterations_of_emission_model = np.max([int(f.split('-')[-1].split('.')[0]) for f in files_to_load])
            idx_of_emission_model = np.argmax([int(f.split('-')[-1].split('.')[0]) == number_of_iterations_of_emission_model for f in files_to_load])
            state_dict_emission = torch.load(files_to_load[idx_of_emission_model])            
            for (key, val) in self.named_parameters():
                if 'emission' in key:
                    with torch.no_grad():
                        val[:] = state_dict_emission[key]
            print('loaded the emission model with {} iterations'.format(number_of_iterations_of_emission_model))

        if self.train_in_cat_task_from_scratch:
            self.optimizer = torch.optim.RMSprop(
                list(self.gru_association.parameters()) + list(self.gru_emission.parameters()) + [self.W_output_association, self.W_output_emission], lr=1e-3
            )
        else:            
            if self.train_w_emission:
                self.optimizer = torch.optim.RMSprop(list(self.gru_association.parameters()) + list(self.gru_emission.parameters()), lr=1e-4)
            else:
                self.optimizer = torch.optim.RMSprop(
                    list(self.gru_association.parameters()) + [self.W_output_association], lr=1e-3
                )
        self.__post_init__()

    def __post_init__(self):
        if (not self.train_in_cat_task_from_scratch) and self.w_emission:
            self.load_model(trained_w_emission=False)
            print('loaded the association model')

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
        if self.w_emission:
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
            logpredict = logpredict - torch.logsumexp(logpredict, dim=-1, keepdims=True) # p(z_t | c_t) = p(z_t, c_t) / \sum_{z_t'} p(z_t', c_t)

            # compute emission probabilities if needed
            if self.w_emission and not use_probabilitistic_reward:
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
            logalphas[:, i_trial] = logpredict + emission_probs.log() # p(y_t, z_t | c_t) = p(z_t | c_t) • p(y_t | z_t)
            logpredicts[:, i_trial] = logpredict # p(z_t | c_t)

            # update RNN state if needed
            if update_state:
                # update emission RNN state if needed
                if self.w_emission and (not use_probabilitistic_reward):
                    # update emission RNN state
                    pfiltering = (logpredict - torch.logsumexp(logpredict, dim=-1, keepdims=True))
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
                elif self.w_emission:
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
            if self.w_emission:
                all_params_emission[:, i_trial] = compute_emission(rnn_state_emission, self.W_output_emission)

        return {
            'rnn_state_association': rnn_state_association,
            'rnn_state_emission': rnn_state_emission if self.w_emission else None,
            'greedy': self.env.greedy,
            'logalphas': logalphas,
            'probas_association': all_probas_association,
            'params_emission': all_params_emission if self.w_emission else None,
            "true_association_probas": self.env.probas,
            "logpredicts": logpredicts,
        }

    def load_model(self, nb_episodes=None, trained_w_emission=None):
        if trained_w_emission is None:
            trained_w_emission = self.train_w_emission
        nb_episodes = nb_episodes if nb_episodes is not None else self.episode_count_max
        model_dir = f"{self.model_path}/{self.model_name}".replace('_trainWithEmission_True', f"_trainWithEmission_{trained_w_emission}")
        # load the model
        model_file = f"{model_dir}/model-{int(nb_episodes)}.pth"
        state_dict = torch.load(model_file)
        print(f"loading model {model_file}")
        for (key, val) in self.named_parameters():
            with torch.no_grad():
                if key in state_dict.keys():
                    val[:] = state_dict[key]
                    print(f"loaded {key}")
                else:
                    print(f"key {key} not found in state_dict")
           

    def train(self, num_trials=500, num_steps=5):
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
            if self.train_in_cat_task_from_scratch: # if not loading a pretrained model, train the association and the emission model. So we need to compute the slope loss.
                slope_loss = torch.relu(result['params_emission'][:, :, :100].sum(axis=-1) - 0.5).sum() * 1e6
                if self.entropy_reg is not None:
                    logpredicts = result['logpredicts']
                    logpredicts_norm = logpredicts - torch.logsumexp(logpredicts, dim=-1, keepdims=True)
                    entropy_loss = -torch.sum(torch.exp(logpredicts_norm) * logpredicts_norm, dim=-1).sum()
                    total_loss = marginal_loss + slope_loss - self.entropy_reg * entropy_loss
                else:
                    total_loss = marginal_loss + slope_loss
            else:
                total_loss = marginal_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            probas = result['true_association_probas'].squeeze()
            categorical_probs = result['probas_association']
                
            correct = (result['logpredicts'].argmax(-1).detach() == self.env.correct_weather).float().mean()

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

                if self.train_in_cat_task_from_scratch and self.entropy_reg is not None:
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
        index = 5

    entropy_regs = [0.1, 0.3, 0.5, 1, 2, 4]

    index_agent = index % 30
    index_reg = index // 30

    np.random.seed(index_agent)
    torch.manual_seed(index_agent)

    self = Worker(
        probabilistic_task(),
        "results/source/saved_models",
        "WP_GRU_agent{0}".format(index_agent),
        w_emission=True,
        train_w_emission=True,
        train_in_cat_task_from_scratch=True,
        entropy_reg=None
    )

    #self.load_model(trained_w_emission=False)
    self.train()
    ffs = [0.1] * 500 + [0.3] * 500
    self.env.generate_test_task(num_tasks=1000, ffs=ffs, tau=0.05)
    result = self.evaluate(use_probabilitistic_reward=False)
    estimated_false_positive_rate = result['params_emission'][:, -1, :100].sum(axis=-1)
    print(estimated_false_positive_rate[:500].mean())
    print(estimated_false_positive_rate[500:].mean())
