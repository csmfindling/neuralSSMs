import numpy as np
import torch
import sys
from task import Mastermind
from torch.utils import tensorboard
import os
from tqdm import tqdm
from scipy.stats import truncnorm
from scipy.special import logsumexp
from task import truncated_exponential_logpdf, sample_truncated_exponential
from sobol_seq import sobol_seq

class Optimality():
    def __init__(
            self, game
        ):
        super().__init__()
        self.env = game

    
    def compute_log_lh(self, particles, t, xx):
        nb_particles, n_tasks, _ = particles.shape
        log_alphas = np.ones([nb_particles, n_tasks, self.env.K]) * np.log(1./self.env.K)
        vol, ffrate = particles[:, :, 0], particles[:, :, 1]
        logvol, log_1_minus_vol = np.log(vol), np.log1p(-vol)

        for i_trial in range(t + 1):
            log_predict_probs = np.stack([
                np.logaddexp(
                    log_alphas[:, :, k] + log_1_minus_vol, 
                    logsumexp(log_alphas[:, :, [j for j in range(self.env.K) if j != k]], axis=-1) + logvol - np.log(self.env.K - 1)
                    )
                for k in range(self.env.K)
            ], axis=-1)

            log_predict_probs_norm = log_predict_probs - logsumexp(log_predict_probs, axis=-1, keepdims=True) 
            log_weights = logsumexp(log_alphas, axis=-1)
            weights_stable = np.exp(log_weights - np.max(log_weights, axis=0, keepdims=True))
            selected_combinations = np.sum(np.exp(log_predict_probs_norm) * weights_stable[:,:,None], axis=0).argmax(axis=-1)
            selected_mapping = self.env.state_space_mapping[selected_combinations]  # Fixed: convert tensor to numpy for indexing
            selected_action = selected_mapping[np.arange(n_tasks), self.env.stimulus[xx, i_trial]][None]

            #selected_actions[:, i_trial] = logsumexp(log_predict_probs, axis=0).argmax(axis=-1)
            reward = (
                (selected_action == self.env.correct_action[xx, i_trial]) * self.env.feedback_when_correct[xx, i_trial] +
                (selected_action != self.env.correct_action[xx, i_trial]) * (1 - self.env.feedback_when_correct[xx, i_trial])
            ).T
            
            predicted_actions = np.stack([self.env.state_space_mapping[i, self.env.stimulus[xx, i_trial]] for i in range(self.env.K)], axis=0)
            bernoulli_rate = 1 - ffrate.T
            
            emission_probs = (
                (predicted_actions == selected_action)[:, :, None] * (bernoulli_rate * reward + (1 - bernoulli_rate) * (1 - reward))[None] +
                (predicted_actions != selected_action)[:, :, None] * (bernoulli_rate * (1 - reward) + (1 - bernoulli_rate) * reward)[None]
            ).T

            log_alphas = log_predict_probs + np.log(emission_probs)

        return log_alphas, logsumexp(log_alphas, axis=-1)

    def infer(self, nb_particles=500):
        log_alphas = np.ones([nb_particles, self.env.n_tasks, self.env.K]) * np.log(1./self.env.K)
        particles = np.zeros([nb_particles, self.env.n_tasks, 2])
        sobol = sobol_seq.i4_sobol_generate(2, nb_particles)
        particles[:, :, 0] = sample_truncated_exponential(size=nb_particles, lambda_param=5, ub=0.2, u=sobol[:, 0])[:, None]
        particles[:, :, 1] = sample_truncated_exponential(size=nb_particles, lambda_param=3, ub=0.5, u=sobol[:, 1])[:, None]
        log_weights = np.zeros([nb_particles, self.env.n_tasks])
        all_map_particles = np.zeros([self.env.n_trials, self.env.n_tasks, 2])
        selected_actions = np.zeros([self.env.n_tasks, self.env.n_trials])
        selected_combinations = np.zeros([self.env.n_tasks, self.env.n_trials], dtype=int)
        outcome_of_selected_actions = np.zeros([self.env.n_tasks, self.env.n_trials])

        for i_trial in tqdm(range(self.env.n_trials), desc="Inferring"):
            vol, ffrate = particles[:, :, 0], particles[:, :, 1]
            logvol, log_1_minus_vol = np.log(vol), np.log1p(-vol)            

            log_predict_probs = np.stack([
                np.logaddexp(
                    log_alphas[:, :, k] + log_1_minus_vol, 
                    logsumexp(log_alphas[:, :, [j for j in range(self.env.K) if j != k]], axis=-1) + logvol - np.log(self.env.K - 1)
                    )
                for k in range(self.env.K)
            ], axis=-1)

            log_predict_probs_norm = log_predict_probs - logsumexp(log_predict_probs, axis=-1, keepdims=True) 
            weights_stable = np.exp(log_weights - np.max(log_weights, axis=0, keepdims=True))
            selected_combinations[:, i_trial] = np.sum(np.exp(log_predict_probs_norm) * weights_stable[:,:,None], axis=0).argmax(axis=-1)
            selected_mapping = self.env.state_space_mapping[selected_combinations[:, i_trial]]  # Fixed: convert tensor to numpy for indexing
            selected_actions[:, i_trial] = selected_mapping[np.arange(self.env.n_tasks), self.env.stimulus[:, i_trial]]
            selected_action = selected_actions[:, i_trial][None]

            #selected_actions[:, i_trial] = logsumexp(log_predict_probs, axis=0).argmax(axis=-1)
            outcome_of_selected_actions[:, i_trial] = (
                (selected_actions[:, i_trial] == self.env.correct_action[:, i_trial]) * self.env.feedback_when_correct[:, i_trial] +
                (selected_actions[:, i_trial] != self.env.correct_action[:, i_trial]) * (1 - self.env.feedback_when_correct[:, i_trial])
            )
            reward = outcome_of_selected_actions[:, i_trial][:, None]
            
            predicted_actions = np.stack([self.env.state_space_mapping[i, self.env.stimulus[:, i_trial]] for i in range(self.env.K)], axis=0)
            bernoulli_rate = 1 - ffrate.T
            
            emission_probs = (
                (predicted_actions == selected_action)[:, :, None] * (bernoulli_rate * reward + (1 - bernoulli_rate) * (1 - reward))[None] +
                (predicted_actions != selected_action)[:, :, None] * (bernoulli_rate * (1 - reward) + (1 - bernoulli_rate) * reward)[None]
            ).T

            prev_llk = logsumexp(log_alphas, axis=-1) # p(y_{1:(t-1)})
            log_alphas = log_predict_probs + np.log(emission_probs) # p(z_t , y_{1:t}) = p(y_t | z_t) • p(z_t , y_{1:(t-1)})
            current_llk = logsumexp(log_alphas, axis=-1) # p(y_{1:t})
            inc_log_weights = current_llk - prev_llk # p(y_t | y_{1:(t-1)})

            weights_trial = np.exp(log_weights - np.max(log_weights, axis=0, keepdims=True))
            normalized_weights_trial = weights_trial / np.sum(weights_trial, axis=0, keepdims=True)
            all_map_particles[i_trial] = np.sum(particles * normalized_weights_trial[:, :, None], axis=0)
            
            # Compute ESS (Effective Sample Size)            
            log_weights = log_weights + inc_log_weights
            weights = np.exp(log_weights - np.max(log_weights, axis=0, keepdims=True))
            weights_norm = weights / np.sum(weights, axis=0, keepdims=True)
            ess = 1.0 / np.sum(weights_norm**2, axis=0)
            
            if np.any(ess < 0.5 * nb_particles):
                xx = np.where(ess < 0.5 * nb_particles)[0]
                mean_proposals = np.sum(weights_norm[:, xx, None] * particles[:, xx], axis=0)                
                std_proposals = np.sqrt(np.sum(weights_norm[:, xx, None] * (particles[:, xx] - mean_proposals)**2, axis=0))
                a, b = (0 - mean_proposals) / std_proposals, (np.array([0.4, 0.5])[None] - mean_proposals) / std_proposals
                proposals = truncnorm.rvs(a, b, loc=mean_proposals, scale=std_proposals, size=(nb_particles, len(xx), 2))
                log_llh_proposals, log_llh_proposals_sum = self.compute_log_lh(proposals, i_trial, xx)
                log_q_old = np.sum(truncnorm.logpdf(particles[:, xx], a, b, loc=mean_proposals, scale=std_proposals), axis=-1)
                log_q_new = np.sum(truncnorm.logpdf(proposals, a, b, loc=mean_proposals, scale=std_proposals), axis=-1)
                log_prior_new = (
                    truncated_exponential_logpdf(proposals[:, :, 0], 5, 0.2) + 
                    truncated_exponential_logpdf(proposals[:, :, 1], 3, 0.5)
                )
                log_prior_old = (
                    truncated_exponential_logpdf(particles[:, xx, 0], 5, 0.2) + 
                    truncated_exponential_logpdf(particles[:, xx, 1], 3, 0.5)
                )
                log_acceptance_probas = log_llh_proposals_sum + log_prior_new - current_llk[:, xx] - log_prior_old + log_q_old - log_q_new
                accepted_proposals = (np.log(np.random.rand(*log_acceptance_probas.shape)) < log_acceptance_probas)
                particles[:, xx] = proposals * accepted_proposals[:, :, None] + particles[:, xx] * (1 - accepted_proposals[:, :, None])
                log_weights[:, xx] = np.zeros_like(log_weights[:, xx])
                log_alphas[:, xx] = log_llh_proposals * accepted_proposals[:, :, None] + log_alphas[:, xx] * (1 - accepted_proposals[:, :, None])

        return all_map_particles, selected_actions, outcome_of_selected_actions

if __name__ == "__main__":

    from optimality import Optimality
    import numpy as np
    from task import Mastermind

    np.random.seed(0)

    self = Optimality(
        Mastermind(),
    )

    self.env._generate_task_schedule(nb_tasks=1000)
    all_particles, selected_actions, outcome_of_selected_actions = self.infer(nb_particles=2000)

    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.scatter(self.env.nu.mean(axis=1), all_particles[-1].mean(axis=0)[:, 0], alpha=0.5)
    plt.xlabel('True volatility')
    plt.ylabel('Estimated volatility')
    plt.title('Estimated vs True Volatility')
    plt.show()
    print(all_particles[-1].mean(axis=0)[:10, 0])
    print(self.env.nu.mean(axis=1)[:10])

