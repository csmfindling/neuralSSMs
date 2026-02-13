import numpy as np
import torch
import sys
from task import SwitchingBandit
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
        log_alphas = np.ones([nb_particles, n_tasks, 2]) * np.log(0.5)
        vol, emission_mean, emission_std = particles[:, :, 0], particles[:, :, 1], particles[:, :, 2]
        logvol, log_1_minus_vol = np.log(vol), np.log1p(-vol)
        a, b = (-1 - emission_mean) / emission_std, (1 - emission_mean) / emission_std

        for i_trial in range(t + 1):
            log_predict_probs = np.stack([
                np.logaddexp(log_alphas[:, :, 0] + log_1_minus_vol, log_alphas[:, :, 1] + logvol),
                np.logaddexp(log_alphas[:, :, 1] + log_1_minus_vol, log_alphas[:, :, 0] + logvol)
            ], axis=-1)

            feedback_arm0 = self.env.feedback_arm0[xx, i_trial]
            feedback_arm1 = self.env.feedback_arm1[xx, i_trial]

            log_emission_probs = np.stack([
                truncnorm.logpdf(feedback_arm0, a, b, loc=emission_mean, scale=emission_std), 
                truncnorm.logpdf(feedback_arm1, a, b, loc=emission_mean, scale=emission_std)], 
                axis=-1
            )
            
            log_alphas = log_predict_probs + log_emission_probs

        return log_alphas, logsumexp(log_alphas, axis=-1)

    def infer(self, nb_particles=500):
        log_alphas = np.ones([nb_particles, self.env.n_tasks, self.env.n_arms]) * np.log(0.5)
        particles = np.zeros([nb_particles, self.env.n_tasks, 3])
        sobol = sobol_seq.i4_sobol_generate(3, nb_particles)
        particles[:, :, 0] = sample_truncated_exponential(size=nb_particles, lambda_param=10, ub=0.4, u=sobol[:, 0])[:, None]
        particles[:, :, 1] = sample_truncated_exponential(size=nb_particles, lambda_param=3, ub=1.0, u=sobol[:, 1])[:, None]
        particles[:, :, 2] = sample_truncated_exponential(size=nb_particles, lambda_param=3, ub=1.5, u=sobol[:, 2])[:, None]
        log_weights = np.zeros([nb_particles, self.env.n_tasks])
        all_map_particles = np.zeros([self.env.n_trials, self.env.n_tasks, 3])
        selected_actions = np.zeros([self.env.n_tasks, self.env.n_trials])
        outcome_of_selected_actions = np.zeros([self.env.n_tasks, self.env.n_trials])

        for i_trial in tqdm(range(self.env.n_trials), desc="Inferring"):
            vol, emission_mean, emission_std = particles[:, :, 0], particles[:, :, 1], particles[:, :, 2]
            logvol, log_1_minus_vol = np.log(vol), np.log1p(-vol)

            # compute log predicts
            log_predict_probs = np.stack([
                np.logaddexp(log_alphas[:, :, 0] + log_1_minus_vol, log_alphas[:, :, 1] + logvol),
                np.logaddexp(log_alphas[:, :, 1] + log_1_minus_vol, log_alphas[:, :, 0] + logvol)
            ], axis=-1) # p(z_t , y_{1:(t-1)}) = \sum_s p(z_t | z_{t-1}=s) • p(z_{t-1}=s, y_{1:(t-1)})

            log_predict_probs_norm = log_predict_probs - logsumexp(log_predict_probs, axis=-1, keepdims=True) 
            weights_stable = np.exp(log_weights - np.max(log_weights, axis=0, keepdims=True))
            selected_actions[:, i_trial] = np.sum(np.exp(log_predict_probs_norm) * weights_stable[:,:,None], axis=0).argmax(axis=-1)
            #selected_actions[:, i_trial] = logsumexp(log_predict_probs, axis=0).argmax(axis=-1)
            outcome_of_selected_actions[:, i_trial] = (
                (selected_actions[:, i_trial] == 0) * self.env.feedback_arm0[:, i_trial] + (selected_actions[:, i_trial] == 1) * self.env.feedback_arm1[:, i_trial]
            )

            feedback_arm0 = self.env.feedback_arm0[:, i_trial]
            feedback_arm1 = self.env.feedback_arm1[:, i_trial]
            
            a, b = (-1 - emission_mean) / emission_std, (1 - emission_mean) / emission_std
            log_emission_probs = np.stack([
                truncnorm.logpdf(feedback_arm0, a, b, loc=emission_mean, scale=emission_std), 
                truncnorm.logpdf(feedback_arm1, a, b, loc=emission_mean, scale=emission_std)], 
                axis=-1
            )
            
            prev_llk = logsumexp(log_alphas, axis=-1) # p(y_{1:(t-1)})
            log_alphas = log_predict_probs + log_emission_probs # p(z_t , y_{1:t}) = p(y_t | z_t) • p(z_t , y_{1:(t-1)})
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
                a, b = (0 - mean_proposals) / std_proposals, (np.array([0.4, 1.0, 1.5])[None] - mean_proposals) / std_proposals
                proposals = truncnorm.rvs(a, b, loc=mean_proposals, scale=std_proposals, size=(nb_particles, len(xx), 3))
                log_llh_proposals, log_llh_proposals_sum = self.compute_log_lh(proposals, i_trial, xx)
                log_q_old = np.sum(truncnorm.logpdf(particles[:, xx], a, b, loc=mean_proposals, scale=std_proposals), axis=-1)
                log_q_new = np.sum(truncnorm.logpdf(proposals, a, b, loc=mean_proposals, scale=std_proposals), axis=-1)
                log_prior_new = (
                    truncated_exponential_logpdf(proposals[:, :, 0], 10, 0.4) + 
                    truncated_exponential_logpdf(proposals[:, :, 1], 3, 1.0) + 
                    truncated_exponential_logpdf(proposals[:, :, 2], 3, 1.5)
                )
                log_prior_old = (
                    truncated_exponential_logpdf(particles[:, xx, 0], 10, 0.4) + 
                    truncated_exponential_logpdf(particles[:, xx, 1], 3, 1.0) + 
                    truncated_exponential_logpdf(particles[:, xx, 2], 3, 1.5)
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
    from task import SwitchingBandit

    np.random.seed(0)

    self = Optimality(
        SwitchingBandit(),
    )

    ffs = [0.05] * 500 + [0.3] * 500

    self.env.reset(nb_tasks=1000, ffs=ffs)
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

