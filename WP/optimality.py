import numpy as np
import torch
import sys
from task import probabilistic_task, truncated_exponential_logpdf, sample_truncated_exponential
from torch.utils import tensorboard
import os
from tqdm import tqdm
from scipy.stats import truncnorm
from scipy.special import logsumexp
from sobol_seq import sobol_seq
import itertools
import warnings
#warnings.simplefilter('error')


class Optimality():
    def __init__(
            self, game
        ):
        super().__init__()
        self.env = game
        self.state_space_mapping =  np.array(list(itertools.permutations(np.arange(4), 4)))
    
    def compute_log_lh(self, particles, t, xx, w_emission=False):
        K=24
        correct_weather = self.env.correct_weather.to(torch.int64).numpy()
        association_mappings = np.swapaxes(self.state_space_mapping.T[self.env.context[xx].long().cpu().numpy()], -1, -2)
        nb_particles, n_tasks, nb_parameters = particles.shape
        log_alphas = np.ones([nb_particles, n_tasks, K, 2]) * np.log(1./(K * 2))

        vol, association_probs = particles[:, :, 0], np.array([0.2, 0.4, 0.6, 0.8])
        if w_emission:
            emission_mean, emission_std = particles[:, :, 1], particles[:, :, 2]
        logvol, log_1_minus_vol = np.log(vol), np.log1p(-(vol))
    
        for i_trial in range(t + 1):

            log_predict_probs = np.stack([
                np.logaddexp(log_alphas[:, :, :, 0] + np.log(0.5), log_alphas[:, :, :,1] + np.log(0.5)),
                np.logaddexp(log_alphas[:, :, :, 1] + np.log(0.5), log_alphas[:, :, :, 0] + np.log(0.5))
            ], axis=-1) # p(z_{t+1}, q_t, y_{1:t}, c_{1:t}) = \sum_z p(z_{t+1} | z_t) • p(z_t, q_t, y_{1:t}, c_{1:t})

            # update log predicts of association mapping
            log_predict_probs = np.stack([
                np.logaddexp(
                    log_predict_probs[:, :, k] + log_1_minus_vol[:, :, None], 
                    logsumexp(log_predict_probs[:, :, [j for j in range(K) if j != k]], axis=-2) + logvol[:, :, None] - np.log(K - 1)
                    )
                for k in range(K)
            ], axis=-2) # p(z_{t+1}, q_{t+1}, y_{1:t}, c_{1:t}) = \sum_q p(q_{t+1} | q_t) • p(z_{t+1}, q_t, y_{1:t}, c_{1:t})

            association_probs_cues = association_probs[association_mappings[:, i_trial]]
            prob_cues_weather0 = np.prod(association_probs_cues, axis=-1) # p(c_t | z_t=0, q_t)
            prob_cues_weather1 = np.prod(1 - association_probs_cues, axis=-1) # p(c_t | z_t=1, q_t)
            prob_cues = np.log(np.stack([prob_cues_weather0, prob_cues_weather1], axis=-1)) # p(c_t | q_t)

            if not w_emission:
                logp_outcome = np.where(
                    np.stack([1 - correct_weather[xx, i_trial], correct_weather[xx, i_trial]], axis=-1).astype(bool),
                    0.0,
                    -np.inf,
                )[None]
            else:
                feedback_arm0 = self.env.feedback_arm0[xx, i_trial]
                feedback_arm1 = self.env.feedback_arm1[xx, i_trial]
                emission_mean, emission_std = particles[:, :, 1], particles[:, :, 2]
                a_, b_ = (-1 - emission_mean) / emission_std, (1 - emission_mean) / emission_std
                logp_outcome = np.stack([
                    truncnorm.logpdf(feedback_arm0, a_, b_, loc=emission_mean, scale=emission_std), 
                    truncnorm.logpdf(feedback_arm1, a_, b_, loc=emission_mean, scale=emission_std)], 
                    axis=-1
                )
            log_predict_probs_cues = log_predict_probs + prob_cues[None]

            log_alphas = log_predict_probs_cues + logp_outcome[:,:, None] # p(z_{t+1}, q_{t+1}, y_{1:(t+1)}, c_{1:(t+1)})

        return log_alphas, logsumexp(log_alphas, axis=(-1, -2)) # p(y_{1:t})

    def infer(self, nb_particles=500, gamma=0.5, w_emission=False):
        nb_parameters = 3 if w_emission else 1
        K = 24

        association_mappings = np.swapaxes(self.state_space_mapping.T[self.env.context.long().cpu().numpy()], -1, -2)
        correct_weather = self.env.correct_weather.to(torch.int64).numpy()

        particles = np.zeros([nb_particles, self.env.num_tasks, nb_parameters])
        sobol = sobol_seq.i4_sobol_generate(nb_parameters + 1 * w_emission, nb_particles)
        particles[:, :, 0] = sobol[:, 0][:, None] * 0.1 #possible_taus[(sobol[:, 0][:, None] < np.arange(0, 1, 0.2)[None]).sum(axis=1)][:, None]
        if w_emission:
            particles[:, :, 1] = sample_truncated_exponential(size=nb_particles, lambda_param=3, ub=1.0, u=sobol[:, 1])[:, None]        
            particles[:, :, 2] = sample_truncated_exponential(size=nb_particles, lambda_param=3, ub=1.5, u=sobol[:, 2])[:, None]
            particles[:, :, 1] = particles[:, :, 1] * (1 - 2 * (sobol[:, 3][:, None] > 0.5))

        all_map_particles = np.zeros([self.env.num_trials, self.env.num_tasks, nb_parameters])
        polarity_particles = np.zeros([self.env.num_trials, self.env.num_tasks])
        log_weights = np.zeros([nb_particles, self.env.num_tasks])
        # p(z_t, q_t, y_{1:t}, c_{1:t}) # z_t is the weather, q_t is the mapping permutation, y_{1:t} is the observations, c_{1:t} is the cues (c_t is a vector)
        log_alphas = np.ones([nb_particles, self.env.num_tasks, K, 2]) * np.log(1./(K * 2))
        prediction_weather = np.zeros([self.env.num_tasks, self.env.num_trials, 2])
        association_probs_trials = np.zeros([self.env.num_tasks, self.env.num_trials, 4])

        for i_trial in tqdm(range(self.env.num_trials), desc="Inferring"):
            vol, association_probs = particles[:, :, 0], np.array([0.2, 0.4, 0.6, 0.8])
            if w_emission:
                emission_mean, emission_std = particles[:, :, 1], particles[:, :, 2]

            logvol, log_1_minus_vol = np.log(vol), np.log1p(-(vol))

            # compute log predicts of weather
            log_predict_probs = np.stack([
                np.logaddexp(log_alphas[:, :, :, 0] + np.log(0.5), log_alphas[:, :, :, 1] + np.log(0.5)),
                np.logaddexp(log_alphas[:, :, :, 1] + np.log(0.5), log_alphas[:, :, :, 0] + np.log(0.5))
            ], axis=-1) # p(z_{t+1}, q_t, y_{1:t}, c_{1:t}) = \sum_z p(z_{t+1} | z_t) • p(z_t, q_t, y_{1:t}, c_{1:t})

            # update log predicts of association mapping
            log_predict_probs = np.stack([
                np.logaddexp(
                    log_predict_probs[:, :, k] + log_1_minus_vol[:, :, None], 
                    logsumexp(log_predict_probs[:, :, [j for j in range(K) if j != k]], axis=-2) + logvol[:, :, None] - np.log(K - 1)
                    )
                for k in range(K)
            ], axis=-2) # p(z_{t+1}, q_{t+1}, y_{1:t}, c_{1:t}) = \sum_q p(q_{t+1} | q_t) • p(z_{t+1}, q_t, y_{1:t}, c_{1:t})

            association_probs_cues = association_probs[association_mappings[:, i_trial]] # p(c_t | q_t)
            prob_cues_weather0 = np.prod(association_probs_cues, axis=-1) # p(c_t | z_t=0, q_t)
            prob_cues_weather1 = np.prod(1 - association_probs_cues, axis=-1) # p(c_t | z_t=1, q_t)
            prob_cues = np.log(np.stack([prob_cues_weather0, prob_cues_weather1], axis=-1)) # p(c_t | q_t)

            # Compute marginal probabilities of association mapping
            log_predict_probs_cues = log_predict_probs + prob_cues[None] # p(z_{t+1}, q_{t+1}, y_{1:t}, c_{1:(t+1)}) = p(z_{t+1}, q_{t+1}, y_{1:t}, c_{1:t}) • p(c_{t+1} | z_{t+1}, q_{t+1})
            
            log_joint_predict_weather_cues = logsumexp(log_predict_probs_cues, axis=-2) # p(z_{t+1}, y_{1:t}, c_{1:(t+1)}) = \sum_q p(q_{t+1} | q_t) • p(z_{t+1}, q_t, y_{1:t}, c_{1:t})
            log_predict_weather_cues = log_joint_predict_weather_cues - logsumexp(log_joint_predict_weather_cues, axis=-1, keepdims=True) # p(z_{t+1}| y_{1:t}, c_{1:(t+1)})

            weights_particles = np.exp(log_weights - np.max(log_weights, axis=0, keepdims=True)) # p(theta^m | y_{1:(t-1)}, c_t)
            normalized_weights_particles = weights_particles / np.sum(weights_particles, axis=0, keepdims=True) # p(theta^m | y_{1:(t-1)}, c_t)
            predict_weather_cues = np.exp(log_predict_weather_cues)
            is_pos = (particles[:, :, 1] >= 0) if w_emission else np.ones_like(particles[:, :, 0]).astype(bool)
            aligned_predict_weather_cues = np.stack([
                predict_weather_cues[:, :, 0] * is_pos + predict_weather_cues[:, :, 1] * (~is_pos),
                predict_weather_cues[:, :, 1] * is_pos + predict_weather_cues[:, :, 0] * (~is_pos),
            ], axis=-1)
            marginal_proba_weather = np.sum(
                normalized_weights_particles[:, :, None] * aligned_predict_weather_cues,
                axis=0
            )
            #marginal_proba_weather = np.sum(normalized_weights_particles[:,:,None] * np.exp(log_predict_weather_cues), axis=0) # p(z_t | y_{1:(t-1)}, c_t)
            prediction_weather[:, i_trial] = marginal_proba_weather  

            # Compute MAP particles
            all_map_particles[i_trial] = np.sum(particles * normalized_weights_particles[:, :, None], axis=0)
            polarity_particles[i_trial] = (np.sign(particles[:,:,1] if w_emission else np.ones_like(particles[:, :, 0])) * normalized_weights_particles).sum(axis=0)

            log_joint_predict_association_cues = logsumexp(log_predict_probs_cues, axis=-1)
            normalized_weights_trial_association = np.exp(
                log_joint_predict_association_cues
                - logsumexp(log_joint_predict_association_cues, axis=-1, keepdims=True)
            )
            is_pos = (particles[:, :, 1] >= 0) if w_emission else np.ones_like(particles[:, :, 0]).astype(bool)
            aligned_association_probs = (
                association_probs[self.state_space_mapping][None, None] * is_pos[:, :, None, None]
                + (1 - association_probs[self.state_space_mapping][None, None]) * (~is_pos[:, :, None, None])
            )

            association_probs_trials[:, i_trial] = (
                (
                    normalized_weights_trial_association[:, :, :, None]
                    * aligned_association_probs
                ).sum(axis=-2)
                * normalized_weights_particles[:, :, None]
            ).sum(axis=0)

            if not w_emission:
                logp_outcome = np.where(
                    np.stack([1 - correct_weather[:, i_trial], correct_weather[:, i_trial]], axis=-1).astype(bool),
                    0.0,
                    -np.inf,
                )[None]
            else:
                # Compute log-likelihood of observation
                feedback_arm0 = self.env.feedback_arm0[:, i_trial]
                feedback_arm1 = self.env.feedback_arm1[:, i_trial]
                a_, b_ = (-1 - emission_mean) / emission_std, (1 - emission_mean) / emission_std
                logp_outcome = np.stack([
                    truncnorm.logpdf(feedback_arm0, a_, b_, loc=emission_mean, scale=emission_std), 
                    truncnorm.logpdf(feedback_arm1, a_, b_, loc=emission_mean, scale=emission_std)], 
                    axis=-1
                )

            # Update weights and particles for association mapping
            prev_llk = logsumexp(log_alphas, axis=(-1, -2)) # p(y_{1:t}, c_{1:t})
            log_alphas = log_predict_probs_cues + logp_outcome[:,:, None] # p(z_{t+1}, q_{t+1}, y_{1:(t+1)}, c_{1:(t+1)})
            current_llk = logsumexp(log_alphas, axis=(-1, -2)) # p(y_{1:(t+1)}, c_{1:(t+1)})
            inc_log_weights = current_llk - prev_llk # p(y_{t+1}, c_{t+1} | y_{1:t}, c_{1:t})

            # Compute ESS (Effective Sample Size)            
            log_weights = log_weights + inc_log_weights
            weights = np.exp(log_weights - np.max(log_weights, axis=0, keepdims=True))
            weights_norm = weights / np.sum(weights, axis=0, keepdims=True)
            ess = 1.0 / np.sum(weights_norm**2, axis=0)

            if np.any(ess < gamma * nb_particles):
                xx = np.where(ess < gamma * nb_particles)[0]                
                mean_proposals = np.sum(weights_norm[:, xx, None] * particles[:, xx], axis=0)
                std_proposals = np.sqrt(np.sum(weights_norm[:, xx, None] * (particles[:, xx] - mean_proposals)**2, axis=0))
                if w_emission:
                    _a, _b = (np.array([0, -1, 0])[None] - mean_proposals) / std_proposals, (np.array([0.1, 1, 1.5])[None] - mean_proposals) / std_proposals
                    proposals = truncnorm.rvs(_a, _b, loc=mean_proposals, scale=std_proposals, size=(nb_particles, len(xx), 3))
                    log_prior_new = (
                        truncated_exponential_logpdf(np.abs(proposals[:, :, 1]), 3, 1.0) + 
                        truncated_exponential_logpdf(proposals[:, :, 2], 3, 1.5)
                    )
                    log_prior_old = (
                        truncated_exponential_logpdf(np.abs(particles[:, xx, 1]), 3, 1.0) + 
                        truncated_exponential_logpdf(particles[:, xx, 2], 3, 1.5)
                    )
                else:
                    _a, _b = (np.array([0])[None] - mean_proposals) / std_proposals, (np.array([0.1])[None] - mean_proposals) / std_proposals
                    log_prior_new, log_prior_old = 0, 0
                    proposals = truncnorm.rvs(_a, _b, loc=mean_proposals, scale=std_proposals, size=(nb_particles, len(xx), 1))
                log_llh_proposals, log_llh_proposals_sum = self.compute_log_lh(proposals, i_trial, xx, w_emission=w_emission)
                # _, log_llh_ = self.compute_log_lh(particles, i_trial, np.arange(self.env.num_tasks))
                log_q_old = np.sum(truncnorm.logpdf(particles[:, xx], _a, _b, loc=mean_proposals, scale=std_proposals), axis=-1)
                log_q_new = np.sum(truncnorm.logpdf(proposals, _a, _b, loc=mean_proposals, scale=std_proposals), axis=-1)
                log_acceptance_probas = log_llh_proposals_sum + log_prior_new - current_llk[:, xx] - log_prior_old + log_q_old - log_q_new
                accepted_proposals = (np.log(np.random.rand(*log_acceptance_probas.shape)) < log_acceptance_probas)
                particles[:, xx] = proposals * accepted_proposals[:, :, None] + particles[:, xx] * (1 - accepted_proposals[:, :, None])
                log_weights[:, xx] = np.zeros_like(log_weights[:, xx])
                log_alphas[:, xx] = np.where(
                    accepted_proposals[:, :, None, None],
                    log_llh_proposals,
                    log_alphas[:, xx]
                )
           

        return all_map_particles, prediction_weather, association_probs_trials, polarity_particles

if __name__ == "__main__":

    from optimality_from_banditWP import Optimality
    import numpy as np
    from task import probabilistic_task

    nb_simuls = 50#00

    self = Optimality(
        probabilistic_task(),
    )


    # 3 steps
    self.env.probas = None
    np.random.seed(1)
    self.env.generate_test_task(num_tasks=nb_simuls, num_trials=200, num_steps=5, probas=None)

    all_map_particles, prediction_weather, _, polarity_particles = self.infer(nb_particles=100, gamma=0.5, w_emission=False)
    performances_optimal_ = (np.argmax(prediction_weather, axis=-1) == self.env.correct_weather.numpy())

    print(performances_optimal_.mean())
    #import pickle
    #with open('results/performances_optimal.pkl', 'wb') as f:
    #    pickle.dump([all_map_particles, prediction_weather, polarity_particles, performances_optimal_], f)
