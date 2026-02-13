import torch
from torch.nn.functional import logsigmoid
import numpy as np

def compute_volatility(_rnn_state_transition, _W_output_transition, _return_exp=False):
    logits_transition = torch.matmul(_rnn_state_transition, _W_output_transition).squeeze()
    return (
        logsigmoid(logits_transition).exp() if _return_exp 
        else (logsigmoid(logits_transition), logsigmoid(-logits_transition))
    )

def compute_emission(_rnn_state_emission, _W_output_emission):
    logits_emission = torch.matmul(_rnn_state_emission, _W_output_emission)
    _p_gen = torch.sigmoid(logits_emission)
    return _p_gen / _p_gen.sum(dim=-1, keepdim=True)


def compute_emission_mastermind(_rnn_state_emission, _W_output_emission):
    logits_emission = torch.matmul(_rnn_state_emission, _W_output_emission).squeeze()
    _p_gen = torch.sigmoid(logits_emission)
    return 1 - _p_gen * 0.5

def compute_association(_rnn_state_association, _W_output_association):
    return torch.matmul(_rnn_state_association, _W_output_association).squeeze()

def stimfun_colour(x, alpha, omega):
  """
  Python implementation of the stimulus distortion for the 'category' task.
  """
  # Note: np.arctanh is the equivalent of atanh in MATLAB
  return np.round(np.tanh(np.exp(alpha) * (np.arctanh(x) - omega)), 2)


def stimfun_bandit(x, alpha, omega):
  """
  Python implementation of the stimulus distortion for the 'bandit' task.
  """
  # Calculate the normalization bounds A and B
  A = np.exp(0.5 * alpha) * (-1 - omega)
  B = np.exp(-0.5 * alpha) * (1 - omega)
  
  # Apply the piecewise linear function
  y = np.where(x <= omega, 
               np.exp(0.5 * alpha) * (x - omega), 
               np.exp(-0.5 * alpha) * (x - omega))
  
  # Normalize the output to be between -1 and 1
  # Add a small epsilon to the denominator to avoid division by zero if B-A is close to 0
  epsilon = 1e-9
  y_normalized = -1 + 2 * (y - A) / (B - A + epsilon)
  
  return np.round(y_normalized, 2)