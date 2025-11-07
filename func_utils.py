import torch
from torch.nn.functional import logsigmoid

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

def compute_association(_rnn_state_association, _W_output_association):
    return torch.matmul(_rnn_state_association, _W_output_association).squeeze()