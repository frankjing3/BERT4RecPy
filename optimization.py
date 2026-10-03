"""Port of the original BERT optimizer (Adam without bias correction).

Original optimization.py: Copyright 2018 The Google AI Language Team Authors.
Licensed under the Apache License, Version 2.0; see LICENSE.
Modified for PyTorch in September 2026.
"""
import torch


class BertAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-6):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError('BertAdam requires dense gradients')
                state = self.state[parameter]
                if not state:
                    state['m'] = torch.zeros_like(parameter)
                    state['v'] = torch.zeros_like(parameter)
                m, v = state['m'], state['v']
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = m / (v.sqrt() + group['eps'])
                if group['weight_decay']:
                    update.add_(parameter, alpha=group['weight_decay'])
                parameter.add_(update, alpha=-group['lr'])
        return loss


def create_optimizer(model, learning_rate):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        # All one-dimensional parameters here are biases or LayerNorm parameters.
        (no_decay if parameter.ndim == 1 else decay).append(parameter)
    return BertAdam([{'params': decay, 'weight_decay': 0.01},
                     {'params': no_decay, 'weight_decay': 0.0}], lr=learning_rate)


def learning_rate_at(step, base_lr, total_steps, warmup_steps):
    """step is the zero-based number of updates already completed."""
    if warmup_steps and step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr * max(0.0, 1.0 - step / total_steps)
