from collections.abc import Callable, Iterable
from typing import Optional, Type
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration
        return loss

class AdamW(torch.optim.Optimizer):
    """
    Implementation of the AdamW optimizer.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        
        for group in self.param_groups:
            
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0 
                    state['m'] = torch.zeros_like(p.data) 
                    state['v'] = torch.zeros_like(p.data) 

                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']
                

                m_next = m * beta1 + (1 - beta1) * g 
                v_next = v * beta2 + (1 - beta2) * (g * g)
                state['m'] = m_next
                state['v'] = v_next

                m_t = m_next / (1 - beta1 ** t)
                v_t = v_next / (1 - beta2 ** t)

                ####### Assignment instructions are not correct here. ########
                p_f = p.data - (lr * weight_decay) * p.data
                p_updated = p_f - lr* m_t / (torch.sqrt(v_t) + eps)
                p.data.copy_(p_updated)

        return None


if __name__ == "__main__":

    lr, iter = 0.001, 10
    torch.manual_seed(0)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    for t in range(iter):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer