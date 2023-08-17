import torch
from torch.optim.optimizer import Optimizer
import copy as cp


def _flatten_tensors(tensors):
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def _unflatten_tensors(flat, tensors):
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


class VRL(Optimizer):
    def __init__(self, params, lr, update_period=2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, vrl=True, local=None):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.vrl = vrl
        self.iter_cnt = 0
        self.last_lr = 1
        self.Round = 0
        self.Fac = 0.005 # 0.01
        if not local:
            update_period = 1
        self.update_period = update_period
        super(VRL, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(VRL, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def update_params(self):
        with torch.no_grad():
            for group in self.param_groups:
                momentum = group['momentum']
                self.last_lr = group["lr"]
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if self.vrl:
                        param_state["last_param_buff"] = p.clone().detach_()
                    if 'momentum_buffer' in param_state:
                        param_state['momentum_buffer'].zero_()

    def update_delta(self, local_steps):
            if local_steps > 0:
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        param_state = self.state[p]
                        if self.vrl:
                            param_state["vrl_buff"] = param_state["vrl_buff"] + self.Fac * 1.0 / (self.last_lr * local_steps) * (p - param_state["last_param_buff"])
             

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                if weight_decay != 0:
                    d_p.add_(p.data,alpha=weight_decay)
                 
                param_state = self.state[p]
                
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p,alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf,alpha=momentum)
                    else:
                        d_p = buf
                if self.vrl:
                    if 'vrl_buff' not in param_state:
                        param_state['vrl_buff'] = torch.zeros_like(d_p).detach()
                    d_p = d_p - param_state['vrl_buff']
                p.data.add_(d_p, alpha=-group['lr'])
        return loss
        
        
    
