import torch
from torch.optim.optimizer import Optimizer


class D_SAM(Optimizer):
    """
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): initial learning rate (default: 1e-1)
        beta (float, optional): coefficients used for computing running averages of gradient (default: 0.01)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.)
    """

    def __init__(self,
                 params,
                 lr=1e-1,
                 beta=0.9, 
                 rho=0.05,
                 adaptive=False,
                 weight_decay=0.,
                 ):
        assert lr > 0.0, ValueError("Invalid initial learning rate: {}".format(lr))
        assert 0.0 <= beta <= 1.0, ValueError("Invalid beta value: {}".format(beta))
        assert 0.0 <= rho <= 10.0, ValueError("Invalid rho value: {}".format(rho))
        assert weight_decay >= 0.0, ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            adaptive=adaptive,
            weight_decay=weight_decay,
            beta=beta,
            rho=rho,
        )

        super(D_SAM, self).__init__(params, defaults)   
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def first_step(self, zero_grad=False):
        beta = self.param_groups[0]['beta']
        lr = self.param_groups[0]['lr']
        rho = self.param_groups[0]['rho']
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    continue
                if len(state) == 0:
                    if beta != 0:
                        state['momentum_buffer'] = torch.zeros_like(p)  # 动量

                p.data.add_(-lr * p.grad.data)  # 进行第一次参数更新
                state['p_origin'] = p.data.clone() # 将原位置保存    
                e_w = (torch.pow(p, 2)if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.data.add_(e_w)
                p = p.requires_grad_(True)

        # if zero_grad: self.zero_grad() #注意：D-SAM第一步不需要清空梯度，这样第二步所计算的梯度可以叠加进第一步的梯度

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        beta = self.param_groups[0]['beta']
        lr = self.param_groups[0]['lr']
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                p.data.copy_(state['p_origin'])  # 回到原位置
                if group['weight_decay'] != 0: 
                    p.grad.data.add_(group['weight_decay'] * p.data)
                if beta != 0:
                    v = state['momentum_buffer']
                    v.mul_(beta).add_(p.grad.data)  # 计算动量
                else:
                    v = p.grad.data
                state['momentum_buffer'] = v
                p.data.add_(-lr * v)

        if zero_grad: self.zero_grad()
        
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the net
                and returns the loss.
        """

        loss = None
        beta = self.param_groups[0]['beta']
        lr = self.param_groups[0]['lr']
        rho = self.param_groups[0]['rho']
        if closure is not None: 
            with torch.enable_grad():
                loss, output = closure()

        grad_norm = self._grad_norm()
        
        for group in self.param_groups: 
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    continue
                if len(state) == 0:
                    if beta != 0:
                        state['momentum_buffer'] = torch.zeros_like(p)  # 动量

                p.data.add_(-lr * p.grad.data)  # 进行第一次参数更新
                state['p_origin'] = p.data.clone() # 将原位置保存    
                e_w = (torch.pow(p, 2)if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.data.add_(e_w)
                p = p.requires_grad_(True)

        if closure is not None: ### 计算第二次梯度
            loss, output = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    p.data.copy_(state['p_origin'])  # 回到原位置
                    if group['weight_decay'] != 0: 
                        p.grad.data.add_(group['weight_decay'] * p.data)
                    if beta != 0:
                        v = state['momentum_buffer']
                        v.mul_(beta).add_(p.grad.data)  # 计算动量
                    else:
                        v = p.grad.data
                    state['momentum_buffer'] = v
                    p.data.add_(-lr * v)
                
        return (loss, output)
