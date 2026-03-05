import math
import torch
import torch.nn as nn
import numpy as np

EPS = 1e-6


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        # 数值稳定版本：用 logsumexp 替代 exp().mean().log()，避免 exp 溢出
        input_log_sum_exp = torch.logsumexp(input, 0) - math.log(input.shape[0])

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        # 数值稳定版本：在 log 空间计算梯度，避免 exp 直接溢出
        # 原式: grad_i = exp(input_i) / (running_mean + EPS) / N
        # 等价: log(grad_i) = input_i - log(running_mean + EPS) - log(N)
        log_grad = input - torch.log(running_mean + EPS) - math.log(input.shape[0])
        # clamp 防止 exp 溢出 (exp(80) ≈ 5.5e34，float32 安全范围内)
        log_grad = torch.clamp(log_grad, max=80.0)
        grad = grad_output * log_grad.exp().detach()
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    log_mean_exp = (torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    # clamp 防止 exp 溢出
    t_exp = torch.exp(torch.clamp(log_mean_exp, max=80.0))
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    return t_log, running_mean


class Mine(nn.Module):
    def __init__(self, model, loss_type='mine', alpha=0.01):
        super().__init__()
        self.running_mean = 0
        self.loss_type = loss_type
        self.alpha = alpha
        self.model = model

    def forward(self, x, z, z_marg=None):
        x = x.reshape(x.shape[0], -1)
        z = z.reshape(z.shape[0], -1)

        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        xz = torch.cat((x, z), dim=1)
        xz_marg = torch.cat((x, z_marg), dim=1)
        t = self.model(xz).mean()
        t_marg = self.model(xz_marg)

        if self.loss_type in ['mine']:
            second_term, self.running_mean = ema_loss(t_marg, self.running_mean, self.alpha)
        elif self.loss_type in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss_type in ['mine_biased']:
            second_term = torch.logsumexp(t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def get_mi(self, x, z, z_marg=None):
        mi = -self.forward(x, z, z_marg)
        return mi
