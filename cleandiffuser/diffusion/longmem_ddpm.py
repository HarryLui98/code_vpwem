from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import amp

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition, IdentityCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.nn_diffusion.chitransformer import ChiTransformer
from cleandiffuser.nn_diffusion.chitransformerptp import ChiTransformerPTP
from cleandiffuser.nn_diffusion.mail import MaIL
from cleandiffuser.nn_diffusion.longmem_mail import LongMemMaIL
from cleandiffuser.nn_memory.base_nn_memory import BaseNNMemory
from cleandiffuser.nn_memory.random_memory import RandomMemory
from cleandiffuser.nn_memory.kmeans_memory import KMeansMemory
from cleandiffuser.nn_memory.adjsim_memory import AdjSimMemory
from cleandiffuser.nn_memory.qformer_memory import QFormerMemory
from cleandiffuser.utils import (
    at_least_ndim,
    cosine_beta_schedule,
    linear_beta_schedule,
    to_tensor)
from cleandiffuser.utils.mlp_correlation import MLP
from cleandiffuser.utils.mine_utils import Mine
from .basic import DiffusionModel
from copy import deepcopy
from cleandiffuser.nn_diffusion.longmem_chitransformerptp import LongMemChiTransformerPTP


class LongMemDDPM(DiffusionModel):

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,
            nn_memory: Optional[BaseNNMemory] = None,

            # ----------------- Masks ----------------- #
            # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`

            # ------------------ Plugs ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier: Optional[BaseClassifier] = None,

            # ------------------ Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            diffusion_steps: int = 1000,
            ema_rate: float = 0.999,
            optim_params: Optional[dict] = None,
            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,
            short_cond_dropout: float = 0.0,
            long_cond_dropout: float = 0.0,

            # ------------------- DPM Params ------------------- #
            predict_noise: bool = True,
            beta_schedule: str = "cosine",  # or cosine
            beta_schedule_params: Optional[dict] = None,
            device: Union[torch.device, str] = "cpu",

            # ----- action predictability ------ #
            action_range: Optional[torch.Tensor] = None,
            action_min: Optional[torch.Tensor] = None,

            # ----- others ------ #
            args_dict: Optional[dict] = None,
    ):
        # super().__init__(
        #     nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
        #     diffusion_steps, ema_rate, optim_params, device)

        if optim_params is None:
            optim_params = {"lr": 2e-4, "weight_decay": 1e-5}

        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.diffusion_steps = diffusion_steps
        self.ema_rate = ema_rate

        # nn_condition is None means that the model is not conditioned on any input.
        self.use_condition_model =  nn_condition is not None
        self.nn_condition_drop = IdentityCondition(dropout=short_cond_dropout)
        self.nn_memory_drop = IdentityCondition(dropout=long_cond_dropout)

        # In the code implementation of Diffusion models, it is common to maintain an exponential
        # moving average (EMA) version of the model for inference, as it has been observed that
        # this approach can result in more stable generation outcomes.
        self.model = nn.ModuleDict({
            "diffusion": nn_diffusion.to(self.device),
            "condition": nn_condition.to(self.device) if nn_condition is not None else None,
            "memory": nn_memory.to(self.device) if nn_memory is not None else None,})
        self.model_ema = deepcopy(self.model).requires_grad_(False)
        
        self.model.train()
        self.model_ema.eval()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optim_params)
        self.scaler = amp.GradScaler()

        self.classifier = classifier

        self.fix_mask = to_tensor(fix_mask, self.device)[None, ] if fix_mask is not None else 0.
        self.loss_weight = to_tensor(loss_weight, self.device)[None, ] if loss_weight is not None else 1.

        self.predict_noise = predict_noise

        if beta_schedule_params is None:
            beta_schedule_params = {}
        beta_schedule_params["T"] = self.diffusion_steps

        if beta_schedule == "linear":
            beta = linear_beta_schedule(**beta_schedule_params)
        elif beta_schedule == "cosine":
            beta = cosine_beta_schedule(**beta_schedule_params)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.beta = torch.tensor(beta, device=self.device, dtype=torch.float32)
        self.alpha = 1 - self.beta
        self.bar_alpha = torch.cumprod(self.alpha.clone(), 0)
        self.x_max, self.x_min = x_max, x_min
        if isinstance(nn_diffusion, ChiTransformerPTP) or isinstance(nn_diffusion, LongMemChiTransformerPTP):
            self.action_steps = nn_diffusion.T + 1 - nn_diffusion.To
            self.obs_steps = nn_diffusion.To
        elif isinstance(nn_diffusion, ChiTransformer):
            self.action_steps = nn_diffusion.T
            self.obs_steps = nn_diffusion.To
        elif isinstance(nn_diffusion, LongMemMaIL):
            self.action_steps = nn_diffusion.action_steps
            self.obs_steps = nn_diffusion.obs_steps
        elif isinstance(nn_diffusion, MaIL):
            self.action_steps = nn_diffusion.action_steps
            self.obs_steps = nn_diffusion.obs_steps
        
        self.inference_action_steps = args_dict.inference_action_steps
        # self.use_condition_cache = (self.obs_steps > self.action_steps)
        # if self.use_condition_cache:
        self.condition_cache_list = []
        
        self.pred_reg_coeff = args_dict.pred_reg_coeff
        if self.pred_reg_coeff:
            self.action_pred_criterion = nn.MSELoss(reduction='mean')
            self.dataset_action_pred_ratio = np.load(args_dict.dataset_action_pred_ratio_path).item()
            self.action_dim = x_max.shape[-1]
            self.dataset_action_pred_mlp = MLP(self.action_dim, 512, self.action_dim).to(device)
            dataset_action_pred_mlp_sd = torch.load(args_dict.dataset_action_pred_mlp_path, map_location=self.device)
            self.dataset_action_pred_mlp.load_state_dict(dataset_action_pred_mlp_sd)
            self.dataset_action_pred_mlp.eval()
            for name, param in self.dataset_action_pred_mlp.state_dict().items():
                param.requires_grad = False
            self.action_range = action_range.to(device)
            self.action_min = action_min.to(device)

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ---------------------------------------------------------------------------
    # Training

    def add_noise(self, x0, t=None, eps=None):
        t = torch.randint(self.diffusion_steps, (x0.shape[0],), device=self.device) if t is None else t
        eps = torch.randn_like(x0) if eps is None else eps
        bar_alpha = at_least_ndim(self.bar_alpha[t], x0.dim())
        xt = x0 * bar_alpha.sqrt() + eps * (1 - bar_alpha).sqrt()
        xt = xt * (1. - self.fix_mask) + x0 * self.fix_mask
        return xt, t, eps
    
    def reverse_noise(self, xt, t, eps):
        bar_alpha = at_least_ndim(self.bar_alpha[t], xt.dim())
        x0 = (xt - eps * (1 - bar_alpha).sqrt()) / bar_alpha.sqrt()
        x0 = x0 * (1. - self.fix_mask) + xt * self.fix_mask
        return x0

    def loss(self, x0, condition=None, memory=None, ep_step=None):
        xt, t, eps = self.add_noise(x0)
        memory_tensor = None
        compressed_memory = None
        with amp.autocast('cuda', dtype=torch.float16):
            condition = self.nn_condition_drop(self.model["condition"](condition) if ((condition is not None) and self.use_condition_model) else condition)
            # Process memory: if it's a dict (multi-modal), process through nn_condition first
            # If it's a tensor (e.g., dp_emb_mem), use it directly
            if memory is not None:
                if isinstance(memory, dict):
                    # Multi-modal memory: process through nn_condition first
                    if self.use_condition_model:
                        memory_tensor = self.model["condition"](memory)  # (B, T, D) when keep_horizon_dims=True
                    else:
                        raise ValueError("Memory is a dict but nn_condition is None. Cannot process multi-modal memory.")
                else:
                    # Single tensor memory (e.g., dp_emb_mem): use directly
                    memory_tensor = memory
                
                # Process through memory network
                compressed_memory = self.nn_memory_drop(self.model["memory"](memory_tensor, ep_step) if self.model["memory"] is not None else memory_tensor)
            else:
                compressed_memory = None
        if self.predict_noise:
            eps_hat = self.model["diffusion"](xt, t, condition, compressed_memory)
            loss = (eps_hat - eps) ** 2
            if self.pred_reg_coeff:
                x0_hat = self.reverse_noise(xt, t, eps_hat)
        else:
            x0_hat = self.model["diffusion"](xt, t, condition, compressed_memory)
            loss = (x0_hat - x0) ** 2

        action_loss = (loss * self.loss_weight * (1 - self.fix_mask)).mean()

        if self.pred_reg_coeff:
            x0_hat_unnorm = (x0_hat + 1.) / 2.
            x0_hat_unnorm = x0_hat_unnorm * self.action_range + self.action_min
            x = x0_hat_unnorm[:, :-1, :]  # Shape: (N, S-1, dim)
            y = x0_hat_unnorm[:, 1:, :]   # Shape: (N, S-1, dim)

            # Reshape to combine batch and sequence dimensions
            x = x.reshape(-1, self.action_dim)  # Shape: ((N*(S-1)), dim)
            y = y.reshape(-1, self.action_dim)  # Shape: ((N*(S-1)), dim)
            y_hat = self.dataset_action_pred_mlp(x)
            action_pred_ratio = self.action_pred_criterion(y_hat, y)
            reg_loss = (action_pred_ratio - self.dataset_action_pred_ratio) ** 2
            total_loss = action_loss + self.pred_reg_coeff * reg_loss
        else:
            total_loss = action_loss

        return total_loss, memory_tensor, compressed_memory, action_loss

   
    def update(self, x0, condition=None, memory=None, ep_step=None, update_ema=True, **kwargs):
        # Step 1: Update the main model (diffusion + condition + memory) with MI loss
        total_loss, memory_tensor, compressed_memory, action_loss = self.loss(x0, condition, memory, ep_step)
        self.scaler.scale(total_loss).backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if update_ema: self.ema_update()
        log = {"loss": total_loss.item(), "grad_norm": grad_norm, "action_loss": action_loss.item()}

        return log

    def update_classifier(self, x0, condition):
        xt, t, eps = self.add_noise(x0)
        log = self.classifier.update(xt, t, condition)
        return log

    # ---------------------------------------------------------------------------
    # Inference

    def predict_function(
            self, x, t, bar_alpha,
            use_ema=False, requires_grad=False,
            # ----------------- CFG ----------------- #
            condition_vec_cfg=None,
            w_cfg: float = 0.0,
            # ----------------- CG ----------------- #
            condition_vec_cg=None,
            w_cg: float = 1.0,
            memory=None,
    ):
        b = x.shape[0]
        model = self.model_ema if use_ema else self.model

        # ----------------- CFG ----------------- #
        with torch.set_grad_enabled(requires_grad):
            # if w_cfg != 0.0 and w_cfg != 1.0:
            #     repeat_dim = [2 if i == 0 else 1 for i in range(x.dim())]
            #     condition_vec_cfg = torch.cat([condition_vec_cfg, torch.zeros_like(condition_vec_cfg)], 0)
            #     pred = model["diffusion"](
            #         x.repeat(*repeat_dim), t.repeat(2), condition_vec_cfg)
            #     pred = w_cfg * pred[:b] + (1. - w_cfg) * pred[b:]
            # elif w_cfg == 0.0:
            #     pred = model["diffusion"](x, t, None)
            # else:
            pred = model["diffusion"](x, t, condition_vec_cfg, memory)

        # ----------------- CG ----------------- #
        if self.classifier is not None and w_cg != 0.0 and condition_vec_cg is not None:
            log_p, grad = self.classifier.gradients(x.clone(), t, condition_vec_cg)
            if self.predict_noise:
                pred = pred - w_cg * (1 - bar_alpha).sqrt() * grad
            else:
                pred = pred + w_cg * (1 - bar_alpha) / bar_alpha.sqrt() * grad
        else:
            log_p = None

        if self.predict_noise:
            if self.clip_pred:
                upper_bound = (x - bar_alpha.sqrt() * self.x_min) / (1 - bar_alpha).sqrt() \
                    if self.x_min is not None else None
                lower_bound = (x - bar_alpha.sqrt() * self.x_max) / (1 - bar_alpha).sqrt() \
                    if self.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
            pred = pred * (1 - self.fix_mask)
        else:
            if self.clip_pred:
                pred = pred.clip(self.x_min, self.x_max)
            pred = pred * (1 - self.fix_mask) + x * self.fix_mask

        return pred, {"log_p": log_p}

    def reset_cache(self):
        self.condition_cache_list = []

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = None,
            use_ema: bool = True,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        # initialize logger
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        # choose the model
        model = self.model_ema if use_ema else self.model

        # check `sample_steps`
        if sample_steps != self.diffusion_steps:
            import warnings
            warnings.warn(f"sample_steps != diffusion_steps, sample_steps will be set to diffusion_steps.")
            sample_steps = self.diffusion_steps

        # initialize the samples
        xt = torch.randn_like(prior, device=self.device) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()

        # preprocess the conditions and memories
        # if self.use_condition_cache:
        if len(self.condition_cache_list) == 0:
            with torch.set_grad_enabled(requires_grad):
                condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
                condition_vec_cg = condition_cg
                if condition_vec_cfg.shape[1] > self.obs_steps:
                    compressed_memory = model["memory"].inference(condition_vec_cfg[:, :-self.obs_steps, :])
                else:
                    compressed_memory = None
                for i in range(self.obs_steps, 0, -1):
                    self.condition_cache_list.append(condition_vec_cfg[:, -i, :])
                condition_vec_cfg = condition_vec_cfg[:, -self.obs_steps:, :]
        else:
            with torch.set_grad_enabled(requires_grad):
                condition_cfg_new = dict()
                for keys in condition_cfg.keys():
                    condition_cfg_new[keys] = condition_cfg[keys][:, -self.inference_action_steps:, ...]
                    mask_cfg_new = mask_cfg
                condition_vec_cfg_new = model["condition"](condition_cfg_new, mask_cfg_new)
                for i in range(self.inference_action_steps):
                    self.condition_cache_list.append(condition_vec_cfg_new[:, i, :])
                condition_to_compress = self.condition_cache_list[:-self.obs_steps]
                self.condition_cache_list = self.condition_cache_list[-self.obs_steps:]
                condition_vec_cfg = torch.stack(self.condition_cache_list, dim=1)
                condition_vec_cg = condition_cg
                compressed_memory = self.model["memory"].inference(torch.stack(condition_to_compress, dim=1))
        # else:
        #     with torch.set_grad_enabled(requires_grad):
        #         condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
        #         condition_vec_cg = condition_cg
        
        # enter the sampling loop
        for t in range(self.diffusion_steps - 1, -1, -1):

            t_batch = torch.tensor(t, device=self.device, dtype=torch.long).repeat(n_samples)
            bar_alpha = self.bar_alpha[t]
            bar_alpha_prev = self.bar_alpha[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            alpha = self.alpha[t]
            beta = self.beta[t]

            # predict eps_theta or x_theta with CG/CFG
            pred_theta, log = self.predict_function(
                xt, t_batch, bar_alpha,
                use_ema=use_ema,
                requires_grad=requires_grad,
                condition_vec_cfg=condition_vec_cfg,
                condition_vec_cg=condition_vec_cg,
                w_cfg=w_cfg, w_cg=w_cg,
                memory=compressed_memory)

            # one step denoise
            if self.predict_noise:

                xt = 1 / alpha.sqrt() * (xt - beta / (1 - bar_alpha).sqrt() * pred_theta)

            else:

                xt = 1 / (1 - bar_alpha) * (
                    alpha.sqrt() * (1 - bar_alpha_prev) * xt +
                    beta * bar_alpha_prev.sqrt() * pred_theta)

            if t != 0:
                xt = xt + (beta * (1 - bar_alpha_prev) / (1 - bar_alpha)).sqrt() * torch.randn_like(xt)

            # Fix the known portion
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history: log["sample_history"][:, 1] = xt.cpu().numpy()

        # calculate the final log_p
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t[-1].repeat(n_samples), condition_vec_cg)
            log["log_p"] = logp

        return xt, log

    def sample_x(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = None,
            extra_sample_steps: int = 8,
            use_ema: bool = True,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        # initialize logger
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        # choose the model
        model = self.model_ema if use_ema else self.model

        # check `sample_steps`
        if sample_steps != self.diffusion_steps:
            import warnings
            warnings.warn(f"sample_steps != diffusion_steps, sample_steps will be set to diffusion_steps.")
            sample_steps = self.diffusion_steps

        # initialize the samples
        xt = torch.randn_like(prior, device=self.device) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()

        # preprocess the conditions
        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # enter the sampling loop
        for t in range(self.diffusion_steps - 1, -1, -1):

            t_batch = torch.tensor(t, device=self.device, dtype=torch.long).repeat(n_samples)
            bar_alpha = self.bar_alpha[t]
            bar_alpha_prev = self.bar_alpha[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            alpha = self.alpha[t]
            beta = self.beta[t]

            # predict eps_theta or x_theta with CG/CFG
            pred_theta, log = self.predict_function(
                xt, t_batch, bar_alpha,
                use_ema=use_ema,
                requires_grad=requires_grad,
                condition_vec_cfg=condition_vec_cfg,
                condition_vec_cg=condition_vec_cg,
                w_cfg=w_cfg, w_cg=w_cg)

            # one step denoise
            if self.predict_noise:

                xt = 1 / alpha.sqrt() * (xt - beta / (1 - bar_alpha).sqrt() * pred_theta)

            else:

                xt = 1 / (1 - bar_alpha) * (
                    alpha.sqrt() * (1 - bar_alpha_prev) * xt +
                    beta * bar_alpha_prev.sqrt() * pred_theta)

            if t != 0:
                xt = xt + (beta * (1 - bar_alpha_prev) / (1 - bar_alpha)).sqrt() * torch.randn_like(xt)

            # Fix the known portion
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history: log["sample_history"][:, 1] = xt.cpu().numpy()

        if extra_sample_steps > 0:

            t_batch = torch.tensor(0, device=self.device, dtype=torch.long).repeat(n_samples)
            bar_alpha = self.bar_alpha[0]
            bar_alpha_prev = torch.tensor(1.0, device=self.device)
            alpha = self.alpha[0]
            beta = self.beta[0]

            for _ in range(extra_sample_steps):

                # predict eps_theta or x_theta with CG/CFG
                pred_theta, log = self.predict_function(
                    xt, t_batch, bar_alpha,
                    use_ema=use_ema,
                    requires_grad=requires_grad,
                    condition_vec_cfg=condition_vec_cfg,
                    condition_vec_cg=condition_vec_cg,
                    w_cfg=w_cfg, w_cg=w_cg)

                # one step denoise
                if self.predict_noise:

                    xt = 1 / alpha.sqrt() * (xt - beta / (1 - bar_alpha).sqrt() * pred_theta)

                else:

                    xt = 1 / (1 - bar_alpha) * (
                            alpha.sqrt() * (1 - bar_alpha_prev) * xt +
                            beta * bar_alpha_prev.sqrt() * pred_theta)

                # Fix the known portion
                xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        # calculate the final log_p
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t[-1].repeat(n_samples), condition_vec_cg)
            log["log_p"] = logp

        return xt, log
