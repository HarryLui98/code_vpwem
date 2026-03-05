from typing import Optional
from functools import partial
import math
import torch
import torch.nn as nn

import hydra
from omegaconf import DictConfig, OmegaConf

from cleandiffuser.utils import (FourierEmbedding, PositionalEmbedding,
                                 SinusoidalEmbedding)

from .base_nn_diffusion import BaseNNDiffusion
from einops import rearrange

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# def init_weight(module):
#     ignore_types = (
#         nn.Dropout,
#         SinusoidalEmbedding,
#         FourierEmbedding,
#         PositionalEmbedding,
#         nn.TransformerEncoderLayer,
#         nn.TransformerDecoderLayer,
#         nn.TransformerEncoder,
#         nn.TransformerDecoder,
#         nn.ModuleList,
#         nn.Mish,
#         nn.Sequential)

#     if isinstance(module, (nn.Linear, nn.Embedding)):
#         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             torch.nn.init.zeros_(module.bias)

#     elif isinstance(module, nn.MultiheadAttention):
#         weight_names = [
#             'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
#         for name in weight_names:
#             weight = getattr(module, name)
#             if weight is not None:
#                 torch.nn.init.normal_(weight, mean=0.0, std=0.02)

#         bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
#         for name in bias_names:
#             bias = getattr(module, name)
#             if bias is not None:
#                 torch.nn.init.zeros_(bias)

#     elif isinstance(module, nn.LayerNorm):
#         torch.nn.init.zeros_(module.bias)
#         torch.nn.init.ones_(module.weight)

#     elif isinstance(module, MaIL):
#         torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
#         if module.obs_emb is not None:
#             torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)

#     elif isinstance(module, ignore_types):
#         # no param
#         pass
#     else:
#         pass
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls=nn.Identity,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)
    

class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, inference_params=None, cond=None):

        # hidden_states = self.embedding(input_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

            if cond is not None:
                hidden_states = hidden_states + cond

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MaIL(BaseNNDiffusion):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            mamba: DictConfig,
            obs_dim: int,
            action_dim: int,
            embed_dim: int,
            obs_seq_len: int,
            action_seq_len: int,
            embed_pdrob: float = 0,
            goal_conditioned: bool = False,
            goal_seq_len: int = 0,
            goal_drop: float = 0.1,
            linear_output: bool = False,
            device: str = 'cuda:0',
    ):
        super().__init__(emb_dim=embed_dim)

        self.mamba = hydra.utils.instantiate(mamba)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        self.obs_steps = obs_seq_len
        self.action_steps = action_seq_len
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len + action_seq_len - 1

        self.tok_emb = nn.Linear(obs_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    # def configure_optimizers(self, train_config):
    #     """
    #     This long function is unfortunately doing something very simple and is being very defensive:
    #     We are separating out all parameters of the model into two buckets: those that will experience
    #     weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    #     We are then returning the PyTorch optimizer object.
    #     """

    #     # separate out all parameters to those that will and won't experience regularizing weight decay
    #     decay = set()
    #     no_decay = set()
    #     whitelist_weight_modules = (torch.nn.Linear,)
    #     blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    #     for mn, m in self.named_modules():
    #         for pn, p in m.named_parameters():
    #             fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

    #             if pn.endswith("bias"):
    #                 # all biases will not be decayed
    #                 no_decay.add(fpn)
    #             elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
    #                 # weights of whitelist modules will be weight decayed
    #                 decay.add(fpn)
    #             elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
    #                 # weights of blacklist modules will NOT be weight decayed
    #                 no_decay.add(fpn)

    #     # special case the position embedding parameter in the root GPT module as not decayed
    #     no_decay.add("pos_emb")

    #     # validate that we considered every parameter
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     inter_params = decay & no_decay
    #     union_params = decay | no_decay
    #     assert (
    #             len(inter_params) == 0
    #     ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    #     assert (
    #             len(param_dict.keys() - union_params) == 0
    #     ), "parameters %s were not separated into either decay/no_decay set!" % (
    #         str(param_dict.keys() - union_params),
    #     )

    #     # create the pytorch optimizer object
    #     optim_groups = [
    #         {
    #             "params": [param_dict[pn] for pn in sorted(list(decay))],
    #             "weight_decay": train_config.weight_decay,
    #         },
    #         {
    #             "params": [param_dict[pn] for pn in sorted(list(no_decay))],
    #             "weight_decay": 0.0,
    #         },
    #     ]
    #     optimizer = torch.optim.AdamW(
    #         optim_groups, lr=train_config.learning_rate, betas=train_config.betas
    #     )
    #     return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals = None,
            uncond: Optional[bool] = False,
    ):

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])

        input_seq = torch.cat([emb_t, state_x, action_x], dim=1)

        mamba_output = self.mamba(input_seq)

        pred_actions = self.action_pred(mamba_output)[:, (t+1):, :]

        return pred_actions

    def get_params(self):
        return self.parameters()