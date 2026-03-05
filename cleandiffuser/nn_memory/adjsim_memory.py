import torch
import torch.nn as nn

from cleandiffuser.utils import at_least_ndim
from cleandiffuser.nn_memory.base_nn_memory import BaseNNMemory
# from sklearn.cluster import KMeans
import torch.nn.functional as F


def get_mask(mask: torch.Tensor, mask_shape: tuple, dropout: float, train: bool, device: torch.device):
    if train:
        mask = (torch.rand(mask_shape, device=device) > dropout).float()
    else:
        mask = 1. if mask is None else mask
    return mask

def compress_one_step(memory: torch.Tensor, compression_size: torch.Tensor):
        B, T, N, D = memory.shape
        # Compute cosine similarity between adjacent time steps
        sim_matrix = F.cosine_similarity(memory[:, :-1, :], memory[:, 1:, :], dim=-1)
        _, max_sim_indices = torch.max(sim_matrix, dim=1, keepdim=True)

        # Calculate source and dst indices for compression
        src_indices = max_sim_indices + 1 # (B, T-1, N)
        dst_indices = torch.arange(T - 1).to(memory.device)[None, :, None].repeat(B, 1, N)
        dst_indices[dst_indices > max_sim_indices] += 1

        # Gather source and dst memory banks and sizes
        src_memory_bank = memory.gather(dim=1, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, D))
        dst_memory_bank = memory.gather(dim=1, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, D))
        src_size = compression_size.gather(dim=1, index=src_indices)
        dst_size = compression_size.gather(dim=1, index=dst_indices)

        # Multiply the memory banks by their corresponding sizes
        src_memory_bank *= src_size.unsqueeze(-1)
        dst_memory_bank *= dst_size.unsqueeze(-1)

        # Compress the memory bank by adding the source memory bank to the dst memory bank
        dst_memory_bank.scatter_add_(dim=1, index=max_sim_indices.unsqueeze(-1).expand(-1, -1, -1, D), src=src_memory_bank)
        dst_size.scatter_add_(dim=1, index=max_sim_indices, src=src_size)

        # Normalize the dst memory bank by its size
        memory = dst_memory_bank / dst_size.unsqueeze(-1)
        compression_size = dst_size

        return memory, compression_size

class AdjSimMemory(BaseNNMemory):
    """
    From https://github.com/boheumd/MA-LMM

    Input:
        - condition: Any
        - mask : Optional, (b, ) or None, None means no mask

    Output:
        - condition: (b, *cond_out_shape)
    """

    def __init__(self, mem_compress_length=4):
        super().__init__()
        self.mem_compress_length = mem_compress_length
        self.condition_mem_cache_list = []
        self.condition_mem_cache_step_list = []
        self.step = 0
        self.dropout = 0.25
    
    def forward(self, memory: torch.Tensor, mask: torch.Tensor = None):
        B, T, D = memory.shape
        compression_size = torch.ones((B, T), device=memory.device)
        for _compress_step in range(T - self.mem_compress_length):
            memory, compression_size = compress_one_step(memory, compression_size)
        mask = at_least_ndim(get_mask(
            mask, (memory.shape[0],), self.dropout, self.training, memory.device), memory.dim())
        return memory * mask

    def reset(self):
        self.condition_mem_cache_list = []
        self.condition_mem_cache_step_list = []
        self.step = 0
        self.compression_size = None
    
    def inference(self, new_memory: torch.Tensor):
        B, T, D = new_memory.shape
        for t_idx in range(T):
            i = len(self.condition_mem_cache_list)
            if i < self.mem_compress_length:
                current_step = self.step * torch.ones(B, device=new_memory.device, dtype=torch.int)
                self.condition_mem_cache_list.append(new_memory[:, t_idx, :])
                self.condition_mem_cache_step_list.append(current_step)
            else:
                if i == self.mem_compress_length and self.compression_size is None:
                    self.compression_size = torch.ones((B, self.mem_compress_length+1), device=new_memory.device)
                self.condition_mem_cache_list.append(new_memory[:, t_idx, :])
                memory = torch.stack(self.condition_mem_cache_list, dim=1) # (B, T_max+1, D)
                compression_size = self.compression_size
                memory, compression_size = compress_one_step(memory, compression_size)
                self.condition_mem_cache_list = [memory[:, t, :] for t in range(memory.shape[1])]
                self.compression_size = compression_size
        
        return memory