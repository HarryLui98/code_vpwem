import torch
import torch.nn as nn

from cleandiffuser.utils import at_least_ndim
from cleandiffuser.nn_memory.base_nn_memory import BaseNNMemory


def get_mask(mask: torch.Tensor, mask_shape: tuple, dropout: float, train: bool, device: torch.device):
    if train:
        mask = (torch.rand(mask_shape, device=device) > dropout).float()
    else:
        mask = 1. if mask is None else mask
    return mask


class RandomMemory(BaseNNMemory):
    """
    In decision-making tasks, generating condition selections can be very diverse,
    including cumulative rewards, languages, images, demonstrations, and so on.
    It can even be a combination of these conditions. Therefore, we aim for
    nn_condition to handle diverse condition selections flexibly and
    ultimately output a tensor of shape (b, *cond_out_shape).

    Input:
        - condition: Any
        - mask : Optional, (b, ) or None, None means no mask

    Output:
        - condition: (b, *cond_out_shape)
    """

    def __init__(self, mem_compress_length=4):
        super().__init__()
        self.condition_mem_cache_list = []
        self.condition_mem_cache_step_list = []
        self.step = 0
        self.mem_compress_length = mem_compress_length
        self.dropout = 0.25

    def forward(self, memory: torch.Tensor, mask: torch.Tensor = None):
        B, T, D = memory.shape
        random_indices = torch.randperm(T)[:self.mem_compress_length]
        sorted_random_indices, _ = torch.sort(random_indices)
        cp_memory = memory[:, sorted_random_indices, :]
        mask = at_least_ndim(get_mask(
            mask, (cp_memory.shape[0],), self.dropout, self.training, cp_memory.device), cp_memory.dim())
        return cp_memory * mask

    def reset(self):
        self.condition_mem_cache_list = []
        self.condition_mem_cache_step_list = []
        self.step = 0
    
    def inference(self, new_memory: torch.Tensor):
        # reservoir_sampling
        B, T, D = new_memory.shape
        # new_memory (B, T, D)
        # new_memory_list = [new_memory[:, t, :] for t in range(T)]
        # 处理前 k 个元素
        for t_idx in range(T):
            i = len(self.condition_mem_cache_list)
            if i < self.mem_compress_length:
                current_step = self.step * torch.ones(B, device=new_memory.device, dtype=torch.int)
                self.condition_mem_cache_list.append(new_memory[:, t_idx, :])
                self.condition_mem_cache_step_list.append(current_step)
            else:
                j = torch.randint(0, self.step, (B,))  # 生成随机数 j
                # 创建掩码，检查哪些 j 值小于 mem_compress_length
                mask = j < self.mem_compress_length

                # 使用索引更新 condition_mem_cache_list 和 condition_mem_cache_step_list
                # 选取满足条件的索引
                selected_indices = j[mask]
                selected_b = torch.arange(B)[mask]

                # 更新缓存
                if len(selected_indices) > 0:  # 确保有满足条件的索引
                    for idx in range(len(selected_indices)):
                        self.condition_mem_cache_list[selected_indices[idx]][selected_b[idx]] = new_memory[selected_b[idx], t_idx, :]
                        self.condition_mem_cache_step_list[selected_indices[idx]][selected_b[idx]] = self.step
            self.step += 1
        # 将缓存转换为张量
        memory_tensor = torch.stack(self.condition_mem_cache_list, dim=1)
        memory_step_tensor = torch.stack(self.condition_mem_cache_step_list, dim=1)
        # 根据 step 排序
        sorted_indices = torch.argsort(memory_step_tensor, dim=1)
        memory_sorted_by_step = memory_tensor[torch.arange(B).unsqueeze(1), sorted_indices]
        # for debug only
        memory_step_tensor_sorted_by_step = memory_step_tensor[torch.arange(B).unsqueeze(1), sorted_indices]
        return memory_sorted_by_step