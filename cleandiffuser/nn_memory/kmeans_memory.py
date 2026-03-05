import torch
import torch.nn as nn

from cleandiffuser.utils import at_least_ndim
from cleandiffuser.nn_memory.base_nn_memory import BaseNNMemory
# from sklearn.cluster import KMeans


def get_mask(mask: torch.Tensor, mask_shape: tuple, dropout: float, train: bool, device: torch.device):
    if train:
        mask = (torch.rand(mask_shape, device=device) > dropout).float()
    else:
        mask = 1. if mask is None else mask
    return mask


class KMeans:
    def __init__(self, k, max_iters=10, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centers = None

    def initial_fit(self, data):
        """
        初始化并拟合数据
        Args:
            data: (B, T, N, D) 张量
        """
        B, T, N, D = data.size()
        device = data.device
        
        # 随机初始化聚类中心，确保设备匹配
        if T < self.k:
            # 如果数据点少于聚类数，重复使用数据点
            indices = torch.arange(T, device=device).repeat((self.k + T - 1) // T)[:self.k]
        else:
            indices = torch.randperm(T, device=device)[:self.k]
        sorted_indices, _ = torch.sort(indices)
        self.centers = data[:, sorted_indices, :, :].clone()  # (B, K, N, D)
        
        # 使用全部数据进行初始拟合
        self.online_fit(data)

    def online_fit(self, new_data):
        """
        在线更新聚类中心
        Args:
            new_data: (B, T, N, D) 张量，新数据或全部数据
        """
        if self.centers is None:
            raise ValueError("Centers not initialized. Call initial_fit first.")
        
        B, T, N, D = new_data.size()
        device = new_data.device
        
        # 确保 centers 和设备匹配
        if self.centers.device != device:
            self.centers = self.centers.to(device)

        for iteration in range(self.max_iters):
            # 计算距离：对每个 query token (N维度) 分别计算到各聚类中心的距离
            # 原始逻辑：new_data.transpose(1,2): (B, N, T, D), centers.transpose(1,2): (B, N, K, D)
            # 使用 cdist 对最后一个维度计算距离，得到 (B, N, T, K)
            new_data_transposed = new_data.transpose(1, 2).contiguous()  # (B, N, T, D)
            centers_transposed = self.centers.transpose(1, 2).contiguous()  # (B, N, K, D)
            
            # 使用 cdist 计算距离（更高效）
            distances = torch.cdist(new_data_transposed, centers_transposed)  # (B, N, T, K)
            
            # 找到最近的聚类中心并转置回 (B, T, N)
            labels = torch.argmin(distances, dim=-1).transpose(1, 2).contiguous()  # (B, T, N)

            # 保存旧的中心用于收敛检测
            old_centers = self.centers.clone()

            # 向量化更新聚类中心
            # 对每个 batch b，每个 query n，每个聚类 k，计算属于该聚类的所有时间步的平均值
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.k).float()  # (B, T, N, K)
            
            # 计算每个聚类的成员数量（按 batch 和 query 维度）
            cluster_counts = labels_one_hot.sum(dim=1)  # (B, N, K) - sum over time dimension
            cluster_counts = torch.clamp(cluster_counts, min=1e-8)  # 避免除零
            
            # 向量化更新：对每个聚类，计算其成员的平均值
            # new_data: (B, T, N, D), labels_one_hot: (B, T, N, K)
            weighted_data = torch.einsum('btnd,btnk->bnkd', new_data, labels_one_hot).contiguous()  # (B, N, K, D)
            # cluster_counts: (B, N, K), need to expand to (B, N, K, 1) for broadcasting with (B, N, K, D)
            cluster_counts_expanded = cluster_counts.unsqueeze(-1)  # (B, N, K, 1)
            new_centers = (weighted_data / cluster_counts_expanded).contiguous()  # (B, N, K, D) / (B, N, K, 1) -> (B, N, K, D)
            self.centers = new_centers.transpose(1, 2).contiguous()  # (B, K, N, D)
            
            # 检查收敛：计算最大变化量
            center_diff = torch.norm(self.centers - old_centers, dim=-1)  # (B, K, N)
            max_diff = center_diff.max().item()
            if max_diff < self.tol:
                break

    def get_centers(self):
        """返回聚类中心"""
        if self.centers is None:
            raise ValueError("Centers not initialized. Call initial_fit first.")
        return self.centers


class KMeansMemory(BaseNNMemory):
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
        self.mem_compress_length = mem_compress_length
        self.online_kmeans = KMeans(k=self.mem_compress_length)
        self.step = 0
        self.dropout = 0.25

    def forward(self, memory: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            memory: (B, T, D) 张量
        Returns:
            cp_memory: (B, mem_compress_length, D) 压缩后的记忆
        """
        B, T, D = memory.shape
        # 添加 N 维度以匹配 KMeans 的输入格式 (B, T, N, D)
        # 这里 N=1，表示单个 query token
        memory_expanded = memory.unsqueeze(2)  # (B, T, 1, D)
        
        # k-means clustering
        kmeans = KMeans(k=self.mem_compress_length)
        kmeans.initial_fit(memory_expanded)
        cp_memory = kmeans.get_centers()  # (B, K, N, D)
        
        # 移除 N 维度，返回 (B, K, D)
        cp_memory = cp_memory.squeeze(2)  # (B, mem_compress_length, D)
        
        mask = at_least_ndim(get_mask(
            mask, (cp_memory.shape[0],), self.dropout, self.training, cp_memory.device), cp_memory.dim())
        return cp_memory * mask

    def reset(self):
        self.condition_mem_cache_list = []
        self.condition_mem_cache_step_list = []
        self.online_kmeans = KMeans(k=self.mem_compress_length)
        self.step = 0
    
    def inference(self, new_memory: torch.Tensor):
        """
        Args:
            new_memory: (B, T, D) 张量
        Returns:
            memory_sorted_by_step: (B, mem_compress_length, D) 压缩后的记忆
        """
        B, T, D = new_memory.shape
        # 添加 N 维度以匹配 KMeans 的输入格式 (B, T, N, D)
        new_memory_expanded = new_memory.unsqueeze(2)  # (B, T, 1, D)
        
        if self.step == 0:
            self.online_kmeans.initial_fit(new_memory_expanded)
        else:
            self.online_kmeans.online_fit(new_memory_expanded)
        
        memory_sorted_by_step = self.online_kmeans.get_centers()  # (B, K, N, D)
        # 移除 N 维度，返回 (B, K, D)
        memory_sorted_by_step = memory_sorted_by_step.squeeze(2)  # (B, mem_compress_length, D)
        return memory_sorted_by_step