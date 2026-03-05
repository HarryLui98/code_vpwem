from typing import Optional, Tuple
import torch
import torch.nn as nn

from cleandiffuser.utils import at_least_ndim
from cleandiffuser.nn_memory.base_nn_memory import BaseNNMemory
# from sklearn.cluster import KMeans
import torch.nn.functional as F
from cleandiffuser.nn_memory.kmeans_memory import KMeans
from cleandiffuser.nn_memory.adjsim_memory import compress_one_step
from cleandiffuser.utils.utils import SinusoidalEmbedding

def cache_compress_fifo(cache: torch.Tensor):
    cache = cache[:, 1:, ...]
    return cache

def cache_compress_random(cache: torch.Tensor, step: int):
    B, T = cache.shape[0], cache.shape[1]
    if T <= 1:
        return cache
    j = torch.randint(0, T - 1, (B,), device=cache.device)  # 生成随机数 j
    # 向量化操作：为每个 batch 选择要替换的位置
    batch_indices = torch.arange(B, device=cache.device)
    # 直接使用索引替换，避免 Python 循环
    cache[batch_indices, j, ...] = cache[batch_indices, -1, ...]
    cache = cache[:, :(T-1), ...]
    return cache

def get_mask(mask: torch.Tensor, mask_shape: tuple, dropout: float, train: bool, device: torch.device):
    if train:
        mask = (torch.rand(mask_shape, device=device) > dropout).float()
    else:
        mask = 1. if mask is None else mask
    return mask

class QFormerLayer(nn.Module):
    """QFormer单层transformer块"""
    def __init__(self, d_model: int, nhead: int, d_ffn: int, 
                 dropout: float = 0.1, activation: str = "gelu",
                 cache_compress: str = "fifo",
                 cache_max_length: int = 128,):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Cross-attention (用于与输入特征交互)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = getattr(F, activation)

        self.cache_compress = cache_compress
        self.cache_max_length = cache_max_length
        
    def forward(self, 
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if hasattr(self, 'query_cache'):
            B, T, N, D = self.query_cache.shape
            query_cache = self.query_cache.contiguous().view(B, -1, D) #[B, T*num_query, C]
            all_query = torch.cat([query_cache, query.detach()], dim=1) #[B, T*num_query + num_query, C]
            self.query_cache = all_query.contiguous().view(B, T + 1, N, D)  #[B, T+1, N, C]
            if T + 1 > self.cache_max_length:
                if self.cache_compress == "fifo":
                    self.query_cache = cache_compress_fifo(self.query_cache)
                elif self.cache_compress == "random":
                    self.query_cache = cache_compress_random(self.query_cache, self.step)
                elif self.cache_compress == "kmeans":
                    if not hasattr(self, 'kmeans'):
                        self.kmeans = KMeans(k=self.cache_max_length)
                        self.kmeans.initial_fit(self.query_cache)
                    else:
                        self.kmeans.online_fit(self.query_cache[:, -1:, ...])
                    self.query_cache = self.kmeans.get_centers()
                elif self.cache_compress == "adjsim":
                    if not hasattr(self, 'compression_size'):
                        self.compression_size = torch.ones((B, T + 1, N), device=self.query_cache.device)
                    else:
                        size_constant = torch.ones(B, 1, N).to(self.query_cache.device) # [B, 1, N]
                        self.compression_size = torch.cat([self.compression_size, size_constant], dim=1)
                    self.query_cache, self.compression_size = compress_one_step(self.query_cache, self.compression_size)
                elif self.cache_compress == "none":
                    pass
                else:
                    raise NotImplementedError(f"Invalid cache_compress method {self.cache_compress}")
        else:
            all_query = query
            self.query_cache = query.detach().unsqueeze(1)  # [B, 1, N, C]
        self.step += 1

        # Self-attention
        all_q = self.norm1(all_query)
        q = all_q[:, -query.size(1):, :]
        attn_output, self_attn_weights = self.self_attn(
            q, all_q, all_q, 
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True
        )
        query = query + self.dropout1(attn_output)
        
        # Cross-attention (如果提供了key和value)
        if key is not None and value is not None:
            q = self.norm2(query)
            attn_output, cross_attn_weights = self.cross_attn(
                q, key, value,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=True
            )
            query = query + self.dropout2(attn_output)
        
        # Feed-forward
        q = self.norm3(query)
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(q))))
        query = query + self.dropout3(ffn_output)
        
        return query, self_attn_weights, cross_attn_weights

    def reset(self):
        self.step = 0
        if hasattr(self, 'query_cache'):
            del self.query_cache

class QFormer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 d_model: int,
                 n_queries: int,
                 n_layers: int = 2,
                 nhead: int = 8,
                 d_ffn: int = 1024,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 cache_compress: str = "fifo",
                 cache_max_length: int = 128,):
        
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_queries = n_queries
        self.n_layers = n_layers
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.query_tokens = nn.Parameter(torch.randn(1, n_queries, d_model))

        self.layers = nn.ModuleList([
            QFormerLayer(d_model, nhead, d_ffn, dropout, activation, 
                         cache_compress, cache_max_length)
            for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.normal_(self.query_tokens, mean=0., std=self.d_model ** -0.5)
        
        for layer in self.layers:
            if hasattr(layer.self_attn, 'in_proj_weight'):
                nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            if hasattr(layer.cross_attn, 'in_proj_weight'):
                nn.init.xavier_uniform_(layer.cross_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
    
    def forward(self, 
                x: torch.Tensor,
                return_weights: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: [B, T, input_dim]
            
        Returns:
            compressed_features: [B, N, d_model]
            (optional) attention_weights
        """
        B, T, _ = x.shape
        
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]
        
        query = self.query_tokens.expand(B, -1, -1)  # [batch_size, n_queries, d_model]
        
        self_attn_weights_list = []
        cross_attn_weights_list = []
        
        for layer in self.layers:
            query, self_attn_weights, cross_attn_weights = layer(
                query, key=x, value=x
            )
            
            if return_weights:
                self_attn_weights_list.append(self_attn_weights)
                cross_attn_weights_list.append(cross_attn_weights)
        
        query = self.norm(query)
        compressed_features = self.output_proj(query)
        
        if return_weights:
            return compressed_features, self_attn_weights_list, cross_attn_weights_list
        else:
            return compressed_features

class QFormerMemory(BaseNNMemory):
    """
    Input:
        - condition: Any
        - mask : Optional, (b, ) or None, None means no mask

    Output:
        - condition: (b, *cond_out_shape)
    """

    def __init__(self, mem_compress_length=4, 
                 emb_dim=256, num_layers=2, 
                 cache_compress='fifo', cache_max_length=128,
                 use_pos_emb='sin_pos', subsample_ratio=10,
                 ):
        super().__init__()
        self.step = 0
        self.mem_compress_length = mem_compress_length
        self.emb_dim = emb_dim
        self.cache_compress = cache_compress
        self.cache_max_length = cache_max_length
        self.subsample_ratio = subsample_ratio
        if num_layers > 0:
            self.qformer = QFormer(
                input_dim=emb_dim,
                d_model=emb_dim,
                n_queries=mem_compress_length,
                n_layers=num_layers,
                cache_compress=cache_compress,
                cache_max_length=cache_max_length,
            )
        else:
            assert cache_max_length == mem_compress_length, "When num_layers=0, cache_max_length must equal mem_compress_length"
        if use_pos_emb == 'sin_pos':
            self.mem_pos_emb = SinusoidalEmbedding(emb_dim)
        elif use_pos_emb == 'learned_pos':
            self.mem_pos_emb = nn.Embedding(2048, emb_dim)
            nn.init.constant_(self.mem_pos_emb.weight, 0.0)
        # self.dropout = dropout
    
    def _compress_condition_cache(self, condition_cache: torch.Tensor, B: int, T_cache: int, N: int):
        """压缩 condition_cache 的辅助方法"""
        if self.cache_compress == "fifo":
            return cache_compress_fifo(condition_cache).contiguous()
        elif self.cache_compress == "random":
            return cache_compress_random(condition_cache, self.step).contiguous()
        elif self.cache_compress == "kmeans":
            if not hasattr(self, 'kmeans'):
                self.kmeans = KMeans(k=self.cache_max_length)
                self.kmeans.initial_fit(condition_cache)
            else:
                self.kmeans.online_fit(condition_cache[:, -1:, ...])
            return self.kmeans.get_centers().contiguous()
        elif self.cache_compress == "adjsim":
            # compression_size 应该在 _update_condition_cache 中已经更新
            # 这里只需要确保长度匹配（作为安全检查）
            if not hasattr(self, 'compression_size') or self.compression_size.shape[1] != T_cache:
                self.compression_size = torch.ones((B, T_cache, N), device=condition_cache.device)
            compressed_cache, self.compression_size = compress_one_step(condition_cache, self.compression_size)
            return compressed_cache.contiguous()
        elif self.cache_compress == "none":
            return condition_cache.contiguous()
        else:
            raise NotImplementedError(f"Invalid cache_compress method {self.cache_compress}")
    
    def _update_condition_cache(self, condition: torch.Tensor):
        """更新 condition_cache 的辅助方法，优化了张量操作"""
        B = condition.shape[0]
        condition_to_add = condition.detach()
        
        if hasattr(self, 'condition_cache'):
            # 直接使用 cat，避免不必要的 view
            self.condition_cache = torch.cat([self.condition_cache, condition_to_add], dim=1)
            T_cache = self.condition_cache.shape[1]
            N = self.condition_cache.shape[2]
            
            # 对于 adjsim，需要同步更新 compression_size
            if self.cache_compress == "adjsim":
                if not hasattr(self, 'compression_size'):
                    self.compression_size = torch.ones((B, T_cache, N), device=self.condition_cache.device)
                else:
                    # 确保长度匹配，每次只添加一个元素
                    size_constant = torch.ones(B, 1, N, device=self.condition_cache.device)
                    self.compression_size = torch.cat([self.compression_size, size_constant], dim=1)
            
            # 只在需要时压缩
            if T_cache > self.cache_max_length:
                self.condition_cache = self._compress_condition_cache(self.condition_cache, B, T_cache, N)
        else:
            self.condition_cache = condition_to_add
    
    def forward(self, memory: torch.Tensor, ep_step: torch.Tensor, mask: torch.Tensor = None):
        self.reset()
        B, T, D = memory.shape
        memory = memory + self.mem_pos_emb(ep_step)
        for t_idx in range(T):
            condition = memory[:, t_idx:t_idx+1, :].unsqueeze(2)  # [B, 1, 1, D]
            self._update_condition_cache(condition)
            self.step += 1
            
            # 缓存形状信息，避免重复 view
            cp_memory = self.condition_cache.contiguous().view(B, -1, D)  # [B, T, D]
            if hasattr(self, 'qformer'):
                cp_memory = self.qformer.forward(cp_memory)  # (B, mem_compress_length, D)
            
        # 清理临时状态
        if self.cache_compress == "kmeans":
            if hasattr(self, 'kmeans'):
                del self.kmeans
            if hasattr(self, 'qformer'):
                for layer in self.qformer.layers:
                    if hasattr(layer, 'kmeans'):
                        del layer.kmeans
        elif self.cache_compress == "adjsim":
            if hasattr(self, 'compression_size'):
                del self.compression_size
        del self.condition_cache
        
        # mask = at_least_ndim(get_mask(
        #     mask, (cp_memory.shape[0],), self.dropout, self.training, cp_memory.device), cp_memory.dim())
        return cp_memory

    def reset(self):
        self.step = 0
        self.cp_memory = None
        if hasattr(self, 'condition_cache'):
            del self.condition_cache
        if hasattr(self, 'qformer'):
            for layer in self.qformer.layers:
                layer.reset()
    
    def inference(self, new_memory: torch.Tensor):
        B, T, D = new_memory.shape
        # only process each subsample_ratio step
        for t_idx in range(T):
            if self.step % self.subsample_ratio != 0:
                self.step += 1
                continue
            
            condition = new_memory[:, t_idx:t_idx+1, :]
            ep_step = self.step * torch.ones((1), device=new_memory.device, dtype=torch.int64)
            condition = (condition + self.mem_pos_emb(ep_step)).unsqueeze(2)  # [B, 1, 1, D]
            
            self._update_condition_cache(condition)
            self.step += 1
            
            cp_memory = self.condition_cache.contiguous().view(B, -1, D)  # [B, T, D]
            if hasattr(self, 'qformer'):
                cp_memory = self.qformer.forward(cp_memory)  # (B, mem_compress_length, D)
            self.cp_memory = cp_memory
        
        # 确保输出长度正确
        if hasattr(self, 'cp_memory') and self.cp_memory.shape[1] < self.mem_compress_length:
            # pad to zero, zero first
            cp_memory = F.pad(self.cp_memory, (0, 0, self.mem_compress_length - self.cp_memory.shape[1], 0), "constant", 0)
        else:
            cp_memory = self.cp_memory if hasattr(self, 'cp_memory') else self.condition_cache.contiguous().view(B, -1, D)
        return cp_memory