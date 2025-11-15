from typing import final
import torch
from torch import nn

from models.utils import DotProductAttention, SingleHeadAttentionLayer
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, code_num, code_size):
        super().__init__()
        self.code_num = code_num
        self.c_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))

    def forward(self):
        return self.c_embeddings




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveVisitPooling(nn.Module):
    """
    以注意力替代 mean 的就诊级聚合：
      - context='learnable'：每次就诊共享一个可学习查询向量 q
      - context='mean'：用该次就诊的节点均值作为上下文
      - use_prior=True 时，可叠加节点先验（如 log(deg)）到打分中
      - 可选 topk 截断，抑制噪声
    """
    def __init__(self, dim, context='learnable', use_prior=False, topk_ratio=None, dropout=0.0):
        super().__init__()
        assert context in ('learnable', 'mean')
        self.context = context
        self.use_prior = use_prior
        self.topk_ratio = topk_ratio

        self.W = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, 1, bias=False)   # 打分头 v^T tanh(...)
        self.q = nn.Parameter(torch.randn(dim)) if context == 'learnable' else None
        self.beta = nn.Parameter(torch.tensor(0.1)) if use_prior else None
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, node_embeddings: torch.Tensor, H: torch.Tensor, prior_logit: torch.Tensor = None):
        """
        node_embeddings: [N, d] 节点（疾病）嵌入
        H: [N, E] 该病人的关联矩阵（列=就诊），二值/权重皆可
        prior_logit: [N] 节点先验（可选），建议用 log(1+deg) 或频次标准化
        返回：每个病人的就诊聚合后（再做 max-pool）得到的向量 [d]
        """
        N, E = H.shape
        result = []
        X_all = node_embeddings

        for e in range(E):
            mask = H[:, e] > 0
            X = X_all[mask]             # [n_e, d]
            if X.numel() == 0:
                # 保险：空就诊时返回 0 向量
                result.append(torch.zeros(X_all.size(1), device=X_all.device, dtype=X_all.dtype))
                continue

            # 上下文 q
            if self.context == 'learnable':
                q = self.q.unsqueeze(0).expand(X.size(0), -1)  # [n_e, d]
            else:  # mean context
                q = X.mean(dim=0, keepdim=True).expand(X.size(0), -1)

            # 打分 e_i = v^T tanh(W x_i + q)
            s = self.v(torch.tanh(self.W(X) + q)).squeeze(-1)   # [n_e]

            # 加先验
            if self.use_prior and (prior_logit is not None):
                s = s + self.beta * prior_logit[mask]

            # 可选 top-k 截断
            if self.topk_ratio is not None:
                k = max(1, int(self.topk_ratio * X.size(0)))
                vals, idx = torch.topk(s, k=k, dim=0)
                X = X.index_select(0, idx)
                s = vals

            alpha = torch.softmax(s, dim=0).unsqueeze(-1)       # [n_e,1]
            pooled = (alpha * X).sum(dim=0)                     # [d]
            result.append(pooled)

        V = torch.stack(result, dim=0)                          # [E, d]
        V = self.norm(V)
        V = self.dropout(V)
        V, _ = torch.max(V, dim=0)                              # [d] 与你原逻辑对齐
        return V

class CasualGraph(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim, num_heads=1,num_layers=4):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(p=0.4)
    def forward(self, node_embeddings, target_martrix,hypergraph_matrix,num_layers=3):
        last_embedding=node_embeddings
        for i in range(num_layers):
            target_embedding=torch.matmul(target_martrix, last_embedding) #[4880x5] x[4880x150]=5x150 x5x4880=4880x150  
            source_embedding=torch.matmul(target_martrix.T, target_embedding)
            last_embedding=source_embedding
            last_embedding=self.norm(last_embedding)
            last_embedding=self.dropout(last_embedding)
        
        H = hypergraph_matrix  # [N, E]
        N, E = H.shape
        result=[]
        for i in range(E):
            mask=H[:,i]>0
            code_in_visit=last_embedding[mask]
            codes_vals = torch.mean(code_in_visit, dim=0)  # [d]
            result.append(codes_vals)
        result=torch.stack(result,dim=0)
        result,_=torch.max(result,dim=0) 
        return result
class embedding_conv(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim, num_heads=1,num_layers=4):
        super().__init__()
        
    def forward(self, node_embeddings, hypergraph_matrix):
        H = hypergraph_matrix  # [N, E]
        N, E = H.shape
        result=[]
        for i in range(E):
            mask=H[:,i]>0
            code_in_visit=node_embeddings[mask]
            codes_vals = torch.mean(code_in_visit, dim=0)  # [d]
            result.append(codes_vals)
        result=torch.stack(result,dim=0)
        result,_=torch.max(result,dim=0) 
        return result
class HyperGraph(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim, num_heads=1,num_layers=4):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = nn.ReLU()
        self.global_edge_attention = DotProductAttention(node_dim, 32)
        self.dropout = nn.Dropout(p=0.35)
        # self.dropout = 0.35
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(node_dim)
        # self.normalization = nn.LayerNorm(node_dim)
    @staticmethod
    def _safe_degree(x, dim, eps=1e-12):
        # x: dense or sparse (coalesce before sum if sparse)
        deg = torch.sum(x, dim=dim)  # dense sum
        return torch.clamp(deg, min=eps)
    # def _visit_pool(self, X: torch.Tensor, prior_logit: torch.Tensor = None, context: str = 'learnable'):
    #     """
    #     X: [n_e, d] 该次就诊包含的疾病节点嵌入
    #     prior_logit: [n_e] 可选先验（如 log(1+deg)）
    #     """
    #     if X.numel() == 0:
    #         return torch.zeros(self.node_dim, device=self.visit_W.weight.device, dtype=X.dtype)

    #     if context == 'learnable':
    #         q = self.visit_q.unsqueeze(0).expand(X.size(0), -1)  # [n_e, d]
    #     else:  # 'mean'：用该就诊节点的均值作为上下文
    #         q = X.mean(dim=0, keepdim=True).expand(X.size(0), -1)

    #     s = self.visit_v(torch.tanh(self.visit_W(X) + q)).squeeze(-1)  # [n_e]

    #     if self.use_prior and (prior_logit is not None):
    #         s = s + self.beta * prior_logit

    #     if self.visit_topk_ratio is not None:
    #         k = max(1, int(self.visit_topk_ratio * X.size(0)))
    #         vals, idx = torch.topk(s, k=k, dim=0)
    #         X = X.index_select(0, idx)
    #         s = vals

    #     alpha = torch.softmax(s, dim=0).unsqueeze(-1)  # [n_e,1]
    #     pooled = (alpha * X).sum(dim=0)                # [d]
    #     return pooled
    def forward(self, node_embeddings, hypergraph_matrix, edge_weight=None, return_all=False, p_drop=0.2, min_keep=1,gamma=1.0, self_loop=0.0):
        """
        Args:
            node_embeddings: [N, node_dim]   节点(疾病/POI)嵌入
            hypergraph_matrix: [N, E]        0/1 或权重的关联矩阵 H，H[i,j]=1 表示节点 i 属于超边 j
            edge_weight: [E] or None         每条超边的权重，可选
            return_all: 若 True，返回 (node_out, edge_out, global_edge_vec)，否则只返回 node_out

        Returns:
            node_out: [N, output_dim]        节点级更新（完成 V->E->V）
            edge_out: [E, hidden_dim]        每条超边的表示（可选）
            global_edge_vec: [hidden_dim]    对所有超边注意力汇总（可选）
        """
        H = hypergraph_matrix  # [N, E]
        N, E = H.shape
        dv = self._safe_degree(H, dim=1)               # [N]
        dv_inv = 1.0 / dv    
        de = self._safe_degree(H, dim=0)              # [E]
        de_inv = 1.0 / de                           # [E]
        
        last_embedding=node_embeddings
        for i in range(self.num_layers):
            edge_sum = torch.matmul(H.T, last_embedding)  # [E, node_dim] 这里只是对每次就诊的疾病潜入进行相加
            edge_mean = edge_sum * de_inv.unsqueeze(-1)    # [E, node_dim] 这里处以节点的数量防止某条超边的数值过高
            edge_hid=edge_mean    # [N]
            # T = build_causal_T_prev_only(E, self_loop=self_loop,device=node_embeddings.device)
            # edge_hid=T @ edge_hid
            node_msg = torch.matmul(H, edge_hid)           # [N, hidden_dim]
            node_msg = node_msg * dv_inv.unsqueeze(-1)     # [N, hidden_dim]
            last_embedding=node_msg
            last_embedding=last_embedding+node_embeddings
            last_embedding=self.norm(last_embedding)
            last_embedding=self.dropout(last_embedding)
        node_msg=last_embedding
        result=[]
        for i in range(E):
            mask=H[:,i]>0
            code_in_visit=node_msg[mask]
            codes_vals = torch.mean(code_in_visit, dim=0)  # [d]
            result.append(codes_vals)
        result=torch.stack(result,dim=0)
        global_attention_output=self.global_edge_attention(result)
        
        
        
        return global_attention_output

