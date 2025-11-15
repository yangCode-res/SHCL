from typing import final
import torch
from torch import nn

from models.utils import DotProductAttention
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

class NodeHypergraphAttention(nn.Module):
    """
    结点->结点的多头注意力，注入与被注意结点 j 相关的超边信息 (b)，
    以及一个可学习的静态超边偏置 (d)。不使用相对位置项 (c)。
    """
    def __init__(self, d_in, d_model, num_heads=2, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d = d_model // num_heads

        # 结点的 Q,K,V
        self.Wq = nn.Linear(d_in, d_model, bias=False)
        self.Wk = nn.Linear(d_in, d_model, bias=False)
        self.Wv = nn.Linear(d_in, d_model, bias=False)


        # (d) 静态 key：每个头一个向量 u_h
        # self.static_key = nn.Parameter(torch.randn(self.h, self.d))

        # self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X_node, E_on_node):
        """
        X_node:   [N, d_in]         结点特征（作为 Q/K/V）
        E_on_node:[N, d_in]         “每个结点自己的超边聚合表示 e_j”
        返回:
            X_out: [N, d_model]     结点更新后的表示
            A:     [H, N, N]        每头的注意力矩阵（可选返回便于调试）
        """
        N, _ = X_node.shape

        # 线性投影并分头
        Q = self.Wq(X_node).view(N, self.h, self.d)          # [N,H,d]
        K = self.Wk(X_node).view(N, self.h, self.d)          # [N,H,d]
        V = self.Wv(X_node).view(N, self.h, self.d)          # [N,H,d]
 
        # E = self.We(E_on_node).view(N, self.h, self.d)       # [N,H,d]

        # (a) node-node
        attn_a = torch.einsum('ihd,jhd->hij', Q, K) / (self.d ** 0.5)   # [H,N,N]
        # (b) node -> "node j 的超边表示"
        # (d) 静态超边偏置（与 i 无关，只与列 j 有关）
        # bias_d = torch.einsum('hd,jhd->hj', self.static_key, E)         # [H,N]
        # bias_d = bias_d.unsqueeze(1).expand(-1, N, -1)                  # [H,N,N]

        A = attn_a                                      # [H,N,N]
        A = torch.softmax(A, dim=-1)

        Y = torch.einsum('hij, jhd -> ihd', A, V)                        # [N,H,d]
        Y = Y.reshape(N, self.h * self.d)                                # [N,d_model]
        # Y = self.proj_drop(self.proj(Y))                                 # [N,d_model]
        return Y, A
@torch.no_grad()
def drop_nodes_in_hyperedges(
    H: torch.Tensor,
    drop_prob: float = 0.2,
    min_keep: int = 1,
    rescale: bool = True,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    对 dense 关联矩阵 H ∈ {0,1}^{N×E} 做随机删点增广：
      - 对每条超边 e 的成员节点独立以 p 丢弃；
      - 若某列保留数 < min_keep，则强制保留若干个（随机挑回）；
      - 可选 rescale：对保留下来的 1 乘以 1/(1-p) 做期望校正（类似 Dropout）。

    返回：H_tilde （与 H 同形状、同设备、同 dtype）
    """
    if drop_prob <= 0.0:
        return H
    N, E = H.shape
    device, dtype = H.device, H.dtype
    keep_p = 1.0 - drop_prob

    # 只在 H==1 的位置采样是否保留；0 位置始终为 0
    # mask_keep ∈ {0,1}^{N×E}
    bern = torch.rand((N, E), generator=generator, device=device) #随机种子生成
    mask_keep = (bern < keep_p) & (H > 0)

    # 保证每列至少保留 min_keep 个 1
    col_kept = mask_keep.sum(dim=0)  # [E]
    need_fix = (col_kept < min_keep) & (H.sum(dim=0) > 0)  # 只对非空列修复
    if need_fix.any():
        cols = need_fix.nonzero(as_tuple=False).flatten()  # 需要修的列索引
        for e in cols.tolist():
            idx = (H[:, e] > 0).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            # 需要补回的数量
            k = int(min_keep) - int(mask_keep[:, e].sum().item())
            k = max(0, min(k, idx.numel()))
            if k > 0:
                # 随机挑回 k 个属于该超边且当前被丢弃的节点
                dropped_idx = idx[~mask_keep[idx, e]]
                if dropped_idx.numel() == 0:
                    # 全部都已经保留，不需补
                    pass
                else:
                    # 如果可补的数量小于 k，就全补
                    choice = dropped_idx
                    if dropped_idx.numel() > k:
                        perm = torch.randperm(dropped_idx.numel(), generator=generator, device=device)[:k]
                        choice = dropped_idx[perm]
                    mask_keep[choice, e] = True
    # 构造增强后的 H
    H_tilde = torch.zeros_like(H)
    H_tilde[mask_keep] = 1.0

    # 期望重标定（可选）：保留的 1 乘以 1/(1-p)
    if rescale and keep_p > 0:
        H_tilde = H_tilde * (1.0 / keep_p)

    return H_tilde.to(dtype)
def build_causal_T_prev_only(E, self_loop: float = 0.0, device=None, dtype=torch.float32):
    """
    只从 j-1 流向 j（一步马尔可夫式），可选自环，再行归一化。
    """
    T = torch.zeros(E, E, device=device, dtype=dtype)
    for j in range(E):
        if j > 0:
            T[j, j-1] = 1.0
        if self_loop > 0:
            T[j, j] += self_loop
        s = T[j].sum()
        if s <= 0:
            T[j, j] = 1.0
        else:
            T[j] /= s
    return T
class CasualGraph(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim, num_heads=1,num_layers=4):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(p=0.35)
        # self.attetion=DotProductAttention(node_dim, 32)
    def forward(self, node_embeddings, target_martrix,hypergraph_matrix, edge_weight=None, return_all=False, p_drop=0.2, min_keep=1,gamma=1.0, self_loop=0.0,num_layers=3):
        last_embedding=node_embeddings
        for i in range(num_layers):
            target_embedding=torch.matmul(target_martrix, last_embedding)
            source_embedding=torch.matmul(target_martrix.T, target_embedding)
            last_embedding=source_embedding
            last_embedding=last_embedding+node_embeddings
            last_embedding=self.norm(last_embedding)
            last_embedding=self.dropout(last_embedding)
        H = hypergraph_matrix  # [N, E]
        N, E = H.shape
        result=[]
        for i in range(E):
            mask=H[:,i]>0
            code_in_visit=source_embedding[mask]
            codes_vals = torch.mean(code_in_visit, dim=0)  # [d]
            # codeVisitCat = torch.cat([codes_vals, edge_hid[i]],dim=0)
            result.append(codes_vals)
        result=torch.stack(result,dim=0)
        result,_=torch.max(result,dim=0) 
        # result=self.attetion(result)
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
        # self.node_attn = NodeHypergraphAttention(
        #     d_in=node_dim, d_model=output_dim, num_heads=num_heads,
        #     attn_drop=0.1, proj_drop=0.1
        # )
        self.norm = nn.LayerNorm(node_dim)
        # self.normalization = nn.LayerNorm(node_dim)
    @staticmethod
    def _safe_degree(x, dim, eps=1e-12):
        # x: dense or sparse (coalesce before sum if sparse)
        deg = torch.sum(x, dim=dim)  # dense sum
        return torch.clamp(deg, min=eps)

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
        # if self.training and p_drop > 0.0:
        #     H = drop_nodes_in_hyperedges(H, drop_prob=p_drop, min_keep=min_keep, rescale=True)
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

        # final_node_msg=[node_embeddings]
        # final_node_msg.append(node_msg)
        # node_msg=torch.mean(torch.stack(final_node_msg),dim=0)
        # node_msg=self.dropout(node_msg)
        node_msg=last_embedding
        result=[]
        for i in range(E):
            mask=H[:,i]>0
            code_in_visit=node_msg[mask]
            # code_in_visit,A=self.node_attn(code_in_visit,edge_hid[i])
            # print(A)
            # code_in_visit=self.norm(code_in_visit)
            codes_vals = torch.mean(code_in_visit, dim=0)  # [d]
            # codeVisitCat = torch.cat([codes_vals, edge_hid[i]],dim=0)
            result.append(codes_vals)
        result=torch.stack(result,dim=0)
        global_attention_output=self.global_edge_attention(result)
        
        
        
        return global_attention_output

