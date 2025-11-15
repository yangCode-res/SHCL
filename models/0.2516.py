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

class HyperGraph(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim, num_heads=1,num_layers=2):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = nn.ReLU()
        self.global_edge_attention = DotProductAttention(node_dim*2, 32)
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

    def forward(self, node_embeddings, hypergraph_matrix, edge_weight=None, return_all=False):
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
        
        edge_sum = torch.matmul(H.T, node_embeddings)  # [E, node_dim] 这里只是对每次就诊的疾病潜入进行相加
        edge_mean = edge_sum * de_inv.unsqueeze(-1)    # [E, node_dim] 这里处以节点的数量防止某条超边的数值过高
        edge_hid=edge_mean    # [N]
        node_msg = torch.matmul(H, edge_hid)           # [N, hidden_dim]
        node_msg = node_msg * dv_inv.unsqueeze(-1)     # [N, hidden_dim]
        final_node_msg=[node_embeddings]
        final_node_msg.append(node_msg)
        node_msg=torch.mean(torch.stack(final_node_msg),dim=0)
        node_msg=self.dropout(node_msg)
        result=[]
        for i in range(E):
            mask=H[:,i]>0
            code_in_visit=node_msg[mask]
            # code_in_visit,A=self.node_attn(code_in_visit,edge_hid[i])
            # print(A)
            # code_in_visit=self.norm(code_in_visit)
            codes_vals = torch.mean(code_in_visit, dim=0)  # [d]
            codeVisitCat = torch.cat([codes_vals, edge_hid[i]],dim=0)
            result.append(codeVisitCat)
        result=torch.stack(result,dim=0)
        global_attention_output=self.global_edge_attention(result)
        
        
        
        return global_attention_output

