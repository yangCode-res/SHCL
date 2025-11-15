import torch
from torch import nn

from models.utils import SingleHeadAttentionLayer,DotProductAttention


class EmbeddingLayer(nn.Module):
    def __init__(self, code_num, code_size):
        super().__init__()
        self.code_num = code_num
        self.c_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))

    def forward(self):
        return self.c_embeddings





class HyperGraph(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim, num_heads=8):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # V->E 前的特征线性映射（可选，不想变维也可以直接用 node_dim）
        # self.hyperedge_aggregation = nn.Linear(node_dim, hidden_dim, bias=True)
        self.activation = nn.ReLU()

        # 超边级自注意力（用于全局汇总）
        # 你已有的 DotProductAttention(hidden_dim, 32) 也可以继续用
        # self.global_edge_attention = DotProductAttention(hidden_dim*2, 32)
        self.global_edge_attention = DotProductAttention(node_dim*2, 32)
        self.dropout = nn.Dropout(p=0.35)
        # E->V 回传后的节点投影
        # self.output_projection = nn.Linear(hidden_dim, output_dim, bias=True)

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

        # --------- 1) V -> E：把节点聚合到超边（带度归一化）---------
        # 超边度：每条超边包含多少个节点  de[e] = sum_i H[i,e]
        de = self._safe_degree(H, dim=0)              # [E]
        de_inv = 1.0 / de                              # [E]

        # 若有超边权重 W，可在此处乘上（等价于谱式的 W D_e^{-1}）
        if edge_weight is not None:
            # 将权重并入到“平均”之前或之后都可以，这里选择先平均后再乘权
            # 也可做：de_inv = edge_weight * de_inv
            w = edge_weight.view(E, 1)                 # [E,1]
        else:
            w = None

        # (a) 先求每条超边聚合的“节点和”  H^T @ X    => [E, node_dim]
        edge_sum = torch.matmul(H.T, node_embeddings)  # [E, node_dim] 这里只是对每次就诊的疾病潜入进行相加
        # (b) 做平均：除以超边度
        edge_mean = edge_sum * de_inv.unsqueeze(-1)    # [E, node_dim] 这里处以节点的数量防止某条超边的数值过高
        # (c) 线性+激活，得到超边隐藏表示
        edge_hid=edge_mean
        # edge_hid = self.activation(self.hyperedge_aggregation(edge_mean))  # [E, hidden_dim] 这里通过一个线性层来将节点的嵌入映射到隐藏层的嵌入

        # (d) 若有超边权重，作用在超边隐藏表示上
        if w is not None:
            edge_hid = edge_hid * w                    # [E, hidden_dim]

        # --------- 2) 全局超边注意力（保留你原来的全局机制）---------
        # if E > 0:
        #     global_edge_vec = self.global_edge_attention(edge_hid)  # [hidden_dim] #这里通过一个全局的注意力机制来对每条超边进行注意力
        # else:
        #     global_edge_vec = torch.zeros(self.hidden_dim, device=device, dtype=edge_hid.dtype)

        # --------- 3) E -> V：把超边消息广播回节点（带度归一化）---------
        # 节点度：每个节点属于多少条超边  dv[i] = sum_e H[i,e]
        dv = self._safe_degree(H, dim=1)               # [N]
        dv_inv = 1.0 / dv                               # [N]

        # 将超边消息回传到节点：  H @ edge_hid   => [N, hidden_dim]
        node_msg = torch.matmul(H, edge_hid)           # [N, hidden_dim]
        # 平均化：每个节点除以自身度
        node_msg = node_msg * dv_inv.unsqueeze(-1)     # [N, hidden_dim]
        final_node_msg=[node_embeddings]
        final_node_msg.append(node_msg)
        node_msg=torch.mean(torch.stack(final_node_msg),dim=0)
        node_msg=self.dropout(node_msg)
        # --------- 4) 节点级输出投影（可在外部再加残差/归一化）---------
        # node_out = self.output_projection(node_msg)     # [N, output_dim]
        
        result=[]
        for i in range(E):
            mask=H[:,i]>0
            code_in_visit=node_msg[mask]
            codes_vals = torch.mean(code_in_visit, dim=0)  # [d]
            
            # codeVisitCat = torch.cat([codes_vals, edge_hid[i]],dim=0)
            codeVisitCat = torch.cat([codes_vals, edge_hid[i]],dim=0)
            result.append(codeVisitCat)
        result=torch.stack(result,dim=0)
        global_attention_output=self.global_edge_attention(result)
        
        # codes_sum=torch.matmul(H.T,node_msg)
        # codes_vals=codes_sum*de_inv.unsqueeze(-1)
        # fused_embedding= torch.cat([codes_vals, edge_hid], dim=-1)
        # global_attention_output=self.global_edge_attention(fused_embedding)
        
        
        return global_attention_output


# class HyperGraph(nn.Module):
#     def __init__(self, node_dim, hidden_dim, output_dim, num_heads=8):
#         super().__init__()
#         self.node_dim = node_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim

#         # 超边聚合层：将超边中的节点embedding聚合
#         self.hyperedge_aggregation = nn.Linear(node_dim, hidden_dim)
#         self.activation = nn.ReLU()

#         # 注意力机制：聚合多个超边的向量
#         self.attention = DotProductAttention(hidden_dim, 32)

#         # 输出投影层
#         self.output_projection = nn.Linear(hidden_dim, output_dim)

#     def forward(self, node_embeddings, hypergraph_matrix):
#         """
#         Args:
#             node_embeddings: 疾病节点的embedding, shape (num_nodes, node_dim)
#             hypergraph_matrix: 超图矩阵, shape (num_nodes, num_hyperedges)
#                              hypergraph_matrix[i,j] = 1 表示节点i在超边j中
#         Returns:
#             aggregated_embedding: 聚合后的embedding, shape (output_dim,)
#         """
#         # 1. 计算每个超边的聚合表示
#         # hypergraph_matrix.T @ node_embeddings 得到每个超边的节点embedding之和
#         hyperedge_embeddings = torch.matmul(hypergraph_matrix.T, node_embeddings)  # (num_hyperedges, node_dim)

#         # 对每个超边进行平均聚合（考虑超边大小）
#         hyperedge_degrees = torch.sum(hypergraph_matrix, dim=0)  # (num_hyperedges,)
#         hyperedge_degrees = torch.clamp(hyperedge_degrees, min=1.0)  # 避免除零
#         hyperedge_embeddings = hyperedge_embeddings / hyperedge_degrees.unsqueeze(-1)  # (num_hyperedges, node_dim)
#         # 2. 超边特征变换
#         hyperedge_hidden = self.activation(self.hyperedge_aggregation(hyperedge_embeddings))  # (num_hyperedges, hidden_dim)

#         # 3. 使用注意力机制聚合多个超边的向量
#         # 这里使用超边自身的表示作为query, key, value
#         if hyperedge_hidden.size(0) > 0:
#             aggregated = self.attention(hyperedge_hidden)  # (hidden_dim,)
#         else:
#             # 如果没有超边，返回零向量
#             aggregated = torch.zeros(self.hidden_dim, device=node_embeddings.device, dtype=node_embeddings.dtype)

#         # 4. 输出投影
#         output = self.output_projection(aggregated)  # (output_dim,)
#         return output