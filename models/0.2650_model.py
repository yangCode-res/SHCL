import torch
from torch import nn

from models.layers import EmbeddingLayer,HyperGraph,CasualGraph
from models.utils import DotProductAttention


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0., activation=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        if self.activation is not None:
            output = self.activation(output)
        return output

import torch

def _safe_inv(x: torch.Tensor, eps: float = 1e-12):
    return 1.0 / torch.clamp(x, min=eps)

def _safe_inv_sqrt(x: torch.Tensor, eps: float = 1e-12):
    return torch.rsqrt(torch.clamp(x, min=eps))
def build_adjacent_transition_binary2(H: torch.Tensor, include_self: bool = True, normalize: str = "row"):
    """
    H: [N, E]，行=疾病，列=就诊（列已按时间顺序）；H[i,t]>0 表示疾病 i 出现在第 t 次就诊
    现在仅构建：第一次就诊(源) -> 最后一次就诊(目标) 的二部迁移
    返回:
      HG_poi_src: [N, N] 浮点矩阵（可归一化），行=源疾病(首访)，列=目标疾病(末访)
      HG_poi_tar: [N, N] = HG_poi_src.T
    normalize: "row" | "col" | "sym" | "none"/None
    """
    assert H.dim() == 2
    N, E = H.shape
    device, dtype = H.device, H.dtype

    # 若没有就诊列，直接返回零矩阵
    if E == 0:
        Z = torch.zeros((N, N), dtype=dtype, device=device)
        return Z, Z.t()

    Hb = (H > 0)  # 布尔出现矩阵
    A = torch.zeros((N, N), dtype=torch.bool, device=device)

    # 只用第一列作为 source，最后一列作为 target
    src = Hb[:, 0]        # 首次就诊出现的疾病
    dst = Hb[:, -1]       # 最后一次就诊出现的疾病
    if src.any() and dst.any():
        # 外积：i 在首访出现 且 j 在末访出现 => i->j
        A |= (src.unsqueeze(1) & dst.unsqueeze(0))

    if not include_self:
        A.fill_diagonal_(False)

    A = A.to(dtype)  # 转数值型以便归一化

    norm = (normalize or "none").lower()
    if norm == "row":
        # D_out^{-1} A
        dout = A.sum(dim=1, keepdim=True)                     # [N,1]
        A = A * _safe_inv(dout)                               # 按行归一
    elif norm == "col":
        # A D_in^{-1}
        din = A.sum(dim=0, keepdim=True)                      # [1,N]
        A = A * _safe_inv(din)                                # 按列归一
    elif norm == "sym":
        # D_out^{-1/2} A D_in^{-1/2}
        dout = A.sum(dim=1, keepdim=True)                     # [N,1]
        din  = A.sum(dim=0, keepdim=True)                     # [1,N]
        A = (_safe_inv_sqrt(dout) * A) * _safe_inv_sqrt(din)  # 左右乘（广播）
    elif norm in ("none", "no", "off"):
        pass
    else:
        raise ValueError(f"Unknown normalize='{normalize}', choose from 'row'|'col'|'sym'|None.")

    HG_poi_src = A
    HG_poi_tar = A.t()
    return HG_poi_src, HG_poi_tar
def build_adjacent_transition_binary(H: torch.Tensor, include_self: bool = True, normalize: str = "row"):
    """
    H: [N, E]，行=疾病，列=就诊（列已按时间顺序）；H[i,t]>0 表示疾病 i 出现在第 t 次就诊
    返回:
      HG_poi_src: [N, N] 浮点矩阵（可归一化），行=源疾病，列=目标疾病，只考虑相邻就诊 t->t+1
      HG_poi_tar: [N, N] = HG_poi_src.T
    normalize: "row" | "col" | "sym" | "none"/None
    """
    assert H.dim() == 2
    N, E = H.shape
    device, dtype = H.device, H.dtype

    Hb = (H > 0)  # 布尔出现矩阵
    A = torch.zeros((N, N), dtype=torch.bool, device=device)

    for t in range(E - 1):
        src = Hb[:, t]       # 本次就诊出现的疾病
        dst = Hb[:, t + 1]   # 下一次就诊出现的疾病
        if src.any() and dst.any():
            # 外积：i 在 t 出现 且 j 在 t+1 出现 => i->j
            A += (src.unsqueeze(1) & dst.unsqueeze(0))

    if not include_self:
        A.fill_diagonal_(False)

    A = A.to(dtype)  # 转成数值型以便归一化

    norm = (normalize or "none").lower()
    if norm == "row":
        # D_out^{-1} A
        dout = A.sum(dim=1, keepdim=True)                     # [N,1]
        A = A * _safe_inv(dout)                               # 广播按行归一
    elif norm == "col":
        # A D_in^{-1}
        din = A.sum(dim=0, keepdim=True)                      # [1,N]
        A = A * _safe_inv(din)                                # 广播按列归一
    elif norm == "sym":
        # D_out^{-1/2} A D_in^{-1/2}
        dout = A.sum(dim=1, keepdim=True)                     # [N,1]
        din  = A.sum(dim=0, keepdim=True)                     # [1,N]
        A = (_safe_inv_sqrt(dout) * A) * _safe_inv_sqrt(din)  # 先左乘，再右乘（广播）
    elif norm in ("none", "no", "off"):
        pass
    else:
        raise ValueError(f"Unknown normalize='{normalize}', choose from 'row'|'col'|'sym'|None.")

    HG_poi_src = A
    HG_poi_tar = A.t()
    return HG_poi_src, HG_poi_tar
class Model(nn.Module):
    def __init__(self, code_num, code_size,
                 adj, graph_size, hidden_size, t_attention_size, t_output_size,
                 output_size, dropout_rate, activation):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(code_num, code_size)
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)
        self.hyper_graph = HyperGraph(code_size, hidden_size, hidden_size)
        self.casual_graph = CasualGraph(code_size, hidden_size, hidden_size)
        # self.attention =DotProductAttention(hidden_size, 32)
        self.hyper_gate = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.casual_gate = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.adj=adj
    def forward(self, code_x, lens):
        embeddings = self.embedding_layer()
        c_embeddings= embeddings
        output = []
        for code_x_i, len_i in zip(code_x, lens):
            # code_x_i 是 (max_visits, num_codes) 格式，但实际有效就诊次数是 len_i
            # 只取前 len_i 次有效就诊记录
            valid_visits = code_x_i[:len_i]  # (len_i, num_codes)
            # 转换成超图矩阵格式 (num_codes, len_i)
            hypergraph_matrix = valid_visits.T.float()  # 转置并转换为float
            HG_poi_src, HG_poi_tar = build_adjacent_transition_binary(H=hypergraph_matrix,include_self=False,normalize="sym")
            
            output_i = self.hyper_graph(c_embeddings, hypergraph_matrix)
            output_casual=self.casual_graph(c_embeddings,hypergraph_matrix=hypergraph_matrix,target_martrix=HG_poi_src)
            # res=self.attention(torch.stack([output_i,output_casual],dim=0))
            # res=torch.cat([output_i,output_casual],dim=0)
            hyper_coef = self.hyper_gate(output_i)
            casual_coef = self.casual_gate(output_casual)
            res = hyper_coef * output_i + casual_coef * output_casual
            # print(res.shape)
            output.append(res)
        output = torch.vstack(output)
        output = self.classifier(output)
        return output
