import torch
from torch import nn

from models.layers import EmbeddingLayer,HyperGraph,CasualGraph,embedding_conv
from models.utils import DotProductAttention
import torch.nn.functional as F

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

def _safe_inv(x: torch.Tensor, eps: float = 1e-12):
    return 1.0 / torch.clamp(x, min=eps)

def _safe_inv_sqrt(x: torch.Tensor, eps: float = 1e-12):
    return torch.rsqrt(torch.clamp(x, min=eps))
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

class icd_conv(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim, num_heads=1, num_layers=2, k=35, norm='sym', dropout=0.4, residual=True):
        super().__init__()
        self.num_layers = num_layers
        self.k = k
        self.norm_kind = norm
        self.residual = residual
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(p=dropout)

        # 缓存图对象（注册为 buffer，随模型走 device）
        self.register_buffer('L_icd', None)   # [d, d]
        self.register_buffer('A_icd', None)   # [d, d]

    def reset_graph(self, S: torch.Tensor):
        # S: [d, d] ICD 相似矩阵（越大越相近）
        L, A = build_knn_laplacian(S, k=self.k, norm=self.norm_kind)
        self.L_icd = L
        self.A_icd = A

    def forward(self, node_embeddings: torch.Tensor, target_martrix: torch.Tensor, num_layers: int = None):
        """
        node_embeddings: [d, m] 疾病嵌入（即 E）
        target_martrix: [d, d] ICD 相似（仅首次用于 reset_graph）
        """
        if (self.A_icd is None) or (self.L_icd is None):
            self.reset_graph(target_martrix)

        A = self.A_icd    # 规范化后的相似矩阵
        last = node_embeddings
        Ls = self.num_layers if num_layers is None else num_layers

        for _ in range(Ls):
            msg = A @ last                          # 一跳传播
            if self.residual:
                msg = msg + node_embeddings         # 残差稳训练
            msg = self.norm(msg)
            msg = self.dropout(msg)
            last = msg
        return last  # [d, m]
def build_knn_laplacian(S: torch.Tensor, k: int = 20, norm: str = 'sym', eps: float = 1e-12):
    """
    S: [d, d] 疾病相似度矩阵（值越大越相近）
    返回:
      L: [d, d] 归一化图拉普拉斯
      A: [d, d] 稀疏化且归一化后的相似矩阵（训练中用于传播）
    """
    d = S.size(0)
    # 对称化 + 去对角
    S = 0.5 * (S + S.t())
    S = S.clone()
    S.fill_diagonal_(0.0)

    # top-k 稀疏化（强烈建议，稳+省显存）
    if k is not None and k < d:
        vals, idx = torch.topk(S, k=k, dim=1)
        mask = torch.zeros_like(S, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        S = torch.where(mask, S, torch.zeros_like(S))
        S = torch.maximum(S, S.t())  # 保证对称

    # 行归一（避免大度数病种主导）
    row_sum = S.sum(dim=1, keepdim=True).clamp_min(eps)
    A = S / row_sum

    # 拉普拉斯
    deg = A.sum(dim=1).clamp_min(eps)
    if norm == 'sym':
        D_inv_sqrt = torch.diag(deg.pow(-0.5))
        L = torch.eye(d, device=S.device, dtype=S.dtype) - D_inv_sqrt @ A @ D_inv_sqrt
    elif norm == 'rw':
        D_inv = torch.diag(deg.reciprocal())
        L = torch.eye(d, device=S.device, dtype=S.dtype) - D_inv @ A
    else:
        D = torch.diag(deg)
        L = D - A
    return L, A
def masked_info_nce(
    z1: torch.Tensor,
    z2: torch.Tensor,
    tau: float = 0.2,
    sim_th: float = 0.5,         # 相似度阈值，越大屏蔽越少
    symmetric: bool = True,
    cosine_norm: bool = True,
    eps: float = 1e-12,
    min_neg: int = 1             # 每行至少保留的负样本数
) -> torch.Tensor:
    """
    z1,z2: [B,d] 两视角投影；只在这些向量上做对比
    """
    B = z1.size(0)
    if B < 2:
        return z1.new_tensor(0.0, requires_grad=True)

    if cosine_norm:
        z1 = z1 / z1.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        z2 = z2 / z2.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)

    # 相似度与 logits
    sim = z1 @ z2.t()                 # [-1,1], [B,B]
    logits = sim / tau

    # 基础掩码：对角线为正样本，其余初始视作负样本（True=保留）
    neg_mask = ~torch.eye(B, dtype=torch.bool, device=z1.device)

    # 阈值屏蔽：把跨病人且“太像”的 pair 去掉（False=屏蔽）
    neg_mask &= (sim <= sim_th)

    # 确保每行至少有 min_neg 个负样本被保留，避免整行被屏蔽
    # 如果某行保留的负样本太少，就按相似度从低到高补足
    with torch.no_grad():
        keep_counts = neg_mask.sum(dim=1)
        need = (min_neg - keep_counts).clamp_min(0)          # [B]
        if need.any():
            # argsort by sim ascending（越不相似越安全）
            idx = torch.argsort(sim, dim=1, descending=False)  # [B,B]
            for i in torch.nonzero(need).flatten().tolist():
                # 跳过对角 & 已经保留的，将最不相似的若干个置 True
                fill_list = []
                for j in idx[i].tolist():
                    if i == j:      # 跳过正对
                        continue
                    if not neg_mask[i, j]:
                        fill_list.append(j)
                    if len(fill_list) >= int(need[i].item()):
                        break
                if fill_list:
                    neg_mask[i, fill_list] = True

    # 把被屏蔽位置的 logit 置为 -inf，softmax 时相当于忽略
    masked_logits = logits.masked_fill(~neg_mask, float('-inf'))

    labels = torch.arange(B, device=z1.device)
    loss = F.cross_entropy(masked_logits, labels)            # A→B

    if symmetric:
        masked_logits_t = logits.t().masked_fill(~neg_mask.t(), float('-inf'))
        loss = 0.5 * (loss + F.cross_entropy(masked_logits_t, labels))  # B→A
    return loss
class Model(nn.Module):
    def __init__(self, code_num, code_size,
                 adj, graph_size, hidden_size, t_attention_size, t_output_size,
                 output_size, dropout_rate, activation,proj_dim=256,icd_k=200, lap_norm='sym'):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(code_num, code_size)
        # self.proj_h = Projector(hidden_size, proj_dim)  # 超图视角 → 投影空间
        # self.proj_c = Projector(hidden_size, proj_dim)  # 转移动因视角 → 投影空间
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)
        self.hyper_graph = HyperGraph(code_size, hidden_size, hidden_size)
        self.casual_graph = CasualGraph(code_size, hidden_size, hidden_size)
        self.embedding_conv=embedding_conv(code_size, hidden_size, hidden_size)
        # self.attention =DotProductAttention(hidden_size, 32)
        self.hyper_gate = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.casual_gate = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.icd_gate = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
          # === ICD 图 ===
        self.adj = adj  # [d, d] ICD 相似
        self.icd_conv = icd_conv(code_size, hidden_size, hidden_size, num_layers=2, k=icd_k, norm=lap_norm)
        self.icd_conv.reset_graph(self.adj)  # 预构建 L/A

        # 也在 Model 缓存一份 L，便于做拉普拉斯正则
        L, A = build_knn_laplacian(self.adj, k=icd_k, norm=lap_norm)
        self.register_buffer('L_icd', L)   # [d, d]
        self.register_buffer('A_icd', A)   # [d, d]
    def laplacian_loss_on_E(self):
        # 对疾病嵌入 E 做拉普拉斯正则：tr(E^T L E) / d
        E = self.embedding_layer.c_embeddings            # [d, m]
        return torch.trace(E.t() @ self.L_icd @ E) / E.size(0)
    def info_nce(self,emb1, emb2, tau=0.4, symmetric=False, cosine_norm=True):
        """#第一个0.4，第二个0.5
        emb1, emb2: [B, D]
        tau: 温度
        symmetric: 是否两方向对称
        cosine_norm: 是否先做 L2 归一化（余弦相似）
        """
        if cosine_norm:
            emb1 = F.normalize(emb1, p=2,dim=-1)
            emb2 = F.normalize(emb2, p=2,dim=-1)

        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / tau)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / tau), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]
        return loss

    
    def forward(self, code_x, lens):
        E = self.embedding_layer()                       # [d, m]
        E_icd = self.icd_conv(E, self.adj)               # [d, m] ICD 先验平滑后的嵌入

        out_list, e_hg_list, e_cas_list, e_icd_list = [], [], [], []
        for code_x_i, len_i in zip(code_x, lens):
            H = code_x_i[:len_i].T.float()               # [d, T]
            HG_src, HG_tar = build_adjacent_transition_binary(H=H, include_self=False, normalize="sym")

            v_hg   = self.hyper_graph(E, H)              # [m]
            v_cas  = self.casual_graph(E, target_martrix=HG_src, hypergraph_matrix=H)  # [m]
            v_icd  = self.embedding_conv(E_icd, H)       # [m]

            a_hg   = self.hyper_gate(v_hg)
            a_cas  = self.casual_gate(v_cas)
            a_icd  = self.icd_gate(v_icd)

            v = a_hg * v_hg + a_cas * v_cas + a_icd * v_icd
            out_list.append(v)
            e_hg_list.append(v_hg)
            e_cas_list.append(v_cas)
            e_icd_list.append(v_icd)

        feat = torch.vstack(out_list)                    # [B, m]
        logits = self.classifier(feat)                   # [B, d] or [B, 1]

        embd1 = torch.vstack(e_hg_list)
        embd2 = torch.vstack(e_cas_list)
        embd3 = torch.vstack(e_icd_list)
        cl12 = self.info_nce(embd1, embd2)
        cl13 = self.info_nce(embd1, embd3)
        cl_loss = cl12 + cl13

        lap_loss = self.laplacian_loss_on_E()            # tr(E^T L E)/d

        return logits, cl_loss, lap_loss