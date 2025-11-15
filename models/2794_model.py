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
    def __init__(self, node_dim, hidden_dim, output_dim, num_heads=1,num_layers=4):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(p=0.4)
    def forward(self, node_embeddings, target_martrix,num_layers=3):
        last_embedding=node_embeddings
        for i in range(num_layers):
            target_embedding=torch.matmul(target_martrix, last_embedding) #[4880x5] x[4880x150]=5x150 x5x4880=4880x150  
            source_embedding=torch.matmul(target_martrix.T, target_embedding)
            last_embedding=source_embedding
            last_embedding=self.norm(last_embedding)
            last_embedding=self.dropout(last_embedding)
        
       
        return last_embedding

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
                 output_size, dropout_rate, activation,proj_dim=256):
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
        self.adj=adj

        self.icd_conv=icd_conv(code_size, hidden_size, hidden_size)
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
        embeddings = self.embedding_layer()
        c_embeddings= embeddings
        output = []
        embd1=[]
        embd2=[]
        embd3=[]
        icd_code_embedding=self.icd_conv(c_embeddings,self.adj)
        for code_x_i, len_i in zip(code_x, lens):
            valid_visits = code_x_i[:len_i]  # (len_i, num_codes)
            hypergraph_matrix = valid_visits.T.float()  # 转置并转换为float
            HG_poi_src, HG_poi_tar = build_adjacent_transition_binary(H=hypergraph_matrix,include_self=False,normalize="sym")
            output_i = self.hyper_graph(c_embeddings, hypergraph_matrix)
            output_casual=self.casual_graph(c_embeddings,hypergraph_matrix=hypergraph_matrix,target_martrix=HG_poi_src)
            output_icd=self.embedding_conv(icd_code_embedding,hypergraph_matrix)
            hyper_coef = self.hyper_gate(output_i)
            casual_coef = self.casual_gate(output_casual)
            icd_coef = self.icd_gate(output_icd)
            embd1.append(output_i)
            embd2.append(output_casual)
            embd3.append(output_icd)
            res = hyper_coef * output_i + casual_coef * output_casual + icd_coef * output_icd
            # print(res.shape)
            output.append(res)
        output = torch.vstack(output)
        output = self.classifier(output)
        embd1=torch.vstack(embd1)
        embd2=torch.vstack(embd2)
        embd3=torch.vstack(embd3)
        loss=self.info_nce(embd1,embd2)
        loss2=self.info_nce(embd1,embd3)
        # loss3=self.info_nce(embd2,embd3)
        return output ,loss+loss2#第一个是三个loss加一起，然后lamda为0.005 第二个是两个loss加一起看看结果 ,第三个是0.003，三个loss加一起，第四个是0.004，三个loss加一起
