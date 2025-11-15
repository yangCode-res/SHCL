# global_hypergraph_pretrain.py
import math
import random
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ================ 数据接口 ================
# 你需要准备：全局超图的“超边列表”，每个超边是一个代码ID集合。
# 例如：hyperedges = [[12, 35, 79], [5, 12], [101, 35, 999, 7], ...]
# 所有ID范围在 [0, num_codes)

class GlobalHypergraphDataset:
    def __init__(self,
                 hyperedges: List[List[int]],
                 num_codes: int,
                 text_prior: Optional[torch.Tensor] = None  # (num_codes, d_text) 可选
                 ):
        self.hyperedges = [sorted(set(e)) for e in hyperedges if len(e) >= 2]
        self.num_codes = num_codes
        self.text_prior = text_prior  # 若提供，将用于先验对齐损失

        # 建立 (node, edge) 正样本列表，和 node 的边列表 加速采样
        self.node2edges: List[List[int]] = [[] for _ in range(num_codes)] #[【code,edge】,【code,edge】,....]
        self.pos_pairs: List[Tuple[int,int]] = []  # (node_id, edge_id)
        for eid, e in enumerate(self.hyperedges):
            for v in e:
                self.node2edges[v].append(eid)  #将code2edge中的节点v加入边id
                self.pos_pairs.append((v, eid)) #将边和节点添加映射

        # 每条超边的节点集合（set 形式，便于过滤负样本/共现采样）
        self.edge_nodes = [set(e) for e in self.hyperedges]

        # 为 node–node 共现对采样做缓存：每个节点的同边邻居集合
        self.node2co = [set() for _ in range(num_codes)]
        for e in self.hyperedges:
            for i in range(len(e)):
                vi = e[i]
                for j in range(len(e)):
                    if i != j:
                        self.node2co[vi].add(e[j])

    def sample_incidence_batch(self, batch_size: int, num_neg: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样 (node, edge) 的正样本 + 负样本（节点固定，采负边）"""
        batch = random.sample(self.pos_pairs, k=min(batch_size, len(self.pos_pairs)))
        nodes_pos = torch.tensor([b[0] for b in batch], dtype=torch.long)
        edges_pos = torch.tensor([b[1] for b in batch], dtype=torch.long)

        # 负采样：对每个正样本 (v, e_pos)，随机采 num_neg 个 e_neg，不在该节点的 incident 边集合中
        neg_edges_list = []
        E = len(self.hyperedges)
        for v in nodes_pos.tolist():
            bad = set(self.node2edges[v])
            cand = []
            trials = 0
            while len(cand) < num_neg and trials < num_neg * 10:
                eid = random.randrange(E)
                if eid not in bad:
                    cand.append(eid)
                trials += 1
            if len(cand) < num_neg:
                # 兜底：若太稀疏，重复采样
                while len(cand) < num_neg:
                    cand.append(random.randrange(E))
            neg_edges_list.append(cand)
        edges_neg = torch.tensor(neg_edges_list, dtype=torch.long)  # (B, num_neg)
        return nodes_pos, edges_pos, edges_neg

    def sample_cooccur_batch(self, batch_size: int, num_neg: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样 node–node 共现对 (vi, vj) 作为正样本；负样本从非共现集合采"""
        vi_list, vj_pos_list, vj_neg_list = [], [], []
        for _ in range(batch_size):
            # 随机节点 vi，要求它至少有一个共现邻居
            vi = random.randrange(self.num_codes)
            if len(self.node2co[vi]) == 0:
                continue
            vj_pos = random.choice(list(self.node2co[vi]))
            vi_list.append(vi)
            vj_pos_list.append(vj_pos)

            # 负样本：不在共现集合里
            bad = set(self.node2co[vi]) | {vi}
            negl = []
            trials = 0
            while len(negl) < num_neg and trials < num_neg * 20:
                vn = random.randrange(self.num_codes)
                if vn not in bad:
                    negl.append(vn)
                trials += 1
            if len(negl) < num_neg:
                while len(negl) < num_neg:
                    negl.append(random.randrange(self.num_codes))
            vj_neg_list.append(negl)

        if len(vi_list) == 0:
            # 极端兜底
            vi_list = [0]; vj_pos_list = [1]; vj_neg_list = [[2]*num_neg]

        vi = torch.tensor(vi_list, dtype=torch.long)
        vj_pos = torch.tensor(vj_pos_list, dtype=torch.long)
        vj_neg = torch.tensor(vj_neg_list, dtype=torch.long)  # (B, num_neg)
        return vi, vj_pos, vj_neg


# ================ 编码器 & 损失 ================

class HypergraphEncoder(nn.Module):
    """
    全局超图编码器（最小）：学习
      - 节点嵌入 table: (num_codes, d)
      - 超边嵌入 table: (num_edges, d)
    并提供一层轻量“节点->超边->节点”的消息传递，作为 refinement（可选）
    """
    def __init__(self, num_codes: int, num_edges: int, dim: int, refine: bool = True):
        super().__init__()
        self.node_emb = nn.Embedding(num_codes, dim)
        self.edge_emb = nn.Embedding(num_edges, dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.edge_emb.weight)

        self.refine = refine
        if refine:
            self.lin_e = nn.Linear(dim, dim)
            self.lin_v = nn.Linear(dim, dim)

    def forward_refine(self,
                       node_ids_per_edge: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        可选 refinement：用当前 node_emb 对每个 edge 做平均得到 edge_ctx，再回注到节点平均
        node_ids_per_edge: 超边 -> 节点id列表
        """
        # edge ctx
        edge_ctx = []
        for e_idx, nodes in enumerate(node_ids_per_edge):
            if len(nodes) == 0:
                edge_ctx.append(self.edge_emb.weight[e_idx:e_idx+1, :])  # 退化
            else:
                nvec = self.node_emb.weight[nodes, :]  # (k,d)
                edge_ctx.append(nvec.mean(dim=0, keepdim=True))          # (1,d)
        edge_ctx = torch.cat(edge_ctx, dim=0)                            # (E,d)
        edge_ctx = self.lin_e(edge_ctx)

        # node ctx（把各自incident的edge_ctx平均回去）
        num_nodes = self.node_emb.weight.shape[0]
        node_sum = torch.zeros_like(self.node_emb.weight)
        node_cnt = torch.zeros(num_nodes, device=node_sum.device).unsqueeze(-1).clamp_min(1.0)
        for e_idx, nodes in enumerate(node_ids_per_edge):
            if len(nodes) == 0: 
                continue
            node_sum[nodes] += edge_ctx[e_idx:e_idx+1, :].expand(len(nodes), -1)
            node_cnt[nodes] += 1.0
        node_ctx = node_sum / node_cnt
        node_ctx = self.lin_v(node_ctx)

        # 残差更新（轻微 refinement）
        node_out = self.node_emb.weight + node_ctx
        edge_out = self.edge_emb.weight + edge_ctx
        return node_out, edge_out

    def forward_tables(self, node_ids_per_edge: List[List[int]]):
        if not self.refine:
            return self.node_emb.weight, self.edge_emb.weight
        return self.forward_refine(node_ids_per_edge)


def bce_incidence_loss(nodes: torch.Tensor,    # (B,d) node emb
                       edges: torch.Tensor,    # (B,d) edge emb (pos)
                       edges_neg: torch.Tensor # (B,k,d) neg edge emb
                       ) -> torch.Tensor:
    """重建 H：score = σ(<v, e>)，正样本对=1，负样本对=0"""
    pos_logit = (nodes * edges).sum(-1)                    # (B,)
    pos_loss = F.binary_cross_entropy_with_logits(pos_logit, torch.ones_like(pos_logit))

    neg_logit = (nodes.unsqueeze(1) * edges_neg).sum(-1)   # (B,k)
    neg_loss = F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(neg_logit))
    return pos_loss + neg_loss.mean()


def info_nce(node_i: torch.Tensor,       # (B,d)
             node_pos: torch.Tensor,     # (B,d)
             node_neg: torch.Tensor,     # (B,k,d)
             tau: float = 0.2) -> torch.Tensor:
    """Node–Node 对比：vi 作为 query，对 (vj_pos, vj_neg*) 做 InfoNCE"""
    # logits: (B, 1+k)
    pos = torch.sum(node_i * node_pos, dim=-1, keepdim=True) / tau        # (B,1)
    neg = torch.sum(node_i.unsqueeze(1) * node_neg, dim=-1) / tau         # (B,k)
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def cosine_align(node_emb: torch.Tensor, text_prior: torch.Tensor, ids: torch.Tensor, alpha: float=1.0):
    """把节点嵌入对齐到文本/本体先验（同ID对齐）；用 1 - cos 作为小正则"""
    e = node_emb[ids]                      # (B,d)
    t = text_prior[ids].to(e.device)       # (B,d)
    e = F.normalize(e, dim=-1)
    t = F.normalize(t, dim=-1)
    return alpha * (1.0 - (e * t).sum(-1)).mean()


# ================ 训练脚本 ================

class GlobalPretrainer:
    def __init__(self,
                 dataset: GlobalHypergraphDataset,
                 dim: int = 128,
                 refine: bool = True,
                 lr: float = 2e-3,
                 device: Optional[torch.device] = None):
        self.ds = dataset
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enc = HypergraphEncoder(dataset.num_codes, len(dataset.hyperedges), dim, refine=refine).to(self.device)
        self.opt = torch.optim.Adam(self.enc.parameters(), lr=lr)

    def train(self,
              steps: int = 5000,
              batch_incidence: int = 512,
              batch_cooccur: int = 512,
              neg_k_inc: int = 5,
              neg_k_co: int = 5,
              align_weight: float = 0.1,
              log_interval: int = 100):
        for step in range(1, steps+1):
            self.opt.zero_grad()

            # 取当前表；可选 refinement
            node_table, edge_table = self.enc.forward_tables(self.ds.hyperedges)  # (N,d), (E,d)

            # 1) H 重建（正负采样）
            v_ids, e_pos_ids, e_neg_ids = self.ds.sample_incidence_batch(batch_incidence, neg_k_inc)
            v_ids = v_ids.to(self.device); e_pos_ids = e_pos_ids.to(self.device); e_neg_ids = e_neg_ids.to(self.device)
            v = node_table[v_ids]                                # (B,d)
            e_pos = edge_table[e_pos_ids]                        # (B,d)
            e_neg = edge_table[e_neg_ids]                        # (B,k,d)
            loss_inc = bce_incidence_loss(v, e_pos, e_neg)

            # 2) Node–Node 共现对比
            vi, vj_pos, vj_neg = self.ds.sample_cooccur_batch(batch_cooccur, neg_k_co)
            vi = vi.to(self.device); vj_pos = vj_pos.to(self.device); vj_neg = vj_neg.to(self.device)
            loss_co = info_nce(node_table[vi], node_table[vj_pos], node_table[vj_neg])

            # 3) 文本/本体先验对齐（可选）
            loss_align = torch.tensor(0.0, device=self.device)
            if self.ds.text_prior is not None:
                # 抽一批 id 做对齐（也可以全量）
                ids = torch.randint(0, self.ds.num_codes, (batch_incidence,), device=self.device)
                loss_align = cosine_align(node_table, self.ds.text_prior.to(self.device), ids, alpha=1.0)

            loss = loss_inc + loss_co + align_weight * loss_align
            loss.backward()
            self.opt.step()

            if step % log_interval == 0:
                print(f"[step {step}] loss={loss.item():.4f}  (inc={loss_inc.item():.4f}, co={loss_co.item():.4f}, align={loss_align.item():.4f})")

        # 训练完成，导出节点静态表
        with torch.no_grad():
            node_table, _ = self.enc.forward_tables(self.ds.hyperedges)
        return node_table.detach().cpu()

# ================ 一个可跑的玩具例子 ================
def _make_fake_global_edges(num_codes=5000, num_edges=2000, min_sz=3, max_sz=6):
    rng = random.Random(0)
    edges = []
    for _ in range(num_edges):
        sz = rng.randint(min_sz, max_sz)
        e = rng.sample(range(num_codes), sz)
        edges.append(e)
    return edges

def _make_fake_text_prior(num_codes, dim):
    # 伪文本先验：随机向量（真实应用里用 Bio-Clinical BERT/LLM 对 ICD 描述编码得到）
    t = torch.randn(num_codes, dim)
    t = F.normalize(t, dim=-1)
    return t

if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    NUM_CODES = 5000
    DIM = 128

    # 1) 准备全局超边（真实应用：用 train 集统计得到；勿混入 valid/test 未来信息）
    hyperedges = _make_fake_global_edges(num_codes=NUM_CODES, num_edges=4000, min_sz=3, max_sz=6)

    # 2) （可选）准备 ICD 文本/本体嵌入先验
    text_prior = _make_fake_text_prior(NUM_CODES, DIM)  # 真实应用替换为你的文本向量

    ds = GlobalHypergraphDataset(hyperedges=hyperedges, num_codes=NUM_CODES, text_prior=text_prior)
    trainer = GlobalPretrainer(ds, dim=DIM, refine=True, lr=2e-3)
    static_table = trainer.train(steps=1000, batch_incidence=512, batch_cooccur=512,
                                 neg_k_inc=5, neg_k_co=5, align_weight=0.1, log_interval=100)

    # 3) 保存静态表以便主模型加载
    torch.save(static_table, "static_table.pt")
    print("Saved static_table.pt with shape:", tuple(static_table.shape))