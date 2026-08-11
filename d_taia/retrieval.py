import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class DATLEncoder(nn.Module):

    def __init__(
        self,
        num_activities,
        feature_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_ff=1024,
        dropout=0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.activity_emb = nn.Embedding(num_activities, d_model, padding_idx=0)
        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.combine = nn.Linear(2 * d_model, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, activities, features, lengths):
        act_emb = self.activity_emb(activities)
        feat_emb = self.feature_proj(features)
        x = self.combine(torch.cat([act_emb, feat_emb], dim=-1))
        x = self.pos_enc(x)
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        x = self.transformer(x, src_key_padding_mask=mask)
        mask_float = (~mask).float().unsqueeze(-1)
        h = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        return self.out_norm(h)


class TripletLoss(nn.Module):

    def __init__(self, margin=1.0, distance="cosine"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, anchor, positive, negative):
        if self.distance == "cosine":
            d_ap = 1.0 - F.cosine_similarity(anchor, positive)
            d_an = 1.0 - F.cosine_similarity(anchor, negative)
        else:
            d_ap = (anchor - positive).pow(2).sum(dim=1).sqrt()
            d_an = (anchor - negative).pow(2).sum(dim=1).sqrt()
        return F.relu(d_ap - d_an + self.margin).mean()


def mine_domain_aware_triplets(domain_ids, rt_buckets):
    B = domain_ids.size(0)
    device = domain_ids.device
    domain_ids_np = domain_ids.cpu().numpy()
    rt_buckets_np = rt_buckets.cpu().numpy()
    anchors, positives, negatives = ([], [], [])
    rng = np.random.RandomState(0)
    for i in range(B):
        same_rt_diff_domain = np.where(
            (rt_buckets_np == rt_buckets_np[i]) & (domain_ids_np != domain_ids_np[i])
        )[0]
        diff_rt_same_domain = np.where(
            (rt_buckets_np != rt_buckets_np[i]) & (domain_ids_np == domain_ids_np[i])
        )[0]
        if len(same_rt_diff_domain) == 0 or len(diff_rt_same_domain) == 0:
            continue
        anchors.append(i)
        positives.append(rng.choice(same_rt_diff_domain))
        negatives.append(rng.choice(diff_rt_same_domain))
    if not anchors:
        return None
    return (
        torch.tensor(anchors, device=device, dtype=torch.long),
        torch.tensor(positives, device=device, dtype=torch.long),
        torch.tensor(negatives, device=device, dtype=torch.long),
    )


class DomainAwareTripletLoss(nn.Module):

    def __init__(
        self,
        num_activities,
        feature_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_ff=1024,
        dropout=0.3,
        margin=1.0,
        distance="cosine",
    ):
        super().__init__()
        self.encoder = DATLEncoder(
            num_activities, feature_dim, d_model, nhead, num_layers, dim_ff, dropout
        )
        self.triplet_loss = TripletLoss(margin=margin, distance=distance)

    def forward(self, activities, features, lengths, domain_ids, rt_buckets):
        h = self.encoder(activities, features, lengths)
        mined = mine_domain_aware_triplets(domain_ids, rt_buckets)
        if mined is None:
            stats = {"n_triplets": 0, "d_ap_mean": None, "d_an_mean": None}
            return (torch.tensor(0.0, device=h.device, requires_grad=True), h, stats)
        a_idx, p_idx, n_idx = mined
        loss = self.triplet_loss(h[a_idx], h[p_idx], h[n_idx])
        with torch.no_grad():
            if self.triplet_loss.distance == "cosine":
                d_ap = 1.0 - F.cosine_similarity(h[a_idx], h[p_idx])
                d_an = 1.0 - F.cosine_similarity(h[a_idx], h[n_idx])
            else:
                d_ap = (h[a_idx] - h[p_idx]).pow(2).sum(dim=1).sqrt()
                d_an = (h[a_idx] - h[n_idx]).pow(2).sum(dim=1).sqrt()
        stats = {
            "n_triplets": len(a_idx),
            "d_ap_mean": d_ap.mean().item(),
            "d_an_mean": d_an.mean().item(),
        }
        return (loss, h, stats)


class FAISSIndex:

    def __init__(self, dim, index_type="flat", nprobe=10):
        self.dim = dim
        self.index_type = index_type
        self._embeddings = None
        self._rt_values = None
        if HAS_FAISS:
            if index_type == "ivf":
                quantiser = faiss.IndexFlatL2(dim)
                self.index = faiss.IndexIVFFlat(quantiser, dim, min(64, 4))
                self.index.nprobe = nprobe
            else:
                self.index = faiss.IndexFlatL2(dim)
        else:
            self.index = None

    def build(self, embeddings, rt_values):
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._embeddings = embeddings
        self._rt_values = np.asarray(rt_values, dtype=np.float32)
        if HAS_FAISS:
            if self.index_type == "ivf" and (not self.index.is_trained):
                self.index.train(embeddings)
            self.index.add(embeddings)

    def search(self, query, k):
        query = np.ascontiguousarray(query, dtype=np.float32)
        k = min(k, len(self._rt_values))
        if HAS_FAISS:
            distances, indices = self.index.search(query, k)
        else:
            d2 = ((query[:, None, :] - self._embeddings[None, :, :]) ** 2).sum(-1)
            indices = np.argsort(d2, axis=1)[:, :k]
            distances = np.take_along_axis(d2, indices, axis=1)
        rt_neighbors = self._rt_values[indices]
        return (distances, rt_neighbors)


class FAISSRTIndex:

    def __init__(self, dim, index_type="flat", nprobe=10, top_k=10):
        self.index = FAISSIndex(dim=dim, index_type=index_type, nprobe=nprobe)
        self.top_k = top_k

    def build(self, embeddings, rt_values):
        self.index.build(embeddings, rt_values)

    def query(self, embeddings):
        distances, rt_neighbors = self.index.search(embeddings, self.top_k)
        weights = np.exp(-distances)
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_sum = np.where(weights_sum < 1e-12, 1.0, weights_sum)
        return (weights * rt_neighbors).sum(axis=1) / weights_sum.squeeze(-1)

    def save(self, path):
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(
            path / "faiss_rt_index.npz",
            embeddings=self.index._embeddings,
            rt_values=self.index._rt_values,
        )

    def load(self, path):
        from pathlib import Path

        data = np.load(Path(path) / "faiss_rt_index.npz")
        self.build(data["embeddings"], data["rt_values"])