import numpy as np
import torch
import torch.nn as nn

from .backbone import (
    load_tinyllm,
    apply_lora,
    drop_ffn_deltas,
    TinyLLMEncoder,
    TaskEmbeddingEncoder,
    LSTMBackbone,
)
from .retrieval import DomainAwareTripletLoss, FAISSRTIndex
from .heads import ActivityHead, TimeHead


class FusionGate:

    def __init__(self, beta=0.5):
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.beta = beta

    def fuse(self, rt_direct, rt_retrieved):
        return self.beta * rt_direct + (1.0 - self.beta) * rt_retrieved


class DTAIAModel(nn.Module):

    def __init__(self, cfg, num_activities, activity_vocab):
        super().__init__()
        self.cfg = cfg
        self.num_activities = num_activities
        self.activity_vocab_inv = {v: k for k, v in activity_vocab.items()}

        if cfg.backbone_lstm:
            self.backbone_kind = "lstm"
            self.backbone = LSTMBackbone(
                num_activities,
                cfg.feature_dim,
                cfg.lstm_hidden_dim,
                cfg.lstm_num_layers,
                dropout=cfg.dropout,
            )
            backbone_dim = self.backbone.output_dim
            self.tokenizer = None
            self.datl = None
            self.retrieval_index = None
        else:
            hf_model, tokenizer = load_tinyllm(
                model_name=cfg.hf_model_name,
                cache_dir=cfg.hf_cache_dir,
                device_map=cfg.hf_device_map,
                torch_dtype=cfg.hf_torch_dtype,
                load_in_4bit=cfg.hf_load_in_4bit,
            )
            hf_model = apply_lora(
                hf_model,
                r=cfg.lora_r,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
            )
            if cfg.oyamada_input:
                self.backbone_kind = "tinyllm_embedding"
                self.backbone = TaskEmbeddingEncoder(hf_model, num_activities, cfg.feature_dim)
                self.tokenizer = None
            else:
                self.backbone_kind = "tinyllm_text"
                self.backbone = TinyLLMEncoder(hf_model, tokenizer, max_length=cfg.hf_max_length)
                self.tokenizer = tokenizer
            backbone_dim = hf_model.config.hidden_size
            self.datl = DomainAwareTripletLoss(
                num_activities,
                cfg.feature_dim,
                d_model=cfg.datl_encoder_dim,
                nhead=cfg.datl_encoder_heads,
                num_layers=cfg.datl_encoder_layers,
                dim_ff=cfg.datl_encoder_ff_dim,
                dropout=cfg.dropout,
                margin=cfg.triplet_margin,
                distance=cfg.triplet_distance,
            )
            self.retrieval_index = (
                None
                if cfg.no_faiss
                else FAISSRTIndex(
                    dim=cfg.datl_encoder_dim,
                    index_type=cfg.faiss_index_type,
                    nprobe=cfg.faiss_nprobe,
                    top_k=cfg.faiss_top_k,
                )
            )

        self.activity_head = ActivityHead(backbone_dim, num_activities, dropout=cfg.dropout)
        self.time_head = TimeHead(backbone_dim, hidden_dim=cfg.head_hidden_dim, dropout=cfg.dropout)
        self.fusion = FusionGate(beta=cfg.fusion_beta)

    def serialize_prefixes(self, batch):
        activities = batch["activities"].cpu().numpy()
        features = batch["features"].cpu().numpy()
        lengths = batch["lengths"].cpu().numpy()
        domain_ids = batch["domain_id"].cpu().numpy()

        texts = []
        for i in range(len(lengths)):
            L = int(lengths[i])
            parts = []
            if not self.cfg.no_domain_id:
                parts.append(f"[DOMAIN={int(domain_ids[i])}]")
            for t in range(L):
                act_name = self.activity_vocab_inv.get(int(activities[i, t]), "UNK")
                feat_str = ",".join((f"{v:.3f}" for v in features[i, t]))
                parts.append(f"{act_name}:[{feat_str}]")
            texts.append(" ".join(parts))
        return texts

    def forward(self, batch):
        if self.backbone_kind == "lstm":
            h = self.backbone(batch["activities"], batch["features"], batch["lengths"])
            logits = self.activity_head(h)
            rt_direct = self.time_head(h).squeeze(-1)
            return {
                "activity_logits": logits,
                "rt_direct": rt_direct,
                "rt_final": rt_direct,
                "datl_loss": None,
            }

        if self.backbone_kind == "tinyllm_embedding":
            h_last = self.backbone(batch["activities"], batch["features"], batch["lengths"])
        else:
            texts = self.serialize_prefixes(batch)
            h_last = self.backbone(texts)

        logits = self.activity_head(h_last)
        rt_direct = self.time_head(h_last).squeeze(-1)

        datl_loss, rt_embed = (None, None)
        if self.datl is not None:
            datl_loss, rt_embed, _ = self.datl(
                batch["activities"],
                batch["features"],
                batch["lengths"],
                batch["domain_id"],
                batch["rt_bucket"],
            )

        rt_final = rt_direct
        if (
            self.retrieval_index is not None
            and self.retrieval_index.index._embeddings is not None
            and (rt_embed is not None)
        ):
            rt_retrieved = self.retrieval_index.query(rt_embed.detach().cpu().numpy())
            rt_retrieved_t = torch.from_numpy(rt_retrieved).to(rt_direct.device).float()
            rt_final = self.fusion.fuse(rt_direct, rt_retrieved_t)

        return {
            "activity_logits": logits,
            "rt_direct": rt_direct,
            "rt_final": rt_final,
            "datl_loss": datl_loss,
            "rt_embed": rt_embed,
        }

    def build_retrieval_index(self, embeddings, rt_values):
        if self.retrieval_index is not None:
            self.retrieval_index.build(embeddings, rt_values)

    def apply_taia_inference(self):
        if self.backbone_kind in ("tinyllm_text", "tinyllm_embedding") and (not self.cfg.no_taia):
            drop_ffn_deltas(self.backbone.model)