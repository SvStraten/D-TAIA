import torch
import torch.nn as nn


def load_tinyllm(
    model_name="arnir0/Tiny-LLM",
    cache_dir=None,
    device_map=None,
    torch_dtype="float16",
    load_in_4bit=False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(torch_dtype, torch.float16)
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        dtype=dtype,
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    return (model, tokenizer)


def apply_lora(model, r=16, alpha=32, dropout=0.05, target_modules=None):
    from peft import LoraConfig, get_peft_model, TaskType

    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, lora_cfg)


_FFN_KEYWORDS = {"gate_proj", "up_proj", "down_proj"}


def drop_ffn_deltas(peft_model):
    for name, param in peft_model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_ffn = any((kw in name for kw in _FFN_KEYWORDS))
        if is_lora and is_ffn:
            param.data.zero_()
            param.requires_grad = False


class TinyLLMEncoder(nn.Module):
    def __init__(self, model, tokenizer, max_length=1024):
        super().__init__()
        self.model = model
        self.base_model = model.get_base_model().model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"  # force known layout
        self.max_length = max_length

    def forward(self, texts):
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        )
        device = next(self.model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = self.base_model(**tokens)
        hidden = outputs.last_hidden_state
        lengths = tokens["attention_mask"].sum(dim=1) - 1  # now safe: right-padded
        idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        h_last = hidden.gather(1, idx).squeeze(1)
        return h_last.float()


class TaskEmbeddingEncoder(nn.Module):

    def __init__(self, model, num_activities, feature_dim):
        super().__init__()
        self.model = model
        self.base_model = model.get_base_model().model
        hidden_size = model.config.hidden_size
        self.activity_emb = nn.Embedding(num_activities, hidden_size, padding_idx=0)
        self.feature_proj = nn.Linear(feature_dim, hidden_size)
        self.combine = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, activities, features, lengths):
        act_emb = self.activity_emb(activities)
        feat_emb = self.feature_proj(features)
        inputs_embeds = self.combine(torch.cat([act_emb, feat_emb], dim=-1))
        inputs_embeds = inputs_embeds.to(next(self.model.parameters()).dtype)

        max_len = activities.size(1)
        positions = torch.arange(max_len, device=activities.device).unsqueeze(0)
        attention_mask = (positions < lengths.unsqueeze(1)).long()

        outputs = self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        h_last = hidden.gather(1, idx).squeeze(1)
        return h_last.float()


class LSTMBackbone(nn.Module):

    def __init__(self, num_activities, feature_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.activity_emb = nn.Embedding(num_activities, hidden_dim, padding_idx=0)
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.input_combine = nn.Linear(2 * hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2

    def forward(self, activities, features, lengths):
        act_emb = self.activity_emb(activities)
        feat_emb = self.feature_proj(features)
        x = self.input_combine(torch.cat([act_emb, feat_emb], dim=-1))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return h_last