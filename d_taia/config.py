from pathlib import Path


class DTAIAConfig:
    def __init__(
        self,
        raw_data_dir=Path("data_functions/raw_data"),
        clean_data_dir=Path("data_functions/clean_data"),
        model_dir=Path("models"),
        results_dir=Path("results"),
        faiss_dir=Path("faiss_indices"),
        dataset_name="bpi2012",
        test_size=0.20,
        val_size=0.15,
        min_case_length=2,
        time_unit="days",
        max_sequence_length=20,
        min_prefix_length=2,
        max_prefix_length=20,
        hf_model_name="arnir0/Tiny-LLM",
        hf_cache_dir=None,
        hf_device_map=None,
        hf_torch_dtype="bfloat16",
        hf_load_in_4bit=False,
        hf_max_length=1024,
        backbone_lstm=False,
        oyamada_input=False,
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.10,
        lora_target_modules=None,
        finetune_epochs=3,
        finetune_lr=2e-4,
        finetune_batch_size=8,
        no_taia=False,
        no_datl=False,
        datl_encoder_dim=256,
        datl_encoder_heads=8,
        datl_encoder_layers=4,
        datl_encoder_ff_dim=1024,
        dropout=0.3,
        datl_lr=1e-3,
        datl_epochs=30,
        datl_batch_size=64,
        triplet_margin=1.0,
        triplet_distance="cosine",
        no_domain_id=False,
        n_entropy_bins=3,
        n_length_bins=3,
        domain_bin_strategy="quantile",
        n_rt_buckets=3,
        rt_bucket_quantiles=None,
        no_faiss=False,
        faiss_index_type="flat",
        faiss_nprobe=10,
        faiss_top_k=10,
        feature_dim=None,
        activity_embedding_dim=128,
        head_hidden_dim=256,
        fusion_beta=0.5,
        loss_alpha=1.0,
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=10,
        gradient_clip=1.0,
        seed=42,
        device="cuda",
        num_workers=4,
    ):
        self.raw_data_dir = raw_data_dir
        self.clean_data_dir = clean_data_dir
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.faiss_dir = faiss_dir

        self.dataset_name = dataset_name
        self.test_size = test_size
        self.val_size = val_size
        self.min_case_length = min_case_length
        self.time_unit = time_unit
        self.max_sequence_length = max_sequence_length
        self.min_prefix_length = min_prefix_length
        self.max_prefix_length = max_prefix_length

        self.hf_model_name = hf_model_name
        self.hf_cache_dir = hf_cache_dir
        self.hf_device_map = hf_device_map
        self.hf_torch_dtype = hf_torch_dtype
        self.hf_load_in_4bit = hf_load_in_4bit
        self.hf_max_length = hf_max_length

        self.backbone_lstm = backbone_lstm
        self.oyamada_input = oyamada_input
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.finetune_batch_size = finetune_batch_size

        self.no_taia = no_taia

        self.no_datl = no_datl
        self.datl_encoder_dim = datl_encoder_dim
        self.datl_encoder_heads = datl_encoder_heads
        self.datl_encoder_layers = datl_encoder_layers
        self.datl_encoder_ff_dim = datl_encoder_ff_dim
        self.dropout = dropout
        self.datl_lr = datl_lr
        self.datl_epochs = datl_epochs
        self.datl_batch_size = datl_batch_size
        self.triplet_margin = triplet_margin
        self.triplet_distance = triplet_distance

        self.no_domain_id = no_domain_id
        self.n_entropy_bins = n_entropy_bins
        self.n_length_bins = n_length_bins
        self.domain_bin_strategy = domain_bin_strategy

        self.n_rt_buckets = n_rt_buckets
        self.rt_bucket_quantiles = rt_bucket_quantiles or [0.33, 0.66]

        self.no_faiss = no_faiss
        self.faiss_index_type = faiss_index_type
        self.faiss_nprobe = faiss_nprobe
        self.faiss_top_k = faiss_top_k

        self.feature_dim = feature_dim
        self.activity_embedding_dim = activity_embedding_dim
        self.head_hidden_dim = head_hidden_dim
        self.fusion_beta = fusion_beta

        self.loss_alpha = loss_alpha

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip = gradient_clip

        self.seed = seed
        self.device = device
        self.num_workers = num_workers

        if not 0.0 <= self.fusion_beta <= 1.0:
            raise ValueError(f"fusion_beta must be in [0, 1], got {self.fusion_beta}")
        if len(self.rt_bucket_quantiles) != self.n_rt_buckets - 1:
            raise ValueError(
                f"rt_bucket_quantiles must have n_rt_buckets - 1 = "
                f"{self.n_rt_buckets - 1} entries, got {len(self.rt_bucket_quantiles)}"
            )
        if self.domain_bin_strategy not in ("quantile", "equal_width"):
            raise ValueError(f"unknown domain_bin_strategy: {self.domain_bin_strategy}")

    def ensure_dirs(self):
        for d in (
            self.raw_data_dir,
            self.clean_data_dir,
            self.model_dir,
            self.results_dir,
            self.faiss_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)