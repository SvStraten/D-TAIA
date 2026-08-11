import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import DTAIAConfig
from .data import (
    EventLogPreprocessor,
    build_prefixes,
    encode_activities,
    FEATURE_COLUMNS,
    fit_domain_thresholds,
    fit_rt_bucket_thresholds,
    DTAIADataset,
    collate_fn,
)
from .model import DTAIAModel
from .engine import (
    set_seed,
    resolve_device,
    train_datl_stage,
    train_finetune_stage,
    prepare_stage3,
    evaluate,
)


def split_cases(df, test_size, val_size, seed=None):
    case_start = df.groupby("case_id")["timestamp"].min().sort_values()
    case_ids = case_start.index.to_numpy()
    n = len(case_ids)
    n_test, n_val = (int(n * test_size), int(n * val_size))
    n_train = n - n_test - n_val
    train_ids = set(case_ids[:n_train])
    val_ids = set(case_ids[n_train : n_train + n_val])
    test_ids = set(case_ids[n_train + n_val :])
    return (
        df[df["case_id"].isin(train_ids)].copy(),
        df[df["case_id"].isin(val_ids)].copy(),
        df[df["case_id"].isin(test_ids)].copy(),
    )


def build_loaders(cfg, train_df, val_df, test_df):
    train_prefixes = build_prefixes(train_df, cfg.min_prefix_length, cfg.max_prefix_length)
    val_prefixes = build_prefixes(val_df, cfg.min_prefix_length, cfg.max_prefix_length)
    test_prefixes = build_prefixes(test_df, cfg.min_prefix_length, cfg.max_prefix_length)
    domain_thresholds = fit_domain_thresholds(
        train_prefixes,
        cfg.n_entropy_bins,
        cfg.n_length_bins,
        cfg.domain_bin_strategy,
        cfg.max_sequence_length,
    )
    rt_thresholds = fit_rt_bucket_thresholds(train_prefixes, cfg.rt_bucket_quantiles)

    def make_loader(prefixes, shuffle):
        ds = DTAIADataset(prefixes, domain_thresholds, rt_thresholds, cfg.max_sequence_length)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_fn(b, cfg.max_sequence_length),
            num_workers=0,
        )

    return (
        make_loader(train_prefixes, True),
        make_loader(val_prefixes, False),
        make_loader(test_prefixes, False),
    )


def _backbone_tag(cfg):
    if cfg.backbone_lstm:
        return "lstm"
    # short slug from e.g. "arnir0/Tiny-LLM" -> "tiny-llm", "Qwen/Qwen2.5-0.5B" -> "qwen2.5-0.5b"
    return cfg.hf_model_name.split("/")[-1].lower()


def _run_tag(cfg):
    if cfg.backbone_lstm:
        return "mtrnn"
    if cfg.no_datl and cfg.no_taia and cfg.no_faiss:
        return "ftllm"
    flags = [
        f
        for f, on in [
            ("no_datl", cfg.no_datl),
            ("no_domain_id", cfg.no_domain_id),
            ("no_faiss", cfg.no_faiss),
            ("no_taia", cfg.no_taia),
        ]
        if on
    ]
    return "_".join(flags) if flags else "dtaia"


def run(cfg, filepath, skip_data_prep):
    set_seed(cfg.seed)
    cfg.ensure_dirs()
    device = resolve_device(cfg)

    clean_csv = cfg.clean_data_dir / f"{cfg.dataset_name}_engineered.csv"
    preprocessor = EventLogPreprocessor(
        time_unit=cfg.time_unit, min_case_length=cfg.min_case_length
    )
    if skip_data_prep:
        df = pd.read_csv(clean_csv)
        # NOTE: read_csv(parse_dates=[...]) can silently leave the column as
        # strings when a file mixes timestamp formats (e.g. some rows with
        # microseconds, some without -- this happens in bpi12_engineered.csv).
        # It doesn't raise, it just doesn't convert, and everything downstream
        # that does datetime arithmetic (e.g. _add_activity_stats) breaks with
        # a confusing "unsupported operand type(s) for -: 'str' and 'str'".
        # format="ISO8601" handles mixed-but-still-ISO8601 formats explicitly.
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    else:
        assert filepath is not None, "--filepath is required unless --skip-data-prep"
        df = preprocessor.engineer_features(filepath)
        df.to_csv(clean_csv, index=False)

    df, activity_vocab = encode_activities(df)
    num_activities = len(activity_vocab) + 1
    cfg.feature_dim = len(FEATURE_COLUMNS)

    train_df, val_df, test_df = split_cases(df, cfg.test_size, cfg.val_size, cfg.seed)
    train_df = preprocessor.fit_transform_train(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)
    train_loader, val_loader, test_loader = build_loaders(cfg, train_df, val_df, test_df)

    model = DTAIAModel(cfg, num_activities, activity_vocab).to(device)
    print(f"model parameters on: {next(model.parameters()).device}")

    start_time = time.time()
    train_datl_stage(model, train_loader, cfg, device)
    train_finetune_stage(model, train_loader, val_loader, cfg, device)
    prepare_stage3(model, train_loader, device)
    metrics = evaluate(model, test_loader, preprocessor, device, num_activities)
    metrics["runtime_seconds"] = time.time() - start_time
    print(json.dumps(metrics, indent=2))

    tag = _run_tag(cfg)
    backbone = _backbone_tag(cfg)
    results_path = cfg.results_dir / f"{cfg.dataset_name}_{tag}_{backbone}_seed{cfg.seed}.json"
    results_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def build_arg_parser():
    p = argparse.ArgumentParser(description="D-TAIA pipeline")
    p.add_argument(
        "--filepath",
        type=str,
        default=None,
        help="raw .xes file (required unless --skip-data-prep)",
    )
    p.add_argument("--dataset", type=str, required=True, dest="dataset_name")
    p.add_argument("--skip-data-prep", action="store_true")

    p.add_argument("--no-datl", action="store_true")
    p.add_argument("--no-domain-id", action="store_true")
    p.add_argument("--no-faiss", action="store_true")
    p.add_argument("--no-taia", action="store_true")
    p.add_argument("--backbone-lstm", action="store_true")
    p.add_argument("--oyamada-input", action="store_true")
    p.add_argument(
        "--hf-model-name",
        type=str,
        default=None,
        help="HF backbone id, e.g. arnir0/Tiny-LLM | Qwen/Qwen2.5-0.5B | meta-llama/Llama-3.2-1B",
    )

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--finetune-epochs", type=int, default=None)
    p.add_argument("--finetune-lr", type=float, default=None)
    p.add_argument("--loss-alpha", type=float, default=None)
    p.add_argument("--datl-epochs", type=int, default=None)
    p.add_argument("--datl-lr", type=float, default=None)
    p.add_argument("--triplet-margin", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--lstm-hidden-dim", type=int, default=None)
    p.add_argument("--lstm-num-layers", type=int, default=None)
    return p


def main():
    args = build_arg_parser().parse_args()

    overrides = {}
    if args.finetune_epochs is not None:
        overrides["finetune_epochs"] = args.finetune_epochs
    if args.finetune_lr is not None:
        overrides["finetune_lr"] = args.finetune_lr
    if args.loss_alpha is not None:
        overrides["loss_alpha"] = args.loss_alpha
    if args.datl_epochs is not None:
        overrides["datl_epochs"] = args.datl_epochs
    if args.datl_lr is not None:
        overrides["datl_lr"] = args.datl_lr
    if args.triplet_margin is not None:
        overrides["triplet_margin"] = args.triplet_margin
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    if args.lora_r is not None:
        overrides["lora_r"] = args.lora_r
    if args.lora_alpha is not None:
        overrides["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        overrides["lora_dropout"] = args.lora_dropout
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.early_stopping_patience is not None:
        overrides["early_stopping_patience"] = args.early_stopping_patience
    if args.lstm_hidden_dim is not None:
        overrides["lstm_hidden_dim"] = args.lstm_hidden_dim
    if args.lstm_num_layers is not None:
        overrides["lstm_num_layers"] = args.lstm_num_layers
    if args.hf_model_name is not None:
        overrides["hf_model_name"] = args.hf_model_name

    cfg = DTAIAConfig(
        dataset_name=args.dataset_name,
        no_datl=args.no_datl,
        no_domain_id=args.no_domain_id,
        no_faiss=args.no_faiss,
        no_taia=args.no_taia,
        backbone_lstm=args.backbone_lstm,
        oyamada_input=args.oyamada_input,
        seed=args.seed,
        **overrides,
    )
    run(cfg, args.filepath, args.skip_data_prep)


if __name__ == "__main__":
    main()