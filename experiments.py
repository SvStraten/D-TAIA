import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from d_taia.config import DTAIAConfig
from d_taia.data import (
    EventLogPreprocessor,
    build_prefixes,
    encode_activities,
    FEATURE_COLUMNS,
    fit_domain_thresholds,
    fit_rt_bucket_thresholds,
    DTAIADataset,
    collate_fn,
)
from d_taia.model import DTAIAModel
from d_taia.engine import set_seed, resolve_device, train_joint, prepare_stage3, evaluate
from d_taia.metrics import accuracy, macro_f1, mae
from d_taia.pipeline import split_cases


def make_loader(prefixes, domain_thresholds, rt_thresholds, cfg, shuffle):
    ds = DTAIADataset(prefixes, domain_thresholds, rt_thresholds, cfg.max_sequence_length)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, cfg.max_sequence_length),
        num_workers=0,
    )


def load_and_prep(cfg, filepath, skip_data_prep):
    clean_csv = cfg.clean_data_dir / f"{cfg.dataset_name}_engineered.csv"
    if skip_data_prep:
        df = pd.read_csv(clean_csv, parse_dates=["timestamp"])
    else:
        preprocessor = EventLogPreprocessor(
            time_unit=cfg.time_unit, min_case_length=cfg.min_case_length
        )
        df = preprocessor.engineer_features(filepath)
        df.to_csv(clean_csv, index=False)
    df, activity_vocab = encode_activities(df)
    return (df, activity_vocab)


def run_once(cfg, train_df, val_df, test_df, activity_vocab, device, return_lengths=False):
    preprocessor = EventLogPreprocessor(
        time_unit=cfg.time_unit, min_case_length=cfg.min_case_length
    )
    train_df = preprocessor.fit_transform_train(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)
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
    train_loader = make_loader(train_prefixes, domain_thresholds, rt_thresholds, cfg, True)
    val_loader = make_loader(val_prefixes, domain_thresholds, rt_thresholds, cfg, False)
    test_loader = make_loader(test_prefixes, domain_thresholds, rt_thresholds, cfg, False)
    num_activities = len(activity_vocab) + 1
    cfg.feature_dim = len(FEATURE_COLUMNS)
    model = DTAIAModel(cfg, num_activities, activity_vocab).to(device)
    train_joint(model, train_loader, val_loader, cfg, device)
    prepare_stage3(model, train_loader, device)
    model.eval()
    all_true_act, all_pred_act, all_true_rt, all_pred_rt, all_prefix_len = ([], [], [], [], [])
    with torch.no_grad():
        for batch, prefixes_batch in zip(test_loader, _batched(test_prefixes, cfg.batch_size)):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            all_true_act.append(batch["next_activity"].cpu().numpy())
            all_pred_act.append(out["activity_logits"].argmax(dim=-1).cpu().numpy())
            all_true_rt.append(batch["remaining_time"].cpu().numpy())
            all_pred_rt.append(out["rt_final"].cpu().numpy())
            all_prefix_len.extend((len(p["activities"]) for p in prefixes_batch))
    y_true_act = np.concatenate(all_true_act)
    y_pred_act = np.concatenate(all_pred_act)
    y_true_rt = preprocessor.inverse_transform_rt(np.concatenate(all_true_rt))
    y_pred_rt = preprocessor.inverse_transform_rt(np.concatenate(all_pred_rt))
    metrics = {
        "accuracy": accuracy(y_true_act, y_pred_act),
        "macro_f1": macro_f1(y_true_act, y_pred_act, num_activities),
        "mae": mae(y_true_rt, y_pred_rt),
    }
    if return_lengths:
        return (metrics, y_true_act, y_pred_act, y_true_rt, y_pred_rt, np.array(all_prefix_len))
    return metrics


def _batched(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def sweep_backbone(dataset_name, filepath, skip_data_prep, seed, backbones, out_path):
    cfg = DTAIAConfig(dataset_name=dataset_name, seed=seed)
    cfg.ensure_dirs()
    device = resolve_device(cfg)
    set_seed(seed)
    df, activity_vocab = load_and_prep(cfg, filepath, skip_data_prep)
    train_df, val_df, test_df = split_cases(df, cfg.test_size, cfg.val_size)
    rows = []
    for backbone in backbones:
        run_cfg = DTAIAConfig(dataset_name=dataset_name, seed=seed, hf_model_name=backbone)
        metrics = run_once(run_cfg, train_df, val_df, test_df, activity_vocab, device)
        rows.append({"backbone": backbone, **metrics})
        print(backbone, metrics)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def sweep_data_pct(dataset_name, filepath, skip_data_prep, seed, pcts, out_path):
    cfg = DTAIAConfig(dataset_name=dataset_name, seed=seed)
    cfg.ensure_dirs()
    device = resolve_device(cfg)
    set_seed(seed)
    df, activity_vocab = load_and_prep(cfg, filepath, skip_data_prep)
    train_df, val_df, test_df = split_cases(df, cfg.test_size, cfg.val_size)
    case_ids = train_df["case_id"].unique()
    rows = []
    for pct in pcts:
        rng = np.random.RandomState(seed)
        n = int(len(case_ids) * pct / 100)
        subset_ids = rng.choice(case_ids, size=n, replace=False)
        sub_train_df = train_df[train_df["case_id"].isin(subset_ids)]
        run_cfg = DTAIAConfig(dataset_name=dataset_name, seed=seed)
        metrics = run_once(run_cfg, sub_train_df, val_df, test_df, activity_vocab, device)
        rows.append({"train_pct": pct, **metrics})
        print(pct, metrics)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def sweep_prefix_length(dataset_name, filepath, skip_data_prep, seed, out_path, n_buckets=5):
    cfg = DTAIAConfig(dataset_name=dataset_name, seed=seed)
    cfg.ensure_dirs()
    device = resolve_device(cfg)
    set_seed(seed)
    df, activity_vocab = load_and_prep(cfg, filepath, skip_data_prep)
    train_df, val_df, test_df = split_cases(df, cfg.test_size, cfg.val_size)
    _, y_true_act, y_pred_act, y_true_rt, y_pred_rt, prefix_len = run_once(
        cfg, train_df, val_df, test_df, activity_vocab, device, return_lengths=True
    )
    max_len = prefix_len.max()
    bucket_edges = np.linspace(0, max_len, n_buckets + 1)
    bucket_idx = np.digitize(prefix_len, bucket_edges[1:-1])
    rows = []
    for b in range(n_buckets):
        mask = bucket_idx == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bucket": f"{int(100 * b / n_buckets)}-{int(100 * (b + 1) / n_buckets)}",
                "n": int(mask.sum()),
                "accuracy": accuracy(y_true_act[mask], y_pred_act[mask]),
                "macro_f1": macro_f1(y_true_act[mask], y_pred_act[mask], len(activity_vocab) + 1),
                "mae": mae(y_true_rt[mask], y_pred_rt[mask]),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_trial(cfg, train_df, val_df, activity_vocab, device):
    preprocessor = EventLogPreprocessor(
        time_unit=cfg.time_unit, min_case_length=cfg.min_case_length
    )
    train_df = preprocessor.fit_transform_train(train_df)
    val_df = preprocessor.transform(val_df)

    train_prefixes = build_prefixes(train_df, cfg.min_prefix_length, cfg.max_prefix_length)
    val_prefixes = build_prefixes(val_df, cfg.min_prefix_length, cfg.max_prefix_length)

    domain_thresholds = fit_domain_thresholds(
        train_prefixes,
        cfg.n_entropy_bins,
        cfg.n_length_bins,
        cfg.domain_bin_strategy,
        cfg.max_sequence_length,
    )
    rt_thresholds = fit_rt_bucket_thresholds(train_prefixes, cfg.rt_bucket_quantiles)

    train_loader = make_loader(train_prefixes, domain_thresholds, rt_thresholds, cfg, True)
    val_loader = make_loader(val_prefixes, domain_thresholds, rt_thresholds, cfg, False)

    num_activities = len(activity_vocab) + 1
    cfg.feature_dim = len(FEATURE_COLUMNS)

    model = DTAIAModel(cfg, num_activities, activity_vocab).to(device)
    train_joint(model, train_loader, val_loader, cfg, device)
    prepare_stage3(model, train_loader, device)

    return evaluate(model, val_loader, preprocessor, device, num_activities)


def grid_search(dataset_name, filepath, skip_data_prep, seed, epochs_per_trial, out_path):
    base_cfg = DTAIAConfig(dataset_name=dataset_name, seed=seed)
    base_cfg.ensure_dirs()
    device = resolve_device(base_cfg)
    set_seed(seed)

    df, activity_vocab = load_and_prep(base_cfg, filepath, skip_data_prep)
    train_df, val_df, test_df = split_cases(df, base_cfg.test_size, base_cfg.val_size)

    grid = {
        "finetune_lr": [1e-4, 5e-4, 1e-3],
        "loss_alpha": [1.0, 2.0, 5.0],
        "triplet_weight": [0.05, 0.10, 0.20],
        "dropout": [0.10],
        "lora_r": [16],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    print(f"{len(combos)} combinations, {epochs_per_trial} epochs per trial")

    rows = []
    for i, combo in enumerate(combos):
        overrides = dict(zip(keys, combo))
        run_cfg = DTAIAConfig(
            dataset_name=dataset_name,
            seed=seed,
            finetune_epochs=epochs_per_trial,
            early_stopping_patience=epochs_per_trial,
            triplet_margin=1.5,
            lora_dropout=0.15,
            batch_size=64,
            lora_alpha=overrides["lora_r"] * 2,
            **overrides,
        )
        metrics = run_trial(run_cfg, train_df.copy(), val_df.copy(), activity_vocab, device)
        rows.append({**overrides, **metrics})
        print(i + 1, len(combos), overrides, metrics)

    result_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    result_df.to_csv(out_path, index=False)
    print(result_df.iloc[0])


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    common = dict(add_help=False)
    for name in ("backbone", "data-pct", "prefix-length"):
        sp = sub.add_parser(name)
        sp.add_argument("--dataset", required=True, dest="dataset_name")
        sp.add_argument("--filepath", default=None)
        sp.add_argument("--skip-data-prep", action="store_true")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--out", default=None)

    grid_sp = sub.add_parser("grid-search")
    grid_sp.add_argument("--dataset", required=True, dest="dataset_name")
    grid_sp.add_argument("--filepath", default=None)
    grid_sp.add_argument("--skip-data-prep", action="store_true")
    grid_sp.add_argument("--seed", type=int, default=42)
    grid_sp.add_argument("--out", default=None)
    grid_sp.add_argument("--epochs-per-trial", type=int, default=5)

    args = p.parse_args()
    out = args.out or f"results/{args.dataset_name}_{args.command.replace('-', '_')}.csv"
    if args.command == "backbone":
        backbones = ["arnir0/Tiny-LLM", "Qwen/Qwen2.5-0.5B", "meta-llama/Llama-3.2-1B"]
        sweep_backbone(
            args.dataset_name, args.filepath, args.skip_data_prep, args.seed, backbones, out
        )
    elif args.command == "data-pct":
        sweep_data_pct(
            args.dataset_name,
            args.filepath,
            args.skip_data_prep,
            args.seed,
            [20, 40, 60, 80, 100],
            out,
        )
    elif args.command == "prefix-length":
        sweep_prefix_length(args.dataset_name, args.filepath, args.skip_data_prep, args.seed, out)
    elif args.command == "grid-search":
        grid_search(
            args.dataset_name,
            args.filepath,
            args.skip_data_prep,
            args.seed,
            args.epochs_per_trial,
            out,
        )


if __name__ == "__main__":
    main()