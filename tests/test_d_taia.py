import numpy as np
import pandas as pd
import torch
from d_taia.config import DTAIAConfig
from d_taia.data import (
    FEATURE_COLUMNS,
    RT_TARGET_COLUMN,
    EventLogPreprocessor,
    build_prefixes,
    encode_activities,
    fit_domain_thresholds,
    fit_rt_bucket_thresholds,
    DTAIADataset,
    collate_fn,
)
from d_taia.model import DTAIAModel
from d_taia.engine import set_seed, train_joint, prepare_stage3, evaluate
from d_taia.pipeline import split_cases


def test_no_leakage():
    assert RT_TARGET_COLUMN not in FEATURE_COLUMNS
    assert "accumulated_time" in FEATURE_COLUMNS


def test_no_datl_no_faiss_independent():
    DTAIAConfig(no_datl=True, no_faiss=False)
    DTAIAConfig(no_datl=True, no_faiss=True)


def make_synthetic_df(n_cases=40, seed=0):
    rng = np.random.RandomState(seed)
    activities = ["A", "B", "C", "D", "E"]
    rows = []
    for case_id in range(n_cases):
        n_events = rng.randint(3, 10)
        ts = pd.Timestamp("2020-01-01") + pd.Timedelta(days=case_id)
        for i in range(n_events):
            ts = ts + pd.Timedelta(hours=int(rng.randint(1, 12)))
            rows.append(
                {
                    "case_id": f"case_{case_id}",
                    "activity": rng.choice(activities),
                    "timestamp": ts,
                    "resource": f"res_{rng.randint(0, 3)}",
                }
            )
    return pd.DataFrame(rows)


def engineer_synthetic(df, cfg):
    pre = EventLogPreprocessor(time_unit=cfg.time_unit, min_case_length=cfg.min_case_length)
    df = pre._filter_short_cases(df)
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    df = pre._add_time_features(df)
    df = pre._add_case_time_targets(df)
    df = pre._add_workload_features(df)
    df = pre._add_case_dynamics(df)
    return df


def test_temporal_split_orders_train_before_test():
    df = make_synthetic_df()
    train_df, val_df, test_df = split_cases(df, test_size=0.2, val_size=0.2)
    train_starts = train_df.groupby("case_id")["timestamp"].min()
    val_starts = val_df.groupby("case_id")["timestamp"].min()
    test_starts = test_df.groupby("case_id")["timestamp"].min()
    assert train_starts.max() <= val_starts.min()
    assert val_starts.max() <= test_starts.min()
    all_ids = set(train_df["case_id"]) | set(val_df["case_id"]) | set(test_df["case_id"])
    assert len(all_ids) == df["case_id"].nunique()
    assert not set(train_df["case_id"]) & set(test_df["case_id"])


def test_stats_fit_on_train_only():
    df = make_synthetic_df()
    df = engineer_synthetic(df, DTAIAConfig())
    train_df, val_df, test_df = split_cases(df, test_size=0.2, val_size=0.2)
    pre = EventLogPreprocessor()
    try:
        pre.transform(test_df)
        assert False, "transform() should refuse to run before fit_transform_train()"
    except RuntimeError:
        pass
    raw_train_rt_mean = train_df[RT_TARGET_COLUMN].mean()
    pre.fit_transform_train(train_df)
    pre.transform(test_df)
    assert abs(pre._norm_stats[RT_TARGET_COLUMN][0] - raw_train_rt_mean) < 1e-06


def test_lstm_pipeline_smoke():
    cfg = DTAIAConfig(
        backbone_lstm=True,
        no_datl=True,
        no_taia=True,
        min_prefix_length=2,
        max_prefix_length=8,
        max_sequence_length=8,
        batch_size=8,
        finetune_epochs=1,
        lstm_hidden_dim=16,
        lstm_num_layers=1,
        head_hidden_dim=16,
        device="cpu",
    )
    set_seed(cfg.seed)
    raw_df = make_synthetic_df()
    df = engineer_synthetic(raw_df, cfg)
    df, activity_vocab = encode_activities(df)
    cfg.feature_dim = len(FEATURE_COLUMNS)
    num_activities = len(activity_vocab) + 1
    train_df, val_df, test_df = split_cases(df, cfg.test_size, cfg.val_size)
    preprocessor = EventLogPreprocessor(
        time_unit=cfg.time_unit, min_case_length=cfg.min_case_length
    )
    train_df = preprocessor.fit_transform_train(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)
    prefixes = build_prefixes(train_df, cfg.min_prefix_length, cfg.max_prefix_length)
    assert len(prefixes) > 0
    domain_thresholds = fit_domain_thresholds(
        prefixes,
        cfg.n_entropy_bins,
        cfg.n_length_bins,
        cfg.domain_bin_strategy,
        cfg.max_sequence_length,
    )
    rt_thresholds = fit_rt_bucket_thresholds(prefixes, cfg.rt_bucket_quantiles)
    ds = DTAIADataset(prefixes, domain_thresholds, rt_thresholds, cfg.max_sequence_length)
    from torch.utils.data import DataLoader

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, cfg.max_sequence_length),
    )
    device = torch.device("cpu")
    model = DTAIAModel(cfg, num_activities, activity_vocab).to(device)
    train_joint(model, loader, loader, cfg, device)
    prepare_stage3(model, loader, device)
    metrics = evaluate(model, loader, preprocessor, device, num_activities)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["mae"] >= 0.0