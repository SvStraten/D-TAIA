import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import pm4py
    HAS_PM4PY = True
except ImportError:
    HAS_PM4PY = False


FEATURE_COLUMNS = [
    "accumulated_time",
    "day_of_month", "day_of_week", "hour_of_day",
    "min_of_hour", "sec_of_min", "week_of_year",
    "month_of_year", "day_of_year", "secs_within_day",
    "avg_duration_activity", "std_duration_activity",
    "hour_sin", "hour_cos",
    "is_business_hours",
    "concurrent_cases", "workload_ratio",
    "velocity", "acceleration",
]

RT_TARGET_COLUMN = "remaining_time"

TIME_UNIT_DIVISORS = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}


class EventLogPreprocessor:

    def __init__(self, time_unit="days", min_case_length=2):
        self.time_unit = time_unit
        self.min_case_length = min_case_length
        self._norm_stats = {}

    def load_xes(self, filepath):
        if not HAS_PM4PY:
            raise ImportError("pm4py is required to parse XES logs: pip install pm4py")
        log = pm4py.read_xes(str(filepath))
        df = pm4py.convert_to_dataframe(log)
        df = df.rename(columns={
            "case:concept:name": "case_id",
            "concept:name": "activity",
            "time:timestamp": "timestamp",
            "org:resource": "resource",
        })
        keep = [c for c in ("case_id", "activity", "timestamp", "resource") if c in df.columns]
        df = df[keep].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if "resource" not in df.columns:
            df["resource"] = "unknown"
        return df

    def _filter_short_cases(self, df):
        lengths = df.groupby("case_id")["activity"].transform("size")
        return df[lengths >= self.min_case_length].copy()

    def _add_time_features(self, df):
        ts = df["timestamp"]
        df["day_of_month"] = ts.dt.day
        df["day_of_week"] = ts.dt.dayofweek
        df["hour_of_day"] = ts.dt.hour
        df["min_of_hour"] = ts.dt.minute
        df["sec_of_min"] = ts.dt.second
        df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
        df["month_of_year"] = ts.dt.month
        df["day_of_year"] = ts.dt.dayofyear
        df["secs_within_day"] = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["is_business_hours"] = (
            (df["hour_of_day"] >= 9) & (df["hour_of_day"] < 17)
            & (df["day_of_week"] < 5)
        ).astype(float)
        return df

    def _add_case_time_targets(self, df):
        divisor = TIME_UNIT_DIVISORS[self.time_unit]
        case_start = df.groupby("case_id")["timestamp"].transform("min")
        case_end = df.groupby("case_id")["timestamp"].transform("max")
        df["accumulated_time"] = (df["timestamp"] - case_start).dt.total_seconds() / divisor
        df[RT_TARGET_COLUMN] = (case_end - df["timestamp"]).dt.total_seconds() / divisor
        return df

    def _add_activity_stats(self, df, fit):
        df = df.sort_values(["case_id", "timestamp"])
        next_ts = df.groupby("case_id")["timestamp"].shift(-1)
        gap = (next_ts - df["timestamp"]).dt.total_seconds().fillna(0.0)
        if fit:
            stats = gap.groupby(df["activity"]).agg(["mean", "std"]).fillna(0.0)
            self._activity_duration_stats = stats
        stats = getattr(self, "_activity_duration_stats", None)
        if stats is None:
            df["avg_duration_activity"] = 0.0
            df["std_duration_activity"] = 0.0
        else:
            df["avg_duration_activity"] = df["activity"].map(stats["mean"]).fillna(0.0)
            df["std_duration_activity"] = df["activity"].map(stats["std"]).fillna(0.0)
        return df

    def _add_workload_features(self, df):
        bounds = df.groupby("case_id")["timestamp"].agg(["min", "max"])
        starts = bounds["min"].sort_values()
        ends = bounds["max"].sort_values()
        events = [(t, 1) for t in starts] + [(t, -1) for t in ends]
        events.sort(key=lambda x: (x[0], -x[1]))
        running = 0
        concurrency_at = {}
        for t, delta in events:
            running += delta
            concurrency_at[t] = running

        conc_series = pd.Series(concurrency_at).sort_index()
        conc_df = pd.DataFrame({"timestamp": conc_series.index,
                                 "concurrent_cases": conc_series.values})
        df = df.sort_values("timestamp")
        df = pd.merge_asof(df, conc_df, on="timestamp", direction="backward")
        df["concurrent_cases"] = df["concurrent_cases"].fillna(0)
        max_conc = max(df["concurrent_cases"].max(), 1)
        df["workload_ratio"] = df["concurrent_cases"] / max_conc
        return df

    def _add_case_dynamics(self, df):
        df = df.sort_values(["case_id", "timestamp"])
        df["_event_idx"] = df.groupby("case_id").cumcount()
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity = np.where(
                df["accumulated_time"] > 0,
                df["_event_idx"] / df["accumulated_time"].replace(0, np.nan),
                0.0,
            )
        df["velocity"] = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
        df["acceleration"] = df.groupby("case_id")["velocity"].diff().fillna(0.0)
        df = df.drop(columns=["_event_idx"])
        return df

    def normalize(self, df, fit):
        cols = FEATURE_COLUMNS + [RT_TARGET_COLUMN]
        for col in cols:
            if fit:
                mu, sigma = df[col].mean(), df[col].std()
                sigma = sigma if sigma > 1e-8 else 1.0
                self._norm_stats[col] = (mu, sigma)
            mu, sigma = self._norm_stats[col]
            df[col] = (df[col] - mu) / sigma
        return df

    def inverse_transform_rt(self, values):
        mu, sigma = self._norm_stats[RT_TARGET_COLUMN]
        return values * sigma + mu

    def engineer_features(self, filepath):
        df = self.load_xes(filepath)
        df = self._filter_short_cases(df)
        df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
        df = self._add_time_features(df)
        df = self._add_case_time_targets(df)
        df = self._add_workload_features(df)
        df = self._add_case_dynamics(df)
        return df

    def fit_transform_train(self, train_df):
        df = self._add_activity_stats(train_df, fit=True)
        df = self.normalize(df, fit=True)
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        assert not missing, f"FEATURE_COLUMNS references missing columns: {missing}"
        return df

    def transform(self, df):
        if not self._norm_stats:
            raise RuntimeError("call fit_transform_train on the training split first")
        df = self._add_activity_stats(df, fit=False)
        df = self.normalize(df, fit=False)
        return df


def build_prefixes(df, min_len, max_len):
    prefixes = []
    for case_id, cdf in df.groupby("case_id"):
        cdf = cdf.sort_values("timestamp").reset_index(drop=True)
        acts = cdf["activity_encoded"].values
        feats = cdf[FEATURE_COLUMNS].values.astype(np.float32)
        rt = cdf[RT_TARGET_COLUMN].values.astype(np.float32)
        n = len(cdf)
        for k in range(min_len, n):  # full range per Definition 1/2, not capped at max_len
            start = max(0, k - max_len)  # keep only the most recent max_len events
            prefixes.append({
                "case_id": case_id,
                "k": k,
                "activities": acts[start:k].copy(),
                "features": feats[start:k].copy(),
                "next_activity": int(acts[k]),
                "remaining_time": float(rt[k - 1]),
            })
    return prefixes


def encode_activities(df):
    vocab = {a: i + 1 for i, a in enumerate(sorted(df["activity"].unique()))}
    df = df.copy()
    df["activity_encoded"] = df["activity"].map(vocab)
    return df, vocab


def compute_prefix_entropy(activities):
    if len(activities) == 0:
        return 0.0
    counts = np.bincount(activities)
    probs = counts[counts > 0] / len(activities)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


class DomainThresholds:
    def __init__(self, entropy_bin_edges, length_bin_edges, n_entropy_bins, n_length_bins):
        self.entropy_bin_edges = entropy_bin_edges
        self.length_bin_edges = length_bin_edges
        self.n_entropy_bins = n_entropy_bins
        self.n_length_bins = n_length_bins

    def assign(self, entropy, length):
        e_bin = int(np.digitize(entropy, self.entropy_bin_edges[1:-1]))
        l_bin = int(np.digitize(length, self.length_bin_edges[1:-1]))
        e_bin = min(e_bin, self.n_entropy_bins - 1)
        l_bin = min(l_bin, self.n_length_bins - 1)
        return e_bin * self.n_length_bins + l_bin


def fit_domain_thresholds(prefixes, n_entropy_bins, n_length_bins, strategy, max_len):
    entropies = np.array([compute_prefix_entropy(p["activities"]) for p in prefixes])
    lengths = np.array([min(len(p["activities"]), max_len) for p in prefixes])

    def edges(values, n_bins):
        if strategy == "quantile":
            qs = np.linspace(0, 1, n_bins + 1)
            e = np.quantile(values, qs)
            e[0], e[-1] = -np.inf, np.inf
            return np.unique(e) if len(np.unique(e)) == n_bins + 1 else np.linspace(
                values.min() - 1e-6, values.max() + 1e-6, n_bins + 1
            )
        else:
            e = np.linspace(values.min(), values.max(), n_bins + 1)
            e[0], e[-1] = -np.inf, np.inf
            return e

    return DomainThresholds(
        entropy_bin_edges=edges(entropies, n_entropy_bins),
        length_bin_edges=edges(lengths.astype(float), n_length_bins),
        n_entropy_bins=n_entropy_bins,
        n_length_bins=n_length_bins,
    )


class RTBucketThresholds:
    def __init__(self, q_edges):
        self.q_edges = q_edges

    def assign(self, rt):
        return int(np.digitize(rt, self.q_edges))


def fit_rt_bucket_thresholds(prefixes, quantiles):
    rts = np.array([p["remaining_time"] for p in prefixes])
    return RTBucketThresholds(q_edges=np.quantile(rts, quantiles))


class DTAIADataset(Dataset):

    def __init__(self, prefixes, domain_thresholds, rt_thresholds, max_len):
        self.prefixes = prefixes
        self.domain_thresholds = domain_thresholds
        self.rt_thresholds = rt_thresholds
        self.max_len = max_len

    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        p = self.prefixes[idx]
        entropy = compute_prefix_entropy(p["activities"])
        length = min(len(p["activities"]), self.max_len)
        domain_id = self.domain_thresholds.assign(entropy, length)
        rt_bucket = self.rt_thresholds.assign(p["remaining_time"])
        return {
            "activities": torch.from_numpy(p["activities"]).long(),
            "features": torch.from_numpy(p["features"]).float(),
            "length": length,
            "next_activity": p["next_activity"],
            "remaining_time": p["remaining_time"],
            "domain_id": domain_id,
            "rt_bucket": rt_bucket,
        }


def collate_fn(batch, max_len):
    B = len(batch)
    feat_dim = batch[0]["features"].shape[1]
    activities = torch.zeros(B, max_len, dtype=torch.long)
    features = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, item in enumerate(batch):
        L = min(item["length"], max_len)
        activities[i, :L] = item["activities"][:L]
        features[i, :L] = item["features"][:L]
        lengths[i] = L
    return {
        "activities": activities,
        "features": features,
        "lengths": lengths,
        "next_activity": torch.tensor([b["next_activity"] for b in batch], dtype=torch.long),
        "remaining_time": torch.tensor([b["remaining_time"] for b in batch], dtype=torch.float32),
        "domain_id": torch.tensor([b["domain_id"] for b in batch], dtype=torch.long),
        "rt_bucket": torch.tensor([b["rt_bucket"] for b in batch], dtype=torch.long),
    }