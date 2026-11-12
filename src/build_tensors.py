#!/usr/bin/env python3
"""
Turn the per-day parquet files from preprocess.py into:
- tensors.npz        station x time delay tensor (mean delay per bin) + mask + counts
- events.parquet     filtered stop events with split / bin / node columns
- splits.json        which days went to train / val / test
- baseline_results.csv, baseline_by_route.csv   the three trivial baselines

Usage:
    python build_tensors.py --proc-dir processed --out tensors_out --seed 42

Task used for the baselines (event level, so per-line results are possible):
  For every stop event (a train arriving at a station with delay y), forecast y
  from information available --horizon-min minutes earlier:
    zero         : always 0
    persistence  : last observed station delay (mean over all trains in a bin) before
                   the forecast origin, if seen within --lookback-min, else 0
    historical   : mean train delay for (station, time-of-day slot, weekday/weekend)
                   computed on the training days only
Splits are randomized over whole local days using --seed. Days with very few events
(edge effects) are dropped before splitting.
Filtering (thresholds are judgement calls, report them): trips whose median delay is
outside [trip-lo, trip-hi] and single stops outside [stop-lo, stop-hi] are dropped.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BREAKS = [180, 900]  # <3 min minor, 3-15 min moderate, >=15 min major


# ------------------------------------------------------------------ loading
def load_events(proc_dir, trip_lo, trip_hi, stop_lo, stop_hi):
    files = sorted(glob.glob(f"{proc_dir}/delays_*.parquet"))
    if not files:
        raise SystemExit(f"No delays_*.parquet in {proc_dir}")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    m = d[d["stop_matched"] & ~d["secondary_record"]].copy()
    for c in ("delay_s", "event_sec"):
        m[c] = m[c].astype("float64")
    med = m.groupby("trip_uid")["delay_s"].transform("median")
    keep = med.between(trip_lo, trip_hi) & m["delay_s"].between(stop_lo, stop_hi)
    print(f"{len(m)} matched stops, keeping {keep.mean():.2%} after filters")
    m = m[keep]
    m["route_id"] = m["route_id"].astype(str)
    m["station"] = m["station"].astype(str)
    return m[["trip_uid", "route_id", "direction_id", "station", "event_sec", "delay_s"]].reset_index(drop=True)


# ------------------------------------------------------------------ tensors
def build_tensor(ev, n_days, bins_per_day, bin_sec, t0):
    stations = np.sort(ev["station"].unique())
    sidx = {s: i for i, s in enumerate(stations)}
    ev["node"] = ev["station"].map(sidx).astype(int)
    ev["bin"] = ((ev["event_sec"] - t0) // bin_sec).astype(int)
    T, N = n_days * bins_per_day, len(stations)
    g = ev.groupby(["bin", "node"])["delay_s"].agg(["sum", "count"])
    b = g.index.get_level_values(0).to_numpy()
    n = g.index.get_level_values(1).to_numpy()
    delay = np.zeros((T, N), dtype=np.float32)
    count = np.zeros((T, N), dtype=np.int32)
    count[b, n] = g["count"].to_numpy()
    delay[b, n] = (g["sum"] / g["count"]).to_numpy()
    return delay, count > 0, count, stations


def last_observed_index(mask):
    T = mask.shape[0]
    idx = np.where(mask, np.arange(T, dtype=np.int32)[:, None], -1).astype(np.int32)
    return np.maximum.accumulate(idx, axis=0)


# --------------------------------------------------------------- baselines
def persistence(df, last, delay, t0, bin_sec, horizon_min, lookback_min):
    b0 = ((df["event_sec"].to_numpy() - horizon_min * 60 - t0) // bin_sec).astype(int) - 1
    node = df["node"].to_numpy()
    li = last[np.clip(b0, 0, None), node]
    age_min = (b0 - li) * bin_sec / 60
    ok = (b0 >= 0) & (li >= 0) & (age_min <= lookback_min)
    pred = np.where(ok, delay[np.clip(li, 0, None), node], 0.0)
    return pred, ok


def historical(train, df, slot_min):
    def feats(x):
        f = x[["station", "event_sec"]].copy()
        f["slot"] = ((f["event_sec"] % 86400) // (slot_min * 60)).astype(int)
        f["wk"] = (((f["event_sec"] // 86400).astype(int) + 3) % 7) >= 5  # Mon=0
        return f
    tr = feats(train)
    tr["delay_s"] = train["delay_s"].to_numpy()
    te = feats(df)
    p1 = tr.groupby(["station", "slot", "wk"])["delay_s"].mean().rename("p1")
    p2 = tr.groupby(["slot", "wk"])["delay_s"].mean().rename("p2")
    out = te.join(p1, on=["station", "slot", "wk"]).join(p2, on=["slot", "wk"])
    return out["p1"].fillna(out["p2"]).fillna(tr["delay_s"].mean()).to_numpy()


def metrics(y, p):
    if len(y) == 0:
        return {"n": 0}
    err = p - y
    yc, pc = np.digitize(y, BREAKS), np.digitize(p, BREAKS)
    f1 = []
    for c in (0, 1, 2):
        tp = ((yc == c) & (pc == c)).sum()
        fp = ((yc != c) & (pc == c)).sum()
        fn = ((yc == c) & (pc != c)).sum()
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return {"n": len(y), "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "f1_minor": f1[0], "f1_moderate": f1[1], "f1_major": f1[2], "f1_macro": float(np.mean(f1))}


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", default="processed")
    ap.add_argument("--out", default="tensors_out")
    ap.add_argument("--bin-min", type=int, default=2)
    ap.add_argument("--hist-bin-min", type=int, default=30, help="time-of-day slot for the historical baseline")
    ap.add_argument("--horizon-min", type=float, default=10)
    ap.add_argument("--lookback-min", type=float, default=30)
    ap.add_argument("--trip-lo", type=float, default=-600)
    ap.add_argument("--trip-hi", type=float, default=3600)
    ap.add_argument("--stop-lo", type=float, default=-1800)
    ap.add_argument("--stop-hi", type=float, default=7200)
    ap.add_argument("--seed", type=int, default=42, help="random seed for day-level train/val/test split")
    a = ap.parse_args()
    assert 1440 % a.bin_min == 0, "--bin-min must divide 1440"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ev = load_events(a.proc_dir, a.trip_lo, a.trip_hi, a.stop_lo, a.stop_hi)
    bin_sec, bpd = a.bin_min * 60, 1440 // a.bin_min
    t0 = (ev["event_sec"].min() // 86400) * 86400
    ev["didx"] = ((ev["event_sec"] - t0) // 86400).astype(int)

    # drop edge days with very few events (e.g. only a late-evening tail)
    per_day = ev.groupby("didx").size()
    good = per_day[per_day >= 0.2 * per_day.median()].index
    dropped = sorted(set(per_day.index) - set(good))
    if dropped:
        print("dropping sparse days (index from first day):", dropped)
    ev = ev[ev["didx"].isin(good)].copy()
    days = sorted(good)
    if len(days) < 3:
        raise SystemExit("Need at least 3 full days for a train/val/test split.")

    n_val = max(1, round(0.1 * len(days)))
    n_test = n_val
    n_train = len(days) - n_val - n_test
    names = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    rng = np.random.default_rng(a.seed)
    shuffled_days = rng.permutation(days).tolist()
    day2split = dict(zip(shuffled_days, names))
    ev["split"] = ev["didx"].map(day2split)

    split_days = {s: sorted(int(d) for d, n in day2split.items() if n == s) for s in ("train", "val", "test")}
    print(split_days)

    delay, mask, count, stations = build_tensor(ev, int(max(days)) + 1, bpd, bin_sec, t0)
    print(f"tensor {delay.shape} (time x stations), {mask.mean():.2%} of cells observed")
    np.savez_compressed(out / "tensors.npz", delay=delay, mask=mask, count=count,
                        stations=stations, t0=t0, bin_min=a.bin_min)
    ev.to_parquet(out / "events.parquet", index=False)
    with open(out / "splits.json", "w") as f:
        json.dump({"day_index_from_t0": split_days,
                   "t0_naive_local_epoch": float(t0), "bins_per_day": bpd, "seed": a.seed}, f, indent=1)

    last = last_observed_index(mask)
    train = ev[ev["split"] == "train"]
    rows, route_rows = [], []
    for split in ("val", "test"):
        df = ev[ev["split"] == split].reset_index(drop=True)
        y = df["delay_s"].to_numpy()
        pers, ok = persistence(df, last, delay, t0, bin_sec, a.horizon_min, a.lookback_min)
        preds = {"zero": np.zeros(len(df)), "persistence": pers,
                 "historical": historical(train, df, a.hist_bin_min)}
        print(f"{split}: {len(df)} events, persistence has an observation for {ok.mean():.1%}")
        hour = ((df["event_sec"].to_numpy() % 86400) // 3600).astype(int)
        subsets = {"all": np.ones(len(df), bool), "daytime_4-24h": hour >= 4, "overnight_0-4h": hour < 4}
        for name, p in preds.items():
            for sub, sel in subsets.items():
                rows.append({"baseline": name, "split": split, "subset": sub, **metrics(y[sel], p[sel])})
            if split == "test":
                for route, idx in df.groupby("route_id").indices.items():
                    if len(idx) >= 100:
                        route_rows.append({"baseline": name, "route": route, **metrics(y[idx], p[idx])})

    res = pd.DataFrame(rows)
    res.to_csv(out / "baseline_results.csv", index=False)
    pd.DataFrame(route_rows).to_csv(out / "baseline_by_route.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(f"\nWrote files to {out}/")


if __name__ == "__main__":
    main()