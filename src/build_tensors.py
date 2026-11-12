#!/usr/bin/env python3
"""
Turn the per-day parquet files from preprocess.py into:
- tensors.npz        station x time delay tensor (mean delay per bin) + mask + counts
- events.parquet     filtered stop events with split / bin / node columns
- splits.json        which days went to train / val / test
- baseline_results.csv, baseline_by_route.csv   the three trivial baselines

Usage:
    python build_tensors.py --proc-dir processed --out tensors_out --seed 42
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BREAKS = [180, 900]  # <3 min minor, 3-15 min moderate, >=15 min major
READ_COLS = [
    "trip_uid",
    "route_id",
    "direction_id",
    "station",
    "event_sec",
    "delay_s",
    "stop_matched",
    "secondary_record",
]


# ------------------------------------------------------------------ helpers
def read_filtered_day(path, trip_lo, trip_hi, stop_lo, stop_hi):
    """Read only required columns for one day and apply stop/trip filters."""
    d = pd.read_parquet(path, columns=READ_COLS)
    m = d[d["stop_matched"] & ~d["secondary_record"]].copy()
    n_matched = len(m)
    if n_matched == 0:
        return m.iloc[0:0], 0, 0

    for c in ("delay_s", "event_sec"):
        m[c] = m[c].astype("float64")

    med = m.groupby("trip_uid")["delay_s"].transform("median")
    keep = med.between(trip_lo, trip_hi) & m["delay_s"].between(stop_lo, stop_hi)
    m = m.loc[keep, ["trip_uid", "route_id", "direction_id", "station", "event_sec", "delay_s"]].copy()

    for c in ("trip_uid", "route_id", "direction_id", "station"):
        m[c] = m[c].astype(str)
    return m, n_matched, len(m)


def last_observed_index(mask):
    T = mask.shape[0]
    idx = np.where(mask, np.arange(T, dtype=np.int32)[:, None], -1).astype(np.int32)
    return np.maximum.accumulate(idx, axis=0)


def persistence(df, last, delay, t0, bin_sec, horizon_min, lookback_min):
    b0 = ((df["event_sec"].to_numpy() - horizon_min * 60 - t0) // bin_sec).astype(int) - 1
    node = df["node"].to_numpy()
    li = last[np.clip(b0, 0, None), node]
    age_min = (b0 - li) * bin_sec / 60
    ok = (b0 >= 0) & (li >= 0) & (age_min <= lookback_min)
    pred = np.where(ok, delay[np.clip(li, 0, None), node], 0.0)
    return pred, ok


# ------------------------------------------------------- streaming metrics
def new_accum():
    return {
        "n": 0,
        "se": 0.0,
        "ae": 0.0,
        "tp": np.zeros(3, dtype=np.int64),
        "fp": np.zeros(3, dtype=np.int64),
        "fn": np.zeros(3, dtype=np.int64),
    }


def update_accum(acc, y, p):
    if len(y) == 0:
        return
    err = p - y
    acc["n"] += len(y)
    acc["se"] += float(np.sum(err ** 2))
    acc["ae"] += float(np.sum(np.abs(err)))
    yc, pc = np.digitize(y, BREAKS), np.digitize(p, BREAKS)
    for c in (0, 1, 2):
        acc["tp"][c] += int(((yc == c) & (pc == c)).sum())
        acc["fp"][c] += int(((yc != c) & (pc == c)).sum())
        acc["fn"][c] += int(((yc == c) & (pc != c)).sum())


def finalize_accum(acc):
    n = acc["n"]
    if n == 0:
        return {"n": 0}
    f1 = []
    for c in (0, 1, 2):
        tp, fp, fn = acc["tp"][c], acc["fp"][c], acc["fn"][c]
        f1.append(0.0 if tp == 0 else float(2 * tp / (2 * tp + fp + fn)))
    return {
        "n": n,
        "rmse": float(np.sqrt(acc["se"] / n)),
        "mae": float(acc["ae"] / n),
        "f1_minor": f1[0],
        "f1_moderate": f1[1],
        "f1_major": f1[2],
        "f1_macro": float(np.mean(f1)),
    }


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

    files = sorted(glob.glob(f"{a.proc_dir}/delays_*.parquet"))
    if not files:
        raise SystemExit(f"No delays_*.parquet in {a.proc_dir}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bin_sec, bpd = a.bin_min * 60, 1440 // a.bin_min

    # ---------------- PASS 1: scan per-day counts and unique stations ----------------
    day_counts = Counter()
    stations_set = set()
    total_matched = 0
    total_kept = 0

    for f in files:
        m, nm, nk = read_filtered_day(f, a.trip_lo, a.trip_hi, a.stop_lo, a.stop_hi)
        total_matched += nm
        total_kept += nk
        if nk == 0:
            continue
        stations_set.update(m["station"].unique())
        abs_days = (m["event_sec"].to_numpy() // 86400).astype(np.int64)
        uniq_d, cnt_d = np.unique(abs_days, return_counts=True)
        for d_val, c_val in zip(uniq_d.tolist(), cnt_d.tolist()):
            day_counts[d_val] += c_val

    if total_matched == 0:
        raise SystemExit("No matched stop events found.")
    print(f"{total_matched} matched stops, keeping {total_kept / total_matched:.2%} after filters")

    abs_day0 = min(day_counts.keys())
    t0 = float(abs_day0 * 86400)
    per_day = pd.Series({d - abs_day0: c for d, c in day_counts.items()}).sort_index()

    good = per_day[per_day >= 0.2 * per_day.median()].index
    dropped = sorted(set(per_day.index) - set(good))
    if dropped:
        print("dropping sparse days (index from first day):", dropped)
    days = sorted(int(d) for d in good)
    if len(days) < 3:
        raise SystemExit("Need at least 3 full days for a train/val/test split.")

    n_val = max(1, round(0.1 * len(days)))
    n_test = n_val
    n_train = len(days) - n_val - n_test
    names = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    rng = np.random.default_rng(a.seed)
    shuffled_days = rng.permutation(days).tolist()
    day2split = dict(zip(shuffled_days, names))
    split_days = {s: sorted(int(d) for d, n in day2split.items() if n == s) for s in ("train", "val", "test")}
    print(split_days)

    stations = np.array(sorted(stations_set))
    sidx = {s: i for i, s in enumerate(stations)}
    n_days = int(max(days)) + 1
    T, N = n_days * bpd, len(stations)

    # ---------------- PASS 2: build tensor + stream events.parquet + train hist ----------------
    delay_sum = np.zeros((T, N), dtype=np.float64)
    count = np.zeros((T, N), dtype=np.int32)

    hist_sum_1, hist_cnt_1 = Counter(), Counter()
    hist_sum_2, hist_cnt_2 = Counter(), Counter()
    train_total_sum = 0.0
    train_total_cnt = 0

    events_path = out / "events.parquet"
    writer = None

    for f in files:
        m, _, nk = read_filtered_day(f, a.trip_lo, a.trip_hi, a.stop_lo, a.stop_hi)
        if nk == 0:
            continue
        m["didx"] = ((m["event_sec"] - t0) // 86400).astype(int)
        m = m[m["didx"].isin(day2split)].copy()
        if len(m) == 0:
            continue

        m["split"] = m["didx"].map(day2split)
        m["node"] = m["station"].map(sidx).astype(int)
        m["bin"] = ((m["event_sec"] - t0) // bin_sec).astype(int)

        # Accumulate tensor cells
        b_arr = m["bin"].to_numpy()
        n_arr = m["node"].to_numpy()
        y_arr = m["delay_s"].to_numpy()
        np.add.at(delay_sum, (b_arr, n_arr), y_arr)
        np.add.at(count, (b_arr, n_arr), 1)

        # Accumulate historical baseline stats from training split
        tr = m[m["split"] == "train"]
        if len(tr) > 0:
            tr_sec = tr["event_sec"].to_numpy()
            tr_y = tr["delay_s"].to_numpy()
            slot = ((tr_sec % 86400) // (a.hist_bin_min * 60)).astype(int)
            wk = (((tr_sec // 86400).astype(int) + 3) % 7) >= 5
            df_tr_stats = pd.DataFrame({
                "station": tr["station"].to_numpy(),
                "slot": slot,
                "wk": wk,
                "delay_s": tr_y,
            })
            g1 = df_tr_stats.groupby(["station", "slot", "wk"])["delay_s"].agg(["sum", "count"])
            for k, row in g1.iterrows():
                hist_sum_1[k] += float(row["sum"])
                hist_cnt_1[k] += int(row["count"])

            g2 = df_tr_stats.groupby(["slot", "wk"])["delay_s"].agg(["sum", "count"])
            for k, row in g2.iterrows():
                hist_sum_2[k] += float(row["sum"])
                hist_cnt_2[k] += int(row["count"])

            train_total_sum += float(tr_y.sum())
            train_total_cnt += len(tr_y)

        # Stream filtered events to disk
        table = pa.Table.from_pandas(m, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(events_path, table.schema, compression="snappy")
        writer.write_table(table)

    if writer is not None:
        writer.close()

    mask = count > 0
    delay = np.where(mask, delay_sum / np.maximum(count, 1), 0.0).astype(np.float32)
    del delay_sum

    print(f"tensor {delay.shape} (time x stations), {mask.mean():.2%} of cells observed")
    np.savez_compressed(
        out / "tensors.npz",
        delay=delay,
        mask=mask,
        count=count,
        stations=stations,
        t0=t0,
        bin_min=a.bin_min,
    )
    with open(out / "splits.json", "w") as f:
        json.dump(
            {
                "day_index_from_t0": split_days,
                "t0_naive_local_epoch": float(t0),
                "bins_per_day": bpd,
                "seed": a.seed,
            },
            f,
            indent=1,
        )

    # ---------------- PASS 3: batched baseline evaluation ----------------
    last = last_observed_index(mask)
    p1_map = {k: hist_sum_1[k] / hist_cnt_1[k] for k in hist_sum_1}
    p2_map = {k: hist_sum_2[k] / hist_cnt_2[k] for k in hist_sum_2}
    global_train_mean = train_total_sum / max(1, train_total_cnt)

    accums = {
        (name, split, sub): new_accum()
        for name in ("zero", "persistence", "historical")
        for split in ("val", "test")
        for sub in ("all", "daytime_4-24h", "overnight_0-4h")
    }
    route_accums = {}
    split_counts = {"val": 0, "test": 0}
    pers_ok_counts = {"val": 0, "test": 0}

    pf = pq.ParquetFile(events_path)
    for batch in pf.iter_batches(
        batch_size=500_000,
        columns=["split", "route_id", "station", "node", "event_sec", "delay_s"],
    ):
        dfb = batch.to_pandas()
        for split in ("val", "test"):
            df = dfb[dfb["split"] == split].reset_index(drop=True)
            if len(df) == 0:
                continue

            y = df["delay_s"].to_numpy()
            pers, ok = persistence(df, last, delay, t0, bin_sec, a.horizon_min, a.lookback_min)

            sec = df["event_sec"].to_numpy()
            slot = ((sec % 86400) // (a.hist_bin_min * 60)).astype(int)
            wk = (((sec // 86400).astype(int) + 3) % 7) >= 5
            st_arr = df["station"].to_numpy()
            hist = np.array(
                [
                    p1_map.get((s, int(sl), bool(w)), p2_map.get((int(sl), bool(w)), global_train_mean))
                    for s, sl, w in zip(st_arr, slot, wk)
                ],
                dtype=np.float64,
            )

            preds = {"zero": np.zeros(len(df)), "persistence": pers, "historical": hist}
            split_counts[split] += len(df)
            pers_ok_counts[split] += int(ok.sum())

            hour = ((sec % 86400) // 3600).astype(int)
            subsets = {
                "all": np.ones(len(df), bool),
                "daytime_4-24h": hour >= 4,
                "overnight_0-4h": hour < 4,
            }

            for name, p in preds.items():
                for sub, sel in subsets.items():
                    update_accum(accums[(name, split, sub)], y[sel], p[sel])
                if split == "test":
                    for route, idx in df.groupby("route_id").indices.items():
                        key = (name, route)
                        if key not in route_accums:
                            route_accums[key] = new_accum()
                        update_accum(route_accums[key], y[idx], p[idx])

    for split in ("val", "test"):
        ok_rate = pers_ok_counts[split] / max(1, split_counts[split])
        print(f"{split}: {split_counts[split]} events, persistence has an observation for {ok_rate:.1%}")

    rows = [
        {"baseline": name, "split": split, "subset": sub, **finalize_accum(accums[(name, split, sub)])}
        for split in ("val", "test")
        for name in ("zero", "persistence", "historical")
        for sub in ("all", "daytime_4-24h", "overnight_0-4h")
    ]
    route_rows = [
        {"baseline": name, "route": route, **finalize_accum(st)}
        for (name, route), st in sorted(route_accums.items())
        if st["n"] >= 100
    ]

    res = pd.DataFrame(rows)
    res.to_csv(out / "baseline_results.csv", index=False)
    pd.DataFrame(route_rows).to_csv(out / "baseline_by_route.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(f"\nWrote files to {out}/")


if __name__ == "__main__":
    main()