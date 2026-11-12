#!/usr/bin/env python3
"""
Build one DataFrame per day that combines actual stop times (subwaydata.nyc)
with scheduled stop times (static GTFS), and save it as parquet.

Usage:
    python preprocess.py --rt-dir data/ --gtfs-dir gtfs_archive/ --out processed/
    python preprocess.py --rt-dir data/ --gtfs-dir gtfs_archive/ --out processed/ \
        --start 2024-01-01 --end 2024-01-31
"""
import argparse
import json
import os
import re
import sys
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

EPOCH = pd.Timestamp("1970-01-01")
KEY_RE = r"(?P<origin>\d{6})_(?P<route>[^._]+)\.{1,2}(?P<dir>[NS])(?P<path>[A-Za-z0-9]*)$"
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

TRIPS_DTYPES = {
    "trip_uid": str,
    "trip_id": str,
    "route_id": str,
    "direction_id": str,
}
ST_DTYPES = {
    "trip_uid": str,
    "stop_id": str,
    "track": str,
}
STRING_OUT_COLS = (
    "trip_uid",
    "stop_id",
    "track",
    "trip_id",
    "route_id",
    "direction_id",
    "static_trip_id",
    "station",
    "day",
)


# ----------------------------------------------------------------- helpers
def to_local_sec(s, mode):
    """Unix seconds -> naive local seconds."""
    s = pd.to_numeric(s, errors="coerce")
    if mode == "naive":
        return s.astype(float)
    dt = pd.to_datetime(s, unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return (dt - EPOCH).dt.total_seconds()


def hms_to_sec(s):
    p = s.astype("string").str.strip().str.split(":", expand=True)
    return (
        pd.to_numeric(p[0], errors="coerce") * 3600
        + pd.to_numeric(p[1], errors="coerce") * 60
        + pd.to_numeric(p[2], errors="coerce")
    )


def midnight_sec(d):
    return (pd.Timestamp(d) - EPOCH).total_seconds()


def is_fully_processed(out_path):
    """Return True only if out_path exists, is non-empty, and has a valid Parquet footer with rows."""
    if not out_path.exists() or out_path.stat().st_size == 0:
        return False
    try:
        pf = pq.ParquetFile(out_path)
        return pf.metadata is not None and pf.metadata.num_rows > 0
    except Exception:
        return False


# ------------------------------------------------------------ static GTFS
_cache = {}


def load_static(zip_path):
    zip_path = str(zip_path)
    if zip_path in _cache:
        return _cache[zip_path]
    _cache.clear()  # keep only one version in memory
    print(f"  loading static GTFS {Path(zip_path).name} ...")
    with zipfile.ZipFile(zip_path) as z:
        files = {Path(n).name: n for n in z.namelist()}

        def rd(name, **kw):
            with z.open(files[name]) as f:
                return pd.read_csv(f, **kw)

        trips = rd("trips.txt", usecols=["trip_id", "service_id", "route_id"], dtype=str)
        st = rd(
            "stop_times.txt",
            usecols=["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"],
            dtype={"trip_id": str, "stop_id": str, "arrival_time": str, "departure_time": str},
        )
        cal = rd("calendar.txt", dtype={"service_id": str}) if "calendar.txt" in files else None
        cald = rd("calendar_dates.txt", dtype={"service_id": str}) if "calendar_dates.txt" in files else None

    st["arr"] = hms_to_sec(st["arrival_time"])
    st["dep"] = hms_to_sec(st["departure_time"])
    st["dep"] = st["dep"].fillna(st["arr"])
    st["arr"] = st["arr"].fillna(st["dep"])
    st = st.sort_values(["trip_id", "stop_sequence"])
    origin_sec = st.groupby("trip_id")["dep"].first()

    trips = pd.concat([trips, trips["trip_id"].str.extract(KEY_RE)], axis=1)
    trips["route"] = trips["route_id"]
    trips["origin_sec_sched"] = trips["trip_id"].map(origin_sec)
    trips = trips.dropna(subset=["origin", "origin_sec_sched"])

    stops = st[["trip_id", "stop_id", "arr", "dep"]].drop_duplicates(["trip_id", "stop_id"])
    out = {"trips": trips, "stops": stops, "cal": cal, "cald": cald}
    _cache[zip_path] = out
    return out


def active_services(cal, cald, d):
    ymd = int(d.strftime("%Y%m%d"))
    active = set()
    if cal is not None:
        wd = d.strftime("%A").lower()
        m = (
            (cal["start_date"].astype(int) <= ymd)
            & (cal["end_date"].astype(int) >= ymd)
            & (cal[wd].astype(int) == 1)
        )
        active |= set(cal.loc[m, "service_id"])
    if cald is not None:
        x = cald[cald["date"].astype(int) == ymd]
        active |= set(x.loc[x["exception_type"].astype(int) == 1, "service_id"])
        active -= set(x.loc[x["exception_type"].astype(int) == 2, "service_id"])
    return active


# ------------------------------------------------------- timezone detection
def detect_tz(trips):
    """Compare start_time with the origin time encoded in trip_id."""
    code = pd.to_numeric(trips["trip_id"].str.extract(r"^(\d{6})_")[0], errors="coerce")
    expected = code * 0.6  # hundredths of a minute -> seconds
    scores = {}
    for mode in ("naive", "utc"):
        s = to_local_sec(trips["start_time"], mode)
        diff = ((s - expected + 43200) % 86400) - 43200
        scores[mode] = float((diff.abs() <= 120).mean())
    best = max(scores, key=scores.get)
    print(f"  timezone check (share of trips whose start_time matches trip_id): {scores}")
    if scores[best] < 0.5:
        print("  WARNING: neither interpretation matches well; check your timestamps.")
    return best


def detect_stop_tz(trips, st, trips_mode):
    """stop_times may use a different timezone convention than trips.start_time.
    Compare each trip's first departure with its start_time."""
    start = to_local_sec(trips.set_index("trip_uid")["start_time"], trips_mode)
    first = st.dropna(subset=["departure_time"]).drop_duplicates("trip_uid").set_index("trip_uid")
    scores = {}
    for m in ("naive", "utc"):
        dep = to_local_sec(first["departure_time"], m)
        scores[m] = float(((dep - start.reindex(first.index)).abs() <= 7200).mean())
    best = max(scores, key=scores.get)
    print(f"  stop_times timezone check (first departure within 2h of start_time): {scores}")
    if scores[best] < 0.5:
        print("  WARNING: stop_times timestamps match neither interpretation.")
    return best


# ------------------------------------------------------------- per-day work
def process_day(day, trips, st, static, mode, tol, st_mode):
    trips = trips.reset_index(drop=True)
    st = st.copy()

    trips["start_local"] = to_local_sec(trips["start_time"], mode)
    first_dep = (
        st.dropna(subset=["departure_time"]).drop_duplicates("trip_uid")
        .set_index("trip_uid")["departure_time"]
    )
    first_local = to_local_sec(first_dep, st_mode).reindex(trips["trip_uid"]).reset_index(drop=True)
    trips["day_shift"] = ((first_local - trips["start_local"]) / 86400).round().fillna(0)
    trips["start_local"] = trips["start_local"] + trips["day_shift"] * 86400
    trips = pd.concat([trips, trips["trip_id"].str.extract(KEY_RE)], axis=1)
    rt = trips.dropna(subset=["origin"])

    # 1. candidate static trips (day and previous service day)
    cands = []
    for sd in (day, day - timedelta(days=1)):
        svc = active_services(static["cal"], static["cald"], sd)
        s = static["trips"][static["trips"]["service_id"].isin(svc)]
        c = rt[["trip_uid", "origin", "route", "dir", "path", "start_local"]].merge(
            s, on=["route", "dir"], suffixes=("_rt", "_st")
        )
        c["service_midnight"] = midnight_sec(sd)
        c["resid"] = c["start_local"] - (c["service_midnight"] + c["origin_sec_sched"])
        cands.append(c[["trip_uid", "trip_id", "service_midnight", "resid", "path_rt", "path_st"]])
    cand = pd.concat(cands, ignore_index=True)
    cand = cand[cand["resid"].abs() <= tol].rename(columns={"trip_id": "static_trip_id"})
    cand["exact"] = cand["path_rt"] == cand["path_st"]

    # 2. pick the candidate sharing the most stops
    rt_stops = st[["trip_uid", "stop_id"]].drop_duplicates()
    st_stops = static["stops"][["trip_id", "stop_id"]].rename(columns={"trip_id": "static_trip_id"})
    pairs = cand[["trip_uid", "static_trip_id"]].drop_duplicates()
    ov = (
        pairs.merge(rt_stops, on="trip_uid")
        .merge(st_stops, on=["static_trip_id", "stop_id"])
        .groupby(["trip_uid", "static_trip_id"])
        .size()
        .rename("overlap")
        .reset_index()
    )
    cand = cand.merge(ov, on=["trip_uid", "static_trip_id"], how="left")
    cand["overlap"] = cand["overlap"].fillna(0)
    cand["abs_resid"] = cand["resid"].abs()
    best = (
        cand.sort_values(["overlap", "abs_resid", "exact"], ascending=[False, True, False])
        .drop_duplicates("trip_uid")
    )[["trip_uid", "static_trip_id", "service_midnight", "resid", "overlap"]]

    # 3. join stops
    st = st[["trip_uid", "stop_id", "track", "arrival_time", "departure_time", "last_observed"]].copy()
    st["arr_sec"] = to_local_sec(st["arrival_time"], st_mode)
    st["dep_sec"] = to_local_sec(st["departure_time"], st_mode)
    st = st.drop(columns=["arrival_time", "departure_time"])

    meta_cols = ["trip_uid", "trip_id", "route_id", "direction_id", "start_local", "day_shift"]
    for extra in ("num_schedule_changes", "num_schedule_rewrites"):
        if extra in trips.columns:
            meta_cols.append(extra)
    df = st.merge(trips[meta_cols], on="trip_uid", how="left")
    df = df.merge(best, on="trip_uid", how="left")
    sched = static["stops"].rename(
        columns={"trip_id": "static_trip_id", "arr": "sched_arr_rel", "dep": "sched_dep_rel"}
    )
    df = df.merge(sched, on=["static_trip_id", "stop_id"], how="left")

    df["sched_arr_sec"] = df["service_midnight"] + df["sched_arr_rel"]
    df["sched_dep_sec"] = df["service_midnight"] + df["sched_dep_rel"]
    df = df.drop(columns=["sched_arr_rel", "sched_dep_rel"])

    # 4. delays (origin stops have no arrival, so fall back to departure)
    has_arr = df["arr_sec"].notna()
    actual = df["arr_sec"].where(has_arr, df["dep_sec"])
    planned = df["sched_arr_sec"].where(has_arr, df["sched_dep_sec"])
    df["event_sec"] = actual
    df["delay_s"] = actual - planned
    df["dep_delay_s"] = df["dep_sec"] - df["sched_dep_sec"]
    df["trip_matched"] = df["static_trip_id"].notna()
    df["stop_matched"] = df["delay_s"].notna()

    df["station"] = df["stop_id"].str.replace(r"[NS]$", "", regex=True)
    df = df.sort_values(["trip_uid", "event_sec"], kind="stable")
    df["stop_idx"] = df.groupby("trip_uid").cumcount()
    df["event_time"] = EPOCH + pd.to_timedelta(df["event_sec"], unit="s")
    df["day"] = day.isoformat()

    run = df["trip_uid"].str.split("_").str[0] + "|" + df["route_id"].fillna("") + "|" + df["direction_id"].fillna("")
    rank = pd.DataFrame({
        "run": run, "trip_uid": df["trip_uid"],
        "n": df.groupby("trip_uid")["stop_id"].transform("size"),
        "L": df["trip_uid"].str.len(),
    }).drop_duplicates("trip_uid")
    primary = set(rank.sort_values(["n", "L"], ascending=False).drop_duplicates("run")["trip_uid"])
    df["secondary_record"] = ~df["trip_uid"].isin(primary)

    # Enforce uniform nullable string schema across all text/ID columns for PyArrow
    for col in STRING_OUT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    n_trips = df["trip_uid"].nunique()
    stats = {
        "day": day.isoformat(),
        "trips": n_trips,
        "trip_match_rate": df.groupby("trip_uid")["trip_matched"].first().mean(),
        "stop_match_rate": df["stop_matched"].mean(),
        "secondary_trips": int(df.loc[df["secondary_record"], "trip_uid"].nunique()),
        "day_shifted_trips": int((df.groupby("trip_uid")["day_shift"].first() != 0).sum()),
        "delay_p01": df["delay_s"].quantile(0.01),
        "delay_median": df["delay_s"].median(),
        "delay_p99": df["delay_s"].quantile(0.99),
    }
    return df, stats


# -------------------------------------------------------------------- main
def find_days(rt_dir):
    days = set()
    for p in rt_dir.rglob("*"):
        if p.is_file() and "trips" in p.name and "stop_times" not in p.name:
            m = DATE_RE.search(p.name)
            if m:
                days.add(datetime.fromisoformat(m.group()).date())
    return sorted(days)


def find_file(rt_dir, day, kind):
    ds = day.isoformat()
    hits = [
        p for p in rt_dir.rglob("*")
        if p.is_file() and ds in p.name and kind in p.name
        and (kind == "stop_times" or "stop_times" not in p.name)
    ]
    return sorted(hits)[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt-dir", required=True)
    ap.add_argument("--gtfs-dir", required=True)
    ap.add_argument("--out", default="processed")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--tz", choices=["auto", "naive", "utc"], default="auto")
    ap.add_argument("--st-tz", choices=["auto", "naive", "utc"], default="auto",
                    help="timezone convention of stop_times timestamps (trips use --tz)")
    ap.add_argument("--tol", type=float, default=300,
                    help="max |start_time - scheduled origin| in s for a trip to count as matched")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rt_dir, gtfs_dir, out_dir = Path(args.rt_dir), Path(args.gtfs_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(gtfs_dir / "version_for_day.json") as f:
        version_for_day = json.load(f)

    days = find_days(rt_dir)
    if args.start:
        days = [d for d in days if d >= date.fromisoformat(args.start)]
    if args.end:
        days = [d for d in days if d <= date.fromisoformat(args.end)]
    if not days:
        sys.exit("No days found. Check --rt-dir and the file names.")
    print(f"{len(days)} days to process.")

    mode = args.tz
    st_mode = args.st_tz
    all_stats = []
    for day in days:
        out_path = out_dir / f"delays_{day}.parquet"
        if not args.overwrite and is_fully_processed(out_path):
            print(f"{day}: already processed, skipping")
            continue
        if out_path.exists() and not args.overwrite:
            print(f"{day}: found incomplete/corrupt parquet, reprocessing")

        tp, sp = find_file(rt_dir, day, "trips"), find_file(rt_dir, day, "stop_times")
        zname = version_for_day.get(day.isoformat())
        if not (tp and sp):
            print(f"{day}: missing trips or stop_times file, skipping")
            continue
        if not zname:
            print(f"{day}: no static GTFS version mapped, skipping")
            continue

        print(f"{day}:")
        trips_df = pd.read_csv(tp, dtype=TRIPS_DTYPES).drop_duplicates("trip_uid")
        st_df = pd.read_csv(sp, dtype=ST_DTYPES)

        if mode == "auto":
            mode = detect_tz(trips_df)
            print(f"  using tz mode '{mode}' for all days")
        static = load_static(gtfs_dir / zname)
        if st_mode == "auto":
            st_mode = detect_stop_tz(trips_df, st_df, mode)
            print(f"  using stop_times tz mode '{st_mode}' for all days")

        df, stats = process_day(day, trips_df, st_df, static, mode, args.tol, st_mode)

        # Atomic write so interrupted runs never leave partial files
        tmp_path = out_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, out_path)

        all_stats.append(stats)
        print(
            f"  {stats['trips']} trips, trip match {stats['trip_match_rate']:.1%}, "
            f"stop match {stats['stop_match_rate']:.1%}, "
            f"delay p01/median/p99 = {stats['delay_p01']:.0f}/{stats['delay_median']:.0f}/{stats['delay_p99']:.0f} s"
        )

    if all_stats:
        new_summary = pd.DataFrame(all_stats)
        summary_path = out_dir / "summary.csv"
        if summary_path.exists() and not args.overwrite:
            old_summary = pd.read_csv(summary_path)
            old_summary = old_summary[~old_summary["day"].isin(new_summary["day"])]
            summary = pd.concat([old_summary, new_summary], ignore_index=True).sort_values("day")
        else:
            summary = new_summary.sort_values("day")
        summary.to_csv(summary_path, index=False)
        print(f"\nWrote summary.csv ({len(summary)} total days). Lowest trip match rate: {summary['trip_match_rate'].min():.1%}")


if __name__ == "__main__":
    main()