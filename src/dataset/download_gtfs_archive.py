#!/usr/bin/env python3
"""
Download historical MTA subway static GTFS versions from Transitland.

Usage:
    export TRANSITLAND_API_KEY=...        # free key from transit.land (account -> API keys)
    python download_gtfs_archive.py --start 2024-01-01 --end 2024-06-30 --out gtfs_archive

What it does:
  1. Lists every archived version of feed f-dr5r-nyctsubway.
  2. Keeps versions whose calendar range overlaps [start, end].
  3. Downloads them (skips files that already exist).
  4. Checks each zip has the files you need and prints its calendar range.
  5. Writes version_for_day.json: for each day, the version that was in effect
     (newest version fetched on or before that day whose calendar covers it).

NOTE: written from memory of the Transitland REST API v2 and not tested against
the live service. If a field name differs, run with --debug to print the raw JSON.
"""
import argparse
import json
import os
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

BASE = "https://transit.land/api/v2/rest"
FEED = "f-dr5r-nyctsubway"
REQUIRED = {"stops.txt", "trips.txt", "stop_times.txt", "calendar.txt"}


def parse_date(s):
    if not s:
        return None
    return datetime.fromisoformat(s[:10]).date()


def api_get(url, key, params=None, stream=False, retries=3):
    for attempt in range(retries):
        r = requests.get(
            url,
            params=params,
            headers={"apikey": key},
            stream=stream,
            timeout=120,
        )
        if r.status_code == 429:  # rate limited
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()


def list_versions(key, debug=False):
    versions, after = [], None
    while True:
        params = {"feed_onestop_id": FEED, "limit": 100}
        if after:
            params["after"] = after
        data = api_get(f"{BASE}/feed_versions", key, params).json()
        if debug and not versions:
            print(json.dumps(data, indent=2)[:3000])
        batch = data.get("feed_versions", [])
        versions.extend(batch)
        after = data.get("meta", {}).get("after")
        if not batch or not after:
            break
    return versions


def normalise(v):
    return {
        "sha1": v["sha1"],
        "fetched": parse_date(v.get("fetched_at")),
        "start": parse_date(v.get("earliest_calendar_date")),
        "end": parse_date(v.get("latest_calendar_date")),
    }


def download(v, key, out_dir):
    name = f"{v['fetched']}_{v['sha1'][:8]}.zip"
    path = out_dir / name
    if path.exists() and path.stat().st_size > 0:
        print(f"  have {name}")
        return path
    print(f"  downloading {name} ...")
    r = api_get(f"{BASE}/feed_versions/{v['sha1']}/download", key, stream=True)
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    tmp.rename(path)
    return path


def check_zip(path):
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        missing = REQUIRED - names
        info = f"calendar_dates.txt={'yes' if 'calendar_dates.txt' in names else 'no'}"
        if missing:
            return f"MISSING {sorted(missing)}"
        return "ok (" + info + ")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="first day of your data, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="last day of your data, YYYY-MM-DD")
    ap.add_argument("--out", default="gtfs_archive")
    ap.add_argument("--api-key", default=os.environ.get("TRANSITLAND_API_KEY"))
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--list-only", action="store_true", help="list versions, download nothing")
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("Set TRANSITLAND_API_KEY or pass --api-key (free key at transit.land).")

    start, end = parse_date(args.start), parse_date(args.end)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw = list_versions(args.api_key, args.debug)
    except requests.HTTPError as e:
        sys.exit(
            f"Listing failed: {e}\n"
            "Check the API key, or download manually from "
            "https://www.transit.land/feeds/f-dr5r-nyctsubway (Versions section)."
        )

    versions = [normalise(v) for v in raw]
    versions = [v for v in versions if v["fetched"]]
    versions.sort(key=lambda v: v["fetched"])
    print(f"{len(versions)} archived versions found.")

    relevant = [
        v for v in versions
        if v["start"] and v["end"] and v["start"] <= end and v["end"] >= start
    ]
    print(f"{len(relevant)} overlap {start} .. {end}:")
    for v in relevant:
        print(f"  fetched {v['fetched']}  covers {v['start']} .. {v['end']}  {v['sha1'][:8]}")

    if not relevant:
        sys.exit("No overlapping versions. Run with --debug --list-only to inspect the raw data.")
    if args.list_only:
        return

    paths = {}
    for v in relevant:
        try:
            p = download(v, args.api_key, out_dir)
            paths[v["sha1"]] = p
            print(f"    {p.name}: {check_zip(p)}")
        except requests.HTTPError as e:
            print(f"    FAILED {v['sha1'][:8]}: {e}")

    # Version in effect per day: newest version fetched on/before the day whose calendar covers it.
    mapping, uncovered = {}, []
    day = start
    while day <= end:
        cands = [
            v for v in relevant
            if v["fetched"] <= day and v["start"] <= day <= v["end"] and v["sha1"] in paths
        ]
        if cands:
            mapping[day.isoformat()] = paths[max(cands, key=lambda v: v["fetched"])["sha1"]].name
        else:
            uncovered.append(day)
        day += timedelta(days=1)

    with open(out_dir / "version_for_day.json", "w") as f:
        json.dump(mapping, f, indent=1)
    print(f"\nMapped {len(mapping)} days to a version.")
    if uncovered:
        print(f"{len(uncovered)} days have no covering version, e.g. {uncovered[:5]}")
        print("For those, try versions fetched slightly AFTER the day, or use the run-time delay fallback.")


if __name__ == "__main__":
    main()