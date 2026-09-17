#!/usr/bin/env python3
"""
diagnose_daily -- find where detections disappear between the area-level CSV
and the per-day folders.

An empty daily mask means plot_fire_mask() received an empty frame. That can
happen at four different stages, and the PNG looks identical for all four.
This walks them in order and reports counts, so the answer is a number rather
than a guess.

Run:
    python diagnose_daily.py "C:\\GradResearch\\WildFireData\\data output\\datasets\\firms_palisades_bbox_..."

Point it at the directory containing the area CSV; it finds daily_samples/
underneath.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _import_pipeline():
    """Import the pipeline module whatever its version suffix is."""
    import importlib
    import re
    here = Path(__file__).resolve().parent
    found = sorted(
        (p.stem for p in here.glob("visualize_firms_dataset_v*.py")),
        key=lambda n: int(m.group(1)) if (m := re.search(r"_v(\d+)$", n)) else 0,
        reverse=True,
    )
    for name in found + ["visualize_firms_dataset"]:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as err:
            if err.name != name:
                raise
    raise ImportError(f"no visualize_firms_dataset*.py found in {here}")


def find_area_csv(root: Path) -> Path | None:
    """The area-level CSV, i.e. the one that is not inside daily_samples/."""
    candidates = [p for p in root.glob("*.csv")
                  if "daily_samples" not in p.parts
                  and not p.stem.endswith(("_centroids", "_clusters"))]
    return candidates[0] if candidates else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", help="Directory holding the area CSV")
    ap.add_argument("--confidence", default=None,
                    help="Replay a confidence filter, e.g. n,h")
    ap.add_argument("--min-frp", type=float, default=None)
    ap.add_argument("--daynight", default=None, choices=["D", "N"])
    args = ap.parse_args()

    root = Path(args.dataset_dir).expanduser()
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    V = _import_pipeline()

    # ---- stage 1: what actually got fetched -------------------------------
    csv = find_area_csv(root)
    if csv is None:
        print(f"No area-level CSV in {root}")
        print("  -> nothing was saved at all; the failure is upstream of the "
              "daily split (fetch, bbox subset, or the date-window guard).")
        return 1
    df = pd.read_csv(csv)
    print(f"1. Area CSV: {csv.name}")
    print(f"   rows: {len(df):,}")
    if df.empty:
        print("   -> EMPTY. Every daily mask will be empty. The problem is the "
              "fetch or the bbox subset, not the daily code.")
        return 1
    if "acq_date" in df.columns:
        print(f"   acq_date (UTC): {df['acq_date'].min()} .. {df['acq_date'].max()}")
    if "firms_source" in df.columns:
        print(f"   by source: {df['firms_source'].value_counts().to_dict()}")

    # ---- stage 2: the UTC -> local day shift ------------------------------
    d2 = V.add_acq_datetime(df)
    if "local_date" not in d2.columns:
        print("\n2. local_date could not be derived -- acq_date/acq_time missing "
              "or unparseable. The daily split groups on local_date, so every "
              "day would come out empty.")
        return 1
    shifted = (d2["local_date"] != d2["acq_date"].astype(str)).sum()
    print(f"\n2. Local-day conversion ({V.LOCAL_TZ})")
    print(f"   rows whose local_date differs from acq_date: {shifted:,} "
          f"({shifted / len(d2) * 100:.1f}%)")
    if shifted:
        moved = d2.loc[d2["local_date"] != d2["acq_date"].astype(str)]
        print(f"   those are overpasses at UTC times "
              f"{sorted(moved['acq_time'].unique())[:6]}")
        print("   NOTE: detections land in the local day, so the first "
              "requested day can legitimately lose its pre-dawn rows to the "
              "day before -- which is outside the requested range and so is "
              "never written.")
    print("   local_date distribution:")
    for day, n in d2["local_date"].value_counts().sort_index().items():
        print(f"     {day}  {n:,}")

    # ---- stage 3: filters -------------------------------------------------
    if any([args.confidence, args.min_frp, args.daynight]):
        conf = tuple(args.confidence.split(",")) if args.confidence else None
        filtered = V.apply_confidence_frp_daynight_filters(
            d2, confidence=conf, min_frp=args.min_frp, daynight=args.daynight)
        print(f"\n3. Filters (confidence={conf}, min_frp={args.min_frp}, "
              f"daynight={args.daynight})")
        print(f"   {len(d2):,} -> {len(filtered):,} rows "
              f"({len(d2) - len(filtered):,} removed)")
        if filtered.empty:
            print("   -> FILTERS REMOVED EVERYTHING. This is the cause.")
            return 1
        d2 = filtered
    else:
        print("\n3. Filters: not replayed (pass --confidence / --min-frp / "
              "--daynight to test the ones you used)")

    # ---- stage 4: what the daily folders actually contain -----------------
    daily = root / "daily_samples"
    print(f"\n4. Daily folders under {daily.name}/")
    if not daily.is_dir():
        print("   none -- daily samples were never written")
        return 1
    expected = set(d2["local_date"].dropna().astype(str))
    empty_days, ok_days = [], []
    for day_dir in sorted(p for p in daily.iterdir() if p.is_dir()):
        csvs = [p for p in day_dir.glob("*.csv")
                if not p.stem.endswith(("_centroids", "_clusters"))]
        n = 0
        if csvs:
            try:
                n = len(pd.read_csv(csvs[0]))
            except Exception:
                n = 0
        (ok_days if n else empty_days).append((day_dir.name, n))
        flag = "" if n else "   <-- EMPTY"
        print(f"   {day_dir.name}  {n:>6,} rows{flag}")

    print("\n" + "=" * 62)
    if not empty_days:
        print("Every day has rows. The empty PNG is from an earlier run -- "
              "check\nthe PNG's timestamp against the folder contents.")
        return 0

    for day, _ in empty_days:
        if day in expected:
            print(f"{day}: rows exist for this local date but the folder CSV is "
                  f"empty.\n  -> the frame was emptied between the area save and "
                  f"the daily split.")
        else:
            print(f"{day}: no detections on this local date at all.")
            print("  -> the day was requested but nothing was observed. An "
                  "empty mask is\n     CORRECT here, not a bug -- FIRMS "
                  "reports active fire, so this is a\n     coverage gap or a "
                  "quiet day, and it belongs in the unobserved\n     class "
                  "rather than no-fire (see firegrid.mark_unobserved).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
