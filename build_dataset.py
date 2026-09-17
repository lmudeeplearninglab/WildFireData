#!/usr/bin/env python3
"""
build_dataset -- fire name in, training tensors out.

One command for the whole extraction. Resolves a fire by name, fetches the
FIRMS detections over its grid tile, pulls every Earth Engine feature layer
onto that same tile, and writes one (18, 64, 64) sample per fire-day.

    python build_dataset.py --fire "Palisades" --fire "Eaton" --year 2025

No bounding box anywhere. The tile comes from the CAL FIRE perimeter, and
every layer lands on it by construction.

WHY THIS IS A DRIVER AND NOT ONE BIG FILE
-----------------------------------------
It composes three modules that each own one decision, and keeping them apart
is what makes the pipeline auditable:

    firelookup   where and when          (CAL FIRE: perimeter, dates)
    <pipeline>   what burned             (FIRMS: detections, filters, dedup)
    firegrid     what shape the data is  (projection, resampling, labels)

A methods question about resampling has exactly one file to open. This file
only sequences them and handles failure.

STAGES
------
    1  resolve the fire            -> tile, date window
    2  fetch FIRMS over the tile   -> detections
    3  static EE layers, once      -> terrain, land cover, population
    4  per day: dynamic layers     -> weather, drought, vegetation
    5  per day: label              -> fire / no-fire / unobserved
    6  write .npz + manifest

The lead-in days are fetched but NOT written as samples: they exist so that
day t has a real prev_fire_mask from day t-1 rather than an assumed-empty one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import firegrid as F
import firelookup as L

V = F.V          # the pipeline module, resolved by version suffix

STATIC_LAYERS = ("elevation", "slope", "aspect", "landcover", "population")
DYNAMIC_LAYERS = ("vegetation", "drought", "humidity", "weather_temp",
                  "weather_precip", "wind_speed", "wind_u", "wind_v",
                  "erc", "vpd")


# ---------------------------------------------------------------------------
def daterange(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d")
            for d in pd.date_range(start=start, end=end, freq="D")]


def fetch_static_layers(tile, spec, ee_project, date_str) -> dict[str, np.ndarray]:
    """Terrain, land cover and population. Fetched once per tile, not per day.

    Five of the eighteen channels never change between days. Re-fetching them
    for every day of every fire would be most of the Earth Engine traffic in
    this pipeline and would return identical arrays each time.
    """
    out: dict[str, np.ndarray] = {}
    for layer in STATIC_LAYERS:
        arr = F.ee_layer_on_grid(layer, tile, spec, date_str=date_str,
                                 ee_project=ee_project)
        if arr is None:
            print(f"    {layer}: unavailable")
            continue
        if layer == "aspect":
            sin_u, cos_u, r = F.split_circular_components(arr)
            out["aspect_sin"], out["aspect_cos"] = sin_u, cos_u
            out["aspect_consistency"] = r
        else:
            out[layer] = arr
    return out


def fetch_dynamic_layers(tile, spec, ee_project, day: str) -> dict[str, np.ndarray]:
    """Weather, drought and vegetation for one local day."""
    out: dict[str, np.ndarray] = {}
    for layer in DYNAMIC_LAYERS:
        arr = F.ee_layer_on_grid(layer, tile, spec, date_str=day,
                                 time_mode="daily", ee_project=ee_project)
        if arr is not None:
            out[layer] = arr
    return out


# ---------------------------------------------------------------------------
def build_fire(record: L.FireRecord, args, map_key: str) -> dict:
    """Build every sample for one fire. Returns a summary dict."""
    spec = F.GRID_SPEC
    tile = F.snap_tile(record.centroid[0], record.centroid[1], spec)
    envelope = F.tile_bounds_lonlat(tile, spec)
    start, end = record.date_window(args.lead_in_days, args.tail_days)

    print(f"\n{'=' * 70}\n{record.name} ({record.year})\n{'=' * 70}")
    print(f"  perimeter   {record._bbox_str()}")
    print(f"  tile        origin ({tile[0]:.0f}, {tile[1]:.0f}), "
          f"{spec.tile_cells}x{spec.tile_cells} @ {spec.cell_m:.0f} m")
    print(f"  fetching    {start} .. {end}")

    # --- stage 2: FIRMS over the WHOLE tile ------------------------------
    # The fetch envelope is the tile, not the perimeter. Anything narrower
    # leaves cells that were never queried, and those become no-fire in the
    # label -- indistinguishable from ground that was observed and did not
    # burn.
    df = V.fetch_firms_data(map_key=map_key, bbox=envelope,
                            start_date=start, end_date=end,
                            use_sample_fallback=False)
    if df.empty:
        print("  no detections in this window; skipping")
        return {"fire": record.name, "samples": 0, "reason": "no detections"}

    df = V.add_acq_datetime(df)
    if not all(c in df.columns for c in V.FOOTPRINT_COLUMNS):
        df = V.add_footprint_columns(df)
    if args.confidence or args.min_frp:
        before = len(df)
        df = V.apply_confidence_frp_daynight_filters(
            df,
            confidence=tuple(args.confidence.split(",")) if args.confidence else None,
            min_frp=args.min_frp,
        )
        print(f"  filters     {before:,} -> {len(df):,} detections")
    print(f"  detections  {len(df):,}")

    # --- stage 3: static layers, once ------------------------------------
    out_dir = Path(args.out) / _slug(record)
    print("  static layers...")
    static = {} if args.no_ee else fetch_static_layers(
        tile, spec, args.ee_project, start)
    if static:
        print(f"    got {', '.join(sorted(static))}")

    # --- stages 4-6: one sample per labelled day -------------------------
    all_days = daterange(start, end)
    label_days = all_days[args.lead_in_days:]   # lead-in builds prev only
    prev_mask = np.zeros(spec.shape, dtype=np.uint8)
    carried = 0            # consecutive days prev_mask has been held over
    written, manifest = 0, []

    for day in all_days:
        day_df = df[df["local_date"].astype(str) == day] if "local_date" in df else df.iloc[0:0]
        fire_mask = F.rasterize_fire_mask(day_df, tile, spec)
        label = F.mark_unobserved(fire_mask, prev_mask,
                                  day_had_detections=not day_df.empty, spec=spec)

        if day in label_days:
            features = dict(static)
            features["prev_fire_mask"] = prev_mask.astype(float)
            if not args.no_ee:
                features.update(fetch_dynamic_layers(tile, spec, args.ee_project, day))

            stack, lab = F.assemble_sample(features, label, spec)
            present = [c for c in F.CHANNELS
                       if c in features and np.isfinite(features[c]).any()]
            meta = {
                "fire": record.name, "year": record.year, "date": day,
                "crs": spec.crs, "cell_m": spec.cell_m,
                "tile": list(tile), "tile_lonlat": list(envelope),
                "detections": int(len(day_df)),
                "channels_present": present,
                "source": record.source,
                "exact_footprint": record.exact_footprint,
            }
            path = F.save_sample(out_dir, f"{_slug(record)}_{day}", stack, lab, meta)
            written += 1
            manifest.append({
                "path": str(path.name), "date": day,
                "detections": int(len(day_df)),
                "fire_cells": int((lab == F.LABEL_FIRE).sum()),
                "unobserved_cells": int((lab == F.LABEL_UNOBSERVED).sum()),
                "channels_present": len(present),
            })
            flag = "" if present else "   <-- NO FEATURES"
            print(f"    {day}  {len(day_df):>5,} det  "
                  f"{int((lab == F.LABEL_FIRE).sum()):>5} fire cells  "
                  f"{len(present):>2}/{len(F.CHANNELS)} channels{flag}")
        else:
            print(f"    {day}  {len(day_df):>5,} det  (lead-in, not written)")

        # Carry prev_fire_mask across a coverage gap instead of zeroing it.
        # A day with no detections anywhere is usually cloud, smoke or a
        # missed overpass; resetting prev_mask to empty would tell the next
        # day's sample that nothing was burning, which is the same "learn the
        # satellite, not the fire" error the unobserved class guards against.
        # Bounded by --max-carry-days so a fire that genuinely ended does not
        # propagate forever.
        if day_df.empty and prev_mask.any() and carried < args.max_carry_days:
            carried += 1
        else:
            prev_mask = (fire_mask > 0).astype(np.uint8)
            carried = 0

    if manifest:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"fire": record.name, "samples": written, "dir": str(out_dir)}


def _slug(record: L.FireRecord) -> str:
    name = "".join(c if c.isalnum() else "_" for c in record.name.lower())
    return f"{name}_{record.year or 'unknown'}".strip("_")



# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------
def _prompt(msg: str, default: str = "") -> str:
    """Same prompt style as the FIRMS pipeline, so the two feel like one tool.

    Routed through the pipeline's _prompt rather than input() so that the run
    transcript records both the question and the answer.
    """
    return V._prompt(msg, default)


def _pick_fire(name: str, year: int | None) -> L.FireRecord | None:
    """Look a fire up and, when the name is ambiguous, let the user choose.

    Resolving here rather than at build time is the point of the interface:
    you see the real perimeter, acreage and dates before committing to a
    fetch, instead of discovering afterwards that you picked the wrong
    Palisades.
    """
    print(f"  searching CAL FIRE for '{name}'...")
    hits = L.resolve_fire(name, year)
    if not hits:
        print(f"  no match for '{name}'"
              + (f" in {year}" if year else "")
              + ". FRAP is republished annually, so a very recent fire may "
                "not be in it yet.")
        return None
    if len(hits) == 1:
        rec = hits[0]
        print(f"  found: {rec.describe().splitlines()[0]}")
        if not rec.exact_footprint:
            print("  NOTE: no perimeter published; extent inferred from acreage.")
        return rec
    print(f"  {len(hits)} matches:")
    for i, rec in enumerate(hits):
        print(f"    {i} = {rec.describe().splitlines()[0]}")
    choice = _prompt("Which one? (index)", "0")
    try:
        return hits[int(choice)]
    except (ValueError, IndexError):
        print("  not a valid index; using the largest match")
        return hits[0]


def run_interactive() -> argparse.Namespace:
    """Terminal UI mirroring the FIRMS pipeline's interactive mode."""
    args = argparse.Namespace()

    print("\n" + "=" * 60)
    print("  Wildfire Training Dataset – Interactive Mode")
    print("  (Press Enter to use default where shown)")
    print("=" * 60 + "\n")

    # 1. Fires, by name
    print("1. FIRES (by name – no bounding boxes needed)")
    year_raw = _prompt("Year (blank = any)", "2025")
    args.year = int(year_raw) if year_raw.strip().isdigit() else None

    records: list[L.FireRecord] = []
    while True:
        name = _prompt("Fire name (blank when done)" if records else "Fire name",
                       "" if records else "Palisades")
        if not name:
            break
        rec = _pick_fire(name, args.year)
        if rec and rec.centroid:
            records.append(rec)
        elif rec:
            print("  that record has no geometry; skipping")
        if records and not _prompt("Add another fire? (y/n)", "n").lower().startswith("y"):
            break
    if not records:
        print("\nNo fires selected; nothing to build.")
        sys.exit(1)
    args._records = records
    args.fire = [r.name for r in records]

    # 2. Date window
    print("\n2. DATE WINDOW (relative to each fire's own alarm/containment dates)")
    print("   The lead-in is fetched but NOT written as samples: day t needs")
    print("   day t-1 to build prev_fire_mask from.")
    args.lead_in_days = int(_prompt("Lead-in days before ignition", "2") or 2)
    args.tail_days = int(_prompt("Tail days after containment", "3") or 3)
    args.max_carry_days = int(_prompt(
        "Blank days that keep the previous fire mask alive", "2") or 2)

    # 3. Feature layers
    print("\n3. FEATURE LAYERS")
    print("   1 = All Earth Engine layers (18 channels)")
    print("   2 = Labels only, no Earth Engine (fast; spends no EE quota)")
    args.no_ee = _prompt("Choice", "1").strip() == "2"
    args.ee_project = None
    if not args.no_ee:
        args.ee_project = _prompt("EE project ID",
                                  V.resolve_ee_project()).strip() or None

    # 4. Detection filters
    print("\n4. DETECTION FILTERS (optional)")
    conf = _prompt("Confidence classes (e.g. n,h; blank = all)", "").strip()
    args.confidence = conf or None
    frp = _prompt("Minimum FRP (blank = none)", "").strip()
    try:
        args.min_frp = float(frp) if frp else None
    except ValueError:
        print("  not a number; ignoring FRP filter")
        args.min_frp = None

    # 5. Output
    print("\n5. OUTPUT")
    args.out = _prompt("Output directory", "ml_dataset")

    # 6. Plan, then confirm
    print("\n" + "=" * 60)
    print("  PLAN")
    print("=" * 60)
    total = 0
    for rec in records:
        start, end = rec.date_window(args.lead_in_days, args.tail_days)
        n = len(daterange(start, end)) - args.lead_in_days
        total += max(n, 0)
        tile = F.snap_tile(rec.centroid[0], rec.centroid[1])
        print(f"  {rec.name} ({rec.year}): {n} samples, {start} .. {end}")
        print(f"      tile origin ({tile[0]:.0f}, {tile[1]:.0f})  "
              f"{F.GRID_SPEC.tile_cells}x{F.GRID_SPEC.tile_cells} @ "
              f"{F.GRID_SPEC.cell_m:.0f} m  {F.GRID_SPEC.crs}")
    # The stack is always len(CHANNELS) deep; missing layers are written as
    # fill rather than dropped, so every sample has the same shape. Saying
    # "(1, 64, 64)" for labels-only mode would be a lie about the file.
    print(f"\n  {total} samples of ({len(F.CHANNELS)}, {F.GRID_SPEC.tile_cells}, "
          f"{F.GRID_SPEC.tile_cells}) -> {args.out}")
    if args.no_ee:
        print("  labels only: prev_fire_mask populated, the other "
              f"{len(F.CHANNELS) - 1} channels written as fill")
    if not args.no_ee:
        # Static layers are fetched once per fire, not once per day.
        calls = len(records) * len(STATIC_LAYERS) + total * len(DYNAMIC_LAYERS)
        print(f"  ~{calls:,} Earth Engine requests "
              f"(~{calls * 2.1 / 60:.0f} min at the measured 2.1 s each)")
    print("=" * 60)

    choice = _prompt("Proceed? (y = build, n = cancel, d = dry run)", "y").lower()
    if choice.startswith("n"):
        sys.exit(0)
    args.dry_run = choice.startswith("d")
    return args


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fire", action="append", default=[], metavar="NAME",
                    help="Fire name; repeat for several")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--out", default="ml_dataset")
    ap.add_argument("--lead-in-days", type=int, default=2,
                    help="Days fetched before ignition to seed prev_fire_mask. "
                         "Not written as samples (default 2)")
    ap.add_argument("--tail-days", type=int, default=3)
    ap.add_argument("--max-carry-days", type=int, default=2,
                    help="How many consecutive blank days keep the previous "
                         "fire mask alive before it is treated as burnt out "
                         "(default 2)")
    ap.add_argument("--confidence", default=None, help="e.g. n,h")
    ap.add_argument("--min-frp", type=float, default=None)
    ap.add_argument("--ee-project", default=None)
    ap.add_argument("--no-ee", action="store_true",
                    help="Labels only, no feature layers. Useful for checking "
                         "the label pipeline without spending EE quota.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve fires and print the plan; fetch nothing")
    ap.add_argument("--report-file", default=None, metavar="PATH",
                    help="Where to write the run transcript. Default: "
                         "build_report.txt, moved into --out when the run ends.")
    ap.add_argument("--report-append", action="store_true",
                    help="Append to the transcript instead of overwriting it")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="Prompt for everything instead of using flags. This "
                         "is the default when no --fire is given and you are "
                         "at a terminal.")
    args = ap.parse_args()

    # Interactive unless fires were named. Requires a terminal: prompting a
    # piped or scheduled run would hang it.
    report_file, report_append = args.report_file, args.report_append
    if args.interactive or (not args.fire and sys.stdin.isatty()):
        args = run_interactive()
        args.report_file, args.report_append = report_file, report_append
    elif not args.fire:
        ap.error("give at least one --fire NAME, or run with -i")

    # Start the transcript BEFORE the prompts. The interactive answers are
    # part of what you need when debugging a run later ("which fire did I
    # pick, what lead-in did I use"), and they happen before args.out exists,
    # so the file starts in the working directory and is moved next to the
    # dataset at the end.
    staging = Path(args.report_file) if args.report_file \
        else Path.cwd() / "build_report.txt"
    with V.RunTranscript(staging, append=args.report_append):
        print(f"Grid: {F.GRID_SPEC.crs}, {F.GRID_SPEC.cell_m:.0f} m, "
              f"{F.GRID_SPEC.tile_cells}x{F.GRID_SPEC.tile_cells}, "
              f"{len(F.CHANNELS)} channels")

        # --- stage 1: resolve every fire before fetching anything --------
        records = list(getattr(args, "_records", []))
        for name in (args.fire if not records else []):
            hits = L.resolve_fire(name, args.year)
            if not hits:
                print(f"  '{name}': no match, skipping")
                continue
            rec = hits[0]
            if not rec.centroid:
                print(f"  '{name}': no geometry, skipping")
                continue
            records.append(rec)
            print(f"  {rec.describe().splitlines()[0]}")
        if not records:
            print("Nothing to build.")
            return 1

        if args.dry_run:
            for rec in records:
                s, e = rec.date_window(args.lead_in_days, args.tail_days)
                days = len(daterange(s, e)) - args.lead_in_days
                print(f"\n{rec.name}: {days} samples, {s}..{e}")
            return 0

        map_key = V.get_map_key()
        t0 = time.time()
        summaries = [build_fire(rec, args, map_key) for rec in records]

        print(f"\n{'=' * 70}")
        total = sum(s["samples"] for s in summaries)
        for s in summaries:
            print(f"  {s['fire']:<20} {s['samples']:>4} samples"
                  + (f"   ({s['reason']})" if s.get("reason") else ""))
        print(f"  {'TOTAL':<20} {total:>4} samples in {time.time() - t0:.0f}s")
        print(f"  written to {Path(args.out).resolve()}")
        print("=" * 70)

    # Outside the with-block: the transcript is closed and complete, so it can
    # be moved without truncating the last lines.
    final = _place_report(staging, Path(args.out))
    if final:
        print(f"Transcript saved: {final}")
    return 0 if total else 1


def _place_report(staging: Path, out_dir: Path) -> Path | None:
    """Move the finished transcript next to the dataset it describes.

    Kept together on purpose: a report that lives somewhere else gets
    separated from its data the first time you tidy up, and then you have
    samples whose provenance you cannot reconstruct.
    """
    if not staging.exists():
        return None
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "build_report.txt"
        if staging.resolve() == target.resolve():
            return target
        target.write_text(staging.read_text(encoding="utf-8", errors="replace"),
                          encoding="utf-8")
        staging.unlink()
        return target
    except OSError as err:
        print(f"  could not move the transcript ({err}); it is at {staging}")
        return staging


if __name__ == "__main__":
    raise SystemExit(main())
