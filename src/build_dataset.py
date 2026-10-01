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

import fireattrib as FA
import firefilter as FF
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
# Time steps
# ---------------------------------------------------------------------------
def step_windows(start: str, end: str, step: int) -> list:
    """Window start times, local, aligned to local midnight, covering start..end."""
    tz = V.LOCAL_TZ
    first = pd.Timestamp(f"{start} 00:00").tz_localize(tz)
    last = pd.Timestamp(f"{end} 00:00").tz_localize(tz) + pd.Timedelta(days=1)
    out, t = [], first
    while t < last:
        out.append(t)
        t = t + pd.Timedelta(hours=step)
    return out


def window_key(ts, step: int) -> str:
    """Stable key for a window. Daily keeps the old YYYY-MM-DD naming."""
    return ts.strftime("%Y-%m-%d") if step == 24 else ts.strftime("%Y-%m-%dT%H")


def assign_windows(df: pd.DataFrame, start: str, step: int) -> pd.Series:
    """Map each detection to the key of the window its local time falls in.

    Computed explicitly from local wall-clock hours rather than with
    Series.dt.floor, which floors tz-aware times against the UTC epoch and
    would misalign every window by the UTC offset.
    """
    if df.empty:
        return pd.Series(dtype=object, index=df.index)
    tz = V.LOCAL_TZ
    if "acq_datetime" in df.columns:
        t = pd.to_datetime(df["acq_datetime"], utc=True, errors="coerce")
    else:
        t = pd.to_datetime(df["acq_date"].astype(str) + " " +
                           df["acq_time"].astype(float).astype(int).astype(str).str.zfill(4),
                           format="%Y-%m-%d %H%M", utc=True, errors="coerce")
    local = t.dt.tz_convert(tz)
    day = local.dt.strftime("%Y-%m-%d")
    hour = (local.dt.hour // step) * step
    if step == 24:
        return day
    return day + "T" + hour.astype(int).astype(str).str.zfill(2)


def _rewrite_audit_windows(out_dir: Path, record, annotated: pd.DataFrame) -> None:
    """Re-save the audit CSV with the window column the labels were built on."""
    path = out_dir / "tables" / f"{_slug(record)}.csv"
    if path.exists():
        annotated.to_csv(path, index=False)


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
    if args.min_frp:
        df = df[pd.to_numeric(df["frp"], errors="coerce") >= args.min_frp]

    # --- decide which detections become labels (firefilter) --------------
    # Every detection keeps a verdict, so the saved CSV is an audit trail for
    # the label: rows with used_in_label=True are exactly the ones rasterized.
    cfg = FF.FilterConfig(
        min_confidence=args.min_confidence,
        remove_static=not args.keep_static,
        target_only=args.target_only,
    )
    annotated = FF.annotate(df, cfg, target_bbox=record.bbox,
                            alarm_date=record.alarm_date)
    df = annotated[annotated["used_in_label"]].copy()
    print(f"  quality     {cfg.describe()}")
    print(FF.summarize(annotated))

    out_dir = Path(args.out) / _slug(record)
    _save_fire_record(out_dir, record, tile, envelope, start, end)

    # --- which fire is each detection? (fireattrib) ----------------------
    # The tile holds more than the named fire. Every kept detection gets a
    # fire_id and a name: 1 = this fire, then other mapped fires, then
    # unmapped clusters. Labels are unchanged; the IDs ride alongside.
    df, fires = name_fires(df, record, envelope, start, end, args)
    annotated = annotated.assign(fire_id=0, fire_name="")
    annotated.loc[df.index, "fire_id"] = df["fire_id"]
    annotated.loc[df.index, "fire_name"] = df["fire_name"]
    FA.save_fires(out_dir, fires)

    # --- visuals, from the detections already fetched --------------------
    if args.visualize:
        render_visuals(df, record, envelope, out_dir, args, annotated=annotated,
                       fires=fires)

    # --- stage 3: static layers, once ------------------------------------
    print("  static layers...")
    static = {} if args.no_ee else fetch_static_layers(
        tile, spec, args.ee_project, start)
    if static:
        print(f"    got {', '.join(sorted(static))}")
    elif not args.no_ee:
        # Features were asked for and none arrived. Writing 27 label-only
        # samples and reporting success would be the wrong outcome: they are
        # unusable for training and look identical to a good run in the
        # output directory.
        print("    NO STATIC LAYERS RETURNED. Every sample would carry only "
              "prev_fire_mask.")
        print("    Fix Earth Engine first: python verify_ee_setup.py")
        if not args.allow_empty_features:
            return {"fire": record.name, "samples": 0,
                    "reason": "no Earth Engine features (use "
                              "--allow-empty-features to write anyway)"}

    # --- stages 4-6: one sample per labelled time step ------------------
    # A step is a window [t, t + step) in LOCAL time, aligned to local
    # midnight. step_hours=24 reproduces the original daily behaviour
    # exactly, including file names, so existing datasets stay comparable.
    step = int(args.step_hours)
    per_day = 24 // step
    to_steps = lambda days: max(1, int(np.ceil(days * per_day)))  # noqa: E731

    windows = step_windows(start, end, step)
    keys = [window_key(w, step) for w in windows]
    df = df.assign(window=assign_windows(df, start, step))
    annotated_windows = assign_windows(annotated, start, step)

    lead_steps = args.lead_in_days * per_day
    label_keys = keys[lead_steps:]              # lead-in builds prev only
    # FRAP alarm dates are UTC dates: an evening ignition in California is
    # recorded as the next day (Eaton, 18:18 PST on 01-07 -> "2025-01-08").
    # If the named fire was seen earlier in LOCAL time, labels start there,
    # so its first night is a sample and not just lead-in.
    seen = next((t.first_seen for t in fires if t.fire_id == 1), None)
    if seen and record.alarm_date and seen < record.alarm_date:
        early = [k for k in keys[1:lead_steps] if k[:10] >= seen]
        if early:
            label_keys = early + label_keys
            print(f"  alarm date  CAL FIRE {record.alarm_date}, first seen {seen} "
                  f"(local); labels start {early[0]}")

    # Stop once the fire has gone quiet. Meaningful only after static
    # removal: a refinery lights up every day, so without it a run of blank
    # steps never occurs and this would never trigger.
    blank_steps = to_steps(args.stop_after_blank_days) if args.stop_after_blank_days else 0
    stop_after = FF.blank_run_stop_date(df, keys, blank_steps, date_col="window")
    if stop_after:
        dropped = [k for k in label_keys if k > stop_after]
        label_keys = [k for k in label_keys if k <= stop_after]
        keys = [k for k in keys if k <= stop_after]
        windows = windows[:len(keys)]
        print(f"  stop rule   {args.stop_after_blank_days} blank day(s) "
              f"({blank_steps} steps) after {stop_after}; "
              f"{len(dropped)} later step(s) not written")

    # Which windows had a satellite overhead at all. FIRMS reports detections,
    # not overpasses, so this is inferred from ANY detection in the tile --
    # including the static sources excluded from the label, which light up on
    # every pass and therefore make a good overpass witness. A window with no
    # detection of any kind is treated as unobserved across the whole tile:
    # nobody looked, so no cell can be labelled "no fire". Conservative -- a
    # pass that saw nothing at all is also marked unobserved -- which loses a
    # little data rather than inventing negatives.
    overpass_keys = set(annotated_windows.dropna())

    # Coverage while the fire was active (from the first fire window on).
    # Two different quantities, reported separately:
    #   overpass       a satellite looked (any detection in the tile, including
    #                  filtered static sources). Windows without one become
    #                  unobserved across the whole tile.
    #   fire detected  a satellite looked AND saw fire. The difference is
    #                  passes where the fire was hidden by smoke or cloud, or
    #                  had died down.
    active = set(df["window"])
    live = [k for k in label_keys if k >= min(active)] if active else []
    if step < 24 and live:
        n_pass = sum(1 for k in live if k in overpass_keys)
        n_fire = sum(1 for k in live if k in active)
        print(f"  coverage    {step}-hour windows while active: {len(live)}; "
              f"with an overpass {n_pass} ({n_pass / len(live) * 100:.0f}%), "
              f"with fire detected {n_fire} ({n_fire / len(live) * 100:.0f}%)")

    max_carry = to_steps(args.max_carry_days)
    label_set = set(label_keys)
    prev_mask = np.zeros(spec.shape, dtype=np.uint8)
    carried = 0            # consecutive steps prev_mask has been held over
    written, manifest = 0, []
    dynamic_cache: dict[str, dict] = {}

    for win, key in zip(windows, keys):
        step_df = df[df["window"] == key]
        fire_mask = F.rasterize_fire_mask(step_df, tile, spec)
        label = F.mark_unobserved(fire_mask, prev_mask,
                                  day_had_detections=not step_df.empty, spec=spec)
        fire_ids = FA.fire_id_map(step_df, tile, spec)
        observed = key in overpass_keys
        if not observed:
            label = np.full(spec.shape, F.LABEL_UNOBSERVED, dtype=np.uint8)
        local_day = win.strftime("%Y-%m-%d")

        if key in label_set:
            features = dict(static)
            features["prev_fire_mask"] = prev_mask.astype(float)
            if not args.no_ee:
                # Zero-order hold: daily products (GRIDMET, drought, NDVI) are
                # fetched once per local day and held constant across every
                # sub-daily step in it. Re-fetching per step would return the
                # same image 24 times at hourly resolution.
                if local_day not in dynamic_cache:
                    dynamic_cache[local_day] = fetch_dynamic_layers(
                        tile, spec, args.ee_project, local_day)
                features.update(dynamic_cache[local_day])

            stack, lab = F.assemble_sample(features, label, spec)
            present = [c for c in F.CHANNELS
                       if c in features and np.isfinite(features[c]).any()]
            meta = {
                "fire": record.name, "year": record.year, "date": local_day,
                "window_start": key,
                "window_end": window_key(win + pd.Timedelta(hours=step), step),
                "step_hours": step,
                "crs": spec.crs, "cell_m": spec.cell_m,
                "tile": list(tile), "tile_lonlat": list(envelope),
                "detections": int(len(step_df)),
                "observed": bool(observed),
                "channels_present": present,
                "source": record.source,
                "exact_footprint": record.exact_footprint,
                "filter": cfg.describe(),
                "feature_lag_days": int(V._FEATURE_LAG_DAYS),
                "fires": FA.cells_by_fire(fire_ids, fires),
                "features_hold": "daily layers held (zero-order) across sub-daily steps"
                                 if step < 24 else "daily",
            }
            path = F.save_sample(out_dir, f"{_slug(record)}_{key}", stack, lab, meta,
                                 extras={"fire_id": fire_ids})
            written += 1
            manifest.append({
                "path": str(path.name), "window_start": key, "date": local_day,
                "detections": int(len(step_df)),
                "fire_cells": int((lab == F.LABEL_FIRE).sum()),
                "unobserved_cells": int((lab == F.LABEL_UNOBSERVED).sum()),
                "channels_present": len(present),
            })
            flag = "" if present else "   <-- NO FEATURES"
            others = [f"{f['name']} {f['cells']}" for f in meta["fires"] if f["id"] != 1]
            if others:
                flag += "   other fires: " + ", ".join(others)
            if not observed:
                flag += "   (no overpass: all unobserved)"
            print(f"    {key:<16} {len(step_df):>5,} det  "
                  f"{int((lab == F.LABEL_FIRE).sum()):>5} fire cells  "
                  f"{len(present):>2}/{len(F.CHANNELS)} channels{flag}")
        elif step == 24 or not step_df.empty:
            print(f"    {key:<16} {len(step_df):>5,} det  (lead-in, not written)")

        # Carry prev_fire_mask across a coverage gap instead of zeroing it.
        # At sub-daily steps this matters far more: a 12-hour overpass gap is
        # 12 blank steps at hourly resolution, and resetting after the first
        # would claim the fire vanished every afternoon. Bounded by
        # --max-carry-days (converted to steps) so a real burnout expires.
        if step_df.empty and prev_mask.any() and carried < max_carry:
            carried += 1
        else:
            prev_mask = (fire_mask > 0).astype(np.uint8)
            carried = 0

    # The CSV audit trail needs the same window keys the labels used.
    if args.visualize:
        _rewrite_audit_windows(out_dir, record, annotated.assign(window=annotated_windows))

    if manifest:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"fire": record.name, "samples": written, "dir": str(out_dir)}



def _save_fire_record(out_dir: Path, record, tile, envelope, start, end) -> None:
    """Write the fire's ground truth next to its samples.

    perimeter.geojson is the independent reference the labels are checked
    against: the agency-mapped burn area, from a different source (ground and
    aerial mapping) than the satellite detections the labels come from. Saved
    at build time so verification works offline and always compares against
    the exact record the dataset was built from.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    props = {
        "name": record.name, "year": record.year,
        "alarm_date": record.alarm_date, "contain_date": record.contain_date,
        "acres": record.acres, "source": record.source,
        "exact_footprint": record.exact_footprint,
        "dates_uncertain": record.dates_uncertain,
        "tile": list(tile), "tile_lonlat": list(envelope),
        "fetch_window": [start, end],
    }
    geom = record.geometry
    if geom is None and record.bbox:
        w, s_, e, n = record.bbox
        geom = {"type": "Polygon",
                "coordinates": [[[w, s_], [e, s_], [e, n], [w, n], [w, s_]]]}
        props["geometry_is_bbox"] = True
    (out_dir / "perimeter.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": props, "geometry": geom}],
    }))


def render_visuals(df, record, envelope, out_dir: Path, args,
                   annotated=None, fires=None) -> None:
    """Produce the visualize_firms_dataset_v3 outputs for this fire.

    Reuses the DataFrame build_fire already fetched rather than re-querying
    FIRMS: the detections, bbox and date window are identical, so a second
    fetch would spend transactions to get the same rows back.

    Everything lands under <out>/<fire>/visuals/, on the same tile envelope the
    tensors use, so a PNG and a .npz for the same day describe the same ground.
    """
    vis_dir = out_dir / "visuals"
    data_dir = out_dir / "tables"
    vis_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    base = _slug(record)
    label = f"{record.name} {record.year or ''}".strip()

    print(f"  visuals -> {vis_dir}")
    try:
        # The CSV holds every detection with its verdict (confidence_class,
        # drop_reason, used_in_label), not just the survivors -- so you can
        # see what was excluded and why, and verify_dataset can check that
        # the used rows reproduce each label exactly.
        V.save_outputs(annotated if annotated is not None else df,
                       data_dir, base, save_csv=True)
    except Exception as err:
        print(f"    tables failed: {err}")

    # Overview plots for the whole window.
    jobs = [
        ("points", lambda: V.plot_fire_map(
            df, f"{label} -- detections (FRP)", envelope,
            vis_dir / f"{base}_points.png")),
        ("mask", lambda: V.plot_fire_mask(
            df, f"{label} -- fire mask", envelope,
            vis_dir / f"{base}_mask.png",
            add_basemap=args.basemap,
            cluster_eps_km=args.cluster_eps_km,
            centroids_dir=data_dir)),
        ("time", lambda: V.plot_fire_map_time_based(
            df, f"{label} -- detections over time", envelope,
            vis_dir / f"{base}_time.png")),
        ("fires", lambda: FA.plot_fires(
            df, fires or [], envelope, vis_dir / f"{base}_fires.png",
            f"{label} -- every fire in the tile ({len(fires or [])})",
            basemap=args.basemap)),
    ]
    for name, job in jobs:
        try:
            job()
        except Exception as err:
            # One failed plot must not lose the others, or the tensors.
            print(f"    {name} plot failed: {str(err).splitlines()[0][:120]}")

    if args.daily_visuals:
        try:
            start, end = record.date_window(args.lead_in_days, args.tail_days)
            V.export_daily_visual_samples(
                df, envelope, data_dir, vis_dir, base,
                plot_points=True, fire_mask=True,
                add_basemap=args.basemap,
                start_date=start, end_date=end,
                cluster_eps_km=args.cluster_eps_km,
            )
        except Exception as err:
            print(f"    daily visuals failed: {str(err).splitlines()[0][:120]}")


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
    print("  Wildfire Training Dataset - Interactive Mode")
    print("  (Press Enter to use default where shown)")
    print("=" * 60 + "\n")

    # 1. Fires, by name
    print("1. FIRES (by name - no bounding boxes needed)")
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
    print("   Prediction time step. Finer is faster to act on, but polar")
    print("   orbiters pass in clusters with gaps up to ~12 h:")
    print("   Windows with a satellite overhead while Eaton burned:")
    print("   24 = daily (all)   12 = 94%   6 = 61%   3 = 31%   1 = 20% (needs GOES)")
    print("   Windows with no overpass are labelled unobserved.")
    step_raw = _prompt("Step in hours (24, 12, 6, 3, 1)", "24").strip()
    args.step_hours = int(step_raw) if step_raw in {"1", "2", "3", "4", "6", "8", "12", "24"} else 24

    # 3. Feature layers
    print("\n3. FEATURE LAYERS")
    print("   1 = All Earth Engine layers (18 channels)")
    print("   2 = Labels only, no Earth Engine (fast; spends no EE quota)")
    args.no_ee = _prompt("Choice", "1").strip() == "2"
    args.allow_empty_features = False
    args.ee_project = None
    if not args.no_ee:
        args.ee_project = _prompt("EE project ID",
                                  V.resolve_ee_project()).strip() or None

    # 4. Visual outputs
    print("\n4. VISUALIZATIONS")
    print("   Same fire, same tile, same dates as the tensors -- nothing extra")
    print("   to type, and no second FIRMS fetch.")
    print("   1 = None (tensors only)")
    print("   2 = Overview plots + tables  (points, fire mask, time)")
    print("   3 = Overview + a plot folder for every day")
    vis_choice = _prompt("Choice", "2").strip()
    args.visualize = vis_choice in ("2", "3")
    args.daily_visuals = vis_choice == "3"
    args.basemap = False
    args.cluster_eps_km = 2.0
    if args.visualize:
        args.basemap = _prompt(
            "Add a basemap under the plots? (y/n)", "n").lower().startswith("y")
        eps = _prompt("Cluster radius for the fire mask (km)", "2.0").strip()
        try:
            args.cluster_eps_km = float(eps) if eps else 2.0
        except ValueError:
            print("  not a number; using 2.0 km")

    # 5. Detection filters
    print("\n5. DETECTION QUALITY (which detections become fire in the label)")
    print("   Confidence, normalized across VIIRS (l/n/h) and MODIS (0-100):")
    print("   1 = All      2 = Nominal and high      3 = High only")
    args.min_confidence = {"1": "low", "3": "high"}.get(
        _prompt("Choice", "2").strip(), "nominal")
    print("   Refineries, gas flares and power plants show up as heat every day.")
    args.keep_static = not _prompt(
        "Remove static heat sources? (y/n)", "y").lower().startswith("y")
    args.no_fire_names = False
    args.target_only = _prompt(
        "Label only the named fire, dropping other fires in the tile? (y/n)",
        "n").lower().startswith("y")
    stop = _prompt("Stop after this many consecutive blank days (0 = never)",
                   "3").strip()
    args.stop_after_blank_days = int(stop) if stop.isdigit() else 3
    frp = _prompt("Minimum FRP in MW (blank = none)", "").strip()
    try:
        args.min_frp = float(frp) if frp else None
    except ValueError:
        print("  not a number; ignoring FRP filter")
        args.min_frp = None

    # 6. Output
    print("\n6. OUTPUT")
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
    args.verify = False
    if not args.dry_run:
        args.verify = _prompt(
            "Verify the dataset when the build finishes? (y/n)", "y"
        ).lower().startswith("y")
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
    ap.add_argument("--step-hours", type=int, default=24, choices=[1, 2, 3, 4, 6, 8, 12, 24],
                    help="Prediction time step. 24 = daily (default). While Eaton "
                         "burned, a satellite was overhead in 94%% of 12-hour windows, "
                         "61%% of 6-hour and 20%% of 1-hour windows; windows with no "
                         "overpass are labelled unobserved. Each build prints this "
                         "for its own fire on the coverage line.")
    ap.add_argument("--max-carry-days", type=int, default=2,
                    help="How many consecutive blank days keep the previous "
                         "fire mask alive before it is treated as burnt out "
                         "(default 2)")
    ap.add_argument("--min-confidence", default="nominal",
                    choices=["low", "nominal", "high"],
                    help="Minimum detection confidence, normalized across VIIRS "
                         "(l/n/h) and MODIS (0-100). Default nominal.")
    ap.add_argument("--keep-static", action="store_true",
                    help="Keep refineries, flares and other persistent heat "
                         "sources in the label. Off by default.")
    ap.add_argument("--target-only", action="store_true",
                    help="Label only the named fire; drop other fires in the tile")
    ap.add_argument("--no-fire-names", action="store_true",
                    help="Skip the agency-perimeter lookup that names other fires "
                         "in the tile. They are still separated and numbered "
                         "(Unmapped #1, #2, ...).")
    ap.add_argument("--stop-after-blank-days", type=int, default=3,
                    help="Stop writing samples after this many consecutive days "
                         "with no surviving detections (0 = never). Default 3.")
    ap.add_argument("--min-frp", type=float, default=None)
    ap.add_argument("--ee-project", default=None)
    ap.add_argument("--visualize", action="store_true",
                    help="Also write the visualize_firms_dataset_v3 plots and "
                         "tables for each fire, on the same tile and dates. No "
                         "bounding box or date typing.")
    ap.add_argument("--daily-visuals", action="store_true",
                    help="With --visualize, also write per-day folders of "
                         "point and mask PNGs")
    ap.add_argument("--basemap", action="store_true",
                    help="Add a contextily basemap under the plots")
    ap.add_argument("--cluster-eps-km", type=float, default=2.0,
                    help="DBSCAN neighbourhood radius for fire-mask polygons")
    ap.add_argument("--allow-empty-features", action="store_true",
                    help="Write samples even when Earth Engine returns nothing. "
                         "Off by default: label-only samples cannot train a "
                         "model and are indistinguishable from a good run once "
                         "written.")
    ap.add_argument("--no-ee", action="store_true",
                    help="Labels only, no feature layers. Useful for checking "
                         "the label pipeline without spending EE quota.")
    ap.add_argument("--verify", action="store_true",
                    help="Run verify_dataset (including reality checks and QA "
                         "sheets) as soon as the build finishes")
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

        # Verify inside the transcript, so the build and its verification
        # live in one report.
        verified = None
        if total and getattr(args, "verify", False):
            verified = run_verification(Path(args.out))

    # Outside the with-block: the transcript is closed and complete, so it can
    # be moved without truncating the last lines.
    final = _place_report(staging, Path(args.out))
    if final:
        print(f"Transcript saved: {final}")
    if verified is False:
        return 2          # built, but verification found failures
    return 0 if total else 1


def run_verification(out_dir: Path) -> bool:
    """Run verify_dataset on what was just built. True when every check passed."""
    import verify_dataset
    print(f"\n{'=' * 70}\nVERIFYING {out_dir}\n{'=' * 70}")
    saved = sys.argv
    sys.argv = ["verify_dataset.py", str(out_dir),
                "--json", str(out_dir / "verification.json")]
    try:
        code = verify_dataset.main()
    finally:
        sys.argv = saved
    print(f"\nQA sheets: {out_dir}\\<fire>\\qa\\  -- look at them before training.")
    return code == 0


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


def name_fires(df, record, envelope, start, end, args):
    """Attribute every kept detection to a fire and print the roll call.

    One extra request: every mapped perimeter intersecting the tile during
    the fetch window. If it fails, other fires are still separated, just not
    named.
    """
    neighbours = []
    if not getattr(args, "no_fire_names", False):
        try:
            neighbours = L.perimeters_in_area(envelope, record.year, start, end)
        except Exception as err:
            print(f"  fires       perimeter lookup failed "
                  f"({str(err).splitlines()[0][:100]}); other fires stay unnamed")
    df, fires = FA.attribute(df, record, neighbours, crs=F.GRID_SPEC.crs,
                             tail_days=args.tail_days)
    print(FA.summarize(fires))
    return df, fires


if __name__ == "__main__":
    raise SystemExit(main())
