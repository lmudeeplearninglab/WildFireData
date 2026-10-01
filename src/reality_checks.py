#!/usr/bin/env python3
"""
reality_checks -- does the extracted data match the real world?

verify_dataset's other checks confirm the pipeline is SELF-CONSISTENT: shapes
agree, dates are consecutive, labels match the CSV. A dataset can pass all of
those and still be wrong -- temperatures in Kelvin, wind pointing backwards,
refinery flares labelled as fire. This module compares the data against
things the pipeline did not produce:

  PHYSICS        every channel inside its physically possible range, with
                 unit sniffers for the classic slips (Kelvin, kg/kg,
                 unscaled NDVI)
  GEOMETRY       aspect agrees with the slope of the elevation channel --
                 detects a flipped or rotated grid, which nothing else would
  WIND           |(u, v)| agrees with wind_speed -- detects a broken
                 decomposition
  STATIC/HOLD    terrain identical across every sample; daily layers
                 identical within a day at sub-daily steps (zero-order hold)
  GROUND TRUTH   labels against the agency-mapped burn perimeter, a
                 different measurement (ground and aerial mapping) from the
                 satellite detections the labels come from
  KNOWN EVENTS   conditions recorded for specific historical fires, e.g. the
                 Santa Ana wind during Palisades and Eaton

It also draws QA sheets, because the fastest check of all is a person
looking at the fire and saying "that is not where it burned".
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Physical ranges.  HARD = impossible outside this (FAIL).
# ---------------------------------------------------------------------------
HARD_RANGE = {
    "elevation": (-500.0, 9000.0),        # m; Dead Sea to Everest
    "slope": (0.0, 90.0),                 # degrees
    "aspect_sin": (-1.001, 1.001),
    "aspect_cos": (-1.001, 1.001),
    "aspect_consistency": (0.0, 1.001),
    "vegetation": (-1.0, 1.0),            # NDVI
    "population": (0.0, 2.0e5),           # people / km^2
    "drought": (-15.0, 15.0),             # PDSI
    "humidity": (0.0, 40.0),              # g/kg
    "weather_temp": (-60.0, 60.0),        # deg C, daily max
    "weather_precip": (0.0, 500.0),       # mm/day
    "wind_speed": (0.0, 50.0),            # m/s
    "wind_u": (-50.0, 50.0),
    "wind_v": (-50.0, 50.0),
    "erc": (0.0, 200.0),                  # energy release component
    "vpd": (0.0, 12.0),                   # kPa
    "prev_fire_mask": (0.0, 1.0),
}
UNITS = {
    "elevation": "m", "slope": "deg", "aspect_sin": "east", "aspect_cos": "north",
    "aspect_consistency": "R 0-1", "vegetation": "NDVI", "landcover": "ESA class",
    "population": "per km2", "drought": "PDSI", "humidity": "g/kg",
    "weather_temp": "deg C", "weather_precip": "mm", "wind_speed": "m/s",
    "wind_u": "m/s east", "wind_v": "m/s north", "erc": "index", "vpd": "kPa",
    "prev_fire_mask": "0/1",
}
ESA_WORLDCOVER = {10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100}
ESA_WATER = 80
COAST_CELLS = 2          # water within 2 km of a land cell counts as coastline
STATIC_CHANNELS = ("elevation", "slope", "aspect_sin", "aspect_cos",
                   "aspect_consistency", "landcover", "population")
DAILY_CHANNELS = ("vegetation", "drought", "humidity", "weather_temp",
                  "weather_precip", "wind_speed", "wind_u", "wind_v", "erc", "vpd")

# Known-answer tests from the historical record.
#
# RULES FOR ADDING ONE -- the first version of this table broke two of them:
#   1. The claim must be documented, with a source, at the strength tested.
#      "No significant rain" is not "no rain": the record said the former,
#      and a 1 mm threshold tested the latter.
#   2. The claim must be about the SAME QUANTITY as the channel. "In drought"
#      was a US Drought Monitor statement; testing it against PDSI, a
#      different index with months of memory, failed on correct data.
#   3. The date is the day the CONDITIONS occurred. Features lag the label
#      by `feature_lag_days`, so conditions on 01-07 are in the sample
#      labelled 01-08. The check does that mapping; enter the real date.
# A failure means the data OR the claim is wrong. Check both.
#
# Sources (January 2025 Los Angeles fires):
#   NASA Earth Observatory, "Fuel for California Fires" -- no significant rain
#     May 2024 to early January 2025; downtown LA had one day in eight months
#     above a tenth of an inch (2.54 mm).
#   Wikipedia, "January 2025 Southern California wildfires" (citing NWS) --
#     Santa Ana winds accelerating the afternoon of January 7 through early
#     January 8; driest nine months on record before the wind event.
# Precipitation is tested over the fire (perimeter + 2 km), not the whole
#   tile: the record is about the burn area, and a 64 km tile reaches into the
#   San Gabriels, where isolated mountain snow showers were forecast for
#   2025-01-06 (UCLA weather synopsis). Eaton's tile showed 3.8 mm there.
# PDSI deliberately absent: the two preceding winters were well above
# average, and PDSI's long memory can sit near zero or positive in early
# January despite the Drought Monitor showing moderate drought.
_LA_JAN_2025 = [
    ("wind_toward", "2025-01-07", 225.0, 70.0,
     "Santa Ana: wind from the NE, so vectors point south-west"),
    ("precip_below", "2025-01-07", 2.54, None,
     "no rain above a tenth of an inch at the fire before the wind event"),
]
KNOWN_EVENTS = {
    ("PALISADES", 2025): _LA_JAN_2025,
    ("EATON", 2025): _LA_JAN_2025,
}


def _ch(samples, name):
    """Stack one channel across samples -> (N, H, W), or None if absent."""
    chans = samples[0]["channels"]
    if name not in chans:
        return None
    i = chans.index(name)
    return np.stack([s["features"][i] for s in samples]).astype("float64")


# ---------------------------------------------------------------------------
# PHYSICS
# ---------------------------------------------------------------------------
def check_physics(samples, rep) -> None:
    out_of_range = []
    for name, (lo, hi) in HARD_RANGE.items():
        x = _ch(samples, name)
        if x is None:
            continue
        v = x[np.isfinite(x)]
        if v.size == 0:
            continue
        bad = ((v < lo) | (v > hi)).mean()
        if bad > 0:
            out_of_range.append(f"{name} [{v.min():.3g}, {v.max():.3g}] vs "
                                f"allowed [{lo:g}, {hi:g}] ({bad * 100:.1f}% of cells)")
    rep.check(not out_of_range, "every channel inside its physical range",
              "; ".join(out_of_range[:4]))

    # Unit sniffers: values that are in range but betray a missing conversion.
    sniffs = []
    t = _ch(samples, "weather_temp")
    if t is not None and np.isfinite(t).any() and np.nanmedian(t) > 150:
        sniffs.append(f"weather_temp median {np.nanmedian(t):.0f}: looks like Kelvin")
    h = _ch(samples, "humidity")
    if h is not None and np.isfinite(h).any() and 0 < np.nanmedian(h) < 0.1:
        sniffs.append(f"humidity median {np.nanmedian(h):.4f}: looks like kg/kg, not g/kg")
    n = _ch(samples, "vegetation")
    if n is not None and np.isfinite(n).any() and np.nanmax(np.abs(n)) > 1.5:
        sniffs.append(f"vegetation max {np.nanmax(n):.0f}: NDVI scale factor not applied")
    rep.check(not sniffs, "units look physical (no Kelvin / kg-per-kg / unscaled NDVI)",
              "; ".join(sniffs))

    lc = _ch(samples, "landcover")
    if lc is not None and np.isfinite(lc).any():
        found = set(np.unique(lc[np.isfinite(lc)]).astype(int).tolist())
        stray = sorted(found - ESA_WORLDCOVER)
        rep.check(not stray, f"landcover holds only ESA WorldCover classes {sorted(found)}",
                  "" if not stray else f"non-class values {stray[:6]}: land cover "
                  f"was averaged instead of taking the majority class")

    pm = _ch(samples, "prev_fire_mask")
    if pm is not None:
        vals = set(np.unique(pm[np.isfinite(pm)]).tolist())
        rep.check(vals <= {0.0, 1.0}, "prev_fire_mask is binary",
                  "" if vals <= {0.0, 1.0} else f"values {sorted(vals)[:6]}")

    flat = []
    for name in ("elevation", "slope", "vegetation", "weather_temp"):
        x = _ch(samples, name)
        if x is None or not np.isfinite(x).any():
            continue
        per_sample_std = np.nanstd(x.reshape(len(x), -1), axis=1)
        if np.nanmax(per_sample_std) == 0:
            flat.append(name)
    rep.check(not flat, "spatially varying channels actually vary across the tile",
              "" if not flat else f"constant over the whole tile: {', '.join(flat)} "
              f"(a scalar was broadcast, or the fetch returned a fill value)")


# ---------------------------------------------------------------------------
# GEOMETRY and WIND
# ---------------------------------------------------------------------------
def check_orientation(samples, rep) -> None:
    """Aspect must point downhill on the elevation channel.

    Both come from Earth Engine, but aspect is computed there at 30 m and the
    elevation gradient here from the resampled 1 km grid. If the grid were
    flipped north-south or transposed anywhere in extraction, the elevation
    field flips but the aspect VALUES (which encode compass directions) do
    not, and the two disagree. Nothing else in the pipeline detects that.
    """
    elev, s, c = (_ch(samples[:1], n) for n in ("elevation", "aspect_sin", "aspect_cos"))
    if elev is None or s is None or c is None:
        return
    z = np.nan_to_num(elev[0])
    d_row, d_col = np.gradient(z)
    # rows run north -> south, so north-gradient = -d_row
    down_e, down_n = -d_col, d_row
    mag = np.hypot(down_e, down_n)
    a_e, a_n = np.nan_to_num(s[0]), np.nan_to_num(c[0])
    w = mag * (np.hypot(a_e, a_n) > 0)
    if w.sum() < 1e-6:
        rep.note("terrain too flat to test grid orientation")
        return
    cos = ((down_e * a_e + down_n * a_n) / np.maximum(mag, 1e-9) * w).sum() / w.sum()
    rep.check(cos > 0.3, f"grid orientation: aspect agrees with the elevation slope "
              f"(weighted cosine {cos:+.2f})",
              "" if cos > 0.3 else "aspect points uphill or sideways: the grid is "
              "flipped or rotated somewhere in extraction")


def check_wind(samples, rep) -> None:
    """|(u, v)| must not exceed wind_speed, and should be close to it.

    Averaging vectors can only shorten them (calm and gusty periods from
    different directions cancel), so |mean vector| <= mean speed always.
    A magnitude well ABOVE the speed means the components are wrong.
    """
    u, v, sp = (_ch(samples, n) for n in ("wind_u", "wind_v", "wind_speed"))
    if u is None or v is None or sp is None:
        return
    mag = np.hypot(u, v)
    ok = np.isfinite(mag) & np.isfinite(sp) & (sp > 0.5)
    if not ok.any():
        return
    ratio = mag[ok] / sp[ok]
    over = (ratio > 1.10).mean()
    rep.check(over < 0.01 and np.median(ratio) > 0.5,
              f"wind components consistent with wind speed "
              f"(median |uv|/speed {np.median(ratio):.2f})",
              "" if over < 0.01 else f"{over * 100:.0f}% of cells have |uv| > speed: "
              "the decomposition is wrong")


# ---------------------------------------------------------------------------
# STATIC and HOLD
# ---------------------------------------------------------------------------
def check_static_and_hold(samples, rep) -> None:
    changed = []
    for name in STATIC_CHANNELS:
        x = _ch(samples, name)
        if x is None or len(x) < 2:
            continue
        if not all(np.array_equal(x[0], xi, equal_nan=True) for xi in x[1:]):
            changed.append(name)
    rep.check(not changed, "static layers identical in every sample",
              "" if not changed else f"varies between samples: {', '.join(changed)}")

    steps = {int(s["meta"].get("step_hours", 24)) for s in samples}
    if steps == {24}:
        return
    by_day: dict[str, list] = {}
    for s in samples:
        by_day.setdefault(s["meta"].get("date"), []).append(s)
    broken = set()
    for day, group in by_day.items():
        for name in DAILY_CHANNELS:
            x = _ch(group, name)
            if x is not None and not all(np.array_equal(x[0], xi, equal_nan=True)
                                         for xi in x[1:]):
                broken.add(name)
    rep.check(not broken, "daily layers held constant within each day "
              "(zero-order hold)", "" if not broken else
              f"change within a day: {', '.join(sorted(broken))}")


# ---------------------------------------------------------------------------
# GROUND TRUTH
# ---------------------------------------------------------------------------
def _load_perimeter(fire_dir: Path):
    p = fire_dir / "perimeter.geojson"
    if not p.exists():
        return None, {}
    feat = json.loads(p.read_text())["features"][0]
    return feat.get("geometry"), feat.get("properties", {})


def perimeter_on_grid(geometry, tile, F, buffer_m: float = 0.0) -> np.ndarray:
    """Rasterize the agency perimeter onto the sample grid."""
    import rasterio.features
    from pyproj import Transformer
    from shapely.geometry import shape, mapping
    from shapely.ops import transform as shp_transform
    fwd = Transformer.from_crs("EPSG:4326", F.GRID_SPEC.crs, always_xy=True)
    poly = shp_transform(lambda x, y, z=None: fwd.transform(x, y), shape(geometry))
    if buffer_m:
        poly = poly.buffer(buffer_m)
    return rasterio.features.rasterize(
        [(mapping(poly), 1)], out_shape=F.GRID_SPEC.shape,
        transform=F.tile_transform(tuple(tile)), fill=0,
        all_touched=True, dtype=np.uint8) > 0


def check_ground_truth(samples, fire_dir: Path, rep, F) -> dict:
    geometry, props = _load_perimeter(fire_dir)
    if geometry is None:
        rep.note("no perimeter.geojson (dataset built before it was saved); "
                 "rebuild to enable the ground-truth check")
        return {}
    if props.get("geometry_is_bbox"):
        rep.note("perimeter is only a bounding box (no published polygon); "
                 "ground-truth overlap not meaningful")
        return {}

    tile = samples[0]["meta"]["tile"]
    fire_any = np.zeros(F.GRID_SPEC.shape, bool)
    # With fire IDs, the perimeter is scored against ITS fire only (id 1):
    # Hurst's cells are not expected near the Palisades perimeter.
    by_id = all(s.get("fire_id") is not None for s in samples)
    target = np.zeros(F.GRID_SPEC.shape, bool)
    first_fire = None
    for s in sorted(samples, key=lambda s: s["meta"].get("window_start") or s["meta"]["date"]):
        hit = s["label"] == F.LABEL_FIRE
        mine = (s["fire_id"] == 1) if by_id else hit
        if mine.any() and first_fire is None:
            first_fire = s["meta"]["date"]
        fire_any |= hit
        target |= mine

    inside = perimeter_on_grid(geometry, tile, F)
    # 2 km covers the VIIRS/MODIS footprint, geolocation error and
    # short-range spotting beyond the final mapped line.
    near = perimeter_on_grid(geometry, tile, F, buffer_m=2000.0)

    n_fire = int(target.sum())
    if n_fire == 0:
        rep.check(False, "labels contain fire at all" if not by_id else
                  "labels contain the named fire at all",
                  "" if not by_id else "no detection lies within 2 km of its perimeter "
                  "during its dates: the wrong fire record, or the perimeter is offset "
                  "from the detections (see visuals/*_fires.png)")
        return {}
    if by_id and (fire_any & ~target).any():
        rep.note(f"{int((fire_any & ~target).sum())} cells belong to other fires in the "
                 f"tile; the perimeter checks below score the named fire only")
    whose = "the named fire's" if by_id else "fire"
    in_share = float((target & near).sum() / n_fire)
    coverage = float((target & inside).sum() / max(1, inside.sum()))

    if in_share >= 0.8:
        rep.check(True, f"labels sit on the mapped burn area: {in_share * 100:.0f}% of "
                  f"{whose} cells within 2 km of the "
                  f"{props.get('source', 'agency perimeter')}")
    else:
        rep.warn(f"only {in_share * 100:.0f}% of {whose} cells are near the mapped perimeter",
                 "the rest are other fires in the tile or unremoved static sources; "
                 "see the red cells outside the outline on qa_labels.png")
    if coverage >= 0.5:
        rep.check(True, f"labels cover the burn area: {coverage * 100:.0f}% of perimeter "
                  f"cells were detected burning at least once")
    else:
        rep.warn(f"labels cover only {coverage * 100:.0f}% of the mapped burn area",
                 "missing days, over-aggressive filtering, or the fire burned "
                 "between overpasses")

    alarm = props.get("alarm_date")
    if alarm and first_fire and not props.get("dates_uncertain"):
        lag = (datetime.strptime(first_fire, "%Y-%m-%d")
               - datetime.strptime(alarm, "%Y-%m-%d")).days
        ok = -1 <= lag <= 1
        (rep.check(True, f"first fire label {first_fire} matches the alarm date "
                   f"{alarm} ({lag:+d} d)") if ok else
         rep.warn(f"first fire label {first_fire} is {lag:+d} days from the "
                  f"alarm date {alarm}",
                  "negative: fire in the tile before ignition (another fire or a "
                  "static source); positive: early days missing"))

    acres = props.get("acres")
    if acres:
        rep.note(f"area: perimeter {acres * 0.004047:,.0f} km2, cells ever burning "
                 f"{n_fire} km2 (1 km cells overstate small fires; expect ratio > 1)")

    water_stats = check_fire_on_water(samples, fire_dir, fire_any, geometry,
                                      props, tile, rep, F)
    return {"inside_share": round(in_share, 3), "coverage": round(coverage, 3),
            "first_fire": first_fire, **water_stats}


# ---------------------------------------------------------------------------
# FIRE ON WATER -- misalignment, or footprint spillover?
# ---------------------------------------------------------------------------
def _shifted(mask: np.ndarray, dr: int, dc: int):
    """mask moved by (dr, dc); cells shifted in from outside are marked invalid."""
    out = np.zeros_like(mask)
    valid = np.zeros(mask.shape, bool)
    H, W = mask.shape
    rs, re_ = max(0, dr), min(H, H + dr)
    cs, ce = max(0, dc), min(W, W + dc)
    out[rs:re_, cs:ce] = mask[rs - dr:re_ - dr, cs - dc:ce - dc]
    valid[rs:re_, cs:ce] = True
    return out, valid


def _audit_rows(fire_dir: Path):
    """Detections that became fire in the labels, from the audit CSV."""
    import pandas as pd
    tables = fire_dir / "tables"
    csvs = [c for c in sorted(tables.glob("*.csv"))
            if not c.stem.endswith(("_centroids", "_clusters"))] if tables.is_dir() else []
    if not csvs:
        return None
    df = pd.read_csv(csvs[0])
    if "used_in_label" in df.columns:
        df = df[df["used_in_label"].astype(str).str.lower().isin(["true", "1"])]
    return df


def check_fire_on_water(samples, fire_dir, fire_any, geometry, props, tile, rep, F) -> dict:
    """Fire labels on water: a registration error, or big footprints at a coast?

    Two very different causes look the same in a count:
      MISALIGNMENT  labels offset from the features. Fatal.
      SPILLOVER     a coastal fire whose detection footprints, rasterized with
                    all_touched, reach cells that are mostly sea. MODIS
                    footprints run to several km across, so this is expected
                    for any fire that burned to the shore. Not fatal, but it
                    is fire painted on the ocean, and worth a decision.
    Evidence gathered to tell them apart:
      1. Registration: the agency perimeter is drawn in the label frame. Its
         interior is land by definition. If shifting the land-cover map a few
         km makes the perimeter sit on land BETTER than no shift does, the
         frames are offset.
      2. Detection centres: spillover touches water cells with footprint
         edges; misalignment puts the detections THEMSELVES on water.
      3. Which sensor's footprints reach the water cells.
    """
    lc = _ch(samples[:1], "landcover")
    if lc is None:
        return {}
    lc = lc[0]
    # ESA WorldCover maps sea as class 80; beyond its tiles there may be no
    # data at all. Neither is land.
    not_land = (lc == ESA_WATER) | ~np.isfinite(lc)
    land = ~not_land
    on_water = fire_any & not_land
    n_fire, n_water = int(fire_any.sum()), int(on_water.sum())
    if n_water <= max(1, 0.02 * n_fire):
        rep.check(True, f"no fire on open water ({n_water} cells)")
        return {"fire_on_water": n_water}

    # Coastal = a land cell within COAST_CELLS. Two, not one: land cover is
    # resampled by MODE, so a shore cell whose land is split between built-up,
    # shrub and beach comes out "water" with only a third of it wet, and the
    # coastline moves about a cell inland. Palisades' 7 "offshore" cells were
    # exactly that (reproduced against an independent land mask).
    land_near = np.zeros_like(land)
    for dr in range(-COAST_CELLS, COAST_CELLS + 1):
        for dc in range(-COAST_CELLS, COAST_CELLS + 1):
            land_near |= _shifted(land, dr, dc)[0]
    coastal = on_water & land_near
    offshore = on_water & ~land_near
    stats = {"fire_on_water": n_water, "coastal": int(coastal.sum()),
             "offshore": int(offshore.sum())}
    rep.note(f"{n_water} fire cells sit on water/no-land cells: "
             f"{stats['coastal']} within {COAST_CELLS} km of land, "
             f"{stats['offshore']} further out")

    # 1. Registration against the perimeter.
    misaligned = None
    if geometry is not None and not props.get("geometry_is_bbox"):
        import rasterio.features
        from pyproj import Transformer
        from shapely.geometry import shape, mapping
        from shapely.ops import transform as shp_transform
        fwd = Transformer.from_crs("EPSG:4326", F.GRID_SPEC.crs, always_xy=True)
        poly = shp_transform(lambda x, y, z=None: fwd.transform(x, y), shape(geometry))
        interior = rasterio.features.rasterize(
            [(mapping(poly), 1)], out_shape=F.GRID_SPEC.shape,
            transform=F.tile_transform(tuple(tile)), fill=0,
            all_touched=False, dtype=np.uint8) > 0
        if interior.sum() >= 5:
            scores = {}
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    moved, valid = _shifted(not_land, dr, dc)
                    cells = interior & valid
                    if cells.sum() >= 5:
                        scores[(dr, dc)] = moved[cells].mean()
            here = scores.get((0, 0), 1.0)
            best_shift, best = min(scores.items(), key=lambda kv: kv[1])
            misaligned = bool(here - best > 0.05)
            stats["perimeter_interior_on_water"] = round(float(here), 3)
            rep.check(not misaligned,
                      f"labels registered to the features: {here * 100:.0f}% of the "
                      f"perimeter interior is water at zero shift",
                      "" if not misaligned else
                      f"shifting land cover by {best_shift} cells (row, col) drops that "
                      f"to {best * 100:.0f}% -- labels and features are offset")

    # 2 and 3. Attribute the water cells to detections.
    rows = _audit_rows(fire_dir)
    centres_offshore = None
    if rows is not None and len(rows):
        from pyproj import Transformer
        fwd = Transformer.from_crs("EPSG:4326", F.GRID_SPEC.crs, always_xy=True)
        x, y = fwd.transform(rows["longitude"].to_numpy(), rows["latitude"].to_numpy())
        x0, _, _, y1 = tile
        c = np.floor((np.asarray(x) - x0) / F.GRID_SPEC.cell_m).astype(int)
        r = np.floor((y1 - np.asarray(y)) / F.GRID_SPEC.cell_m).astype(int)
        H, W = F.GRID_SPEC.shape
        inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        centre_cells = np.zeros(F.GRID_SPEC.shape, bool)
        centre_cells[r[inside], c[inside]] = True
        centres_on_water = int((on_water & centre_cells).sum())
        centres_offshore = int((offshore & centre_cells).sum())

        is_modis = rows["firms_source"].astype(str).str.startswith("MODIS") \
            if "firms_source" in rows.columns else np.zeros(len(rows), bool)
        viirs_fp = F.rasterize_fire_mask(rows[~is_modis], tuple(tile)) > 0
        modis_fp = F.rasterize_fire_mask(rows[is_modis], tuple(tile)) > 0
        modis_only = on_water & modis_fp & ~viirs_fp
        stats.update(centres_on_water=centres_on_water,
                     centres_offshore=centres_offshore,
                     modis_only=int(modis_only.sum()))
        rep.note(f"of the {n_water} water cells, {centres_on_water} contain a detection "
                 f"centre ({centres_offshore} offshore); {int(modis_only.sum())} are "
                 f"reached only by MODIS footprints")
        if "scan" in rows.columns and is_modis.any():
            m = rows[is_modis]
            rep.note(f"MODIS footprints in this fire reach "
                     f"{m['scan'].max():.1f} x {m['track'].max():.1f} km; every cell "
                     f"they touch is labelled fire (all_touched)")
        all_fp = modis_fp | viirs_fp
        if all_fp.any():
            stats["modis_only_fire_share"] = round(
                float((all_fp & modis_fp & ~viirs_fp).sum() / all_fp.sum()), 3)
            rep.note(f"{stats['modis_only_fire_share'] * 100:.0f}% of all fire cells "
                     f"are reached only by MODIS footprints (modis_only_fire_share)")

    if misaligned:
        return stats                     # already failed, with the shift
    if centres_offshore:
        rep.check(False, f"no detections on open water ({centres_offshore} cells "
                  f"hold a detection centre over {COAST_CELLS} km from land)",
                  "detections themselves sit on the sea: label/feature offset, or an "
                  "offshore heat source (platform, ship) that passed the filter")
        return stats
    explained = ("registration passes and no detection centre lies offshore"
                 if misaligned is False and centres_offshore == 0 else
                 "only partly tested -- rebuild with --visualize so the audit CSV "
                 "exists" if rows is None else "no detection centre lies offshore")
    rep.warn(f"{n_water} fire cells on water are footprint spillover at the coast, "
             f"not misalignment",
             f"{explained}. Every cell a detection footprint touches is labelled "
             f"fire, so a fire that burns to the shore paints the first sea cells; "
             f"MODIS footprints, the largest, reach furthest. These cells teach the "
             f"model fire on the ocean.")
    return stats


# ---------------------------------------------------------------------------
# KNOWN EVENTS
# ---------------------------------------------------------------------------
def check_known_events(samples, rep, fire_dir=None, F=None) -> None:
    from datetime import timedelta
    meta = samples[0]["meta"]
    key = (str(meta.get("fire", "")).upper(), meta.get("year"))
    tests = KNOWN_EVENTS.get(key)
    if not tests:
        return
    # Older samples do not record the lag; the pipeline default has been 1.
    lag = int(meta.get("feature_lag_days", 1))
    by_label_day = {}
    for s in samples:
        by_label_day.setdefault(s["meta"].get("date"), s)
    chans = samples[0]["channels"]
    area = _fire_area(samples, fire_dir, F)
    for kind, day, a, b, why in tests:
        label_day = (datetime.strptime(day, "%Y-%m-%d")
                     + timedelta(days=lag)).strftime("%Y-%m-%d")
        s = by_label_day.get(label_day)
        where = f"conditions {day} (sample labelled {label_day}, {lag}-day feature lag)"
        if s is None:
            rep.note(f"known event: {where} not in dataset -- {why}")
            continue
        f = s["features"]
        if kind == "wind_toward":
            u = np.nanmean(f[chans.index("wind_u")])
            v = np.nanmean(f[chans.index("wind_v")])
            heading = float(np.degrees(np.arctan2(u, v)) % 360)
            off = abs((heading - a + 180) % 360 - 180)
            rep.check(off <= b, f"known event, {where}: wind heading {heading:.0f} deg "
                      f"(expected ~{a:.0f}), {np.hypot(u, v):.1f} m/s -- {why}")
        elif kind == "precip_below":
            pr = f[chans.index("weather_precip")]
            p = float(np.nanmax(pr[area] if area is not None else pr))
            scope = "over the fire" if area is not None else "in the tile"
            rep.check(p < a, f"known event, {where}: wettest cell {scope} {p:.1f} mm "
                      f"(limit {a:g}) -- {why}")
            if area is not None and np.nanmax(pr) > p:
                r, c = np.unravel_index(np.nanargmax(pr), pr.shape)
                lat, lon = _cell_latlon(samples[0]["meta"]["tile"], r, c, F)
                rep.note(f"wettest cell in the whole tile: {np.nanmax(pr):.1f} mm at "
                         f"({lat:.2f}, {lon:.2f}), outside the fire area")


def _fire_area(samples, fire_dir, F):
    """Cells within 2 km of the agency perimeter, or None if there is none."""
    if fire_dir is None or F is None:
        return None
    geometry, props = _load_perimeter(Path(fire_dir))
    if geometry is None or props.get("geometry_is_bbox"):
        return None
    area = perimeter_on_grid(geometry, samples[0]["meta"]["tile"], F, buffer_m=2000.0)
    return area if area.any() else None


def _cell_latlon(tile, r, c, F):
    from pyproj import Transformer
    inv = Transformer.from_crs(F.GRID_SPEC.crs, "EPSG:4326", always_xy=True)
    cell = F.GRID_SPEC.cell_m
    lon, lat = inv.transform(tile[0] + (c + 0.5) * cell, tile[3] - (r + 0.5) * cell)
    return lat, lon


# ---------------------------------------------------------------------------
# QA SHEETS
# ---------------------------------------------------------------------------
def _perimeter_pixels(geometry, tile, F):
    """Perimeter rings in (col, row) pixel coordinates for outlining."""
    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", F.GRID_SPEC.crs, always_xy=True)
    x0, _, _, y1 = tile
    cell = F.GRID_SPEC.cell_m
    polys = geometry["coordinates"] if geometry["type"] == "MultiPolygon" \
        else [geometry["coordinates"]]
    rings = []
    for poly in polys:
        for ring in poly[:1]:                          # outer ring
            xy = np.array([fwd.transform(lon, lat) for lon, lat in ring[:]])
            rings.append(((xy[:, 0] - x0) / cell - 0.5, (y1 - xy[:, 1]) / cell - 0.5))
    return rings


def _label_rgba(label, fire_id, FA) -> np.ndarray:
    """Label as an image: fire cells in their fire's colour, unobserved grey."""
    from matplotlib.colors import to_rgba
    img = np.zeros(label.shape + (4,))
    img[label == 2] = (0.55, 0.55, 0.6, 0.55)
    for v in np.unique(fire_id[fire_id > 0]):
        img[fire_id == v] = to_rgba(FA.fire_color(int(v)))
    return img


def draw_qa(samples, fire_dir: Path, F, max_panels: int = 30) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    qa = fire_dir / "qa"
    qa.mkdir(exist_ok=True)
    geometry, props = _load_perimeter(fire_dir)
    tile = samples[0]["meta"]["tile"]
    rings = _perimeter_pixels(geometry, tile, F) \
        if geometry and not props.get("geometry_is_bbox") else []
    # Other mapped fires in the tile, outlined dashed in their own colour.
    import fireattrib as FA
    fires = FA.load_fires(fire_dir)
    others = [(f, _perimeter_pixels(f["geometry"], tile, F)) for f in fires
              if f["kind"] == "mapped" and f.get("geometry", {}).get("type")
              in ("Polygon", "MultiPolygon")]
    elev = _ch(samples[:1], "elevation")
    shade = None
    if elev is not None and np.isfinite(elev).any():
        z = np.nan_to_num(elev[0], nan=np.nanmean(elev[0]))
        dy, dx = np.gradient(z)
        shade = np.clip(0.6 + 0.02 * (dx - dy), 0, 1)       # simple NW light

    ordered = sorted(samples, key=lambda s: s["meta"].get("window_start") or s["meta"]["date"])
    if len(ordered) > max_panels:
        # keep every step with fire, then fill evenly
        fire = [s for s in ordered if (s["label"] == F.LABEL_FIRE).any()]
        pick = fire if len(fire) <= max_panels else \
            [fire[i] for i in np.linspace(0, len(fire) - 1, max_panels).astype(int)]
        ordered = pick

    cols = 6
    rows = int(np.ceil(len(ordered) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8), squeeze=False)
    cmap = ListedColormap([(0, 0, 0, 0), (0.85, 0.1, 0.05, 1), (0.55, 0.55, 0.6, 0.55)])
    pm_i = samples[0]["channels"].index("prev_fire_mask")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, s in zip(axes.ravel(), ordered):
        if shade is not None:
            ax.imshow(shade, cmap="gray", vmin=0, vmax=1)
        prev = np.nan_to_num(s["features"][pm_i]) > 0
        ax.contour(prev, levels=[0.5], colors="orange", linewidths=0.6)
        if s.get("fire_id") is None:
            ax.imshow(s["label"], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        else:
            ax.imshow(_label_rgba(s["label"], s["fire_id"], FA), interpolation="nearest")
        for cx, cy in rings:
            ax.plot(cx, cy, color="cyan", lw=0.8)
        for f, frings in others:
            for cx, cy in frings:
                ax.plot(cx, cy, color=f["color"], lw=0.7, ls="--")
        key = s["meta"].get("window_start") or s["meta"]["date"]
        ax.set_title(f"{key}\n{s['meta'].get('detections', '?')} det, "
                     f"{int((s['label'] == 1).sum())} fire", fontsize=7)
    present = sorted({int(v) for s in ordered if s.get("fire_id") is not None
                      for v in np.unique(s["fire_id"]) if v})
    fire_key = ("fire label, coloured by fire (key below)" if len(present) > 1
                else "red: fire label")
    fig.suptitle(f"{samples[0]['meta'].get('fire')} {samples[0]['meta'].get('year')} -- "
                 f"{fire_key}   grey: unobserved   orange: prev_fire_mask   "
                 "cyan: agency perimeter", fontsize=9)
    if len(present) > 1:
        from matplotlib.patches import Patch
        names = {f["fire_id"]: f["name"] for f in fires}
        fig.legend(handles=[Patch(color=FA.fire_color(v),
                                  label=f"{v} {names.get(v, '?')}") for v in present[:12]],
                   loc="lower center", ncol=min(6, len(present)), fontsize=7,
                   frameon=False, title="fire label colour = fire_id (dashed: its perimeter)",
                   title_fontsize=7)
    foot = 0.0 if len(present) < 2 else 0.5 + 0.2 * ((len(present[:12]) - 1) // 6)
    fig.tight_layout(rect=(0, foot / fig.get_size_inches()[1], 1, 1))
    labels_png = qa / "qa_labels.png"
    fig.savefig(labels_png, dpi=110)
    plt.close(fig)

    # Channels of the peak day, each with its own scale and units.
    peak = max(samples, key=lambda s: int((s["label"] == 1).sum()))
    chans = peak["channels"]
    n = len(chans)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.8), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    fire = peak["label"] == 1
    for ax, (i, name) in zip(axes.ravel(), enumerate(chans)):
        x = peak["features"][i]
        cm = {"landcover": "tab10", "prev_fire_mask": "Reds", "elevation": "terrain",
              "vegetation": "YlGn", "weather_temp": "inferno", "erc": "magma",
              "drought": "BrBG"}.get(name, "viridis")
        im = ax.imshow(x, cmap=cm)
        if fire.any():
            ax.contour(fire, levels=[0.5], colors="red", linewidths=0.5)
        if name == "wind_speed" and "wind_u" in chans:
            u = peak["features"][chans.index("wind_u")]
            v = peak["features"][chans.index("wind_v")]
            yy, xx = np.mgrid[4:64:8, 4:64:8]
            # angles="uv" draws the arrow in SCREEN space: +v is up regardless
            # of imshow's inverted y axis. The grid is north-up, so pass v as
            # is. Negating it here drew every wind arrow mirrored north-south.
            ax.quiver(xx, yy, u[4::8, 4::8], v[4::8, 4::8], color="white",
                      angles="uv", scale=None)
        fin = x[np.isfinite(x)]
        rng = f"{fin.min():.3g} .. {fin.max():.3g}" if fin.size else "all missing"
        ax.set_title(f"{name}\n({UNITS.get(name, '')})  {rng}", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.tick_params(labelsize=6)
    key = peak["meta"].get("window_start") or peak["meta"]["date"]
    fig.suptitle(f"{peak['meta'].get('fire')} -- all channels on the peak step {key} "
                 "(red outline: fire label; arrows: wind, pointing where it blows)",
                 fontsize=9)
    fig.tight_layout()
    channels_png = qa / "qa_channels.png"
    fig.savefig(channels_png, dpi=110)
    plt.close(fig)
    return [labels_png, channels_png]


def run(samples, fire_dir: Path, rep, F, plots: bool = True) -> dict:
    """All reality checks for one fire."""
    check_physics(samples, rep)
    check_orientation(samples, rep)
    check_wind(samples, rep)
    check_static_and_hold(samples, rep)
    truth = check_ground_truth(samples, fire_dir, rep, F)
    check_known_events(samples, rep, fire_dir, F)
    if plots:
        try:
            for p in draw_qa(samples, fire_dir, F):
                rep.note(f"QA sheet: {p}")
        except Exception as err:
            rep.warn(f"QA sheets not drawn: {str(err).splitlines()[0][:120]}")
    return truth
