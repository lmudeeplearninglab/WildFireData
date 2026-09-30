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
STATIC_CHANNELS = ("elevation", "slope", "aspect_sin", "aspect_cos",
                   "aspect_consistency", "landcover", "population")
DAILY_CHANNELS = ("vegetation", "drought", "humidity", "weather_temp",
                  "weather_precip", "wind_speed", "wind_u", "wind_v", "erc", "vpd")

# Known-answer tests from the historical record. Each is something any
# report on the event states plainly, so a failure means the DATA is wrong
# (or the claim is -- check both before trusting either).
KNOWN_EVENTS = {
    ("PALISADES", 2025): [
        ("wind_toward", "2025-01-08", 225.0, 70.0,
         "Santa Ana event: NE wind, so vectors point south-west"),
        ("precip_below", "2025-01-08", 1.0, None,
         "no measurable rain during the Santa Ana event"),
        ("pdsi_below", "2025-01-08", 0.0, None,
         "Southern California was in drought in January 2025"),
    ],
    ("EATON", 2025): [
        ("wind_toward", "2025-01-08", 225.0, 70.0,
         "Santa Ana event: NE wind, so vectors point south-west"),
        ("precip_below", "2025-01-08", 1.0, None,
         "no measurable rain during the Santa Ana event"),
        ("pdsi_below", "2025-01-08", 0.0, None,
         "Southern California was in drought in January 2025"),
    ],
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
    first_fire = None
    for s in sorted(samples, key=lambda s: s["meta"].get("window_start") or s["meta"]["date"]):
        hit = s["label"] == F.LABEL_FIRE
        if hit.any() and first_fire is None:
            first_fire = s["meta"]["date"]
        fire_any |= hit

    inside = perimeter_on_grid(geometry, tile, F)
    # 2 km covers the VIIRS/MODIS footprint, geolocation error and
    # short-range spotting beyond the final mapped line.
    near = perimeter_on_grid(geometry, tile, F, buffer_m=2000.0)

    n_fire = int(fire_any.sum())
    if n_fire == 0:
        rep.check(False, "labels contain fire at all")
        return {}
    in_share = float((fire_any & near).sum() / n_fire)
    coverage = float((fire_any & inside).sum() / max(1, inside.sum()))

    if in_share >= 0.8:
        rep.check(True, f"labels sit on the mapped burn area: {in_share * 100:.0f}% of "
                  f"fire cells within 2 km of the {props.get('source', 'agency perimeter')}")
    else:
        rep.warn(f"only {in_share * 100:.0f}% of fire cells are near the mapped perimeter",
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

    water = _ch(samples[:1], "landcover")
    if water is not None:
        on_water = (fire_any & (water[0] == ESA_WATER)).sum()
        rep.check(on_water <= max(1, 0.02 * n_fire),
                  f"no fire on open water ({int(on_water)} cells)",
                  "" if on_water <= max(1, 0.02 * n_fire) else
                  "labels and features are misaligned -- fire sits on the ocean")
    return {"inside_share": round(in_share, 3), "coverage": round(coverage, 3),
            "first_fire": first_fire}


# ---------------------------------------------------------------------------
# KNOWN EVENTS
# ---------------------------------------------------------------------------
def check_known_events(samples, rep) -> None:
    meta = samples[0]["meta"]
    key = (str(meta.get("fire", "")).upper(), meta.get("year"))
    tests = KNOWN_EVENTS.get(key)
    if not tests:
        return
    by_day = {}
    for s in samples:
        by_day.setdefault(s["meta"].get("date"), s)
    chans = samples[0]["channels"]
    for kind, day, a, b, why in tests:
        s = by_day.get(day)
        if s is None:
            rep.note(f"known event {day} not in dataset: {why}")
            continue
        f = s["features"]
        if kind == "wind_toward":
            u = np.nanmean(f[chans.index("wind_u")])
            v = np.nanmean(f[chans.index("wind_v")])
            heading = float(np.degrees(np.arctan2(u, v)) % 360)
            off = abs((heading - a + 180) % 360 - 180)
            rep.check(off <= b, f"known event {day}: wind heading {heading:.0f} deg "
                      f"(expected ~{a:.0f}), {np.hypot(u, v):.1f} m/s -- {why}")
        elif kind == "precip_below":
            p = float(np.nanmax(f[chans.index("weather_precip")]))
            rep.check(p < a, f"known event {day}: max precip {p:.1f} mm -- {why}")
        elif kind == "pdsi_below":
            d = float(np.nanmean(f[chans.index("drought")]))
            rep.check(d < a, f"known event {day}: PDSI {d:+.1f} -- {why}")


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
        ax.imshow(s["label"], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        for cx, cy in rings:
            ax.plot(cx, cy, color="cyan", lw=0.8)
        key = s["meta"].get("window_start") or s["meta"]["date"]
        ax.set_title(f"{key}\n{s['meta'].get('detections', '?')} det, "
                     f"{int((s['label'] == 1).sum())} fire", fontsize=7)
    fig.suptitle(f"{samples[0]['meta'].get('fire')} {samples[0]['meta'].get('year')} -- "
                 "red: fire label   grey: unobserved   orange: prev_fire_mask   "
                 "cyan: agency perimeter", fontsize=9)
    fig.tight_layout()
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
    check_known_events(samples, rep)
    if plots:
        try:
            for p in draw_qa(samples, fire_dir, F):
                rep.note(f"QA sheet: {p}")
        except Exception as err:
            rep.warn(f"QA sheets not drawn: {str(err).splitlines()[0][:120]}")
    return truth
