#!/usr/bin/env python3
"""
firegrid -- the common-grid resampling stage.

Turns the ragged per-layer arrays the extraction code produces into uniform
(channels, 64, 64) tensors that a CNN can actually consume, one per fire-day.

THE GRID SPEC
-------------
Every decision below is a choice, not a law. They are collected in GRID_SPEC
so they can be changed in one place and cited in a write-up.

  CRS         EPSG:3310 (California Albers, equal-area, metres)
              Not EPSG:4326. At California latitudes a "1 km" cell in lat/lon
              is ~1.00 km tall and ~0.83 km wide, and that anisotropy changes
              between San Diego and the Oregon border. A convolution kernel
              would therefore mean something different in the north than in
              the south, and cell "area" would not be constant -- which
              matters the moment you compute burned area per cell.

  cell        1000 m. Compromise between 90 m terrain and 4000 m GRIDMET.
              Matches Huot et al., "Next Day Wildfire Spread" (IEEE TGRS 60,
              2022), so results are comparable to a published benchmark.

  tile        64 x 64 cells = 64 km x 64 km, fire centred. 4096 cells is far
              under the sampleRectangle cap, so no silent coarsening.

  origin      Tiles snap to a global grid anchored at (0, 0) in EPSG:3310.
              Two overlapping fires therefore share cell boundaries exactly,
              which makes tiles comparable, cacheable and mergeable. Centring
              each tile on its own fire centroid without snapping would give
              every sample a different sub-cell offset.

RESAMPLING RULES
----------------
One method per layer type, not one method for everything:

  continuous    area-weighted mean (reduceResolution) when downsampling,
                bilinear when upsampling from coarser native data.
  categorical   majority class (mode). Land cover must never be averaged --
                the mean of classes 3 and 7 is class 5, an invented category.
  circular      decomposed to sin/cos, averaged separately, kept as two
                channels. Aspect 350 deg and 10 deg are 20 deg apart but
                average to 180 deg, the exact opposite direction. Wind
                direction already gets this treatment upstream as wind_u and
                wind_v; aspect did not, and now does.
  terrain       slope and aspect are derived from the DEM at its native 30 m
                and only then aggregated. Deriving slope from an
                already-coarsened elevation grid measures the slope of a
                smoothed landscape, which is systematically too flat.

LABELS
------
Three classes, not two:

  0  no fire      observed, no detection
  1  fire         detection footprint intersects the cell
  2  unobserved   coverage gap; excluded from the loss

The third class exists because absence of a FIRMS detection is not evidence
of absence of fire -- it can be cloud, smoke, or an overpass gap. Forcing
those cells to 0 teaches the model satellite coverage patterns rather than
fire behaviour. NDWS makes the same distinction.

See mark_unobserved() for the coverage heuristic and its limitations; it is
the weakest link in this module and is deliberately easy to replace.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

def _import_pipeline():
    """Import the main pipeline module regardless of its version suffix.

    The file gets renamed as it is revised (v2 -> v3 -> ...), and hardcoding
    one name here means every rename silently breaks the import and surfaces
    as a confusing "could not be imported" further downstream. Tries the
    known names, then any visualize_firms_dataset_v*.py sitting next to this
    file, newest suffix first.
    """
    import importlib
    import re

    candidates = ["visualize_firms_dataset_v3", "visualize_firms_dataset_v2",
                  "visualize_firms_dataset"]
    here = Path(__file__).resolve().parent
    found = sorted(
        (p.stem for p in here.glob("visualize_firms_dataset_v*.py")),
        key=lambda n: int(m.group(1)) if (m := re.search(r"_v(\d+)$", n)) else 0,
        reverse=True,
    )
    errors = []
    for name in list(dict.fromkeys(found + candidates)):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as err:
            if err.name == name:
                continue          # this filename does not exist; try the next
            errors.append(f"{name}: {err}")   # a DEPENDENCY is missing
        except Exception as err:
            errors.append(f"{name}: {type(err).__name__}: {err}")
    raise ImportError(
        "could not import the pipeline module from "
        f"{here}\n  " + ("\n  ".join(errors) if errors else
                         "no visualize_firms_dataset*.py found there")
    )


V = _import_pipeline()

# Geo stack. Imported lazily-ish so the module can be inspected without them.
try:
    import rasterio.features
    from rasterio.transform import from_origin
    from pyproj import Transformer
    from shapely.geometry import box, mapping
    from shapely.ops import transform as shp_transform
    HAS_GEO = True
except ImportError:  # pragma: no cover - exercised only on bare installs
    HAS_GEO = False


# ---------------------------------------------------------------------------
# Grid specification
# ---------------------------------------------------------------------------
LABEL_NO_FIRE = 0
LABEL_FIRE = 1
LABEL_UNOBSERVED = 2


@dataclass(frozen=True)
class GridSpec:
    crs: str = "EPSG:3310"
    cell_m: float = 1000.0
    tile_cells: int = 64
    fill_value: float = np.nan

    @property
    def tile_m(self) -> float:
        return self.cell_m * self.tile_cells

    @property
    def shape(self) -> tuple[int, int]:
        return (self.tile_cells, self.tile_cells)


GRID_SPEC = GridSpec()

# How each layer is aggregated onto the grid.
# The band name every fetched layer is renamed to before sampling, so the
# pixel array can be found by name in a response that also contains metadata.
VALUE_BAND = "value"

CONTINUOUS = "continuous"
CATEGORICAL = "categorical"
CIRCULAR = "circular"

LAYER_RESAMPLING: dict[str, str] = {
    "elevation": CONTINUOUS,
    "slope": CONTINUOUS,
    "aspect": CIRCULAR,
    "vegetation": CONTINUOUS,
    "landcover": CATEGORICAL,
    "population": CONTINUOUS,
    "drought": CONTINUOUS,
    "humidity": CONTINUOUS,
    "weather_temp": CONTINUOUS,
    "weather_precip": CONTINUOUS,
    "wind_speed": CONTINUOUS,
    "wind_u": CONTINUOUS,
    "wind_v": CONTINUOUS,
    "erc": CONTINUOUS,
    "vpd": CONTINUOUS,
}

# Layers whose source product has a cadence coarser than one day. Asking for
# a single day returns an EMPTY collection, whose reduction is a zero-band
# image, and select(0) on that fails with "Input only contains 0 bands".
# GRIDMET/DROUGHT is a 5-day pentad product, which is why a daily request
# succeeded on exactly every fifth day and failed on the other four.
LAYER_LOOKBACK_DAYS: dict[str, int] = {
    "drought": 20,       # pentad cadence, plus slack for late publication
    "vegetation": 24,    # VNP13A1 is a 16-day composite
}

# Native resolution in metres, used to decide aggregate-vs-interpolate.
LAYER_NATIVE_M: dict[str, float] = {
    "elevation": 30, "slope": 30, "aspect": 30,
    "vegetation": 500, "landcover": 10, "population": 927,
    "drought": 4000, "humidity": 4000, "weather_temp": 4000,
    "weather_precip": 4000, "wind_speed": 4000, "wind_u": 4000,
    "wind_v": 4000, "erc": 4000, "vpd": 4000,
}

# Channel order in the exported tensor. Fixed, because a model trained on one
# ordering silently mispredicts on another. aspect becomes two channels;
# prev_fire_mask is the single most predictive input for next-day spread.
CHANNELS: tuple[str, ...] = (
    "elevation", "slope", "aspect_sin", "aspect_cos", "aspect_consistency",
    "vegetation", "landcover", "population",
    "drought", "humidity", "weather_temp", "weather_precip",
    "wind_speed", "wind_u", "wind_v", "erc", "vpd",
    "prev_fire_mask",
)


# ---------------------------------------------------------------------------
# Tile geometry
# ---------------------------------------------------------------------------
def _transformers(spec: GridSpec = GRID_SPEC):
    """lon/lat -> projected, and back."""
    fwd = Transformer.from_crs("EPSG:4326", spec.crs, always_xy=True)
    inv = Transformer.from_crs(spec.crs, "EPSG:4326", always_xy=True)
    return fwd, inv


def snap_tile(
    lon: float, lat: float, spec: GridSpec = GRID_SPEC
) -> tuple[float, float, float, float]:
    """Tile bounds in projected metres, snapped to the global cell grid.

    Snapping is what makes two tiles comparable: an unsnapped tile centred on
    an arbitrary centroid puts cell boundaries at an arbitrary offset, so the
    same patch of ground lands in different cells in different samples.
    """
    fwd, _ = _transformers(spec)
    cx, cy = fwd.transform(lon, lat)
    half = spec.tile_m / 2.0
    # Snap the tile ORIGIN (not the centre) to the grid.
    x0 = np.floor((cx - half) / spec.cell_m) * spec.cell_m
    y0 = np.floor((cy - half) / spec.cell_m) * spec.cell_m
    return (x0, y0, x0 + spec.tile_m, y0 + spec.tile_m)


def tile_transform(tile: tuple[float, float, float, float], spec: GridSpec = GRID_SPEC):
    """Affine transform for the tile, north-up (row 0 is the top / max y)."""
    x0, _y0, _x1, y1 = tile
    return from_origin(x0, y1, spec.cell_m, spec.cell_m)


def tile_bounds_lonlat(
    tile: tuple[float, float, float, float], spec: GridSpec = GRID_SPEC
) -> tuple[float, float, float, float]:
    """Tile bounds as (w, s, e, n) in degrees -- the envelope, for EE requests.

    The projected tile is a rectangle in 3310 and therefore NOT a rectangle in
    lon/lat; this returns a slightly larger envelope, which is what you want
    when asking an upstream service for enough data to cover the tile.
    """
    _, inv = _transformers(spec)
    x0, y0, x1, y1 = tile
    xs, ys = [], []
    for x in (x0, x1):
        for y in (y0, y1):
            lon, lat = inv.transform(x, y)
            xs.append(lon)
            ys.append(lat)
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# Automated fire-event discovery  (replaces hardcoded bboxes)
# ---------------------------------------------------------------------------
@dataclass
class FireEvent:
    event_id: str
    centroid_lon: float
    centroid_lat: float
    start_date: str
    end_date: str
    n_detections: int
    tile: tuple[float, float, float, float] = field(default=None)

    def to_row(self) -> dict:
        d = asdict(self)
        d["tile"] = list(self.tile) if self.tile else None
        return d


def discover_fire_events(
    df: pd.DataFrame,
    spec: GridSpec = GRID_SPEC,
    eps_km: float = 2.0,
    min_detections: int = 10,
    time_days: float | None = None,
) -> list[FireEvent]:
    """Derive fire events from FIRMS detections instead of hardcoding bboxes.

    Space-time DBSCAN (already in the pipeline) groups detections into events;
    each event becomes one snapped tile centred on its centroid. This is the
    automated replacement for CALIFORNIA_BBOX / PALISADES_BBOX / EATON_BBOX.

    min_detections filters out one-pixel noise. Note the tension with the
    upstream keep_singletons rationale: an isolated detection is a new
    ignition and scientifically interesting, but a 64 km tile built around a
    single pixel is almost all no-fire and will swamp the class balance.
    Default of 10 keeps events worth learning from; lower it deliberately.

    time_days defaults to None -- SPACE-ONLY clustering -- which is the
    opposite of what the plotting code wants and is deliberate. An event is a
    place that burned over some span of days; the day axis is handled by
    building one sample per day within the event. Space-time clustering here
    would shatter a week-long fire into one "event" per day (the time axis
    scales at spread_km_per_day, so consecutive days sit far beyond eps), and
    each fragment would get its own near-identical tile.
    """
    df = V.add_acq_datetime(df)
    # cluster_fire_points_to_polygons returns nothing without footprint
    # corners, and a CSV read straight off disk will not have them. Add them
    # here rather than making every caller remember.
    if not all(c in df.columns for c in V.FOOTPRINT_COLUMNS):
        df = V.add_footprint_columns(df)
    _polys, centroids = V.cluster_fire_points_to_polygons(
        df, eps_km=eps_km, min_samples=2, time_days=time_days, keep_singletons=True
    )
    if centroids.empty:
        return []

    events: list[FireEvent] = []
    for _, row in centroids.iterrows():
        n = int(row.get("n_detections", 0))
        if n < min_detections:
            continue
        # Per-cluster dates come from the centroid table, which the clustering
        # already computed. Re-deriving them by filtering df would need a
        # cluster_id column that df does not carry, and silently fell back to
        # the whole frame's date range when it was missing.
        start = str(row.get("first_detection", "") or "")[:10]
        end = str(row.get("last_detection", "") or "")[:10]
        if not start:
            start = end = str(row.get("first_local_date", "") or "")[:10]
        lon, lat = float(row["centroid_lon"]), float(row["centroid_lat"])
        events.append(FireEvent(
            event_id=f"evt_{int(row['cluster_id'])}",
            centroid_lon=lon,
            centroid_lat=lat,
            start_date=start,
            end_date=end,
            n_detections=n,
            tile=snap_tile(lon, lat, spec),
        ))
    events.sort(key=lambda e: -e.n_detections)
    return events


# ---------------------------------------------------------------------------
# Label rasterization
# ---------------------------------------------------------------------------
def _footprint_shapes(day_df: pd.DataFrame, spec: GridSpec):
    """Detection footprints as projected polygons.

    Uses the true VIIRS/MODIS footprint corners when present rather than
    buffering the centroid, so a 375 m nadir pixel and a 1.5 km edge-of-scan
    pixel are not treated as the same size.
    """
    fwd, _ = _transformers(spec)
    project = lambda geom: shp_transform(  # noqa: E731
        lambda xx, yy, zz=None: fwd.transform(xx, yy), geom
    )
    have_corners = all(c in day_df.columns for c in V.FOOTPRINT_COLUMNS)
    if have_corners:
        polys = V.footprint_polygons(day_df)
        return [project(p) for p in polys if p is not None and not p.is_empty]
    # Fallback: square of one cell centred on the detection.
    half = spec.cell_m / 2.0
    out = []
    for lon, lat in zip(day_df["longitude"], day_df["latitude"]):
        x, y = fwd.transform(lon, lat)
        out.append(box(x - half, y - half, x + half, y + half))
    return out


def rasterize_fire_mask(
    day_df: pd.DataFrame,
    tile: tuple[float, float, float, float],
    spec: GridSpec = GRID_SPEC,
) -> np.ndarray:
    """Burn detection footprints onto the tile grid as a binary fire mask.

    all_touched=True: a cell is fire if any part of a footprint falls in it.
    The alternative (centre-only) would drop small fires entirely at 1 km,
    and under-calling fire is the more costly error here.
    """
    mask = np.zeros(spec.shape, dtype=np.uint8)
    if day_df.empty:
        return mask
    shapes = _footprint_shapes(day_df, spec)
    if not shapes:
        return mask
    burned = rasterio.features.rasterize(
        [(mapping(s), 1) for s in shapes],
        out_shape=spec.shape,
        transform=tile_transform(tile, spec),
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )
    return burned


def mark_unobserved(
    fire_mask: np.ndarray,
    prev_mask: np.ndarray,
    day_had_detections: bool,
    spec: GridSpec = GRID_SPEC,
) -> np.ndarray:
    """Promote no-fire cells to 'unobserved' where coverage is doubtful.

    HEURISTIC, and the weakest part of this module. FIRMS does not publish
    per-cell overpass or cloud coverage, so true observability is unknown.
    The rule implemented is deliberately conservative:

      if the tile recorded no detections at all on this day, but the previous
      day had fire, treat the previously-burning cells as unobserved rather
      than as extinguished.

    A fire that genuinely goes out looks identical to one hidden under smoke,
    and calling the second case "no fire" trains the model to predict
    satellite gaps. Marking those cells unobserved costs a little data and
    removes a systematic bias.

    To do better, join against per-overpass swath geometry or a cloud mask
    (e.g. MODIS MOD35) -- replace this function, not its callers.
    """
    label = fire_mask.astype(np.uint8).copy()
    if not day_had_detections and prev_mask is not None and prev_mask.any():
        label[(label == LABEL_NO_FIRE) & (prev_mask == LABEL_FIRE)] = LABEL_UNOBSERVED
    return label


# ---------------------------------------------------------------------------
# Feature resampling
# ---------------------------------------------------------------------------
def circular_to_components(
    degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a circular field into sin/cos components before any averaging.

    Averaging degrees directly is wrong: mean(350, 10) == 180, pointing
    opposite to both inputs. Averaging the components and recombining gives 0.
    """
    rad = np.radians(np.asarray(degrees, dtype=float))
    return np.sin(rad), np.cos(rad)


# Below this, the averaged direction is noise rather than signal and the
# unit components are zeroed instead of amplified by the division.
CIRCULAR_MIN_R = 0.05


def split_circular_components(
    stacked: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split averaged sin/cos into unit direction plus concentration R.

    Averaging sin and cos over the ~1100 native cells inside one 1 km cell
    gives a resultant whose LENGTH is the circular concentration R (0..1), not
    1. R = 1 means every sub-cell faces the same way; R near 0 means they
    cancel -- a ridge crest, or flat ground where aspect is meaningless.

    Keeping the raw components would hand the model direction and consistency
    multiplied together in two channels. Splitting them gives it a clean
    direction plus an explicit "how much should I trust this direction",
    which is also a serviceable terrain-roughness proxy.

    Returns (sin_unit, cos_unit, R).
    """
    sin_c, cos_c = np.asarray(stacked[0], float), np.asarray(stacked[1], float)
    r = np.sqrt(sin_c ** 2 + cos_c ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sin_u = np.where(r > CIRCULAR_MIN_R, sin_c / r, 0.0)
        cos_u = np.where(r > CIRCULAR_MIN_R, cos_c / r, 0.0)
    sin_u[~np.isfinite(sin_c)] = np.nan
    cos_u[~np.isfinite(cos_c)] = np.nan
    r[~np.isfinite(sin_c)] = np.nan
    return sin_u, cos_u, np.clip(r, 0.0, 1.0)


def resample_to_grid(
    arr: np.ndarray,
    src_bounds: tuple[float, float, float, float],
    tile: tuple[float, float, float, float],
    method: str,
    spec: GridSpec = GRID_SPEC,
) -> np.ndarray:
    """Reproject one already-fetched array (lon/lat bounds) onto the tile grid.

    This is the local fallback path, used for arrays that came back from
    sampleRectangle. The preferred path is ee_layer_on_grid(), which does the
    aggregation server-side at native resolution; this one can only work with
    what was already sampled, so it cannot recover detail lost upstream.
    """
    from rasterio.warp import reproject, Resampling

    w, s, e, n = src_bounds
    rows, cols = arr.shape
    src_transform = from_origin(w, n, (e - w) / cols, (n - s) / rows)
    resampling = {
        CONTINUOUS: Resampling.average,
        CATEGORICAL: Resampling.mode,
        CIRCULAR: Resampling.average,   # caller splits into components first
    }[method]

    dst = np.full(spec.shape, spec.fill_value, dtype="float64")
    reproject(
        source=np.asarray(arr, dtype="float64"),
        destination=dst,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=tile_transform(tile, spec),
        dst_crs=spec.crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def ee_layer_on_grid(
    layer_key: str,
    tile: tuple[float, float, float, float],
    spec: GridSpec = GRID_SPEC,
    date_str: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    time_mode: str = "daily",
    ee_project: str | None = None,
) -> np.ndarray | None:
    """Fetch one layer already aggregated onto the target grid, server-side.

    Aggregating in Earth Engine at native resolution is strictly better than
    fetching a coarse array and resampling locally: reduceResolution averages
    every 30 m DEM cell inside each 1 km output cell, whereas a local resample
    can only work from whatever was already sampled.

    Returns a (tile_cells, tile_cells) float array, or None on failure.
    """
    # ee_project=None is fine: _init_earth_engine resolves it via
    # V.resolve_ee_project (--ee-project -> $EE_PROJECT -> DEFAULT_EE_PROJECT).
    if not V.HAS_EARTH_ENGINE or not V._init_earth_engine(ee_project):
        return None

    # Coarse-cadence products get a lookback window instead of a single day.
    lookback = LAYER_LOOKBACK_DAYS.get(layer_key)
    if lookback and time_mode == "daily" and date_str:
        window_start = (datetime.strptime(date_str, "%Y-%m-%d")
                        - timedelta(days=lookback)).strftime("%Y-%m-%d")
        img = V._get_ee_image_for_layer(
            layer_key, date_str=date_str, start_date=window_start,
            end_date=date_str, time_mode="period",
        )
    else:
        img = V._get_ee_image_for_layer(
            layer_key, date_str=date_str, start_date=start_date,
            end_date=end_date, time_mode=time_mode,
        )
    if img is None:
        return None

    method = LAYER_RESAMPLING.get(layer_key, CONTINUOUS)
    native = LAYER_NATIVE_M.get(layer_key, spec.cell_m)

    # Everything from here to getInfo() can raise, and the informative
    # failures are server-side: a zero-band composite blows up at select(0),
    # not at the fetch. One guard around the whole chain keeps every EE error
    # on the same reporting path.
    try:
        return _fetch_layer_array(img, layer_key, tile, spec, method, native)
    except Exception as ex:
        msg = str(ex)
        hint = ""
        if "0 bands" in msg:
            hint = (f" -- no imagery in the window; consider raising "
                    f"LAYER_LOOKBACK_DAYS['{layer_key}']")
        elif "default projection" in msg:
            hint = " -- composite without a fixed projection"
        print(f"  grid fetch failed ({layer_key}): "
              f"{msg.splitlines()[0][:160]}{hint}")
        return None


def _fetch_layer_array(img, layer_key, tile, spec, method, native):
    """Resample one image onto the tile and pull it back as an array."""
    ee = V.ee
    # Reduce to exactly one band and give it a name we choose. Both halves
    # matter: _get_ee_image_for_layer may hand back a multi-band image, and
    # the band name is how the pixel array is located in the response below.
    img = img.select(0).rename(VALUE_BAND)

    if method == CIRCULAR:
        rad = img.multiply(np.pi / 180.0)
        img = rad.sin().rename("sin").addBands(rad.cos().rename("cos"))

    if native < spec.cell_m:
        # reduceResolution needs a fixed input projection, and an image built
        # by reducing an ImageCollection does not have one -- the composite
        # carries EE's default 1-degree projection instead. Asserting the
        # product's real native grid here is what makes the aggregation
        # meaningful; without it EE raises "does not have a valid default
        # projection".
        img = img.setDefaultProjection(crs="EPSG:4326", scale=native)
        reducer = ee.Reducer.mode() if method == CATEGORICAL else ee.Reducer.mean()
        img = img.reduceResolution(
            reducer=reducer,
            maxPixels=4096,          # (1000/30)^2 ~ 1111 for SRTM
            bestEffort=True,
        )
    else:
        img = img.resample("bilinear" if method != CATEGORICAL else "nearest")

    img = img.reproject(crs=spec.crs, scale=spec.cell_m)

    x0, y0, x1, y1 = tile
    region = ee.Geometry.Rectangle(
        [x0, y0, x1, y1], proj=ee.Projection(spec.crs), geodesic=False
    )
    info = img.unmask(-9999).sampleRectangle(
        region=region, defaultValue=-9999
    ).getInfo()
    if not info or "properties" not in info:
        return None

    # sampleRectangle returns the pixel arrays ALONGSIDE every metadata
    # property the image carries -- SRTM, for instance, ships an HTML
    # 'description'. Select by band name; never by position in properties.
    props = info["properties"]
    bands = ["sin", "cos"] if method == CIRCULAR else [VALUE_BAND]
    out = []
    for band in bands:
        raw = props.get(band)
        if raw is None:
            raw = _find_pixel_array(props)
            if raw is None:
                print(f"  grid fetch ({layer_key}): band '{band}' not in response; "
                      f"properties present: {sorted(props)[:8]}")
                return None
        try:
            a = np.asarray(raw, dtype=float)
        except (ValueError, TypeError) as ex:
            print(f"  grid fetch ({layer_key}): property '{band}' is not numeric "
                  f"pixel data ({ex.__class__.__name__})")
            return None
        a[a == -9999] = np.nan
        out.append(fit_to_shape(a, spec))
    return np.stack(out) if method == CIRCULAR else out[0]


def _find_pixel_array(props: dict):
    """Fallback: locate the one property that looks like a 2-D pixel array.

    Used only when the expected band name is absent, so a future rename
    upstream degrades to a warning path rather than an exception.
    """
    for value in props.values():
        if (isinstance(value, list) and value
                and isinstance(value[0], list) and value[0]
                and isinstance(value[0][0], (int, float))):
            return value
    return None


def fit_to_shape(arr: np.ndarray, spec: GridSpec = GRID_SPEC) -> np.ndarray:
    """Crop or pad to exactly the tile shape.

    sampleRectangle can return one row or column more or fewer than asked
    depending on how the region lands on the projection's pixel grid. Silently
    accepting that would make tensors of inconsistent shape, so it is
    corrected here and the discrepancy is bounded to a single cell.
    """
    target = spec.tile_cells
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    out = np.full(spec.shape, spec.fill_value, dtype=float)
    r = min(arr.shape[0], target)
    c = min(arr.shape[1], target)
    out[:r, :c] = arr[:r, :c]
    return out


# ---------------------------------------------------------------------------
# Sample assembly
# ---------------------------------------------------------------------------
def assemble_sample(
    features: dict[str, np.ndarray],
    label: np.ndarray,
    spec: GridSpec = GRID_SPEC,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack features into (channels, H, W) in the fixed CHANNELS order.

    Missing channels become all-fill rather than being dropped, so every
    sample has the same depth and a model never sees a shifted channel axis.
    """
    stack = np.full((len(CHANNELS), *spec.shape), spec.fill_value, dtype="float32")
    for i, name in enumerate(CHANNELS):
        arr = features.get(name)
        if arr is None:
            continue
        stack[i] = fit_to_shape(arr, spec).astype("float32")
    return stack, label.astype("uint8")


def save_sample(
    out_dir: Path,
    sample_id: str,
    stack: np.ndarray,
    label: np.ndarray,
    meta: dict,
) -> Path:
    """One compressed .npz per fire-day, with the spec embedded.

    The spec travels with the data on purpose: a tensor whose projection and
    cell size are only recorded in a README becomes unusable the moment the
    README drifts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sample_id}.npz"
    np.savez_compressed(
        path,
        features=stack,
        label=label,
        channels=np.array(CHANNELS),
        meta=np.array(json.dumps(meta)),
    )
    return path


def load_sample(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Read a sample back, including its channel names and spec."""
    with np.load(path, allow_pickle=False) as z:
        return (
            z["features"], z["label"], list(z["channels"]),
            json.loads(str(z["meta"])),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Grid spec inspection and event discovery")
    ap.add_argument("--csv", type=str, help="FIRMS CSV to discover events from")
    ap.add_argument("--min-detections", type=int, default=10)
    ap.add_argument("--out", type=str, default=None, help="Write events to this CSV")
    args = ap.parse_args()

    spec = GRID_SPEC
    print(f"Grid spec: {spec.crs}, {spec.cell_m:.0f} m cells, "
          f"{spec.tile_cells}x{spec.tile_cells} = {spec.tile_m / 1000:.0f} km tiles")
    print(f"Channels ({len(CHANNELS)}): {', '.join(CHANNELS)}")
    print(f"Labels: {LABEL_NO_FIRE}=no fire, {LABEL_FIRE}=fire, "
          f"{LABEL_UNOBSERVED}=unobserved")

    if not args.csv:
        return
    df = pd.read_csv(args.csv)
    events = discover_fire_events(df, spec, min_detections=args.min_detections)
    print(f"\n{len(events)} events with >= {args.min_detections} detections:")
    for e in events[:20]:
        print(f"  {e.event_id:<12} {e.n_detections:>6} det  "
              f"{e.start_date}..{e.end_date}  "
              f"({e.centroid_lat:.3f}, {e.centroid_lon:.3f})")
    if args.out:
        pd.DataFrame([e.to_row() for e in events]).to_csv(args.out, index=False)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
