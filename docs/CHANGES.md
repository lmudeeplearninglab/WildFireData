# Changes applied

Every item maps to a finding ID in `REVIEW.md`. Verified with `pytest -q
test_firms.py` (33 tests), `pyflakes` (clean), and an end-to-end smoke run on
synthetic detections. **The Earth Engine code paths could not be executed
here** — no EE credentials in this environment — so those changes are reviewed
and type-consistent but not runtime-verified. Flagged individually below.

---

## Correctness

| ID | Change |
|---|---|
| A-1 | Hardcoded MAP_KEY removed. `get_map_key()` reads `FIRMS_MAP_KEY`, prompts only on a TTY, and raises `SystemExit` under cron/CI instead of hanging on stdin. A test fails if any 32-hex key literal reappears. |
| A-2 | Confidence filter handles both encodings. Numeric MODIS values map through `MODIS_CONFIDENCE_BANDS`; a missing column warns instead of raising `KeyError`. |
| A-3 | `ee_start_date`/`ee_end_date`/`date_str` bound unconditionally before the EE branches in both the single-area and multi-area paths. |
| A-4 | `_fetch_window()` queries every requested source and tags rows with `firms_source`, rather than stopping at the first non-empty one. `--first-source-only` restores the old behaviour. |
| A-5 | `acq_time` parsed numerically, so float promotion during concat no longer raises. Timestamps are UTC-aware. |
| A-6 | Shapefile subsetting builds points in EPSG:4326 and reprojects to the shapefile CRS. |
| A-7 | `fetch_area_csv()` uses a `requests.Session` with `Retry` on 429/5xx, a 60 s timeout, and body inspection for `Invalid MAP_KEY` / quota / HTML responses. New `FirmsRequestError` surfaces these instead of returning an empty frame. |
| A-8 | `plt.cm.get_cmap` → `plt.get_cmap` (removal in matplotlib 3.11). |
| A-9 | `plot_fire_map_time_based` copies before assigning `_day`; asserted in the smoke test. |
| A-10 | Latitudes clipped before `cos()`; verified warning-free under `np.errstate(all="raise")`. |
| — | **New:** the time-plot legend was rendering empty — `GeoDataFrame.plot` produces a `PatchCollection`, which matplotlib won't build handles from. Now uses explicit `Patch` proxies. Found by the smoke test, not in the original review. |

## Algorithm and ML validity

| ID | Change |
|---|---|
| B-1 | `keep_singletons=True` by default: DBSCAN noise points become one-detection clusters with `is_singleton=True`. `--drop-singletons` restores the old behaviour. |
| B-2 | DBSCAN now uses the haversine metric on radians with `eps_km`. Optional `--cluster-time-days` adds a scaled time axis. Tested: two co-located fires six months apart yield 1 cluster space-only, 2 with the time axis. |
| B-3 | `shapely.concave_hull(ratio=0.3)` with automatic convex fallback on Shapely < 2.1. `--convex-hull` forces convex. Tested: concave area ≤ convex on an L-shaped fire. |
| B-4 | `add_acq_datetime()` emits UTC-aware timestamps plus a `local_date` derived via `FIRMS_LOCAL_TZ`. Daily grouping uses `local_date`. Tested: 06:00 UTC Jan 8 → local Jan 7. |
| B-5 | `set_feature_lag_days()` / `--feature-lag-days` (default 1). Both `_get_time_window` and `_get_exact_day_window` are now end-exclusive relative to the label day; `--feature-lag-days 0` prints an explicit leakage warning. |
| B-7 | Wind direction no longer reduced as a scalar. New `wind_u`/`wind_v` layers decompose per-image to vector components before averaging. **Not runtime-verified (EE).** |
| B-8 | Per-layer reducers moved to `EE_TEMPORAL_REDUCER`, so aggregation is declared in one place rather than inferred from an if-chain. |
| B-9 | `EE_SCALE_FACTORS` applies VNP13A1's 1e-4 factor, putting raster NDVI in [-1, 1] to match the colour range. **Not runtime-verified (EE).** |
| B-10 | Footprints use the per-detection `scan`/`track` columns when present, falling back per-row to the nominal coefficient. Tested both paths. |
| B-11 | New `sample_background_points()` / `--background-samples`: matched negatives drawn from the same bbox and date pool, excluding a configurable radius around real detections. The docstring states plainly that the sampling strategy is an unresolved research decision and lists the specific traps. |
| B-12 | Added `slope`, `aspect` (from the DEM already loaded), `erc`, and `vpd` (Tetens equation from tmmx/tmmn/sph). Terrain sampling scale dropped 500 m → 90 m. **Not runtime-verified (EE).** |

## Performance

| ID | Change |
|---|---|
| C-1 | `extract_ee_rasters_parallel()` with `ThreadPoolExecutor`, wired into both export paths. `--ee-workers` (default 8). **Not runtime-verified (EE).** |
| C-3 | `_sample_ee_rectangle_array()` requests `ee.Image.pixelLonLat()` alongside the data band and returns the array's *true* bounds. Per-band pixel budget reduced to ⅓ of the cap to pay for the extra bands. **Not runtime-verified (EE).** |
| C-4 | `footprint_polygons()` uses `shapely.polygons()` on a stacked `(n, 4, 2)` array; `iterrows` removed from the plotting and clustering hot paths. |
| C-5 | `_cache_put()` bounds the in-process caches at 64 entries. `joblib.Memory` wired up via `_disk_cache()` when joblib is installed (`FIRMS_CACHE_DIR` to relocate). |
| C-6 | `save_raster()` writes GeoTIFF with CRS and band description when rasterio is available, falling back to `.npy` + `bounds.txt` otherwise. Per-layer grid CSVs are now opt-in via `--write-grid-csv`. Verified: GeoTIFF reads back at correct shape and CRS. |

## Code health

- Removed ~400 lines of dead code: `fetch_california_from_api`, `save_earth_engine_outputs`, `plot_ee_feature_map`, `plot_ee_feature_grid`, `_get_ee_feature_columns`, `_get_ee_feature_label`, `get_fire_centroids_dataframe`, `sample_ee_at_points`, `enrich_with_earth_engine`. This was the older point-sampling design the raster path superseded; `sample_ee_at_points` also had an unbatched payload bug that would have failed past a few thousand points.
- All pyflakes findings cleared (unused `Polygon`, `im`, `default_end`).
- Repeated inline footprint-column lists collapsed to `FOOTPRINT_COLUMNS`.
- Conflicting `aspect="auto"` + `set_aspect("equal")` in `plot_raster_layer` resolved.
- `test_firms.py`: 33 tests, no network or credentials required.
- `requirements.txt` with pins and optional-group comments.
- Top-level handler turns `FirmsRequestError` and `KeyboardInterrupt` into clean exits.

## Output format changes

Scripts reading the old outputs need updating:

| Before | After |
|---|---|
| `{base}_{layer}.npy` + `{base}_{layer}_bounds.txt` | `{base}_{layer}.tif` (`.npy` pair only without rasterio) |
| `{base}_{layer}_grid.csv` always written | opt-in via `--write-grid-csv` |
| centroids: `cluster_id, centroid_lat, centroid_lon, n_detections` | adds `is_singleton`, `hull_area_km2`, `frp_sum`, `frp_max`, `detected_area_km2`, `first_detection`, `last_detection`, `duration_hours` |
| cluster polygons existed only inside the PNG | `{base}_clusters.geojson` |
| — | `{base}_background.csv` with `--background-samples` |
| detections CSV | adds `firms_source`, `local_date`, `footprint_source` |

Renamed flags: `--cluster-eps` (degrees) → `--cluster-eps-km` (kilometres). Note the unit change — `0.02` is no longer a meaningful value; the equivalent is roughly `2.0`.

---

## Deliberately not done

- **Package split.** Still one ~3,100-line module. Mechanical but it touches every import in whatever else you've built around this, so it's better done when you choose, not bundled into a correctness pass.
- **`logging` migration.** ~60 `print` calls left as-is. Converting them is churn that would obscure the substantive diff.
- **`ee.data.computePixels` migration.** The larger win over `sampleRectangle`, but it can't be validated without EE credentials, and shipping an unverified rewrite of the extraction core is worse than shipping the tiling code with correct georeferencing.
- **Batch export to GCS.** `--ee-bucket`/`--ee-folder` are still parsed and unused. This is the right answer for training-scale extraction and deserves its own design pass.
- **Common-grid resampling (B-6).** GeoTIFF output makes it a short `rioxarray.reproject_match` step, but the target grid is your call and it belongs in the feature-engineering stage rather than here.

## Suggested first run

```bash
export FIRMS_MAP_KEY=your_new_key

python visualize_firms_dataset.py \
  --area palisades --start-date 2025-01-07 --end-date 2025-01-31 \
  --use-archive --fire-mask --time-plot \
  --cluster-eps-km 2.0 --cluster-time-days 2.0 \
  --background-samples 5000 \
  --ee-raster-layers all --ee-project your-project --ee-workers 8
```

Then sanity-check three things before trusting anything downstream:

1. Open `*_clusters.geojson` in QGIS against the CAL FIRE Palisades perimeter. You have ground truth for this fire — it converts the hull-shape question from a guess into a measured error.
2. Confirm `*_vegetation.tif` values fall in roughly [-0.2, 0.9]. Anything in the thousands means the NDVI scale factor regressed.
3. Confirm the GRIDMET rasters are dated `2025-01-06` when the label day is `2025-01-07`. That one-day offset is the leakage fix doing its job.

---

# Session 2 — gridding, automation, and the bugs the transcripts caught

## New modules

The pipeline is now four scripts that compose, rather than one that does
everything. Each owns one decision, so a methods question has one file to open.

| File | Owns |
|---|---|
| `visualize_firms_dataset_v3.py` | FIRMS ingest, filters, CSV/GeoJSON/PNG output |
| `firelookup.py` | Where and when — resolves a fire NAME to perimeter + dates |
| `firegrid.py` | The grid contract — projection, resampling, labels, tensors |
| `build_dataset.py` | The driver — composes the three into `.npz` samples |

Support scripts: `verify_ee_setup.py` (checks the EE path end to end),
`grid_report.py` (audits tile/layer geometry), `diagnose_daily.py` (traces
where detections vanish between fetch and daily folders).

## The grid spec

Decided and documented in `firegrid.py`'s docstring:

- **EPSG:3310**, not 4326. The same 64 km spans 0.683° at San Diego and 0.771°
  at the Oregon border — 13% anisotropy that would make a convolution kernel
  mean different things at different latitudes.
- **1 km cells, 64×64 tiles**, snapped to a global grid so overlapping fires
  share cell boundaries exactly. Matches Huot et al., *Next Day Wildfire
  Spread* (IEEE TGRS 60, 2022).
- **18 channels**, fixed order, written even when a layer is unavailable so
  every sample has identical shape.
- **Three label classes**: 0 no-fire, 1 fire, 2 unobserved. The third exists
  because absence of a FIRMS detection is missing data, not absence of fire.

## Bugs fixed

**Correctness**

- `export_daily_visual_samples()` never accepted `write_grid_csv`, so both call
  sites raised `TypeError`. A test now walks every call site's keyword
  arguments against its callee.
- Earth Engine was initialized lazily from inside the extraction thread pool,
  so an unauthenticated machine started **one browser OAuth flow per worker**,
  all binding the same callback port. Seven of fifteen layers were silently
  lost to the collision. Now serialized behind a lock, never run off the main
  thread, and authenticated once up front by `ensure_earth_engine_ready()`.
- HDF5 writes failed on the mixed VIIRS-letter / MODIS-integer `confidence`
  column. Object columns are now cast to string for the HDF5 copy only.
- `sampleRectangle` returns pixel arrays alongside image metadata; selecting by
  position grabbed SRTM's HTML `description` instead of elevation. Layers are
  now reduced to one band and renamed before sampling.
- `reduceResolution` failed on `vegetation` because a collection composite has
  no fixed projection. `setDefaultProjection` is now asserted from the layer's
  native scale.
- `drought` requested single days from a 5-day pentad product, so four days in
  five returned a zero-band image. Coarse-cadence layers now use a lookback
  window (`LAYER_LOOKBACK_DAYS`).
- `prev_fire_mask` reset to empty after a coverage gap, telling the next
  sample the fire had vanished. It now carries forward for up to
  `--max-carry-days` blank days, then expires.
- Aspect was averaged as a linear quantity: `mean(350°, 10°) = 180°`, the
  opposite direction. Now split into unit sin/cos plus `aspect_consistency`
  (circular concentration R), which doubles as a terrain-roughness feature.

**Corrected guidance**

- `--use-archive` was recommended for historical pulls. It silently drops all
  NOAA-21 detections (~20% of a CA pull) because FIRMS publishes no
  `VIIRS_NOAA21_SP` product. NRT sources serve historical dates fine.
- The suspected NRT date contamination was **not** real — verified against the
  saved CSV. A date-window guard was kept anyway as cheap insurance.

## Operational

- **Run transcripts.** Every run writes a redacted `report.txt` /
  `build_report.txt` capturing stdout, stderr, interactive answers, tracebacks
  and environment. Credentials are masked before they reach the file.
- **Earth Engine** moved to its own project (`ee-lmu-ndp-wildfire`), resolved
  through one function: `--ee-project` → `$EE_PROJECT` → default.
- **`KEY_ROTATION.md`** rewritten PowerShell-first, with the BOM trap that
  breaks `git filter-repo` and `detect-secrets` documented.
- **Secret scanning** via `detect-secrets` pre-commit hook.

## Still open

- **Live EE has never run all 18 layers in sequence.** Verified one layer at a
  time by `verify_ee_setup.py`; the driver's full loop is unproven.
- **`mark_unobserved()` is a heuristic.** FIRMS publishes no per-cell overpass
  or cloud coverage, so "no detections anywhere today" stands in for "not
  observed". Joining against swath geometry or MOD35 would do it properly.
- **Batch export.** ~600k interactive `getInfo` calls for a 180-day statewide
  build (~44 h at 8 workers) versus ~2,300 batch export tasks. `crsTransform`
  would also make co-registration structural rather than checked after the fact.
- **No negative sampling, no target definition, no spatially-blocked CV** —
  B-11 from the original review is still the largest gap.
- **Tile overlap between concurrent fires.** Eaton's tile reaches into the
  Palisades burn area. Split train/test by geography or date, never by fire
  name.
