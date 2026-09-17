# Wildfire Next-Day Prediction — Data Extraction Pipeline

Turns a fire *name* into machine-learning training tensors.

```powershell
python build_dataset.py --fire "Palisades" --year 2025
```

That resolves the fire against CAL FIRE's published perimeter, fetches NASA
FIRMS detections over its grid tile, pulls eighteen Earth Engine feature layers
onto the same tile, and writes one `(18, 64, 64)` sample per fire-day.

No bounding boxes. No dates typed by hand.

---

## Table of contents

1. [Research context](#1-research-context)
2. [Pipeline at a glance](#2-pipeline-at-a-glance)
3. [Installation](#3-installation)
4. [Authentication](#4-authentication)
5. [Quickstart](#5-quickstart)
6. [Module reference](#6-module-reference)
7. [The grid specification](#7-the-grid-specification)
8. [Labels and what they mean](#8-labels-and-what-they-mean)
9. [Channel dictionary](#9-channel-dictionary)
10. [Output layout](#10-output-layout)
11. [Run transcripts](#11-run-transcripts)
12. [Troubleshooting](#12-troubleshooting)
13. [Known limitations](#13-known-limitations)
14. [Security](#14-security)

---

## 1. Research context

The end goal is a model that predicts where a fire will burn tomorrow given
where it burned today and the surrounding conditions. This repository is the
**data acquisition and labeling stage**. It does not train, evaluate, or serve
a model.

It answers two questions for a given fire:

1. **Where and when did fire actually burn?** — NASA FIRMS thermal-anomaly
   detections (VIIRS 375 m, MODIS 1 km). These become the **labels**.
2. **What were the conditions there?** — Earth Engine: terrain, vegetation,
   drought, weather, wind, land cover, population. These become the
   **features**.

The feature set and tile geometry follow Huot et al., *Next Day Wildfire
Spread* (IEEE TGRS 60, 2022), so results are comparable to a published
benchmark.

### The caveat that governs everything else

FIRMS reports **active fire detections**, not burned-area perimeters. A
detection means a satellite observed a thermal anomaly in a pixel at an
overpass time. Absence of a detection does **not** mean absence of fire —
cloud, smoke, and overpass gaps all cause misses, and those misses are
systematic rather than random.

A model trained naively on this data learns *"what VIIRS and MODIS saw"*, which
is a proxy for fire, not fire itself. Two mechanisms in this pipeline exist
specifically to stop that: the third label class (§8) and the previous-mask
carry-forward (§6.4).

---

## 2. Pipeline at a glance

```
     "Palisades"
          |
          v
 [ firelookup.py ]  CAL FIRE FRAP / incidents / NIFC / WFIGS
          |         -> perimeter polygon, alarm + containment dates
          v
 [ firegrid.py ]    snap_tile()
          |         -> 64 x 64 km tile on the EPSG:3310 grid
          |
          +----------------------------+
          |                            |
          v                            v
 [ visualize_firms_    ]     [ firegrid.ee_layer_on_grid ]
 [ dataset_v3.py       ]       Earth Engine -> 17 feature
   FIRMS -> detections          channels, resampled onto
          |                     the same tile
          v                            |
   rasterize_fire_mask()               |
   mark_unobserved()                   |
          |                            |
          +------------+---------------+
                       v
              [ build_dataset.py ]
                assemble_sample()
                       |
                       v
          one .npz per fire-day: (18, 64, 64) + label + manifest
```

### Why four modules and not one

Each owns exactly one kind of decision, so a question has one file to open:

| File | Owns | Lines |
|---|---|---|
| `visualize_firms_dataset_v3.py` | FIRMS ingest, filters, dedup, CSV/GeoJSON/PNG | ~3,400 |
| `firelookup.py` | *Where and when* — name → perimeter + dates | ~560 |
| `firegrid.py` | *What shape* — projection, resampling, labels, tensors | ~780 |
| `build_dataset.py` | The driver — sequences the three, handles failure | ~440 |

If your advisor asks why aspect is resampled the way it is, that is
`firegrid.py` and nothing else. If the answer were spread across a 3,400-line
script, it would be unauditable.

### Support scripts

| File | Purpose |
|---|---|
| `verify_ee_setup.py` | Walks the exact Earth Engine calls the pipeline makes and reports where they break |
| `grid_report.py` | Audits tile extent and per-layer array shapes without needing credentials |
| `diagnose_daily.py` | Traces where detections disappear between the area CSV and the daily folders |

---

## 3. Installation

```powershell
conda create -n wfdata python=3.11
conda activate wfdata
pip install -r requirements_v2.txt
```

Core: `numpy`, `pandas`, `requests`, `matplotlib`.
Geospatial: `geopandas`, `shapely`, `rasterio`, `pyproj`.
Optional: `earthengine-api` (features), `scikit-learn` (clustering),
`tables` (HDF5 output), `contextily` (basemaps).

**On Windows, prefer conda for the geospatial stack.** `pip` installing over
conda-managed binaries is what breaks `cffi`, `cryptography`, and `numpy` —
see §12.

---

## 4. Authentication

### FIRMS

Free key from <https://firms.modaps.eosdis.nasa.gov/api/map_key>.

```powershell
[Environment]::SetEnvironmentVariable("FIRMS_MAP_KEY", "your_key", "User")
$env:FIRMS_MAP_KEY = "your_key"   # this session too
```

`setx` and `SetEnvironmentVariable` do **not** affect the window you run them
in. Never hardcode the key; see `KEY_ROTATION.md`.

### Earth Engine

Register your own Cloud project at
<https://code.earthengine.google.com/register> (Noncommercial → Academia &
Research), complete the eligibility questionnaire, then:

```powershell
earthengine authenticate
earthengine set_project ee-your-project
[Environment]::SetEnvironmentVariable("EE_PROJECT", "ee-your-project", "User")
```

Use your own project rather than a shared one: since 27 April 2026 each
noncommercial project has a monthly EECU-hour quota, and this pipeline is not
a light workload.

Verify before spending an hour on a build:

```powershell
python verify_ee_setup.py --date 2025-01-10
```

Project resolution order everywhere: `--ee-project` → `$EE_PROJECT` →
`DEFAULT_EE_PROJECT` in the pipeline module.

---

## 5. Quickstart

```powershell
# Interactive — prompts for fires, dates, layers, filters
python build_dataset.py

# Non-interactive
python build_dataset.py --fire "Palisades" --fire "Eaton" --year 2025 --out ml_dataset

# See the plan and cost without fetching anything
python build_dataset.py --fire "Palisades" --year 2025 --dry-run

# Labels only, no Earth Engine quota spent
python build_dataset.py --fire "Palisades" --year 2025 --no-ee
```

Loading a sample back:

```python
import firegrid as F
features, label, channels, meta = F.load_sample(
    "ml_dataset/palisades_2025/palisades_2025_2025-01-08.npz")

features.shape   # (18, 64, 64) float32
label.shape      # (64, 64) uint8 — 0 no-fire, 1 fire, 2 unobserved
channels         # ['elevation', 'slope', 'aspect_sin', ...]
meta['crs']      # 'EPSG:3310'
```

The grid spec travels inside every file on purpose. A tensor whose projection
is recorded only in a README becomes unusable the moment the README drifts.

The older per-fire visual workflow still works:

```powershell
python visualize_firms_dataset_v3.py --fire-name "Palisades" --fire-year 2025 --fire-mask
```

---

## 6. Module reference

### 6.1 `firelookup.py` — where and when

Resolves a fire name to a perimeter, dates, and a tile. Talks to CAL FIRE and
NIFC; handles no pixels.

| Function | What it does |
|---|---|
| `FireRecord` | Dataclass: name, year, alarm/containment dates, acreage, bbox, centroid, source, `exact_footprint`, `dates_uncertain` |
| `FireRecord.date_window(lead_in, tail)` | Fetch window = alarm − lead-in … containment + tail |
| `resolve_fire(name, year, state, sources)` | Tries each source in turn, returns matches largest-first |
| `search_perimeters()` | CAL FIRE **FRAP** historic perimeters (California, good dates) |
| `search_incidents()` | CAL FIRE incidents feed (California, near-real-time, **no polygon**) |
| `search_nifc_history()` | **NIFC** Interagency Perimeter History (national, all years) |
| `search_wfigs_current()` | **WFIGS** current-season perimeters (national, real timestamps) |
| `pad_bbox(bbox, km)` | Grows a bbox in real kilometres, latitude-corrected |
| `to_tile()` / `tile_envelope()` | Snapped grid tile, and its lon/lat envelope |
| `emit_command()` | Prints the `visualize_firms_dataset_v3.py` invocation for a fire |

**Source order is deliberate.** The default chain is
`("frap", "incidents", "nifc", "wfigs")` — California first. FRAP carries real
alarm and containment dates; NIFC's history layer often carries only a fire
year, which silently turns a two-week fire into an eight-month fetch window.
Records derived that way set `dates_uncertain=True` and print a warning rather
than pretending the window is known. Pass `--national` to reverse the order.

**`--state` matters more than it looks.** Incident names repeat constantly
across the country. When a service does not report a state, records are
filtered geographically against `_STATE_BOXES`.

Two flags worth knowing:

- `--match-tile` emits a bbox covering the **whole 64 km tile** rather than the
  padded perimeter. Without it the fetch covers only ~10–15% of the tile, and
  every unqueried cell becomes no-fire in the label — indistinguishable from
  ground that was observed and did not burn. **Always use this when the output
  feeds `firegrid`.**
- `--pad-km` (default 5) grows the perimeter because FIRMS detects heat
  *outside* the final mapped boundary — spot fires and the active front running
  ahead of the burn.

### 6.2 `firegrid.py` — the grid contract

Everything about geometry, resampling, labels, and tensor layout. Knows nothing
about FIRMS APIs or matplotlib.

**Geometry**

| Function | What it does |
|---|---|
| `GridSpec` | Frozen dataclass: CRS, cell size, tile size, fill value |
| `snap_tile(lon, lat)` | Tile bounds in projected metres, snapped to a global grid |
| `tile_transform()` | Affine transform for rasterization, north-up |
| `tile_bounds_lonlat()` | Lon/lat envelope of a projected tile (slightly larger — a rectangle in 3310 is not one in 4326) |
| `fit_to_shape()` | Crops or pads to exactly 64×64 |

Snapping is what makes two tiles comparable. An unsnapped tile centred on an
arbitrary centroid puts cell boundaries at an arbitrary offset, so the same
patch of ground lands in different cells in different samples.

**Labels**

| Function | What it does |
|---|---|
| `rasterize_fire_mask(day_df, tile)` | Burns detection footprints onto the grid (`all_touched=True`) |
| `mark_unobserved(mask, prev, had_detections)` | Promotes no-fire cells to *unobserved* across a coverage gap |

`rasterize_fire_mask` uses true VIIRS/MODIS footprint corners when present, so
a 375 m nadir pixel and a 1.5 km edge-of-scan pixel are not treated as the same
size. `all_touched=True` because under-calling fire is the more costly error at
1 km.

**Features**

| Function | What it does |
|---|---|
| `ee_layer_on_grid(layer, tile, ...)` | Fetches one layer already aggregated onto the tile, server-side |
| `split_circular_components(stacked)` | Averaged sin/cos → unit direction + concentration R |
| `circular_to_components(degrees)` | Splits a circular field before any averaging |
| `resample_to_grid(arr, ...)` | Local fallback for arrays already fetched |

`ee_layer_on_grid` is where the resampling rules live:

- **Continuous** → area-weighted mean (`reduceResolution`) when downsampling,
  bilinear when upsampling from coarser native data.
- **Categorical** → majority class. Land cover must never be averaged; the mean
  of classes 3 and 7 is class 5, an invented category.
- **Circular** → decomposed to sin/cos *before* averaging. `mean(350°, 10°)`
  is 180°, pointing opposite to both inputs.
- **Terrain** → slope and aspect derived from the DEM at native 30 m, then
  aggregated. Deriving slope from an already-coarsened grid measures a smoothed
  landscape and runs systematically too flat.

Aggregating server-side is strictly better than fetching coarse arrays and
resampling locally: each 1 km cell averages the ~1,100 30 m DEM cells inside
it, whereas a local resample can only work from what was already sampled.

**Tensors**

| Function | What it does |
|---|---|
| `assemble_sample(features, label)` | Stacks into `(18, 64, 64)` in the fixed `CHANNELS` order |
| `save_sample()` / `load_sample()` | Compressed `.npz` with channel names and spec embedded |

Missing channels are written as fill rather than dropped, so every sample has
the same depth and a model never sees a shifted channel axis.

**Event discovery**

| Function | What it does |
|---|---|
| `discover_fire_events(df)` | DBSCAN over detections → one snapped tile per event |

Clusters in **space only**, deliberately. Space-time clustering shatters a
week-long fire into one "event" per day, because the time axis scales at
`spread_km_per_day` and consecutive days land far beyond `eps`. Days are
handled by building one sample per day within an event.

### 6.3 `visualize_firms_dataset_v3.py` — FIRMS ingest

The original pipeline, still the source of all detection data.

| Function | What it does |
|---|---|
| `fetch_firms_data()` | Chunked, multi-source fetch with dedup |
| `add_acq_datetime()` | UTC timestamp + **local** calendar date |
| `add_footprint_columns()` | Per-detection footprint corners from scan/track |
| `apply_confidence_frp_daynight_filters()` | Confidence / FRP / day-night subsetting |
| `cluster_fire_points_to_polygons()` | Space-time DBSCAN → polygons + centroids |
| `resolve_fire_name_args(args)` | Fills `--bbox` and dates from `--fire-name` |
| `ensure_earth_engine_ready()` | One-time EE auth on the main thread |
| `RunTranscript` | Context manager that mirrors a run into a redacted text file |
| `_enforce_requested_window()` | Drops detections outside the requested range, loudly |

**Group on `local_date`, not `acq_date`.** GRIDMET and the drought products are
indexed by local day; joining on raw UTC misattributes every night overpass.

**`--use-archive` drops NOAA-21.** FIRMS publishes no `VIIRS_NOAA21_SP`
product, so archive-only mode silently loses that satellite — about 20% of a
typical California pull. NRT sources serve historical dates fine. Use the
default for old data.

### 6.4 `build_dataset.py` — the driver

| Function | What it does |
|---|---|
| `main()` | Parses args or runs the interactive UI, opens the transcript |
| `run_interactive()` | Six-section prompt UI matching the FIRMS pipeline's style |
| `build_fire(record, args, map_key)` | Every sample for one fire |
| `fetch_static_layers()` | Terrain, land cover, population — **once per tile** |
| `fetch_dynamic_layers()` | Weather, drought, vegetation — per day |
| `daterange()` | Inclusive day list |

Three behaviours that are easy to get wrong and are handled here:

**Static layers fetch once.** Five of eighteen channels never change between
days. Re-fetching them per day would be most of the Earth Engine traffic and
would return identical arrays.

**Lead-in days are fetched but not written.** Day *t* needs a real
`prev_fire_mask` from day *t−1*. Without a lead-in, the first sample claims
nothing was burning the day before — true for day one of a fire, false for any
window that starts mid-event.

**Coverage gaps carry forward.** After a blank day, `prev_fire_mask` used to
reset to empty, telling the next sample the fire had vanished. It now holds the
last known mask across up to `--max-carry-days` (default 2) blank days, then
expires so a fire that genuinely ended does not propagate forever:

```
date        det  fire  unobs   prev_cells
2025-01-08  150    32      0      32
2025-01-09    0     0     32      32   <- gap, carried
2025-01-10    0     0     32      32   <- still carried
2025-01-12    0     0      0       0   <- expired, treated as out
```

---

## 7. The grid specification

Every choice below is a decision, not a law. They live in `GRID_SPEC` and the
`firegrid.py` docstring so they can be changed in one place and cited in a
write-up.

| Parameter | Value | Why |
|---|---|---|
| CRS | **EPSG:3310** (California Albers) | Equal-area, metres. In EPSG:4326 the same 64 km spans 0.683° at San Diego and 0.771° at the Oregon border — 13% anisotropy, so a convolution kernel would mean different things at different latitudes and cell area would not be constant. |
| Cell | **1000 m** | Compromise between 90 m terrain and 4 km GRIDMET. Matches NDWS. |
| Tile | **64 × 64 cells** | 4,096 cells, far under the 262,144-value `sampleRectangle` cap, so no silent coarsening. |
| Origin | Global grid anchored at (0, 0) | Overlapping fires share cell boundaries exactly. |
| Fill | `NaN` | Distinct from both zero and any label class. |

### Why this mattered

Before the common grid, `grid_report.py` measured what was actually coming
back:

| Area | Extent | Distinct grids | Worst coarsening |
|---|---|---|---|
| California | 919 × 1,054 km | 3 | terrain served **41× coarser** than configured |
| Palisades | 47.8 × 26.6 km | 3 | weather at 7 × 12 cells |
| Eaton | 18 × 11 km | 3 | weather at 3 × 4 cells |

Twelve cells covering an entire fire area meant wind direction — the variable
that most determines where a fire goes tomorrow — was effectively constant
across the domain. Run `python grid_report.py` to reproduce.

---

## 8. Labels and what they mean

Three classes, not two:

| Value | Meaning |
|---|---|
| `0` | **No fire** — observed, no detection |
| `1` | **Fire** — a detection footprint intersects the cell |
| `2` | **Unobserved** — coverage gap; exclude from the loss |

The third class exists because absence of a FIRMS detection is not evidence of
absence of fire. Forcing those cells to 0 teaches the model satellite coverage
patterns rather than fire behaviour, and because coverage gaps are systematic
rather than random, that bias does not wash out with more data.

**`mark_unobserved()` is a heuristic, and the weakest part of the pipeline.**
FIRMS publishes no per-cell overpass or cloud coverage, so the rule
implemented is: *if the tile recorded no detections at all on this day but the
previous day had fire, treat the previously-burning cells as unobserved rather
than extinguished.* Doing it properly means joining against per-overpass swath
geometry or a cloud mask such as MODIS MOD35. Replace that function, not its
callers.

---

## 9. Channel dictionary

Fixed order. A model trained on one ordering silently mispredicts on another.

| # | Channel | Source | Native | Resampling |
|---|---|---|---|---|
| 0 | `elevation` | SRTM GL1 | 30 m | mean |
| 1 | `slope` | SRTM → `ee.Terrain.slope` | 30 m | mean (derived at native) |
| 2 | `aspect_sin` | SRTM → `ee.Terrain.aspect` | 30 m | circular → unit |
| 3 | `aspect_cos` | " | 30 m | circular → unit |
| 4 | `aspect_consistency` | " | 30 m | circular concentration R |
| 5 | `vegetation` | VIIRS VNP13A1 NDVI | 500 m | mean, 24-day lookback |
| 6 | `landcover` | ESA WorldCover | 10 m | **majority class** |
| 7 | `population` | CIESIN GPWv411 | 927 m | mean |
| 8 | `drought` | GRIDMET/DROUGHT PDSI | 4 km | bilinear, 20-day lookback |
| 9 | `humidity` | GRIDMET | 4 km | bilinear |
| 10 | `weather_temp` | GRIDMET | 4 km | bilinear |
| 11 | `weather_precip` | GRIDMET | 4 km | bilinear |
| 12 | `wind_speed` | GRIDMET | 4 km | bilinear |
| 13 | `wind_u` | GRIDMET `th` → components | 4 km | bilinear |
| 14 | `wind_v` | " | 4 km | bilinear |
| 15 | `erc` | GRIDMET energy release component | 4 km | bilinear |
| 16 | `vpd` | GRIDMET vapour pressure deficit | 4 km | bilinear |
| 17 | `prev_fire_mask` | Previous day's label | — | — |

**`aspect_consistency` is a real feature, not bookkeeping.** After averaging
sin and cos over the ~1,100 native cells inside a 1 km cell, the resultant
*length* is the circular concentration R, not 1:

| Terrain inside one cell | R | Direction |
|---|---|---|
| Uniform SW-facing slope | 0.999 | 225° |
| Moderately varied hillside | 0.726 | 225° |
| Ridge crest, aspects oppose | 0.106 | 213° |
| Flat ground, aspect random | 0.033 | zeroed |

R tells the model how much to trust the direction, and doubles as a
terrain-roughness proxy. Below R = 0.05 the direction is meaningless, so the
components are zeroed rather than amplified by dividing through near-zero.

**Lookback windows.** `LAYER_LOOKBACK_DAYS` exists because some products have a
cadence coarser than a day. GRIDMET/DROUGHT is a 5-day pentad product;
requesting a single day returns an empty collection four days in five. VNP13A1
is a 16-day composite.

---

## 10. Output layout

```
ml_dataset/
├── build_report.txt                    # full run transcript
└── palisades_2025/
    ├── manifest.json                   # per-sample stats
    ├── palisades_2025_2025-01-07.npz
    ├── palisades_2025_2025-01-08.npz
    └── ...
```

Each `.npz` holds `features` `(18,64,64) float32`, `label` `(64,64) uint8`,
`channels` (names), and `meta` (JSON: fire, date, CRS, cell size, tile bounds,
detection count, channels actually present, source, whether the footprint was
exact).

`manifest.json` gives per-day `detections`, `fire_cells`, `unobserved_cells`,
and `channels_present` — the fastest way to spot a day where feature extraction
silently degraded.

The visual pipeline writes separately under `data output/`: `datasets/` for
CSV/GeoJSON/HDF5, `visualizations/` for PNGs, and `daily_samples/<date>/`
subfolders.

---

## 11. Run transcripts

Every run writes a transcript capturing stdout, stderr, every interactive
prompt *and answer*, tracebacks, timings, and the environment (Python version,
platform, package versions). On by default.

```powershell
python build_dataset.py --report-file custom.txt   # relocate
python build_dataset.py --report-append            # accumulate runs
python visualize_firms_dataset_v3.py --no-report   # disable
```

The transcript is line-buffered, so it survives a Ctrl-C or a hard kill
mid-run, and it is moved next to the dataset it describes when the run ends.

**Credentials are masked before they reach the file.** Anything matching a
32-hex key, a `MAP_KEY=` query parameter, or an `api_key:` assignment becomes
`<redacted>`. Prompts mentioning key, token, secret, password or credential
have their answers suppressed entirely.

**Send the transcript, not your console output.** Your terminal echoes what you
type; the redaction only applies to the file.

---

## 12. Troubleshooting

### Environment

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: _cffi_backend` | pip overwrote conda-managed `cffi` | `pip install --force-reinstall --no-cache-dir cffi cryptography` |
| `Error importing numpy: you should not try to import numpy from its source directory` | numpy's C extension failed to load (same cause) | Reinstall numpy with whichever manager owns it (`conda list numpy` → check the Channel column) |
| `no visualize_firms_dataset*.py found` | Running from the wrong directory | All scripts must sit in one folder; they import the pipeline by globbing `visualize_firms_dataset_v*.py` |

### Earth Engine

| Symptom | Cause | Status |
|---|---|---|
| Many `Earth Engine auth failed` + `KeyError: 'code'` | One OAuth flow per worker thread, all binding the same callback port | Fixed — auth is serialized and runs on the main thread. Run `earthengine authenticate` once first. |
| `reduceResolution ... does not have a valid default projection` | An ImageCollection composite has no fixed projection | Fixed — `setDefaultProjection` from the layer's native scale |
| `Image.select: Invalid band number (0) ... contains 0 bands` | Empty collection from a coarse-cadence product | Fixed — raise `LAYER_LOOKBACK_DAYS[layer]` if it recurs |
| `could not convert string to float: '<p>The Shuttle Radar...'` | `sampleRectangle` returns metadata alongside pixels | Fixed — bands are renamed and selected by name |

Run `python verify_ee_setup.py --date 2025-01-10` before any long build. It
checks the client library, credentials, project registration, every dataset
family, and a real 64×64 extraction including the aspect unit-vector property.

### Empty daily masks

```powershell
python diagnose_daily.py "path\to\datasets\firms_<area>_bbox_..."
```

Walks the four stages that all produce an identical blank PNG and reports which
one emptied the frame: empty area CSV, the UTC→local day shift, filters, or a
genuine bug. **An empty mask is often correct** — it means nothing was detected
that day, which belongs in the unobserved class rather than no-fire.

---

## 13. Known limitations

- **Live Earth Engine has never run all 18 layers in sequence.** Verified one
  layer at a time; the driver's full loop is unproven at the time of writing.
- **`mark_unobserved()` is a heuristic** — see §8.
- **Interactive `getInfo`, not batch export.** A 180-day statewide build is
  ~600,000 `getInfo` calls (~44 h at 8 workers) versus ~2,300 batch export
  tasks. `Export.image.toDrive` with an explicit `crsTransform` would also make
  co-registration structural rather than checked after the fact. This is the
  highest-value remaining change.
- **No negative sampling, no target definition, no spatially-blocked CV.**
  `--background-samples` draws uniformly over a bbox, which is a weak baseline.
  This is B-11 from `REVIEW.md` and the largest gap before training.
- **Tile overlap between concurrent fires.** Eaton's 64 km tile reaches into
  the Palisades burn area. Split train/test by geography or date, **never by
  fire name**.
- **FRAP is incomplete**, particularly for older and smaller fires. A name that
  returns nothing has not been proven not to exist.
- **`--ee-bucket` / `--ee-folder` are parsed but unused** — placeholders for
  the batch-export path.

See `REVIEW.md` for the full findings list and `CHANGES.md` for what has been
fixed and why.

---

## 14. Security

- **Never commit a MAP_KEY.** `KEY_ROTATION.md` has the full rotation runbook,
  PowerShell-first.
- **`.gitignore`, not `_gitignore`.** Git only reads the dotted name; a file
  called `_gitignore` does nothing at all.
- **Commit** `.pre-commit-config.yaml` and `.secrets.baseline`. The baseline
  stores hashes of reviewed findings, not secrets.
- **Never share** `~/.config/earthengine/credentials` — that is a live OAuth
  token for your whole Google account, not just Earth Engine.
- **On Windows, do not generate machine-read files with `>` or
  `Set-Content -Encoding utf8`.** Windows PowerShell 5.1 writes a BOM, which
  breaks `git filter-repo` rule files and `detect-secrets` baselines — the
  latter with `JSONDecodeError: Expecting value: line 1 column 1`. Use
  `cmd /c "... > file"` or `-Encoding ascii`.
