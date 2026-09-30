# Wildfire Spread Prediction -- Data Extraction Pipeline

Turns a fire **name** into verified machine-learning training data.

```powershell
cd src
python build_dataset.py
```

Answer the prompts, pressing Enter for the defaults. The pipeline looks the
fire up in CAL FIRE's records, fetches NASA FIRMS satellite detections over it,
removes non-fire heat sources, pulls eighteen Earth Engine feature layers onto
a common grid, writes one `(18, 64, 64)` sample per time step, and then checks
the result against the real world.

No bounding boxes. No dates typed by hand.

**New here?** Read this file for how it works, then
[`RUNBOOK.md`](docs/RUNBOOK.md) for how to run a build and verify it.
[`CHANGES.md`](docs/CHANGES.md) lists the latest changes.

---

## Contents

1. [Research context](#1-research-context)
2. [Pipeline at a glance](#2-pipeline-at-a-glance)
3. [Installation](#3-installation)
4. [Authentication](#4-authentication)
5. [Quickstart](#5-quickstart)
6. [Module reference](#6-module-reference)
7. [The grid specification](#7-the-grid-specification)
8. [Time steps](#8-time-steps)
9. [Labels](#9-labels)
10. [Channel dictionary](#10-channel-dictionary)
11. [Verification](#11-verification)
12. [Output layout](#12-output-layout)
13. [Handing off to Swin-UNet](#13-handing-off-to-swin-unet)
14. [Troubleshooting](#14-troubleshooting)
15. [Known limitations](#15-known-limitations)
16. [Security](#16-security)

---

## 1. Research context

The goal is a model that predicts where a fire will burn **next** -- ideally
at short enough horizons for firefighters to act on -- given where it is
burning now and the surrounding conditions. The planned model is
**Swin-UNet**.

This repository is the **data acquisition, labelling and verification** stage.
It does not train or serve a model. For a given fire it answers:

1. **Where and when did fire burn?** NASA FIRMS thermal-anomaly detections
   (VIIRS 375 m, MODIS 1 km). These become the **labels**.
2. **What were the conditions?** Earth Engine: terrain, vegetation, drought,
   weather, wind, land cover, population. These become the **features**.

The tile geometry and feature set follow Huot et al., *Next Day Wildfire
Spread* (IEEE TGRS 60, 2022), so daily results are comparable to a published
benchmark.

### The caveat that shapes everything

FIRMS reports **active fire detections**, not burned area. A detection means a
satellite saw a thermal anomaly at an overpass. **No detection does not mean
no fire**: cloud, smoke and gaps between overpasses all cause misses, and they
are systematic, not random.

A model trained naively learns *what the satellites saw* rather than *where
fire was*. Three mechanisms here exist to prevent that: the **unobserved**
label class (section 9), carrying the previous fire mask across coverage gaps
(section 6.5), and labelling windows with no satellite overhead as unobserved
across the whole tile (section 8).

FIRMS also reports heat that is not wildfire -- refineries, gas flares, power
plants. In the Eaton pull these were **83% of all detections on 2025-01-13**.
`firefilter.py` removes them (section 6.2).

---

## 2. Pipeline at a glance

```
   "Eaton"
      |
      v
 firelookup.py        CAL FIRE FRAP -> perimeter polygon, alarm/containment dates
      |
      v
 firegrid.snap_tile   64 x 64 km tile on the EPSG:3310 grid
      |
      +--------------------------------+
      v                                v
 visualize_firms_dataset_v3      firegrid.ee_layer_on_grid
   FIRMS detections                Earth Engine -> 17 feature channels
      |                            on the same tile
      v                                |
 firefilter.py                         |
   confidence, static sources,         |
   stop rule                           |
      |                                |
      v                                |
 rasterize + mark_unobserved           |
      |                                |
      +---------------+----------------+
                      v
              build_dataset.py
         one .npz per time step, plus CSV audit trail,
         plots, perimeter.geojson and a run transcript
                      |
                      v
              verify_dataset.py
         internal consistency + reality checks + QA sheets
                      |
                      v
       export_hdf5.py -> swin_data.py -> Swin-UNet
```

### Modules

Each module owns one kind of decision, so any methods question has one file
to open.

| File | Owns |
|---|---|
| `build_dataset.py` | **Start here.** The driver and interactive UI; sequences everything below |
| `firelookup.py` | *Where and when*: fire name -> perimeter, dates, tile |
| `visualize_firms_dataset_v3.py` | FIRMS ingest, dedup, plots, tables, run transcripts |
| `firefilter.py` | *Which detections count*: confidence, static sources, stop rule |
| `firegrid.py` | *What shape*: projection, resampling, labels, tensor layout |
| `verify_dataset.py` | Checks a built dataset, internally and against reality |
| `reality_checks.py` | The real-world checks and QA sheets used by `verify_dataset.py` |
| `export_hdf5.py` | Packs a dataset into one HDF5 file for training |
| `swin_data.py` | Swin-UNet geometry, normalization and augmentation |

| Support script | Purpose |
|---|---|
| `verify_ee_setup.py` | Tests the exact Earth Engine calls the pipeline makes |
| `grid_report.py` | Audits tile extents and per-layer shapes, no credentials needed |
| `diagnose_daily.py` | Traces where detections vanish between fetch and daily folders |

All scripts live together in `src/`. They find each other by filename in their
own folder, so **keep them in one directory**.

---

## 3. Installation

```powershell
conda create -n wfdata python=3.11
conda activate wfdata
conda install -c conda-forge geopandas shapely rasterio pyproj
pip install -r requirements.txt
```

**On Windows, install the geospatial stack with conda, not pip.** pip
installing over conda-managed binaries is what breaks `cffi`, `cryptography`
and `numpy` (see section 14).

---

## 4. Authentication

### FIRMS

Free key from <https://firms.modaps.eosdis.nasa.gov/api/map_key>:

```powershell
[Environment]::SetEnvironmentVariable("FIRMS_MAP_KEY", "your_key", "User")
$env:FIRMS_MAP_KEY = "your_key"      # the command above does not affect this window
```

Never hardcode or paste the key. See [`KEY_ROTATION.md`](docs/KEY_ROTATION.md).

### Earth Engine

Use **your own** Cloud project: since April 2026 every noncommercial project
has a monthly compute quota, and this pipeline is not a light workload.
Register at <https://code.earthengine.google.com/register> (Noncommercial ->
Academia & Research) and complete the eligibility questionnaire, then:

```powershell
earthengine authenticate
earthengine set_project ee-your-project
[Environment]::SetEnvironmentVariable("EE_PROJECT", "ee-your-project", "User")
```

Project resolution everywhere: `--ee-project` -> `$EE_PROJECT` ->
`DEFAULT_EE_PROJECT` in `visualize_firms_dataset_v3.py`.

Verify before any long build:

```powershell
python verify_ee_setup.py --date 2025-01-10
```

---

## 5. Quickstart

```powershell
cd src

# Interactive: prompts for fires, dates, time step, layers, visuals, filters,
# output, and whether to verify afterwards. Enter accepts each default.
python build_dataset.py

# Non-interactive equivalent
python build_dataset.py --fire "Palisades" --fire "Eaton" --year 2025 --visualize --verify

# See the plan and Earth Engine cost without fetching anything
python build_dataset.py --fire "Eaton" --year 2025 --dry-run

# Labels only, no Earth Engine quota spent
python build_dataset.py --fire "Eaton" --year 2025 --no-ee
```

**Exit codes:** `0` built (and verified, if verification was requested),
`2` built but verification failed, `1` nothing built.

Loading a sample:

```python
import firegrid as F
features, label, channels, meta = F.load_sample(
    "ml_dataset/eaton_2025/eaton_2025_2025-01-08.npz")

features.shape    # (18, 64, 64) float32
label.shape       # (64, 64) uint8 -- 0 no fire, 1 fire, 2 unobserved
meta["crs"]       # 'EPSG:3310'
meta["filter"]    # which detections were allowed into the label
```

The grid spec travels inside every file, so a sample can always be
interpreted without the documentation that produced it.

---

## 6. Module reference

### 6.1 `firelookup.py` -- where and when

Resolves a fire name against published perimeter records and returns a
`FireRecord`: name, year, alarm and containment dates, acreage, bounding box,
centroid, the **perimeter polygon**, and flags for how trustworthy each is.

| Source | Coverage | Dates | Polygon |
|---|---|---|---|
| **FRAP** (default first) | California | good | yes |
| CAL FIRE incidents | California, near-real-time | good | no -- inferred from acreage |
| NIFC perimeter history | national, all years | often year only | yes |
| WFIGS current | national, current season | real timestamps | yes |

California-first is deliberate: FRAP has real alarm dates, while NIFC often
has only a fire year, which silently turns a two-week fire into an
eight-month fetch window. Such records set `dates_uncertain=True` and warn.
`--national` reverses the order once the project goes beyond California.

`date_window(lead_in, tail)` gives the fetch window: alarm date minus a
lead-in, to containment plus a tail. The lead-in exists because the first
labelled step needs a real previous fire mask.

### 6.2 `firefilter.py` -- which detections count

Before this module, every detection in the tile became fire in the label.
`annotate()` now gives every detection a verdict, applied in order:

1. **Confidence, normalized across sensors.** VIIRS reports `l/n/h`; MODIS
   reports 0-100. Both map onto `low/nominal/high` using the FIRMS MODIS
   thresholds (<30, 30-79, >=80). Default minimum: nominal.
2. **Static heat sources removed.** A site counts as static only if it shows
   what a fire does not: **persistent** across many days, at **low power**.
   Inside the named fire's own perimeter, a site must also have been active
   **before ignition**.
3. **Target fire only** (optional, off by default): drop other real fires that
   share the tile.
4. **Stop rule:** after N consecutive blank days (default 3), the fire is
   treated as out and later steps are not written.

Every detection is kept in the output CSV with `confidence_class`,
`drop_reason` and `used_in_label`, so the CSV is an **audit trail** of the
label rather than a copy of it.

> **FIRMS's own `type` field is not trusted alone.** In the Eaton pull it
> labelled 44 detections *inside the burn perimeter* -- MODIS pixels up to
> 800 MW on ignition night -- as "other static land source", most likely
> because the fire burned into urban Altadena. Trusting it dropped 176 real
> detections on the peak day. Persistence and pre-ignition activity decide
> instead.

The stop rule only works *because* static sources are removed: a refinery
lights up every day, so without step 2 a blank run never occurs.

### 6.3 `firegrid.py` -- the grid contract

Everything about geometry, resampling, labels and tensor shape.

| Function | Does |
|---|---|
| `snap_tile(lon, lat)` | 64 km tile, snapped to a global grid so overlapping fires share cell boundaries |
| `ee_layer_on_grid(layer, tile, ...)` | fetches one layer already aggregated onto the tile, server-side |
| `split_circular_components()` | averaged sin/cos -> unit direction + concentration R |
| `rasterize_fire_mask()` | burns detection footprints onto the grid (`all_touched=True`) |
| `mark_unobserved()` | promotes cells to *unobserved* across a coverage gap |
| `assemble_sample()` | stacks into `(18, 64, 64)` in the fixed channel order |
| `save_sample()` / `load_sample()` | `.npz` with channel names and spec embedded |

Resampling rules in `ee_layer_on_grid`:

- **Continuous:** area-weighted mean when downsampling, bilinear when
  upsampling.
- **Categorical (land cover):** majority class. Averaging classes 3 and 7
  invents class 5.
- **Circular (aspect):** split into sin/cos *before* averaging;
  `mean(350, 10)` is 180, pointing opposite both inputs.
- **Terrain:** slope and aspect computed at native 30 m, then aggregated.
- **Coarse-cadence products** (drought, NDVI) read a lookback window
  (`LAYER_LOOKBACK_DAYS`), because a single day is usually empty.

### 6.4 `visualize_firms_dataset_v3.py` -- FIRMS ingest

The original pipeline and the source of all detections. Also works on its
own for per-fire plots: `--fire-name "Eaton" --fire-year 2025 --fire-mask`.

Two rules worth knowing:

- **Group by `local_date`, not `acq_date`.** FIRMS reports UTC; weather
  products are indexed by local day.
- **`--use-archive` drops NOAA-21.** FIRMS publishes no `VIIRS_NOAA21_SP`
  product, so archive-only loses about 20% of a California pull. NRT sources
  serve historical dates fine.

### 6.5 `build_dataset.py` -- the driver

Sequences the modules and handles the details that are easy to get wrong:

- **Static layers fetch once per fire**, not once per step.
- **Lead-in days are fetched but not written**, so the first labelled step
  has a real previous mask.
- **Coverage gaps carry forward.** After a blank step, `prev_fire_mask` holds
  the last known mask for up to `--max-carry-days` (converted to steps), then
  expires. At hourly steps this matters most: a 12-hour overpass gap would
  otherwise declare the fire out every afternoon.
- **Featureless builds abort.** If Earth Engine returns nothing, the build
  stops instead of writing samples that carry only `prev_fire_mask`
  (override: `--allow-empty-features`).
- **Visuals reuse the same fetch.** `--visualize` writes plots and tables
  from the detections already in memory, on the same tile and dates as the
  tensors.
- **Ground truth is saved** as `perimeter.geojson` beside the samples, so
  verification works offline.

---

## 7. The grid specification

| Parameter | Value | Why |
|---|---|---|
| CRS | **EPSG:3310** (California Albers) | Equal-area metres. In lat/lon the same 64 km spans 0.683 deg at San Diego and 0.771 deg at the Oregon border -- 13% anisotropy |
| Cell | **1000 m** | Between 90 m terrain and 4 km weather; matches NDWS |
| Tile | **64 x 64 cells** | Far below the 262,144-value `sampleRectangle` cap, so nothing is silently coarsened |
| Origin | global grid at (0, 0) | overlapping fires share cell boundaries exactly |
| Fill | `NaN` | distinct from zero and from every label class |

Before the common grid, `grid_report.py` found terrain served **41x coarser**
than configured over California, and Palisades weather at 7 x 12 cells -- wind
direction effectively constant across the fire. Run `python grid_report.py`
to reproduce.

---

## 8. Time steps

`--step-hours` (or the time-step prompt) sets **how much time each sample
covers**: 24, 12, 8, 6, 4, 3, 2 or 1 hours, aligned to local midnight.

This is decided at extraction because it changes the data itself: each label
holds the detections from one window, and `prev_fire_mask` comes from the
window before. It also sets the model's horizon -- daily data trains a
next-day model, 12-hour data a next-12-hour model. A finer step cannot be
recovered from a coarser dataset later.

The cost of going finer is observation. Polar-orbiting satellites pass in
clusters with gaps of up to ~12 hours:

| Step | Windows with a satellite overhead while Eaton burned |
|---|---|
| 24 h | all |
| 12 h | ~65% |
| 6 h | ~45% |
| 1 h | ~14% |

Windows with no overpass are labelled **unobserved across the whole tile**:
nobody looked, so no cell can honestly be labelled "no fire". Hourly steps
need a geostationary source such as GOES-18.

Daily layers (weather, drought, vegetation) are fetched once per day and held
across all sub-daily steps in it (**zero-order hold**), so Earth Engine cost
does not grow with a finer step. Daily file names are unchanged
(`eaton_2025_2025-01-08.npz`); sub-daily ones add the hour
(`eaton_2025_2025-01-08T12.npz`).

---

## 9. Labels

| Value | Meaning |
|---|---|
| `0` | **No fire** -- observed, no detection |
| `1` | **Fire** -- a surviving detection footprint touches the cell |
| `2` | **Unobserved** -- coverage gap; exclude from the loss |

Class 2 exists because a missing detection is missing data, not a negative.
It is assigned in two ways:

- **Whole tile**, when no detection of any kind (including filtered static
  sources, which appear on every pass) falls in the window: no satellite
  looked.
- **Previously burning cells**, when a satellite passed but saw no fire
  anywhere in the tile: likely smoke or cloud over the fire.

`mark_unobserved()` is a heuristic. A per-pixel cloud mask (GOES or MODIS
MOD35) would do it properly; replace that function, not its callers.

---

## 10. Channel dictionary

Fixed order. A model trained on one ordering silently mispredicts on another.

| # | Channel | Source | Native | Resampling | Units |
|---|---|---|---|---|---|
| 0 | `elevation` | SRTM GL1 | 30 m | mean | m |
| 1 | `slope` | SRTM | 30 m | mean, derived at native | deg |
| 2 | `aspect_sin` | SRTM | 30 m | circular -> unit | east component |
| 3 | `aspect_cos` | SRTM | 30 m | circular -> unit | north component |
| 4 | `aspect_consistency` | SRTM | 30 m | circular concentration R | 0-1 |
| 5 | `vegetation` | VIIRS VNP13A1 | 500 m | mean, 24-day lookback | NDVI |
| 6 | `landcover` | ESA WorldCover | 10 m | **majority class** | class code |
| 7 | `population` | CIESIN GPWv411 | 927 m | mean | per km2 |
| 8 | `drought` | GRIDMET/DROUGHT | 4 km | bilinear, 20-day lookback | PDSI |
| 9 | `humidity` | GRIDMET `sph` | 4 km | bilinear | g/kg |
| 10 | `weather_temp` | GRIDMET `tmmx` | 4 km | bilinear | deg C (daily max) |
| 11 | `weather_precip` | GRIDMET `pr` | 4 km | bilinear | mm |
| 12 | `wind_speed` | GRIDMET `vs` | 4 km | bilinear | m/s |
| 13 | `wind_u` | GRIDMET `vs`,`th` | 4 km | bilinear | m/s toward east |
| 14 | `wind_v` | GRIDMET `vs`,`th` | 4 km | bilinear | m/s toward north |
| 15 | `erc` | GRIDMET | 4 km | bilinear | index |
| 16 | `vpd` | GRIDMET | 4 km | bilinear | kPa |
| 17 | `prev_fire_mask` | previous step's label | -- | -- | 0/1 |

Wind vectors point **where the wind is going** (GRIDMET's `th` is where it
comes from, hence the negation in `_gridmet_wind_components`).
`aspect_consistency` is a real feature: 1.0 when every 30 m cell in a 1 km
cell faces the same way, near 0 on a ridge crest. Below 0.05 the direction is
zeroed rather than amplified.

---

## 11. Verification

```powershell
python verify_dataset.py ml_dataset --json ml_dataset\verification.json
```

Or answer `y` to the verify prompt at the end of a build. Two layers of
checks, then pictures:

**Internal consistency** -- did the pipeline do what it claims? Uniform
shapes, CRS and cell size; tiles snapped to the grid; windows consecutive
with no gaps; `prev_fire_mask` matching the previous step; the CSV's used
rows rasterizing to **exactly** each label, cell for cell.

**Reality** -- does the data match the world?

- every channel within its physical range, with sniffers for Kelvin, kg/kg
  and unscaled NDVI
- **grid orientation:** aspect points downhill on the elevation channel
- wind components consistent with wind speed
- terrain identical in every sample; daily layers held within a day
- **labels against the CAL FIRE perimeter** -- an independent measurement
  from ground and aerial mapping
- first fire label near the alarm date; no fire on open water
- **known events** from the historical record, e.g. the January 2025 Santa
  Ana wind over Palisades and Eaton must point south-west

To prove the reality layer is needed, five broken datasets were built --
Kelvin temperatures, a flipped grid, backwards wind, averaged land cover,
labels in the wrong place. **All five pass every internal-consistency check
with zero failures.** The reality checks catch every one.

**QA sheets** in `<fire>/qa/`: `qa_labels.png` (every step, with the
perimeter outlined) and `qa_channels.png` (all 18 channels on the peak day,
with units and wind arrows). Look at them before training.

[`RUNBOOK.md`](docs/RUNBOOK.md) explains what every check proves and what a
failure means.

---

## 12. Output layout

```
ml_dataset/
|-- build_report.txt             run transcript: every prompt, answer and warning
|-- verification.json            every check result
|-- dataset.h5                   after export_hdf5.py
`-- eaton_2025/
    |-- eaton_2025_2025-01-08.npz    one per time step
    |-- ...
    |-- manifest.json                per-sample detections, fire and unobserved cells
    |-- perimeter.geojson            the ground truth labels are checked against
    |-- tables/
    |   |-- eaton_2025.csv           every detection, with keep/drop verdict
    |   |-- eaton_2025_centroids.csv
    |   `-- eaton_2025_clusters.geojson
    |-- visuals/                     overview and per-day plots (--visualize)
    `-- qa/
        |-- qa_labels.png
        `-- qa_channels.png
```

Each `.npz` holds `features`, `label`, `channels` and `meta` (fire, date,
window, step, CRS, tile, detection count, whether a satellite was overhead,
channels present, and the filter settings used).

---

## 13. Handing off to Swin-UNet

```powershell
python export_hdf5.py ml_dataset
python swin_data.py stats ml_dataset\dataset.h5 --train-fires PALISADES -o ml_dataset\norm_stats.json
python swin_data.py check --tile 64 --patch 2 --window 4
python swin_data.py selftest
```

**`export_hdf5.py`** packs the dataset into one file, chunked one sample per
chunk for random access. Its docstring includes a ready PyTorch `Dataset`.

**`swin_data.py`** handles what goes wrong silently between the data and a
transformer:

- **Geometry.** The published Swin-UNet setting (224 input, patch 4,
  window 7) **crashes on 64 x 64 tiles**: the 16 x 16 token grid is not
  divisible by 7. Use **patch 2, window 4**, which also keeps the finest
  token at 2 x 2 km.
- **Missing values.** NaN propagates through attention and makes the loss
  NaN on step one. `normalize()` fills and standardizes using statistics
  from **training fires only**.
- **Channel types.** Wind and aspect are scaled but never mean-centred;
  `prev_fire_mask` stays 0/1; land cover stays as class codes (one-hot or
  embed it in the model).
- **Augmentation rotates the vectors.** Rotating the tile without rotating
  the wind makes wind that blew uphill blow downhill. `augment()` handles all
  16 flips and rotations; `selftest` proves it.

For training: two output classes with `ignore_index=2` for unobserved cells;
fire is under 1% of cells, so use weighted, focal or Dice loss and report F1
or IoU on the fire class, never accuracy. There are no pretrained weights for
18 channels at patch 2, so **the number of fires matters more than
architecture tuning**, and a plain CNN U-Net is worth training as a baseline.

---

## 14. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: _cffi_backend` | pip overwrote conda's `cffi` | `pip install --force-reinstall --no-cache-dir cffi cryptography` |
| `Error importing numpy ... source directory` | numpy's C extension broken, same cause | reinstall numpy with whichever manager owns it (`conda list numpy`) |
| `No module named 'src'` | an editor rewrote `import firegrid` as `import src.firegrid` | remove the `src.` prefix; add `"python.analysis.extraPaths": ["./src"]` to `.vscode/settings.json` |
| `???` in plot titles | a file was re-saved through PowerShell with the wrong encoding | use the current files, which are pure ASCII |
| `EARTH ENGINE UNAVAILABLE` | authentication or project | `earthengine authenticate`, then `python verify_ee_setup.py` |
| fewer than 18/18 channels | a layer failed to fetch | read the `grid fetch failed` line above it |
| empty daily masks | several possible causes | `python diagnose_daily.py <dataset folder>` |
| a verification `[FAIL]` | see the table in `RUNBOOK.md` | -- |

---

## 15. Known limitations

- **Labels come from polar orbiters.** Fine time steps leave most windows
  unobserved; hourly needs GOES-18.
- **`mark_unobserved()` is a heuristic**; there is no per-pixel cloud mask.
- **The perimeter is final, not daily**, so day-by-day progression cannot be
  checked against it.
- **Weather is 4 km** resampled to 1 km.
- **Tiles of concurrent fires overlap.** Eaton's reaches into the Palisades
  burn area. Split train and test by geography or date, **never by fire
  name**.
- **Only two fires so far.** Automated event discovery
  (`firegrid.discover_fire_events`) exists but is not yet wired into the
  driver.
- **Interactive `getInfo`, not batch export.** Fine for a few fires; a
  statewide build needs `Export.image.toDrive`.
- **Known-event checks exist only for Palisades and Eaton.** Add entries to
  `KNOWN_EVENTS` in `reality_checks.py` for other documented fires.

---

## 16. Security

- **Never commit or paste a MAP_KEY.** [`KEY_ROTATION.md`](docs/KEY_ROTATION.md)
  has the rotation runbook. Send `build_report.txt` when asking for help --
  it redacts credentials; raw console output does not.
- **Never share** `~/.config/earthengine/credentials`: it is a live token for
  your whole Google account.
- **Dotfile names matter.** Git reads `.gitignore` and pre-commit reads
  `.pre-commit-config.yaml`; `_gitignore` or `pre-commit-config.yaml` do
  nothing.
- **Commit** `.pre-commit-config.yaml` (with `--baseline .secrets.baseline`)
  and `.secrets.baseline`.
- **In `detect-secrets audit`, "Should this string be committed?" -- answer
  `y` for false positives.** `n` marks it as a real secret and blocks every
  commit.
- **Do not generate machine-read files with `>` or `Set-Content -Encoding
  utf8` in Windows PowerShell 5.1** -- both add a byte-order mark that breaks
  JSON and rule files. Use `cmd /c "... > file"` or `-Encoding ascii`.
