# Changes -- September 2026

Updates since the last `CHANGES.md` (session 2: gridding and automation).
For how the pipeline works, see `README.md`; for how to run and verify a
build, see `RUNBOOK.md`.

---

## Action needed

1. **Replace every file in `src/`.** Several files were corrupted by a
   PowerShell re-encode (plot titles showed `???`). All sources are now pure
   ASCII so this cannot recur.
2. **Reinstall requirements.** `pyproj` and `h5py` were missing, and the
   geospatial stack is now required rather than optional:
   `pip install -r requirements.txt`.
3. **Delete existing datasets and rebuild.** Defaults changed (below), and old
   datasets have no `perimeter.geojson`, so the new ground-truth check skips
   them.
4. **Replace `.pre-commit-config.yaml`** and regenerate the baseline:
   `cmd /c "detect-secrets scan > .secrets.baseline"`.

---

## Headline changes

### Data is now verified against the real world

`verify_dataset.py` previously checked only that the pipeline was internally
consistent. A new reality layer (`reality_checks.py`) compares the data with
things the pipeline did not produce: physical ranges and units, terrain
orientation, the CAL FIRE mapped perimeter, and conditions recorded for
specific historical fires.

Five deliberately broken datasets -- temperatures in Kelvin, the grid flipped
north-south, wind pointing backwards, land cover averaged, labels in the
wrong place -- **all passed the old checks with zero failures.** The new
checks catch every one.

Each fire also gets two QA images in `<fire>/qa/`: every labelled step with
the perimeter outlined, and all 18 channels on the peak day with units and
wind arrows.

### Non-fire heat sources are removed from labels

New module `firefilter.py`. Refineries, gas flares and power plants appear in
FIRMS as heat every day; in the Eaton pull they were 83% of all detections on
2025-01-13, exactly the days where a model learns how fires die down.

It also normalizes confidence across VIIRS (`l/n/h`) and MODIS (0-100), can
drop other fires sharing the tile, and stops a build after consecutive days
with no fire.

### Time steps finer than a day

`--step-hours` (or the new time-step prompt): 24, 12, 8, 6, 4, 3, 2 or 1 hour
samples. This sets the prediction horizon the model will learn. Weather layers
are fetched once per day and held across sub-daily steps, so Earth Engine cost
does not increase. See README section 8 for the trade-off: finer steps leave
more windows with no satellite overhead.

### Swin-UNet handoff

- `export_hdf5.py` packs a dataset into one HDF5 file for training.
- `swin_data.py` validates the model geometry, normalizes channels using
  training fires only, and augments with rotations that rotate the wind with
  the map.

### One-command build and verify

`build_dataset.py` now asks whether to verify when the build finishes, and
optionally writes the plots and tables from `visualize_firms_dataset_v3` for
the same fire and dates. Pressing Enter through every prompt builds, plots and
verifies. Exit code `2` means built but failed verification.

---

## New files

| File | Purpose |
|---|---|
| `firefilter.py` | Which detections become labels |
| `reality_checks.py` | Real-world checks and QA images |
| `verify_dataset.py` | Checks a built dataset (now calls `reality_checks`) |
| `export_hdf5.py` | Dataset -> single HDF5 file |
| `swin_data.py` | Swin-UNet geometry, normalization, augmentation |
| `RUNBOOK.md` | Step-by-step build and verification guide |
| `.gitattributes` | Consistent line endings between Windows and Linux |

---

## Changed behaviour

These change what a build produces. Rebuild before comparing with older
datasets.

| Setting | Before | Now |
|---|---|---|
| Detections in labels | every detection in the tile | confidence >= nominal, static sources removed |
| End of a build | containment date + tail | stops after 3 consecutive blank days |
| Window with no satellite overhead | labelled "no fire" | labelled **unobserved** across the whole tile |
| Earth Engine returns nothing | writes samples with 1/18 channels | aborts with an explanation |
| CSV output | kept detections only | every detection, with keep/drop reason |
| Sample `meta` | fire, date, tile | adds window, step, filter settings, whether observed |

Eaton, for example, now produces 9 daily samples instead of 27: once static
sources were removed, the fire went quiet after 2025-01-16 and the remaining
days held no fire detections.

---

## New interactive prompts

`python build_dataset.py` now asks, in order:

1. **Fires** by name
2. **Date window**, including the **time step**
3. **Feature layers**
4. **Visualizations**: none, overview, or overview plus per-day folders
5. **Detection quality**: confidence, static sources, target fire only, stop
   rule, minimum FRP
6. **Output** directory
7. **Proceed**, then **verify when finished**

---

## Bugs fixed

- **`???` in plot titles.** Caused by re-saving sources through PowerShell
  with the wrong encoding; all sources are now ASCII.
- **Vegetation failed on every step** (`reduceResolution ... no valid default
  projection`): composites now get their native projection set first.
- **Drought failed four days in five** (`Invalid band number (0)`): the
  product is published every 5 days, so single-day requests were empty. Now
  reads a 20-day window.
- **Earth Engine failures were silent.** A build could finish in 20 seconds
  with 1 of 18 channels and report success. Outages are now reported and the
  build aborts.
- **Fine time steps claimed confirmed no-fire when no satellite was looking.**
  In the Eaton test at 1-hour steps, 79.9% of cells are now correctly
  unobserved (was 0.3%).
- **Wind arrows in QA images were mirrored north-south** (caught before
  release).
- **`grid_report.py`** broke after the `v3` rename.
- **Pre-commit blocked every commit.** The hook ignored the baseline and
  scanned the baseline file itself. Public dataset IDs that tripped the
  detector are now marked allowed.
- **`requirements.txt`** was missing `pyproj` and `h5py`.

---

## Findings worth knowing

- **FIRMS mislabels real fire as static.** Its `type` field marked 44
  detections inside the Eaton burn perimeter -- up to 800 MW on ignition
  night -- as "other static land source", probably because the fire reached
  urban Altadena. The filter therefore relies on persistence and pre-ignition
  activity, not on that field.
- **A second fire shares Eaton's tile on 2025-01-08** north of the main fire,
  almost certainly the Lidia Fire near Acton. It is real fire; use
  `--target-only` to exclude it.
- **The published Swin-UNet setting does not run on 64 x 64 tiles.** Patch 4
  with window 7 gives a 16 x 16 token grid that 7 does not divide. Use patch
  2, window 4.
- **Satellite coverage limits the time step.** While Eaton burned, a
  satellite was overhead in about 65% of 12-hour windows and 14% of 1-hour
  windows. Hourly prediction needs GOES-18.

---

## Repository

- Code lives in `src/`. Documentation (`RUNBOOK.md`, `CHANGES.md`,
  `KEY_ROTATION.md`, `REVIEW.md`) belongs in `docs/` -- the README links
  point there.
- Reference notebooks belong in `references/`, with a README explaining they
  are prior art rather than pipeline code.
- `README.md` rewritten for the current pipeline.
- `requirements.txt` replaces `requirements_v2.txt`.

---

## Open items

- Only two fires so far. With 18 channels and no pretrained weights,
  Swin-UNet needs many more; automated fire discovery is the next priority.
- Hourly labels need GOES-18 fire detections, which also provide a real
  per-pixel cloud mask to replace the unobserved heuristic.
- Statewide builds need Earth Engine batch export instead of interactive
  requests.
- No training/test split yet. Tiles of concurrent fires overlap, so split by
  geography or date, never by fire name.
- Decide with the advisor: observed vs forecast wind as a feature, and the
  persistence baseline every fine-step result must beat.
