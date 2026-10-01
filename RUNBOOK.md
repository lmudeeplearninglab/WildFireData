# Runbook -- building and verifying a training dataset

How to run the pipeline, and how to know the data is right before you train
on it. Commands are PowerShell, run from the `src` folder:

```powershell
conda activate wfdata
cd C:\GradResearch_GeoTransformer_Wildfires_2025\WildFireData\src
```

---

## What "verified" means here

A dataset has to pass three different kinds of check. Each catches mistakes
the other two cannot.

| Layer | Question | Done by |
|---|---|---|
| **Internal consistency** | Did the pipeline do what it claims? Shapes, grid alignment, consecutive dates, labels matching the CSV cell for cell | `verify_dataset.py` |
| **Physical reality** | Does the data match the world? Units, ranges, orientation, the agency burn perimeter, known historical conditions | `verify_dataset.py` (reality section) |
| **Human inspection** | Does it *look* right to someone who knows the fire? | the QA sheets |

The second layer exists because the first can be fooled. Five deliberately
broken datasets -- temperatures in Kelvin, the grid flipped north-south, wind
pointing backwards, land cover averaged, labels in the wrong place -- **all
pass every internal-consistency check with zero failures.** Only the reality
checks catch them.

---

## 1. Check Earth Engine (once per session)

```powershell
python verify_ee_setup.py --date 2025-01-10
```

**Expect:** every line `[ok]`, ending in `All checks passed`.
**If not:** fix it now. A build with broken Earth Engine produces samples
with 1/18 channels, which the build will refuse to write.

## 2. Build, then verify -- one command

```powershell
python build_dataset.py
```

Answer the prompts. Pressing Enter everywhere gives sensible defaults. At the
end, the plan is shown; answer `d` the first time to see it without fetching,
then run again and answer `y`. When asked **"Verify the dataset when the build
finishes?"** answer `y` (the default).

### The time-step prompt

```
Step in hours (24, 12, 6, 3, 1) [24]:
```

This asks **how much time each extracted sample covers**. It is decided at
extraction, not at training, because it changes what the data is.

Each sample is a snapshot: a fire mask built from every detection inside one
time window, plus a `prev_fire_mask` from the window before it. The step sets
the length of that window:

- **24** -- every detection in a local day goes into one label. Eaton gives
  9 samples, one per day.
- **12** -- each day is split into 00:00-12:00 and 12:00-24:00. The same
  detections, sliced twice as finely: 17 samples, each with its own label and
  its own previous mask from 12 hours earlier.

The labels and the `prev_fire_mask` channel are physically different depending
on the answer. **A finer step cannot be recovered from a coarser dataset** --
12-hour labels do not exist inside daily ones -- so changing your mind means
re-extracting. The other direction works: two 12-hour labels can be merged
into a daily one.

The prompt calls it a *prediction* step because whatever step you extract at
becomes the horizon the model learns. Swin-UNet is trained on "given this
window, predict the next one", so daily data trains a next-day model and
12-hour data trains a next-12-hour model. Shortening the warning time for
firefighters is decided here, not in the model code.

The percentages on screen are the cost of going finer:

| Step | Windows with a satellite overhead | Windows where it saw fire |
|---|---|---|
| 24 h | all | all |
| 12 h | 94% | 65% |
| 6 h | 61% | 45% |
| 3 h | 31% | 23% |
| 1 h | 20% | 14% |

Measured on the Eaton detections while the fire was active. The first column
is what matters for labels: a window with no satellite overhead is labelled
*unobserved* across the whole tile. The second is lower because a satellite
can pass without seeing the fire -- smoke, cloud, or a lull. Your build prints
both, for your fire, on the `coverage` line.

Polar-orbiting satellites pass in clusters, with gaps of up to ~12 hours.
Unobserved windows contribute nothing to training. Hourly steps need a
geostationary source such as GOES before they are practical.

Earth Engine cost does not grow with a finer step: the daily weather,
drought and vegetation layers are fetched once per day and held constant
across every sub-daily window in it (zero-order hold).

**Which to pick:**

- **24** to compare against the published *Next Day Wildfire Spread*
  benchmark (Huot et al., 2022).
- **12** when working toward faster prediction.
- Both, into separate output folders, if you want to compare. It costs very
  little, since the Earth Engine requests are the same either way.

**While it builds, watch for:**

| Line | Good | Stop and investigate |
|---|---|---|
| `quality  kept X of Y detections` | most kept; drops explained | nearly everything dropped |
| per-step rows | `18/18 channels` on every row | anything less than 18 |
| `EARTH ENGINE UNAVAILABLE` | never appears | appears -> go back to step 1 |
| `stop rule ... after <date>` | a date after the fire quieted | a date during the active fire |

**Exit code:** `0` = built and verified; `2` = built but verification
failed; `1` = nothing built. In PowerShell: `echo $LASTEXITCODE`.

The non-interactive equivalent:

```powershell
python build_dataset.py --fire "Eaton" --year 2025 --visualize --verify
```

To re-verify a dataset later without rebuilding:

```powershell
python verify_dataset.py ml_dataset --json ml_dataset\verification.json
```

## 3. Read the verification

Every line is `[ok]`, `[warn]` or `[FAIL]`. **Any `[FAIL]` means do not
train on this dataset.** A `[warn]` means look at the QA sheet and decide.

### Reality checks -- what each one proves

| Check | Proves | A failure means |
|---|---|---|
| every channel inside its physical range | no impossible values | wrong units, corrupt fetch, fill values leaking through |
| units look physical | no Kelvin, kg/kg or unscaled NDVI | a conversion was skipped |
| landcover holds only ESA classes | land cover was not averaged | categorical resampling is broken |
| spatially varying channels vary | real fields, not a broadcast scalar | a fetch returned one number |
| **grid orientation** | aspect points downhill on the elevation map | the grid is flipped or rotated somewhere |
| wind components consistent | `|(u,v)|` never exceeds wind speed | the wind decomposition is wrong |
| static layers identical | terrain is the same in every sample | static fetch is re-running or corrupted |
| daily layers held within a day | zero-order hold works (sub-daily steps only) | weather changes inside a single day |
| **labels sit on the mapped burn area** | fire cells fall on the agency perimeter | static sources left in, or other fires in the tile |
| **labels cover the burn area** | the fire was actually captured | missing days, over-filtering |
| first fire label matches alarm date | timing is right | days missing, or fire in the tile before ignition |
| fire on open water | labels and features are aligned (registration test) | labels are offset from the features |
| **known events** | conditions match the historical record | the data, or the recorded claim, is wrong |

**Fire on open water** is a `[warn]`, not a `[FAIL]`, for a fire that burned
to the coast. A satellite detection is a footprint -- up to 3.4 x 1.7 km for
MODIS -- and every 1 km cell it touches is labelled fire, so a detection on
the beach also labels the first offshore cell. The check separates that from
real misalignment by sliding the perimeter up to 3 cells in every direction
over the land-cover water: if the unshifted grid fits best, the data is
aligned. It also reports which sensor the offshore cells came from and
whether any detection *centre* is offshore. It fails only when a shift fits
better, or when detection centres sit more than 2 km from any land cell.

The 2 km allowance is deliberate. Land cover is resampled by majority class,
so a shore cell whose land is split between built-up, shrub and beach can come
out "water" while mostly land, which moves the coastline about a cell inland.
On Palisades, 22 low-power detections from 8 January sat about a kilometre off
the Malibu shore and were wrongly failed with a 1 km allowance.

**Known events** are dated by the conditions, not the label. Features lag the
label by one day, so the Santa Ana wind of 2025-01-07 is tested on the
2025-01-08 sample. Each claim and its source is in `KNOWN_EVENTS` in
`reality_checks.py`.

The perimeter is the strongest check. It comes from CAL FIRE's ground and
aerial mapping -- a completely different measurement from the satellite
detections the labels are built from -- so agreement between them is
independent evidence the labels are right.

**Reading the perimeter numbers.** Roughly: above 80% of fire cells near the
perimeter and above 50% of the perimeter covered is healthy. When the pipeline
was tested on the real Eaton detections -- against a stand-in perimeter, since
the real polygon could not be downloaded in testing -- it gave 84% near, 92%
covered, and the first fire label on the alarm date to the day. Your own run
will report figures against the real mapped polygon; expect them to differ.

Of the fire cells outside the perimeter in that test, just over half (14 of 25)
were a second fire on 2025-01-08 north of Eaton -- almost certainly the Lidia
Fire near Acton -- which is real fire, not an error. The rest were scattered
single cells. `--target-only` excludes other fires if the model should learn
one fire at a time.

**Other fires are now named, and scored separately.** The build prints a
`fires` roll call: every fire in the tile with its ID, detection count,
dates and source. The perimeter checks score fire 1 only, and the verifier
notes how many cells belong to other fires. Expect the "near the perimeter"
share to rise once other fires stop counting against it.

## 4. Look at the QA sheets

Every fire gets two images in `ml_dataset\<fire>\qa\`. **Open them before
training.** They take a minute and catch things no automated check can.

### `qa_labels.png` -- one panel per step

Fire cells are coloured by fire: red = the fire you built, other colours =
other fires (key at the bottom, dashed outlines = their perimeters). Grey =
unobserved, orange outline = the previous step's fire (what the model is
given), cyan = the agency perimeter.

- [ ] The red sits inside or against the cyan outline
- [ ] The fire grows, then shrinks, in a plausible order
- [ ] The orange outline in each panel matches the fire of the panel before
- [ ] Every non-red fire is one you can name, or a plausible small fire
- [ ] No other fire sits on a refinery, landfill or power plant

### `visuals\<fire>_fires.png` -- every fire in the tile

One map, every fire in its own colour with its name, dates and detection
count; small fires are ringed so they can be found. Check the named fires
against what you know burned, and look up any `Unmapped` fire that persists
for several days at one spot: that is the signature of an industrial site.
- [ ] Grey appears where you would expect missing observations, not randomly

### `qa_channels.png` -- all 18 channels on the peak day

Each panel shows its units and its min .. max. The red outline is the fire.

- [ ] **Elevation** looks like the real terrain (for Eaton: San Gabriel
      Mountains rising to the north, the basin to the south)
- [ ] **Wind arrows** point where the wind was going. For Palisades and Eaton
      on 2025-01-08, the Santa Ana blew from the north-east, so arrows point
      **south-west**
- [ ] **Temperature** is a believable daily maximum in deg C
- [ ] **Vegetation** is lower over the city than the mountains
- [ ] **Population** is high over the city, near zero in the mountains
- [ ] No panel is blank or a single flat colour (unless that layer is
      genuinely uniform at 4 km, like PDSI over a small area)

## 5. Package for training

Only after steps 3 and 4 pass:

```powershell
python export_hdf5.py ml_dataset
python swin_data.py stats ml_dataset\dataset.h5 --train-fires PALISADES -o ml_dataset\norm_stats.json
python swin_data.py check --tile 64 --patch 2 --window 4
python swin_data.py selftest
```

`--train-fires` must list **training fires only**. Normalization statistics
computed over the test fires leak the test set into training.

## 6. What to keep

Keep these together with the dataset. They are what lets someone else --
or you in three months -- trust it:

- `build_report.txt` -- every prompt, answer, fetch and warning
- `verification.json` -- every check result
- `<fire>\qa\*.png` -- the visual record
- `<fire>\perimeter.geojson` -- the reference the labels were checked against
- `<fire>\fires.geojson` -- which fire every `fire_id` is
- `<fire>\tables\*.csv` -- every detection with its keep/drop verdict

When asking for help, send `build_report.txt`, not console output: the
report has credentials redacted, the console does not.

---

## When a check fails

| Symptom | Likely cause | What to do |
|---|---|---|
| `looks like Kelvin` | temperature conversion skipped | check `_get_ee_image_for_layer` in the pipeline |
| `grid orientation ... cosine` negative | rows reversed during extraction | check `fit_to_shape` and the `sampleRectangle` row order |
| known-event wind heading ~180 deg off | u/v sign flipped | check `_gridmet_wind_components` |
| landcover has non-class values | mean instead of mode | check `LAYER_RESAMPLING["landcover"]` |
| low share of fire near the perimeter | other fires or static sources in the tile | look at `qa_labels.png`; consider `--target-only` |
| low perimeter coverage | days missing or filters too strict | check the stop-rule date and `--min-confidence` |
| `no perimeter.geojson` | dataset built with an older version | rebuild |
| `EARTH ENGINE UNAVAILABLE` | authentication or project | `earthengine authenticate`, then step 1 |
| `footprint spillover at the coast` [warn] | a coastal fire; footprints reach past the shoreline | expected; see "Label size" below if it matters |
| `labels registered to the features` [FAIL], `shifting land cover by` | labels offset from features | check tile origin and `fit_to_shape` |
| known-event rain fails | rain over the fire, or (older builds) anywhere in the tile | the check now scores the fire's area; the note gives the wettest cell elsewhere |
| `alarm date ... first seen ... labels start` | FRAP stores the UTC date; the fire started the previous evening, local time | expected; that night becomes the first sample |
| fire labels on late days far from the fire, at low power | a refinery or flare that escaped the static filter | check `tables\*.csv` for the site; see the static-filter note in CHANGES |
| cells ever burning many times the perimeter area | footprint rasterization (`all_touched`) | see "Label size" below |

### Label size

Labels are rasterized from detection footprints: every 1 km cell a footprint
touches becomes fire. For Palisades this gives 276 km2 of cells ever burning
against a 95 km2 perimeter. Part of that is unavoidable at 1 km; part is
MODIS, whose footprints are several times larger than VIIRS. The
`modis_only_fire_share` note in the verification shows how much of the label
comes from MODIS alone. Whether to keep it, rasterize MODIS by centre only,
or drop MODIS is a modelling decision -- record it with the dataset.

## Adding a new fire

Everything works for any fire by name. The **known-event** checks only exist
for fires listed in `KNOWN_EVENTS` in `reality_checks.py`. When you add a
fire whose conditions are well documented -- a named wind event, a heat
wave, a rain-ending date -- add an entry. Each one turns a historical fact into
an automatic test, and they are the only checks that catch a wind vector
pointing the wrong way.

## What verification cannot prove

Be explicit about these in any write-up:

- **FIRMS misses fire.** Cloud, smoke and gaps between overpasses mean some
  burning is never detected. The unobserved class reduces the damage but
  cannot recover what was never seen.
- **The perimeter is final, not daily.** It shows where the fire burned in
  total, not where it was on a given day. Day-by-day progression cannot be
  checked against it.
- **Weather is at 4 km.** Checks confirm it is physical, not that it is right
  at 1 km.
- **Known-event checks are only as good as the record.** If one fails, check
  the claim as well as the data.
