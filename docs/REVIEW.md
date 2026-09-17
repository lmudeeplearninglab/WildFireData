# Code review — visualize_firms_dataset.py

Findings ordered by how much they threaten the research result, not by how hard they are to fix. Line numbers reference the reviewed version.

Sections: **A** correctness, **B** algorithm and ML validity, **C** performance, **D** infrastructure and hardware, **E** code health.

---

## A. Correctness bugs

### A-1. Hardcoded API key in source — line 292

```python
DEFAULT_FIRMS_MAP_KEY = "<redacted-rotated-key>"
```

A live credential is sitting in a file you're about to hand to other developers. If this ever touches a public repo, assume it's compromised — scrapers find these within minutes, and NASA's quota is per-key, so anyone can exhaust yours.

Rotate that key now, then:

```python
def get_map_key() -> str:
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if key:
        return key
    if not sys.stdin.isatty():
        raise SystemExit(
            "FIRMS_MAP_KEY not set. Get one free at "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key"
        )
    return input("Paste MAP_KEY: ").strip()
```

If it's already in git history, `git filter-repo` or BFG — rotating alone doesn't remove it from past commits.

### A-2. Confidence filter crashes or silently deletes MODIS — lines 96–98

```python
valid = (out["confidence"].isin(confidence)) if "confidence" in out.columns else True
out = out.loc[valid]
```

Two problems. First, when the column is missing, `out.loc[True]` raises:

```
KeyError: 'True: boolean label can not be used without a boolean index'
```

Second and worse: VIIRS reports confidence as `l`/`n`/`h`, MODIS as an integer 0–100. `SOURCES` mixes both. I ran this — `--confidence n,h` against a MODIS frame keeps **0 of 2 rows**. You lose every MODIS detection with no warning, which quietly changes your dataset's sensor composition.

```python
if confidence and "confidence" in out.columns:
    col = out["confidence"]
    if pd.api.types.is_numeric_dtype(col):
        # MODIS: map letters to the documented numeric bands
        bands = {"l": (0, 30), "n": (30, 80), "h": (80, 101)}
        mask = pd.Series(False, index=out.index)
        for c in confidence:
            lo, hi = bands[c.lower()]
            mask |= col.between(lo, hi, inclusive="left")
        out = out[mask]
    else:
        out = out[col.str.lower().isin([c.lower() for c in confidence])]
```

### A-3. `UnboundLocalError` when Earth Engine is requested but not installed — lines 2534/2565 and 2607/2638

`ee_start_date` is bound only inside `if args.enrich_earth_engine and HAS_EARTH_ENGINE:`, then read at:

```python
start_date=ee_start_date if args.enrich_earth_engine else args.start_date,
```

Run `--enrich-earth-engine --daily-visual-samples` without `earthengine-api` installed and it crashes after all the fetching is done. Hoist the assignment above the branch:

```python
ee_start_date, ee_end_date, date_str = _get_fire_date_window(df)
```

### A-4. Source fallback produces an inconsistent sensor mix — lines 421–425

```python
for src in sources:
    chunk = fetch_area_csv(map_key, src, bbox, day_range=n_days, date=chunk_date)
    if not chunk.empty:
        chunks.append(chunk)
        break
```

`break` on first non-empty means chunk 1 might be VIIRS_SNPP while chunk 2 is NOAA-20 and chunk 3 is MODIS. Your detection density then varies by *satellite availability* rather than by fire activity, and a model will happily learn that artifact.

Fetch all requested sources, tag them, and concatenate:

```python
for src in sources:
    chunk = fetch_area_csv(map_key, src, bbox, day_range=n_days, date=chunk_date)
    if not chunk.empty:
        chunks.append(chunk.assign(firms_source=src))
```

Dedup on `["latitude", "longitude", "acq_date", "acq_time", "satellite"]` — the current key omits `satellite` (lines 429–431), so two satellites detecting the same pixel at the same minute collapse to one row.

### A-5. `add_acq_datetime` raises on mixed-dtype `acq_time` — lines 78–79

If any concatenated chunk lacks `acq_time`, pandas promotes the column to float and `str(1234.0)` → `"1234.0"`, which `zfill(4)` won't fix. Verified:

```
dtype after concat: float64
['2025-01-07 1234.0', nan]
add_acq_datetime would RAISE: ValueError
```

```python
t = pd.to_numeric(df["acq_time"], errors="coerce")
time_str = (df["acq_date"].astype(str) + " "
            + t.fillna(0).astype(int).astype(str).str.zfill(4))
df["acq_datetime"] = pd.to_datetime(time_str, format="%Y-%m-%d %H%M",
                                    errors="coerce", utc=True)
```

Note `utc=True` — see B-4.

### A-6. Shapefile subsetting mislabels the CRS — lines 150–154

```python
fire_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(...), crs=states_gdf.crs)
```

This *asserts* your lon/lat points are already in the shapefile's CRS. If the shapefile is in an Albers or State Plane projection — common for US state boundaries — the spatial join returns garbage without erroring. Build in 4326, then reproject:

```python
fire_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(
    df["longitude"], df["latitude"]), crs="EPSG:4326").to_crs(states_gdf.crs)
```

### A-7. API errors are indistinguishable from "no fires" — line 336

`pd.read_csv(url)` with no timeout and no status check. An invalid key, a quota exhaustion, and a genuinely fire-free window all produce the same empty DataFrame. Use `requests` and look at the body:

```python
r = requests.get(url, timeout=30)
r.raise_for_status()
if "Invalid MAP_KEY" in r.text[:500] or "<html" in r.text[:200].lower():
    raise RuntimeError(f"FIRMS rejected the request: {r.text[:200]}")
df = pd.read_csv(io.StringIO(r.text))
```

Wrap with `urllib3.Retry` on 429/5xx and a backoff — FIRMS throttles per rolling 10-minute window, and a multi-month California pull will hit it.

### A-8. Deprecated matplotlib API — lines 524, 558

`plt.cm.get_cmap` is deprecated since 3.7 and **removed in 3.11**. Use `plt.get_cmap("YlOrRd", n)` or `matplotlib.colormaps["YlOrRd"].resampled(n)`.

### A-9. Caller's DataFrame mutated — line 519, 556

`add_acq_datetime()` returns the *original* object when `acq_date`/`acq_time` are absent, then `df["_day"] = ...` writes through to the caller's frame. Add `df = df.copy()` at the top of `plot_fire_map_time_based`.

### A-10. Division-by-zero warning near the poles — line 193

`np.where` evaluates both branches, so `cos(radians(90))` is computed regardless. Not fatal at California latitudes but noisy:

```python
cos_lat = np.cos(np.radians(np.clip(lat, -89.99, 89.99)))
dx = coefs / (cos_lat * KM_PER_DEG) / 2.0
```

---

## B. Algorithm and ML validity

These are the ones that would show up as an unexplained accuracy ceiling six months from now.

### B-1. Isolated detections are silently dropped — lines 249–253

```python
for label in sorted(set(labels)):
    if label < 0:
        continue  # noise points
```

With `min_samples=2`, a lone detection is DBSCAN noise and vanishes from both polygons and the centroids CSV. For a *real-time prediction* system this is exactly backwards: a single fresh detection with no neighbors is a new ignition, and new ignitions are your most valuable positive samples. Large established fires — the ones that cluster easily — are the easy cases.

Either set `min_samples=1` for the label-generation path, or keep noise points as singleton clusters:

```python
noise = df[labels < 0]
for i, (_, row) in enumerate(noise.iterrows()):
    centroid_rows.append({"cluster_id": -(i + 1), "centroid_lat": row["latitude"],
                          "centroid_lon": row["longitude"], "n_detections": 1,
                          "is_singleton": True})
```

### B-2. DBSCAN in degrees is anisotropic and time-blind — line 244

```python
clustering = DBSCAN(eps=eps_deg, min_samples=min_samples, metric="euclidean")
```

Two issues:

**Anisotropy.** 0.02° of latitude is 2.22 km everywhere; 0.02° of longitude at 34°N is 1.84 km. Your cluster radius is ~17% tighter east-west than north-south, and the distortion grows across California's latitude span. Use the haversine metric:

```python
coords = np.radians(df[["latitude", "longitude"]].values)
eps_rad = eps_km / 6371.0
DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine",
       algorithm="ball_tree").fit_predict(coords)
```

**No time dimension.** A fire in January and a different fire at the same location in August merge into one cluster. Over a multi-month archive pull this is guaranteed. Cluster in space-time by appending a scaled time axis:

```python
t_days = (df["acq_datetime"] - df["acq_datetime"].min()).dt.total_seconds() / 86400
X = np.column_stack([x_km, y_km, t_days * spread_km_per_day])   # ~15 km/day
```

`spread_km_per_day` is the knob controlling how aggressively you link detections across days. Alternatively run DBSCAN per rolling window and link clusters between windows by overlap.

### B-3. Convex hull is the wrong shape for a fire — lines 277–286

Real fire perimeters are concave — they run up canyons and stretch downwind into long fingers. A convex hull over an L-shaped or crescent-shaped fire can easily double the reported area. If `footprint_area_km2` or hull area ever becomes a training target or feature, that error propagates directly.

Shapely 2.1 has a concave hull:

```python
from shapely import concave_hull
hull = concave_hull(MultiPoint(points), ratio=0.3)
```

Or union the footprints with a small buffer, which respects actual detection geometry:

```python
from shapely.ops import unary_union
hull = unary_union([p.buffer(0.002) for p in footprint_polys]).buffer(-0.002)
```

Worth validating either against CAL FIRE / NIFC official perimeters for Palisades and Eaton — you have ground truth for both, and it turns a guess into a measured error bar.

### B-4. UTC/local-day misalignment

FIRMS `acq_time` is UTC. GRIDMET days are local. A detection at 06:00 UTC on Jan 8 is 22:00 Jan 7 Pacific — night overpasses systematically land on the wrong day. Since VIIRS has both day and night passes, roughly half your detections are affected. Convert to local time before deriving `acq_date` for joins:

```python
df["acq_datetime_utc"] = pd.to_datetime(..., utc=True)
df["local_date"] = df["acq_datetime_utc"].dt.tz_convert("America/Los_Angeles").dt.date
```

### B-5. Temporal leakage in the period-mode window — lines 1058–1060

```python
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
```

The feature window is *inclusive of the fire day*. Post-fire NDVI collapses and land surface temperature spikes, so a model given these features can identify fires from the burn signature rather than predicting them. Test-set accuracy looks excellent; real-time accuracy doesn't.

Interestingly `_get_composite_for_date` at line 2120 uses `filterDate(start_str, date_str)` — end-exclusive, which is correct. So the two code paths disagree. Make the raster path match, and add an explicit `--feature-lag-days` (default 1) so the gap is a stated experimental parameter rather than an implementation detail.

### B-6. Layers are never resampled to a common grid

Elevation comes back at 500 m, GRIDMET at 4000 m, land cover at 250 m. Each `.npy` has a different shape. Nothing in the codebase aligns them, so "stack these into a training tensor" is still entirely unsolved.

Pick a target grid — 375 m matches VIIRS, which is the natural resolution for your labels — and resample everything to it inside Earth Engine before extraction, so one request returns a co-registered stack:

```python
target = ee.Projection("EPSG:4326").atScale(375)
stack = ee.Image.cat([
    dem.rename("elevation"),
    ee.Terrain.slope(dem).rename("slope"),
    ee.Terrain.aspect(dem).rename("aspect"),
    ndvi.rename("ndvi"),
    gridmet.select(["tmmx", "sph", "vs", "th", "erc", "pr"]),
]).reproject(target)
```

That single change also collapses ten round trips into one, so it's a correctness fix and a 10× speedup at the same time.

### B-7. Wind direction is averaged as a scalar — line 1143

`th` (wind direction, degrees) falls through to `coll.median()`. Direction is circular: the median of 350°, 355°, 5°, 10° is **180°** — the exact opposite of the true mean. I verified this. Every windy multi-day window in period mode produces a wrong value.

Decompose, average, recombine:

```python
u = vs.multiply(th.multiply(np.pi/180).sin()).multiply(-1)
v = vs.multiply(th.multiply(np.pi/180).cos()).multiply(-1)
# average u and v over the window, then optionally recombine
```

For the model itself, **feed u and v directly**, or `sin(th)`/`cos(th)`. Never feed raw degrees — 359 and 1 are adjacent directions but maximally distant numbers, and no tree split or linear weight can express that.

### B-8. Precipitation is summed over the whole lookback — lines 1136–1137

In period mode, `coll.sum()` over the 30-day lookback yields a 30-day total labeled "Precipitation (mm)" with a colorbar capped at 50. In daily mode the same label means one day's total. Two incompatible quantities under one name. Either separate them (`precip_1d`, `precip_7d`, `precip_30d`) or make the aggregation window explicit in the column name. `days_since_rain` is also a stronger fire predictor than raw totals and is cheap to derive.

### B-9. NDVI scale mismatch between the two code paths — lines 1160–1166

The comment says the VNP13A1 scale factor is deliberately skipped. But VNP13A1 NDVI *is* stored scaled by 0.0001, and `EE_RASTER_VIS["vegetation"]` sets the color range to `(-1.0, 1.0)`. So raster NDVI arrives in the thousands and every pixel clamps to the top of the colorbar — the map is uniformly dark green regardless of actual vegetation.

Meanwhile `enrich_with_earth_engine` (line 2085) uses MOD13A2 and *does* multiply by 0.0001. Two paths, two sensors, two scales, two column names (`vegetation` vs `ee_ndvi`). Pick one sensor, apply `.multiply(0.0001)`, and delete the other path.

### B-10. Footprint area ignores the `scan`/`track` columns you already have — line 206

```python
df["footprint_area_km2"] = coefs ** 2
```

This is nadir pixel size. VIIRS pixels grow toward the swath edge — from 0.375 km at nadir to roughly 0.8 km at scan edge, so real area varies by a factor of ~4. FIRMS gives you the actual per-detection dimensions in the `scan` and `track` columns:

```python
if "scan" in df.columns and "track" in df.columns:
    df["footprint_area_km2"] = df["scan"] * df["track"]
    dx = df["scan"] / (cos_lat * KM_PER_DEG) / 2.0
    dy = df["track"] / KM_PER_DEG / 2.0
```

Same fix applies to `footprint_dx`/`dy`, which propagate into every polygon and every area estimate.

### B-11. No negative samples, no defined target

This is the biggest structural gap. The pipeline exports *only* fire locations. A classifier trained on positives alone has nothing to learn from.

You need, roughly:

1. **A target definition.** "Fire at cell (x, y) on day t+1" is the standard next-day formulation. Write it down explicitly — the answer changes what features are legal.
2. **Negative sampling.** Sample no-fire cells from the same region and time distribution as your positives. Sampling uniformly over all of California biases toward ocean and desert; the model then learns geography, not fire risk. Hard negatives — cells adjacent to fires, or cells with high fire weather that didn't ignite — are what actually teach the decision boundary.
3. **Spatially-blocked cross-validation.** Random splits leak, because a fire spans many adjacent cells on consecutive days and neighbors land in both train and test. Split by fire event or by spatial block. Random-split scores on this kind of data are routinely 20+ points optimistic.
4. **Class imbalance handling.** Fire cells are on the order of 0.01–0.1% of cells. Report PR-AUC, not accuracy or ROC-AUC.

### B-12. Missing features that are known-strong for fire

Cheap additions with real predictive value:

- **Slope and aspect** — `ee.Terrain.products(dem)`, free from the DEM you already load. Fire moves ~2× faster uphill; south-facing slopes cure earlier.
- **ERC (Energy Release Component)** — already in `EE_WEATHER_BANDS` for the dead point path but absent from `EE_RASTER_LAYER_CONFIG`. It's arguably the single best-established fire-danger index in the US.
- **VPD** — derivable from `tmmx` and `sph`; correlates with fire activity better than raw humidity.
- **Distance to road / WUI** — most ignitions are human-caused. `TIGER/2016/Roads` gives you this.
- **Time since last fire** — from MTBS or the FIRMS archive itself. Recently burned ground doesn't reburn.

---

## C. Performance

The workload is **network-latency bound**, not compute bound. Profile before optimizing anything else — I'd expect >90% of wall time inside `getInfo()`.

### C-1. Serial `getInfo()` calls are the dominant cost

`--daily-visual-samples` over 30 days × 10 layers = 300 sequential blocking round trips, each a few seconds to a minute. That's hours of pure waiting.

**Parallelize.** The EE Python client is thread-safe and the backend parallelizes well:

```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=12) as ex:
    futures = {ex.submit(extract_ee_raster, bbox, k, **kw): k for k in layers}
    results = {futures[f]: f.result() for f in as_completed(futures)}
```

8–16 workers is the usual sweet spot before EE starts throttling. Expect roughly 10× on the daily-samples path alone.

**Better: collapse days into bands.** Instead of one request per day, build one multi-band image and fetch the whole time series at once:

```python
stack = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET") \
    .filterDate(start, end).select(["tmmx", "vs", "erc"]).toBands()
```

30 requests become 1.

**Best, for training-scale extraction: stop using `getInfo()` entirely.** It's an interactive API with hard payload limits. Use batch export:

```python
task = ee.batch.Export.image.toCloudStorage(
    image=stack, bucket=BUCKET, fileNamePrefix=f"{region}/{date}",
    region=geom, scale=375, fileFormat="GeoTIFF", maxPixels=1e10)
task.start()
```

Asynchronous, no pixel cap, and EE parallelizes it server-side. Your existing `--ee-bucket`/`--ee-folder` flags are already parsed but never used — this is what they were presumably meant for.

### C-2. `sampleRectangle` is the wrong extraction primitive — lines 1222–1243, 1372–1399

The 262,144-pixel cap is why lines 1330–1370 contain a whole tiling subsystem, and why lines 1375–1397 retry up to 5× at 1.5× coarser scale — up to 5 wasted round trips per layer before giving up.

`ee.data.computePixels` (or `getDownloadURL(format="NPY")`) has far higher limits and returns a properly georeferenced array:

```python
pixels = ee.data.computePixels({
    "expression": img,
    "fileFormat": "NUMPY_NDARRAY",
    "grid": {"dimensions": {"width": w_px, "height": h_px},
             "affineTransform": {...}, "crsCode": "EPSG:4326"},
})
```

This deletes the tiling code, the retry loop, *and* fixes B-13 below, because you specify the transform rather than inferring it.

### C-3. Raster georeferencing is inferred, not read — lines 1174–1199

```python
lon_step = (e - w) / max(ncols, 1)
lat_vals = n - lat_step * (np.arange(nrows) + 0.5)
```

This assumes the returned array exactly spans your requested bbox. It doesn't — `sampleRectangle` snaps to the projection's pixel grid, so the true extent is the bbox rounded outward to pixel boundaries. Every pixel center is then off by up to a full pixel: 500 m for elevation, 4 km for GRIDMET.

That's spatial misregistration between your labels and your features, and it's invisible in the PNGs. Fix by getting the real transform:

```python
proj = img.reproject(crs="EPSG:4326", scale=scale).projection().getInfo()
transform = proj["transform"]   # [xScale, xShear, xOrigin, yShear, yScale, yOrigin]
```

Or move to `computePixels`, where you supply the grid.

### C-4. `iterrows()` in every hot loop — lines 267, 564, 667, 2145, 2174

`iterrows` constructs a Series per row. At California scale (100k+ detections) this dominates all non-network time. Shapely 2.0 has vectorized constructors:

```python
import shapely
corners = np.stack([
    np.column_stack([df.footprint_nw_lon, df.footprint_nw_lat]),
    np.column_stack([df.footprint_ne_lon, df.footprint_ne_lat]),
    np.column_stack([df.footprint_se_lon, df.footprint_se_lat]),
    np.column_stack([df.footprint_sw_lon, df.footprint_sw_lat]),
], axis=1)
polys = shapely.polygons(corners)   # entire array at once
```

Typically 50–100× on that step.

### C-5. Caches don't survive process exit — lines 65–67

`_EE_RASTER_CACHE` is a bare dict. Every debugging re-run re-fetches everything from Earth Engine. During development that's the difference between a 40-minute iteration and a 10-second one.

```python
from joblib import Memory
memory = Memory("~/.cache/firms_ee", verbose=0)
extract_ee_raster = memory.cache(extract_ee_raster)
```

The cache key is already well-designed — `_normalize_ee_time_args` collapses equivalent requests — so this is nearly a drop-in. Also add an eviction bound; the in-memory dicts currently grow without limit across a long multi-area run.

### C-6. Homegrown raster format — lines 1608–1613

`.npy` plus a `bounds.txt` sidecar means every consumer has to know your convention, no CRS travels with the data, and QGIS can't open it. Write GeoTIFF:

```python
import rasterio
from rasterio.transform import from_bounds
with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                   dtype="float32", crs="EPSG:4326",
                   transform=from_bounds(w_, s_, e_, n_, w, h),
                   compress="deflate") as dst:
    dst.write(arr.astype("float32"), 1)
```

Then `rioxarray.open_rasterio()` gives you a labeled, CRS-aware array, and reprojection to a common grid becomes one call — directly addressing B-6. For multi-temporal stacks, Zarr with xarray is better still: chunked, compressed, and lazily readable, which matters once you're feeding a training loop.

Also, `f"{base_name}_{layer_key}_grid.csv"` (line 1808) writes the flattened raster to CSV. A 512×512 float grid is ~2 MB as CSV versus ~250 KB as compressed GeoTIFF, and CSV parsing is slow. Drop it or make it opt-in.

---

## D. Infrastructure and hardware

Short answer on hardware: **this is not a hardware problem, and a GPU won't help.** The bottleneck is network round trips to Earth Engine. Buying compute optimizes the 5% of wall time that isn't waiting on a socket.

What actually helps, in order of impact per hour of your effort:

1. **Thread pool around EE calls** (C-1) — ~10×, one afternoon of work.
2. **Persistent disk cache** (C-5) — makes iteration tolerable, ~20 lines.
3. **Batch export to GCS** (C-1) — the only approach that scales to a real training set. `Export.image.toCloudStorage` runs asynchronously and server-side; you poll task status and download when done.
4. **NVMe + enough RAM to hold a region's stack.** For a state-scale multi-year set you're in the tens-of-GB range. RAM matters more than cores.
5. **Colab / Vertex AI for EE-heavy runs.** Not for the compute — for the network locality. EE, GCS, and Colab are all inside Google's network; extraction from a colo VM is often several times faster than from a residential connection, and it's free.

Where GPUs *do* eventually matter: model training, once you have the data. Not extraction. Given your homelab and FPGA background, the genuinely interesting hardware question is the **inference** end — a quantized model on an edge device co-located with sensors, doing local inference and transmitting only alerts. That's a real bandwidth-and-latency argument and it connects directly to the AEGIS work. The extraction pipeline is not that problem.

One caveat if you go the local-processing route: at 375 m over California you're looking at roughly 3,000 × 2,600 cells per layer per day. Keep it chunked (`dask` via `rioxarray`) rather than loading whole arrays, or you'll hit memory limits well before you hit compute limits.

---

## E. Code health

**~400 lines of dead code (~15% of the file).** Never called from `main()`: `fetch_california_from_api` (354), `save_earth_engine_outputs` (956), `plot_ee_feature_map` (841), `plot_ee_feature_grid` (897), `get_fire_centroids_dataframe` (1264), `sample_ee_at_points` (1402), `enrich_with_earth_engine` (2037). These represent an older point-sampling design that the raster path superseded. Reviewers will read them and assume they're live. Delete them, or move them to `legacy/` with a note.

Note that `sample_ee_at_points` also has a latent bug if you ever revive it — it builds one `ee.Feature` per row client-side and passes the whole list in a single request, with no batching, so it fails on payload size past a few thousand points. `enrich_with_earth_engine` does batch at `EE_BATCH_SIZE=1000`; that one doesn't.

**pyflakes findings:** unused `Polygon` import (239), unused `im` (1738), unused `default_end` (2232).

**Split the module.** 2,670 lines in one file is past the point where anyone can hold it in their head:

```
firms/
├── api.py          # fetch, key management, retry
├── ingest.py       # filters, datetime, footprints
├── cluster.py      # DBSCAN, hulls
├── earthengine.py  # EE layers, extraction, caching
├── viz.py          # plotting
├── io.py           # save/load, output dirs
└── cli.py          # argparse, interactive, main
```

**Replace `print` with `logging`.** ~60 print calls with no levels. `logging` gives you `-v`, timestamps, and file output — all three matter when a run takes an hour.

**Config object instead of hand-built Namespace.** `run_interactive()` constructs an `argparse.Namespace` field by field and omits attributes the CLI path sets, which is why `getattr(args, "ee_raster_layers", None)` appears throughout. A dataclass validated once at entry removes that entire class of bug:

```python
@dataclass
class Config:
    start_date: str | None = None
    areas: list[str] = field(default_factory=lambda: ["california"])
    ee_raster_layers: list[str] = field(default_factory=list)
    ...
```

**Tests.** No test suite. Highest-value targets, all pure functions needing no network:

- `add_footprint_columns` — known lat/lon → known corners
- `apply_confidence_frp_daynight_filters` — both confidence encodings, missing column
- `cluster_fire_points_to_polygons` — synthetic two-cluster fixture
- `raster_array_to_dataframe` / `dataframe_to_raster_array` — round-trip identity
- `_get_time_window` / `_get_exact_day_window` — boundary dates

Add `responses` or `requests-mock` for the FIRMS layer.

**Pin dependencies.** `requirements.txt` with versions, plus a note that `geopandas`/`shapely`/`sklearn` are optional. Geospatial stacks break on minor upgrades more than most.

---

## Suggested order of work

**Before sharing the code:** A-1 (rotate the key), A-2, A-3, A-5 — the crashes and the silent data loss.

**Before trusting any result:** B-5 (leakage), B-7 (wind), B-4 (UTC), C-3 (georeferencing). These four determine whether your features mean what you think they mean, and all four are invisible in the output PNGs.

**Before scaling up:** C-1 (parallelism), C-5 (disk cache), B-6 (common grid). Together these are the difference between a script you run overnight and one you iterate on.

**Before training:** B-11. Negative sampling and target definition are unavoidable, and they're design decisions rather than code fixes — worth writing down and defending in the paper before implementing.
