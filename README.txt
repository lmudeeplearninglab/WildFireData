================================================================================
  visualize_firms_dataset.py – Pipeline, Usage, and Output Reference
================================================================================

This document describes what `visualize_firms_dataset.py` does, how the data
pipeline works, and how to use the script for:

  - FIRMS fire-data gathering
  - preprocessing and filtering
  - fire visualization
  - fire-mask / centroid generation
  - standalone Google Earth Engine feature export

The script is intended for wildfire analysis and dataset preparation. It keeps
NASA FIRMS fire detections separate from Google Earth Engine (EE) feature
datasets.


--------------------------------------------------------------------------------
1. WHAT THE SCRIPT DOES
--------------------------------------------------------------------------------

`visualize_firms_dataset.py` is a multi-purpose wildfire data utility that:

  1. downloads NASA FIRMS fire detections for a selected area and time range
  2. filters and preprocesses the detections
  3. derives fire-footprint geometry and cluster centroids
  4. saves fire-only outputs such as CSV, mask PNGs, point PNGs, and time plots
  5. optionally exports standalone EE feature datasets for the same fire period

Important design rule:

  - FIRMS outputs stay as FIRMS fire data
  - EE outputs are exported as separate datasets and visualizations

This means the script does NOT append EE feature columns onto the main FIRMS
acquisition CSV for the normal fire workflow.


--------------------------------------------------------------------------------
2. END-TO-END PIPELINE
--------------------------------------------------------------------------------

The script follows the pipeline below.

2.1 Data gathering

  - Reads the target region from:
      - built-in areas (`california`, `eaton`, `palisades`)
      - custom `--bbox`
      - `--shapefile` + optional `--state`
      - `--wkt-polygon`
  - Reads the requested date window from:
      - `--start-date` and `--end-date`
      - or `--days` when no explicit date range is provided
  - Fetches FIRMS CSV data from the NASA FIRMS area API
  - Tries NRT and/or archive sources depending on CLI flags
  - Falls back to the FIRMS sample CSV only when live API data is unavailable
    and no explicit date range makes the fallback invalid

2.2 Preprocessing

  - Converts `acq_date` + `acq_time` into `acq_datetime`
  - Applies optional filtering:
      - `--confidence`
      - `--min-frp`
      - `--daynight`
  - Optionally subsets the detections spatially using:
      - bounding box
      - shapefile geometry
      - WKT polygon

2.3 Fire geometry generation

  - Computes FIRMS footprint fields from latitude, longitude, and instrument
  - Adds footprint corner columns:
      - `footprint_nw_*`
      - `footprint_ne_*`
      - `footprint_se_*`
      - `footprint_sw_*`
  - Computes `footprint_area_km2`
  - Clusters nearby detections using DBSCAN when possible
  - Derives fire-cluster centroids for mask export and EE centroid sampling

2.4 Fire-only outputs

  Depending on flags, the script saves:

  - point detections map
  - time-based fire spread map
  - fire-mask polygon map
  - centroid CSV
  - fire-only CSV / Shapefile / KML / HDF5

2.5 Standalone Earth Engine export

  If `--enrich-earth-engine` is enabled, the script exports EE features as
  separate datasets for the fire period and area. It does this in two ways:

  - bounding-box raster export
  - centroid-based sampling at fire cluster locations

  Current standalone EE layers:

  - elevation
  - vegetation (NDVI)
  - population density
  - land cover
  - drought (PDSI)
  - weather temperature
  - weather precipitation

  Time-varying EE layers use the fire-period window instead of a single random
  snapshot, so the extracted features better represent conditions during the
  fire interval.


--------------------------------------------------------------------------------
3. REQUIRED AND OPTIONAL DEPENDENCIES
--------------------------------------------------------------------------------

Core:

  pip install matplotlib pandas numpy requests

Optional:

  pip install geopandas shapely
    Required for shapefile and WKT subsetting, and for polygon-style fire masks

  pip install contextily
    Required for `--basemap`

  pip install earthengine-api
    Required for `--enrich-earth-engine` and `--ee-raster-layers`

  pip install scikit-learn
    Recommended for clustered fire-mask polygons and cluster centroids

  pip install tables
    Required for `--save-hdf5`


--------------------------------------------------------------------------------
4. AUTHENTICATION
--------------------------------------------------------------------------------

4.1 FIRMS MAP_KEY

Real FIRMS API data requires a FIRMS MAP_KEY:

  https://firms.modaps.eosdis.nasa.gov/api/map_key

Set it with an environment variable:

  Windows CMD:
    set FIRMS_MAP_KEY=your_key_here

  Windows PowerShell:
    $env:FIRMS_MAP_KEY="your_key_here"

Or let the script prompt for it interactively.

Skip verification if needed:

  python visualize_firms_dataset.py --no-verify-key

4.2 Google Earth Engine

EE export requires:

  pip install earthengine-api
  earthengine authenticate

Also provide a GCP project:

  Windows CMD:
    set EE_PROJECT=your-gcp-project-id

  Windows PowerShell:
    $env:EE_PROJECT="your-gcp-project-id"

Or:

  earthengine set_project your-gcp-project-id


--------------------------------------------------------------------------------
5. INTERACTIVE MODE
--------------------------------------------------------------------------------

Run with no flags:

  python visualize_firms_dataset.py

Or explicitly:

  python visualize_firms_dataset.py -i
  python visualize_firms_dataset.py --interactive

Interactive mode prompts for:

  - date range
  - area(s)
  - source type
  - visualization type(s)
  - filters
  - outputs
  - standalone EE export
  - EE project / bucket / folder


--------------------------------------------------------------------------------
6. REGIONS AND SPATIAL SUBSETTING
--------------------------------------------------------------------------------

Built-in areas:

  - `california`
  - `eaton`
  - `palisades`

Use one or more:

  python visualize_firms_dataset.py --area california
  python visualize_firms_dataset.py --area eaton --area palisades

Custom bounding box:

  python visualize_firms_dataset.py --bbox -124.5,32.5,-114.1,42.0

Shapefile:

  python visualize_firms_dataset.py --shapefile path/to/states.shp --state "California"

WKT polygon:

  python visualize_firms_dataset.py --wkt-polygon "POLYGON((-123 38,-121 38,-121 40,-123 40,-123 38))"

Bbox format:

  west,south,east,north


--------------------------------------------------------------------------------
7. TIME RANGE AND SOURCES
--------------------------------------------------------------------------------

Explicit date range:

  python visualize_firms_dataset.py --start-date 2025-01-07 --end-date 2025-01-31

Single day:

  python visualize_firms_dataset.py --start-date 2025-01-08 --end-date 2025-01-08

Recent rolling window:

  python visualize_firms_dataset.py --days 5

Source selection:

  --source VIIRS_SNPP_NRT
  --use-archive
  --nrt-only

Default behavior:

  - try NRT first
  - then try archive/SP sources when needed


--------------------------------------------------------------------------------
8. PREPROCESSING AND FILTERING
--------------------------------------------------------------------------------

Available filters:

  --confidence n,h
  --min-frp 5
  --daynight D

Example:

  python visualize_firms_dataset.py --confidence n,h --min-frp 5 --daynight D

What preprocessing adds:

  - `acq_datetime`
  - fire-footprint size and corner coordinates
  - optional clustered centroids for fire-mask export

These preprocessing steps are important because the downstream mask plots,
centroid exports, and EE centroid sampling all depend on them.


--------------------------------------------------------------------------------
9. VISUALIZATION MODES
--------------------------------------------------------------------------------

Point map:

  --points

  Produces a scatter plot of detections, colored by FRP when available.

Time-based spread map:

  --time-plot

  Produces a multi-day fire spread visualization across the selected period.

Fire mask:

  --fire-mask

  Produces polygonal fire-mask style output using clustered detections or
  fallback footprint polygons.

All fire visualizations:

  --all-viz

Basemap:

  --basemap


--------------------------------------------------------------------------------
10. STANDALONE EARTH ENGINE PIPELINE
--------------------------------------------------------------------------------

Enable standalone EE export:

  python visualize_firms_dataset.py --enrich-earth-engine --ee-project your-project-id

What it does:

  - uses the same area as the FIRMS run
  - derives the fire-period date window from the FIRMS detections
  - exports EE data separately from FIRMS

Two EE export modes are produced:

10.1 Bounding-box raster export

  For each EE layer, the script attempts to export:

  - `{base}_{layer}.npy`
  - `{base}_{layer}_bounds.txt`
  - `{base}_{layer}_grid.csv`
  - `{base}_{layer}.png`

  This gives a standalone raster-style dataset over the full requested region.

10.2 Fire-centroid sampling

  The script also samples EE values at cluster centroids:

  - `{base}_{layer}_centroids.csv`
  - `{base}_{layer}_centroids.png`

  This is useful when:

  - the full bbox is large
  - you want feature values exactly at detected fire clusters
  - you want quick verification plots at fire locations

10.3 Combined EE summary files

  The script also creates:

  - `{base}_ee_features_standalone.csv`
      Long-form combined bbox EE dataset

  - `{base}_ee_features_grid.png`
      Multi-panel grid of bbox EE feature maps

  - `{base}_ee_features_centroids.csv`
      Long-form combined centroid EE dataset

  - `{base}_ee_features_centroids_grid.png`
      Multi-panel grid of centroid EE feature plots

Important:

  These EE files are standalone. They are not meant to replace the FIRMS
  acquisition CSV and are not appended onto the normal fire-only outputs.


--------------------------------------------------------------------------------
11. OPTIONAL EE RASTER LAYER EXPORT
--------------------------------------------------------------------------------

You can also request specific EE raster layers explicitly:

  python visualize_firms_dataset.py --ee-raster-layers elevation,population
  python visualize_firms_dataset.py --ee-raster-layers all

This path reuses the bbox-based raster export workflow and is useful when you
want selected EE layers even without the full standalone EE export workflow.


--------------------------------------------------------------------------------
12. OUTPUT FILES
--------------------------------------------------------------------------------

Fire-only outputs:

  - `{base}.csv`
  - `{base}.png`
  - `{base}_time.png`
  - `{base}_mask.png`
  - `{base}_centroids.csv`
  - optional `.shp`, `.kml`, `.h5`

Standalone EE outputs:

  - `{base}_{layer}.npy`
  - `{base}_{layer}_bounds.txt`
  - `{base}_{layer}_grid.csv`
  - `{base}_{layer}.png`
  - `{base}_{layer}_centroids.csv`
  - `{base}_{layer}_centroids.png`
  - `{base}_ee_features_standalone.csv`
  - `{base}_ee_features_grid.png`
  - `{base}_ee_features_centroids.csv`
  - `{base}_ee_features_centroids_grid.png`


--------------------------------------------------------------------------------
13. EXAMPLE COMMANDS
--------------------------------------------------------------------------------

Default interactive run:

  python visualize_firms_dataset.py

California, fire mask + points:

  python visualize_firms_dataset.py --area california --fire-mask --points

Multiple areas:

  python visualize_firms_dataset.py --area eaton --area palisades --fire-mask --points

Strict filtering:

  python visualize_firms_dataset.py --area california --start-date 2025-01-07 --end-date 2025-01-31 --confidence n,h --min-frp 5 --daynight D --fire-mask --points

Standalone EE export for a fire period:

  python visualize_firms_dataset.py --area california --start-date 2025-01-07 --end-date 2025-01-31 --points --enrich-earth-engine --ee-project your-project-id

Custom polygon with EE export:

  python visualize_firms_dataset.py --wkt-polygon "POLYGON((-123 38,-121 38,-121 40,-123 40,-123 38))" --start-date 2025-01-07 --end-date 2025-01-31 --fire-mask --enrich-earth-engine --ee-project your-project-id


--------------------------------------------------------------------------------
14. TROUBLESHOOTING
--------------------------------------------------------------------------------

"No FIRMS data retrieved"

  - check MAP_KEY
  - confirm the date range and region
  - try `--use-archive` for historical dates

"WARNING: Polygon subsetting skipped"

  - install `geopandas` and `shapely`

"WARNING: --enrich-earth-engine ignored"

  - install `earthengine-api`
  - authenticate Earth Engine

"ee.Initialize: no project found"

  - set `EE_PROJECT`
  - or run `earthengine set_project your-project-id`

EE raster extraction is very large or slow

  - large regions such as all of California produce big raster outputs
  - use centroid outputs if you want a lighter-weight verification dataset
  - use a smaller bbox or a focused area like `eaton` or `palisades`

Centroid CSV exists but values look sparse

  - some EE layers are coarse or masked in parts of the region
  - compare bbox grid outputs with centroid outputs
  - verify the fire-period dates match the event you want to study


--------------------------------------------------------------------------------
15. QUICK REFERENCE
--------------------------------------------------------------------------------

Key URLs:

  FIRMS MAP_KEY:
    https://firms.modaps.eosdis.nasa.gov/api/map_key

  FIRMS API docs:
    https://firms.modaps.eosdis.nasa.gov/api/area/

  Earth Engine:
    https://code.earthengine.google.com

Key flags:

  --interactive
  --area
  --bbox
  --shapefile
  --wkt-polygon
  --start-date
  --end-date
  --days
  --confidence
  --min-frp
  --daynight
  --points
  --time-plot
  --fire-mask
  --all-viz
  --enrich-earth-engine
  --ee-raster-layers
  --ee-project

================================================================================
End of Guide
================================================================================
