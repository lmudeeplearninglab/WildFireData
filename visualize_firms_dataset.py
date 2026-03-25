# coding=utf-8
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)

# Optional: geopandas, shapely for shapefile/polygon subsetting
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    from shapely import wkt
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import contextily as cx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

try:
    import ee
    HAS_EARTH_ENGINE = True
except ImportError:
    HAS_EARTH_ENGINE = False


# ---------------------------------------------------------------------------
# FIRMS API (from firms_api_use.ipynb)
# ---------------------------------------------------------------------------
FIRMS_AREA_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_SAMPLE_CSV = "https://firms.modaps.eosdis.nasa.gov/content/notebooks/sample_viirs_snpp_071223.csv"

CALIFORNIA_BBOX = (-124.5, 32.5, -114.1, 42.0)  # west, south, east, north
EATON_BBOX = (-118.195, 34.148, -118.0, 34.249)
# 2025 Palisades Fire: Pacific Palisades, Topanga, Malibu (~23,448 acres / 95 km²)
PALISADES_BBOX = (-118.90, 33.96, -118.38, 34.20)  # west, south, east, north

# NRT = Near Real-Time (~last 10 days). SP = Standard Processing (archive, years of history).
SOURCES_NRT = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT")
SOURCES_ARCHIVE = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP")
SOURCES = SOURCES_NRT + SOURCES_ARCHIVE  # Try NRT first, then archive
KM_PER_DEG = 111.32


# ---------------------------------------------------------------------------
# Data Ingest (from firms_data_ingest.ipynb)
# ---------------------------------------------------------------------------
def add_acq_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Combine acq_date and acq_time into acq_datetime (firms_data_ingest)."""
    if df.empty or "acq_date" not in df.columns or "acq_time" not in df.columns:
        return df
    df = df.copy()
    time_str = df["acq_date"].astype(str) + " " + df["acq_time"].astype(str).str.zfill(4)
    df["acq_datetime"] = pd.to_datetime(time_str, format="%Y-%m-%d %H%M")
    return df


def apply_confidence_frp_daynight_filters(
    df: pd.DataFrame,
    confidence: tuple[str, ...] | None = None,
    min_frp: float | None = None,
    daynight: str | None = None,
) -> pd.DataFrame:
    """
    Subset by confidence (n/normal, h/high, l/low), min FRP, daynight (D/N).
    From firms_data_ingest: (confidence='n' or 'h') and frp>=5 and daynight='D'
    """
    if df.empty:
        return df
    out = df.copy()
    if confidence:
        valid = (out["confidence"].isin(confidence)) if "confidence" in out.columns else True
        out = out.loc[valid]
    if min_frp is not None and "frp" in out.columns:
        out = out[out["frp"] >= min_frp]
    if daynight is not None and "daynight" in out.columns:
        out = out[out["daynight"] == daynight]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shapefile / Polygon subsetting (from subSetDataFromShapeFileOrPolygon.ipynb)
# ---------------------------------------------------------------------------
def subset_by_wkt_polygon(df: pd.DataFrame, wkt_polygon: str, crs: str = "EPSG:4326") -> pd.DataFrame:
    """Subset FIRMS data to points inside a WKT polygon (subSetDataFromShapeFileOrPolygon)."""
    if not HAS_GEOPANDAS or not HAS_SHAPELY:
        print("  WARNING: Polygon subsetting skipped. Install: pip install geopandas shapely")
        return df
    try:
        polygon = wkt.loads(wkt_polygon)
    except Exception as e:
        print(f"  Invalid WKT polygon: {e}")
        return df
    fire_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=crs,
    )
    polygon_gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon], crs=crs)
    polygon_gdf = polygon_gdf.to_crs(fire_gdf.crs)
    result = gpd.sjoin(fire_gdf, polygon_gdf, how="inner", predicate="intersects")
    cols = [c for c in result.columns if c not in ("index_right", "geometry")]
    return result[cols].drop_duplicates().reset_index(drop=True)


def subset_by_shapefile(
    df: pd.DataFrame,
    shapefile_path: str | Path,
    state_name: str | None = None,
    name_col: str = "NAME",
) -> pd.DataFrame:
    """Subset FIRMS data to points inside shapefile (optionally a specific state)."""
    if not HAS_GEOPANDAS:
        print("  WARNING: Shapefile subsetting skipped. Install: pip install geopandas")
        return df
    path = Path(shapefile_path)
    if not path.exists():
        print(f"  Shapefile not found: {path}")
        return df
    try:
        states_gdf = gpd.read_file(path)
    except Exception as e:
        print(f"  Failed to read shapefile: {e}")
        return df
    fire_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=states_gdf.crs,
    )
    if state_name is not None and name_col in states_gdf.columns:
        subset_geom = states_gdf[states_gdf[name_col] == state_name]
        if subset_geom.empty:
            print(f"  State '{state_name}' not found in shapefile")
            return df
    else:
        subset_geom = states_gdf
    result = gpd.sjoin(fire_gdf, subset_geom, how="inner", predicate="intersects")
    cols = [c for c in result.columns if c in df.columns]
    return result[cols].drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# API & fetch (from firms_api_use.ipynb, firms_data_ingest.ipynb)
# ---------------------------------------------------------------------------
def get_footprint_coef(instrument: str) -> float:
    """Fire footprint coefficient by sensor (VIIRS=0.375, MODIS=1.0, LANDSAT=0.03)."""
    inst = str(instrument).upper() if isinstance(instrument, str) else ""
    if "MODIS" in inst:
        return 1.0
    if "LANDSAT" in inst:
        return 0.03
    return 0.375


def add_footprint_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add Fire Footprint Area columns (dx, dy, corners, area_km2) per sensor."""
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return df
    lat = df["latitude"].values
    lon = df["longitude"].values
    if "instrument" in df.columns:
        instruments = df["instrument"].fillna("VIIRS")
        coefs = np.array([get_footprint_coef(inst) for inst in instruments])
    else:
        coefs = np.full(len(df), 0.375)
    lat_rad = np.radians(lat)
    abs_lat = np.abs(lat)
    dx = np.where(abs_lat <= 89.99, coefs / (np.cos(lat_rad) * KM_PER_DEG) / 2.0, 180.0)
    dy = (coefs / KM_PER_DEG) / 2.0
    df = df.copy()
    df["footprint_dx"] = dx
    df["footprint_dy"] = dy
    df["footprint_nw_lat"] = lat + dy
    df["footprint_nw_lon"] = lon - dx
    df["footprint_ne_lat"] = lat + dy
    df["footprint_ne_lon"] = lon + dx
    df["footprint_se_lat"] = lat - dy
    df["footprint_se_lon"] = lon + dx
    df["footprint_sw_lat"] = lat - dy
    df["footprint_sw_lon"] = lon - dx
    df["footprint_area_km2"] = coefs ** 2
    return df


def cluster_fire_points_to_polygons(
    df: pd.DataFrame,
    eps_deg: float = 0.02,
    min_samples: int = 2,
) -> tuple[list, pd.DataFrame]:
    """
    Cluster fire detections by spatial proximity and return one convex-hull polygon
    per cluster plus cluster centroids. Uses outermost points (footprint corners)
    to draw the boundary.

    Args:
        df: DataFrame with footprint_nw/ne/se/sw_lat/lon (and latitude, longitude).
        eps_deg: Max distance in degrees for DBSCAN clustering (~0.02 ≈ 2 km).
        min_samples: Min points to form a cluster.

    Returns:
        (polygons, centroids_df) where polygons are Shapely Polygons and
        centroids_df has cluster_id, centroid_lat, centroid_lon, n_detections.
    """
    req = ["footprint_nw_lat", "footprint_nw_lon", "footprint_ne_lat", "footprint_ne_lon",
           "footprint_se_lat", "footprint_se_lon", "footprint_sw_lat", "footprint_sw_lon"]
    if not all(c in df.columns for c in req) or df.empty:
        return [], pd.DataFrame()

    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return [], pd.DataFrame()  # Caller will fall back to individual footprints

    from shapely.geometry import MultiPoint, Polygon

    # Centroids for clustering
    centroids = df[["latitude", "longitude"]].values

    clustering = DBSCAN(eps=eps_deg, min_samples=min_samples, metric="euclidean")
    labels = clustering.fit_predict(centroids)

    polygons = []
    centroid_rows = []
    for label in sorted(set(labels)):
        if label < 0:
            continue  # noise points
        mask = labels == label
        sub = df[mask]

        # Cluster centroid: mean of detection lat/lon
        centroid_lat = sub["latitude"].mean()
        centroid_lon = sub["longitude"].mean()
        centroid_rows.append({
            "cluster_id": int(label),
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "n_detections": len(sub),
        })

        # Collect all footprint corners (outermost points) for this cluster
        points = []
        for _, row in sub.iterrows():
            points.extend([
                (row["footprint_nw_lon"], row["footprint_nw_lat"]),
                (row["footprint_ne_lon"], row["footprint_ne_lat"]),
                (row["footprint_se_lon"], row["footprint_se_lat"]),
                (row["footprint_sw_lon"], row["footprint_sw_lat"]),
            ])

        if len(points) < 3:
            continue
        mp = MultiPoint(points)
        hull = mp.convex_hull
        if hull.is_empty or hull.area == 0:
            continue
        if hull.geom_type == "Polygon":
            polygons.append(hull)
        elif hull.geom_type == "LineString":
            continue  # degenerate
        else:
            polygons.append(hull.convex_hull)

    centroids_df = pd.DataFrame(centroid_rows) if centroid_rows else pd.DataFrame()
    return polygons, centroids_df


DEFAULT_FIRMS_MAP_KEY = "525966a64a09d3ae159039cef88c37d9"


def get_map_key() -> str:
    """FIRMS MAP_KEY from environment, prompt, or default."""
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if key:
        return key
    print("FIRMS MAP_KEY: set FIRMS_MAP_KEY or get a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key")
    key = input(f"Paste MAP_KEY (or Enter for default) [{DEFAULT_FIRMS_MAP_KEY[:8]}...]: ").strip()
    return key or DEFAULT_FIRMS_MAP_KEY


def check_map_key(map_key: str) -> bool:
    """Test MAP_KEY (firms_api_use: mapkey_status)."""
    if not map_key:
        return False
    url = f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={map_key}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        d = r.json()
        print(f"MAP_KEY: {d.get('current_transactions', '?')} / {d.get('transaction_limit', '?')} transactions (10 min window)")
        return True
    except Exception as e:
        print(f"MAP_KEY check failed: {e}")
        return False


def fetch_area_csv(
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    day_range: int = 5,
    date: str | None = None,
) -> pd.DataFrame:
    """Fetch FIRMS area CSV (firms_api_use: area_url)."""
    w, s, e, n = bbox
    area = f"{w},{s},{e},{n}"
    if date:
        url = f"{FIRMS_AREA_BASE}/{map_key}/{source}/{area}/{day_range}/{date}"
    else:
        url = f"{FIRMS_AREA_BASE}/{map_key}/{source}/{area}/{day_range}"
    try:
        df = pd.read_csv(url)
        if "latitude" not in df.columns:
            return pd.DataFrame()
        return df
    except Exception as err:
        print(f"  API error ({source}): {err}")
        return pd.DataFrame()


def subset_bbox(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Subset to bbox (firms_data_ingest style)."""
    w, s, e, n = bbox
    return df[
        (df["longitude"] >= w) & (df["latitude"] >= s)
        & (df["longitude"] <= e) & (df["latitude"] <= n)
    ].copy()


def fetch_california_from_api(
    map_key: str,
    day_range: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
    sources: tuple[str, ...] = SOURCES,
) -> pd.DataFrame:
    """Fetch California data from API; try each source until one returns data."""
    if start_date and end_date:
        start_d = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_d = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start_d > end_d:
            return pd.DataFrame()
        chunks = []
        d = start_d
        while d <= end_d:
            chunk_end = min(d + timedelta(days=4), end_d)
            n_days = (chunk_end - d).days + 1
            chunk_date = d.strftime("%Y-%m-%d")
            for src in sources:
                df = fetch_area_csv(map_key, src, CALIFORNIA_BBOX, day_range=n_days, date=chunk_date)
                if not df.empty:
                    chunks.append(df)
                    break
            d = chunk_end + timedelta(days=1)
        if not chunks:
            return pd.DataFrame()
        out = pd.concat(chunks, ignore_index=True)
        keys = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in out.columns]
        if keys:
            out = out.drop_duplicates(subset=keys, keep="first")
        return out
    date_arg = start_date
    for src in sources:
        df = fetch_area_csv(map_key, src, CALIFORNIA_BBOX, day_range=day_range, date=date_arg)
        if not df.empty:
            return df
    return pd.DataFrame()


def fetch_firms_data(
    map_key: str | None = None,
    bbox: tuple[float, float, float, float] = CALIFORNIA_BBOX,
    start_date: str | None = None,
    end_date: str | None = None,
    days: int = 5,
    sources: tuple[str, ...] = SOURCES,
    use_sample_fallback: bool = True,
) -> pd.DataFrame:
    """
    Fetch FIRMS fire data.
    1) If map_key: call FIRMS API for bbox.
    2) If API empty or no key and use_sample_fallback: use sample CSV, subset to bbox.
    """
    df = pd.DataFrame()
    if map_key:
        print("Fetching from FIRMS API...")
        if start_date and end_date:
            start_d = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_d = datetime.strptime(end_date, "%Y-%m-%d").date()
            if start_d <= end_d:
                chunks = []
                d = start_d
                while d <= end_d:
                    chunk_end = min(d + timedelta(days=4), end_d)
                    n_days = (chunk_end - d).days + 1
                    chunk_date = d.strftime("%Y-%m-%d")
                    for src in sources:
                        chunk = fetch_area_csv(map_key, src, bbox, day_range=n_days, date=chunk_date)
                        if not chunk.empty:
                            chunks.append(chunk)
                            break
                    d = chunk_end + timedelta(days=1)
                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                    keys = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in df.columns]
                    if keys:
                        df = df.drop_duplicates(subset=keys, keep="first")
        else:
            date_arg = start_date
            for src in sources:
                df = fetch_area_csv(map_key, src, bbox, day_range=days, date=date_arg)
                if not df.empty:
                    break
        if not df.empty:
            df = subset_bbox(df, bbox)
            print(f"  API: {len(df)} detections")
    if df.empty and use_sample_fallback:
        # Don't fall back to sample when user specified dates (would return wrong-year data)
        if start_date or end_date:
            print("  No FIRMS data for requested date range. Try --use-archive for older dates.")
        else:
            print("Using FIRMS sample CSV (2023-07-12, no key required)...")
            raw = pd.read_csv(FIRMS_SAMPLE_CSV)
            df = subset_bbox(raw, bbox)
            print(f"  Sample: {len(df)} detections")
    return df


# ---------------------------------------------------------------------------
# Visualization (from firms_visualization.ipynb)
# ---------------------------------------------------------------------------
def plot_fire_map(
    df: pd.DataFrame,
    title: str,
    bbox: tuple[float, float, float, float],
    path: Path,
    color_col: str = "frp",
) -> None:
    """Plot detections (lon/lat) and save figure."""
    fig, ax = plt.subplots(figsize=(10, 8))
    w, s, e, n = bbox
    if df.empty:
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_title(f"{title} (no detections)")
    else:
        if color_col in df.columns:
            sc = ax.scatter(df["longitude"], df["latitude"], c=df[color_col], s=10, cmap="hot", alpha=0.8)
            plt.colorbar(sc, ax=ax, label=color_col)
        else:
            ax.scatter(df["longitude"], df["latitude"], c="red", s=10, alpha=0.8)
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_title(f"{title} ({len(df)} detections)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fire_map_time_based(
    df: pd.DataFrame,
    title: str,
    bbox: tuple[float, float, float, float],
    path: Path,
    add_basemap: bool = False,
) -> None:
    """
    Plot fire footprints layered by day to visualize fire spread over the date range.
    Each day is a separate layer; oldest (yellow) drawn first, newest (dark red) on top.
    """
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        w, s, e, n = bbox
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_title(f"{title} (no detections)")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return
    df = add_acq_datetime(df)
    date_col = "acq_date" if "acq_date" in df.columns else ("acq_datetime" if "acq_datetime" in df.columns else None)
    if not date_col:
        plot_fire_map(df, title, bbox, path)
        return

    req = ["footprint_nw_lat", "footprint_nw_lon", "footprint_ne_lat", "footprint_ne_lon",
           "footprint_se_lat", "footprint_se_lon", "footprint_sw_lat", "footprint_sw_lon"]
    has_footprints = all(c in df.columns for c in req)
    if not has_footprints or not HAS_GEOPANDAS:
        df["_day"] = pd.to_datetime(df[date_col]).dt.date
        days = sorted(df["_day"].unique())
        if not days:
            plot_fire_map(df, title, bbox, path)
            return
        cmap = plt.cm.get_cmap("YlOrRd", len(days) + 1)
        fig, ax = plt.subplots(figsize=(10, 10))
        w, s, e, n = bbox
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_aspect("equal", adjustable="box")
        for i, day in enumerate(days):
            sub = df[df["_day"] == day]
            if not sub.empty:
                color = cmap((i + 1) / (len(days) + 1))
                ax.scatter(sub["longitude"], sub["latitude"], c=color, s=8, alpha=0.7, label=str(day))
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{title} – Fire spread by day ({len(days)} days, {len(df)} detections)")
        ax.grid(True, alpha=0.3)
        if add_basemap and HAS_CONTEXTILY and HAS_GEOPANDAS:
            try:
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
                gdf_proj = gdf.to_crs(epsg=3857)
                ax.set_xlim(gdf_proj.total_bounds[0], gdf_proj.total_bounds[2])
                ax.set_ylim(gdf_proj.total_bounds[1], gdf_proj.total_bounds[3])
                cx.add_basemap(ax, crs=gdf_proj.crs)
            except Exception:
                pass
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return

    from shapely.geometry import Polygon

    df["_day"] = pd.to_datetime(df[date_col]).dt.date
    days = sorted(df["_day"].unique())
    cmap = plt.cm.get_cmap("YlOrRd", len(days) + 1)

    gdf_by_day = []
    for day in days:
        sub = df[df["_day"] == day]
        polygons = []
        for _, row in sub.iterrows():
            coords = [
                (row["footprint_nw_lon"], row["footprint_nw_lat"]),
                (row["footprint_ne_lon"], row["footprint_ne_lat"]),
                (row["footprint_se_lon"], row["footprint_se_lat"]),
                (row["footprint_sw_lon"], row["footprint_sw_lat"]),
                (row["footprint_nw_lon"], row["footprint_nw_lat"]),
            ]
            poly = Polygon(coords)
            if not poly.is_empty:
                polygons.append(poly)
        if polygons:
            gdf_by_day.append((day, gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")))

    fig, ax = plt.subplots(figsize=(10, 10))
    w, s, e, n = bbox
    basemap_ok = False
    if add_basemap and HAS_CONTEXTILY:
        try:
            from shapely.geometry import box
            bbox_geom = box(w, s, e, n)
            bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326").to_crs(epsg=3857)
            extent = bbox_gdf.total_bounds
            ax.set_xlim(extent[0], extent[2])
            ax.set_ylim(extent[1], extent[3])
            cx.add_basemap(ax, crs="EPSG:3857", alpha=0.8)
            basemap_ok = True
        except Exception as ex:
            print(f"  Basemap failed: {ex}")
    if not basemap_ok:
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
    ax.set_aspect("equal", adjustable="box")

    for i, (day, gdf_day) in enumerate(gdf_by_day):
        color = cmap((i + 1) / (len(days) + 1))
        to_plot = gdf_day.to_crs(epsg=3857) if basemap_ok else gdf_day
        to_plot.plot(ax=ax, facecolor=color, edgecolor="black", alpha=0.5, linewidth=0.3, label=str(day))

    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{title} – Fire spread by day ({len(days)} days, {len(df)} footprints)")
    ax.grid(True, alpha=0.3)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fire_mask(
    df: pd.DataFrame,
    title: str,
    bbox: tuple[float, float, float, float],
    path: Path,
    add_basemap: bool = False,
    cluster_eps: float = 0.02,
    cluster_min_samples: int = 2,
    centroids_dir: Path | None = None,
) -> None:
    """
    Plot fire detection mask as clustered polygons (one polygon per fire cluster).
    Clusters detections by spatial proximity and draws the boundary using the
    convex hull of outermost footprint points. Requires sklearn for DBSCAN.
    Note: FIRMS provides active fire detections, not post-fire burned perimeters.
    """
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        w, s, e, n = bbox
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_title(f"{title} (no detections)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return

    req = ["footprint_nw_lat", "footprint_nw_lon", "footprint_ne_lat", "footprint_ne_lon",
           "footprint_se_lat", "footprint_se_lon", "footprint_sw_lat", "footprint_sw_lon"]
    if not all(c in df.columns for c in req):
        print("  WARNING: Footprint columns missing; falling back to point plot.")
        plot_fire_map(df, title, bbox, path)
        return

    if not HAS_GEOPANDAS:
        print("  WARNING: Fire mask requires geopandas. Falling back to point plot.")
        plot_fire_map(df, title, bbox, path)
        return

    from shapely.geometry import Polygon

    # Cluster fire points and build one polygon per cluster (convex hull of outermost points)
    polygons, centroids_df = cluster_fire_points_to_polygons(
        df, eps_deg=cluster_eps, min_samples=cluster_min_samples
    )

    if not polygons:
        # Fallback: individual footprints (e.g. if sklearn not installed)
        polygons = []
        for _, row in df.iterrows():
            coords = [
                (row["footprint_nw_lon"], row["footprint_nw_lat"]),
                (row["footprint_ne_lon"], row["footprint_ne_lat"]),
                (row["footprint_se_lon"], row["footprint_se_lat"]),
                (row["footprint_sw_lon"], row["footprint_sw_lat"]),
                (row["footprint_nw_lon"], row["footprint_nw_lat"]),
            ]
            poly = Polygon(coords)
            if not poly.is_empty:
                polygons.append(poly)
        if not polygons:
            plot_fire_map(df, title, bbox, path)
            return
        print("  Note: Install sklearn for fire clustering. Using individual footprints.")
        # When no clustering: export individual detection centroids (lat/lon)
        centroids_df = df[["latitude", "longitude"]].copy()
        centroids_df = centroids_df.rename(columns={"latitude": "centroid_lat", "longitude": "centroid_lon"})
        centroids_df.insert(0, "cluster_id", np.arange(len(centroids_df)))
        centroids_df["n_detections"] = 1

    gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")

    # Save centroids (cluster centroids when clustered; detection centroids when not)
    if not centroids_df.empty:
        centroids_root = centroids_dir or path.parent
        centroids_root.mkdir(parents=True, exist_ok=True)
        centroids_path = centroids_root / (path.stem.replace("_mask", "") + "_centroids.csv")
        centroids_df.to_csv(centroids_path, index=False)
        print(f"  Saved: {centroids_path} ({len(centroids_df)} centroids)")

    fig, ax = plt.subplots(figsize=(10, 10))
    w, s, e, n = bbox
    basemap_ok = False
    if add_basemap and HAS_CONTEXTILY:
        try:
            from shapely.geometry import box
            gdf_proj = gdf.to_crs(epsg=3857)
            bbox_geom = box(w, s, e, n)
            bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326").to_crs(epsg=3857)
            extent = bbox_gdf.total_bounds
            ax.set_xlim(extent[0], extent[2])
            ax.set_ylim(extent[1], extent[3])
            cx.add_basemap(ax, crs="EPSG:3857", alpha=0.8)
            gdf_proj.plot(ax=ax, facecolor="red", edgecolor="darkred", alpha=0.6, linewidth=0.5)
            basemap_ok = True
        except Exception as ex:
            print(f"  Basemap failed: {ex}")
    if not basemap_ok:
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        gdf.plot(ax=ax, facecolor="red", edgecolor="darkred", alpha=0.6, linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{title} – Fire detection mask ({len(gdf)} clusters, {len(df)} detections)")
    ax.grid(True, alpha=0.3)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_outputs(
    df: pd.DataFrame,
    out_dir: Path,
    base_name: str = "firms",
    save_csv: bool = True,
    save_shp: bool = False,
    save_kml: bool = False,
    save_hdf5: bool = False,
) -> None:
    """Save dataframe to CSV, optionally Shapefile, KML, and HDF5."""
    if df.empty:
        return
    drop_cols = [c for c in ["geometry", "index_right"] if c in df.columns]
    out = df.drop(columns=drop_cols, errors="ignore") if drop_cols else df
    if save_csv:
        csv_path = out_dir / f"{base_name}.csv"
        out.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path} ({len(df)} rows)")
    if save_hdf5:
        h5_path = out_dir / f"{base_name}.h5"
        try:
            out.to_hdf(h5_path, key="firms", mode="w", format="table")
            print(f"  Saved: {h5_path} ({len(df)} rows)")
        except ImportError:
            print("  HDF5 save skipped. Install: pip install tables")
        except Exception as e:
            print(f"  HDF5 save failed: {e}")
    if (save_shp or save_kml) and HAS_GEOPANDAS:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        )
        if save_shp:
            shp_path = out_dir / f"{base_name}.shp"
            try:
                gdf.to_file(shp_path)
                print(f"  Saved: {shp_path}")
            except Exception as e:
                print(f"  Shapefile save failed: {e}")
        if save_kml:
            kml_path = out_dir / f"{base_name}.kml"
            try:
                gdf.to_file(kml_path, driver="KML")
                print(f"  Saved: {kml_path}")
            except Exception as e:
                print(f"  KML save failed: {e}")


def _bbox_slug(bbox: tuple[float, float, float, float]) -> str:
    """Create a filesystem-safe bbox label for organizing outputs."""
    labels = ("w", "s", "e", "n")
    parts = []
    for label, value in zip(labels, bbox):
        token = f"{value:.3f}".replace("-", "m").replace(".", "p")
        parts.append(f"{label}_{token}")
    return "bbox_" + "_".join(parts)


def get_output_dirs(
    root_dir: Path,
    base_name: str,
    bbox: tuple[float, float, float, float],
) -> tuple[Path, Path]:
    """Return dataset and visual output directories for a bbox/area."""
    group_name = f"{base_name}_{_bbox_slug(bbox)}"
    dataset_dir = root_dir / "datasets" / group_name
    visual_dir = root_dir / "visuals" / group_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir, visual_dir


EE_FEATURE_VIS = {
    "elevation": ("terrain", "Elevation"),
    "vegetation": ("RdYlGn", "Vegetation (NDVI)"),
    "population": ("Purples", "Population density"),
    "landcover": ("tab20", "Land cover"),
    "drought": ("BrBG", "Drought (PDSI)"),
    "weather_temp": ("YlOrRd", "Max temperature (C)"),
    "weather_precip": ("Blues", "Precipitation (mm)"),
    "ee_elevation": ("terrain", "Elevation"),
    "ee_ndvi": ("RdYlGn", "NDVI"),
    "ee_population_density": ("Purples", "Population density"),
    "ee_landcover": ("tab20", "Land cover"),
    "ee_pdsi": ("BrBG", "Drought (PDSI)"),
    "ee_pr": ("Blues", "GRIDMET pr"),
    "ee_sph": ("viridis", "GRIDMET sph"),
    "ee_th": ("magma", "GRIDMET th"),
    "ee_tmmn": ("coolwarm", "GRIDMET tmmn"),
    "ee_tmmx": ("coolwarm", "GRIDMET tmmx"),
    "ee_vs": ("cividis", "GRIDMET vs"),
    "ee_erc": ("inferno", "GRIDMET erc"),
}


def _get_ee_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return Earth Engine feature columns present in the dataframe."""
    return [c for c in df.columns if c.startswith("ee_")]


def _get_ee_feature_label(feature_col: str) -> str:
    """Human-readable label for an Earth Engine feature column."""
    if feature_col in EE_FEATURE_VIS:
        return EE_FEATURE_VIS[feature_col][1]
    return feature_col.removeprefix("ee_").replace("_", " ").title()


def plot_ee_feature_map(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    feature_col: str,
    path: Path,
    title: str | None = None,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> None:
    """Save one standalone PNG for a single sampled Earth Engine feature."""
    if (
        df.empty
        or feature_col not in df.columns
        or lon_col not in df.columns
        or lat_col not in df.columns
    ):
        return

    label = _get_ee_feature_label(feature_col)
    plot_title = title or label
    w, s, e, n = bbox

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    sub = df[[lon_col, lat_col, feature_col]].copy()
    sub[feature_col] = pd.to_numeric(sub[feature_col], errors="coerce")
    sub = sub.dropna(subset=[lon_col, lat_col, feature_col])
    if sub.empty:
        ax.set_title(f"{plot_title} (no data)")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return

    cmap = EE_FEATURE_VIS.get(feature_col, ("viridis", label))[0]
    sc = ax.scatter(
        sub[lon_col],
        sub[lat_col],
        c=sub[feature_col],
        s=18,
        cmap=cmap,
        alpha=0.85,
    )
    ax.set_title(f"{plot_title} ({len(sub)} points)")
    plt.colorbar(sc, ax=ax, label=label)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ee_feature_grid(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    path: Path,
    title: str = "Google Earth Engine feature grid",
    feature_cols: list[str] | None = None,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> None:
    """Save one PNG with a subplot for each sampled Earth Engine feature."""
    ee_cols = feature_cols or _get_ee_feature_columns(df)
    ee_cols = [c for c in ee_cols if c in df.columns]
    if df.empty or not ee_cols or lon_col not in df.columns or lat_col not in df.columns:
        return

    n_panels = len(ee_cols)
    ncols = 2 if n_panels <= 4 else 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    w, s, e, n = bbox

    for ax, feature_col in zip(axes, ee_cols):
        label = _get_ee_feature_label(feature_col)
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        sub = df[[lon_col, lat_col, feature_col]].copy()
        sub[feature_col] = pd.to_numeric(sub[feature_col], errors="coerce")
        sub = sub.dropna(subset=[lon_col, lat_col, feature_col])
        if sub.empty:
            ax.set_title(f"{label} (no data)")
            continue

        cmap = EE_FEATURE_VIS.get(feature_col, ("viridis", label))[0]
        sc = ax.scatter(
            sub[lon_col],
            sub[lat_col],
            c=sub[feature_col],
            s=16,
            cmap=cmap,
            alpha=0.85,
        )
        ax.set_title(f"{label} ({len(sub)} points)")
        plt.colorbar(sc, ax=ax, label=label, shrink=0.85)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_earth_engine_outputs(
    df: pd.DataFrame,
    out_dir: Path,
    base_name: str,
    bbox: tuple[float, float, float, float],
) -> None:
    """
    Save Earth Engine enrichment separately from fire-mask/fire-detection outputs.
    Writes one combined CSV, one CSV per EE feature, and one grid PNG.
    """
    if df.empty:
        return

    ee_cols = _get_ee_feature_columns(df)
    if not ee_cols:
        print("  NOTE: No Earth Engine feature columns were available to save.")
        return

    drop_cols = [c for c in ["geometry", "index_right"] if c in df.columns]
    out = df.drop(columns=drop_cols, errors="ignore") if drop_cols else df

    combined_csv_path = out_dir / f"{base_name}_ee_features.csv"
    out.to_csv(combined_csv_path, index=False)
    print(f"  Saved: {combined_csv_path} ({len(out)} rows)")

    base_cols = [c for c in out.columns if c not in ee_cols]
    for feature_col in ee_cols:
        feature_out = out[base_cols + [feature_col]]
        feature_csv_path = out_dir / f"{base_name}_{feature_col}.csv"
        feature_out.to_csv(feature_csv_path, index=False)
        print(f"  Saved: {feature_csv_path} ({len(feature_out)} rows)")
        feature_png_path = out_dir / f"{base_name}_{feature_col}.png"
        plot_ee_feature_map(
            df,
            bbox,
            feature_col,
            feature_png_path,
            title=f"{base_name} – {_get_ee_feature_label(feature_col)}",
        )

    grid_path = out_dir / f"{base_name}_ee_features_grid.png"
    plot_ee_feature_grid(df, bbox, grid_path, title=f"{base_name} – Google Earth Engine features")


# ---------------------------------------------------------------------------
# Earth Engine raster extraction and visualization (like fire footprints)
# ---------------------------------------------------------------------------
EE_RASTER_SCALE = 500   # meters per pixel (min); increased for large regions
EE_RASTER_MAX_PIXELS = 262144  # sampleRectangle limit
EE_RASTER_SCALE_BY_LAYER = {
    "elevation": 500,
    "vegetation": 1000,
    "landcover": 250,
    "population": 1000,
    "drought": 4000,
    "weather_temp": 4000,
    "weather_precip": 4000,
}
EE_STANDALONE_FEATURE_LAYERS = (
    "elevation",
    "vegetation",
    "population",
    "landcover",
    "drought",
    "weather_temp",
    "weather_precip",
)

# Layer config: (EE dataset, band(s), is_time_varying, title)
EE_RASTER_LAYER_CONFIG = {
    "elevation": ("USGS/SRTMGL1_003", ["elevation"], False, "Elevation (m)"),
    "vegetation": ("NASA/VIIRS/002/VNP13A1", ["NDVI"], True, "Vegetation (NDVI)"),
    "landcover": ("ESA/WorldCover/v100", ["Map"], False, "Land cover"),
    "drought": ("GRIDMET/DROUGHT", ["pdsi"], True, "Drought (PDSI)"),
    "weather_temp": ("IDAHO_EPSCOR/GRIDMET", ["tmmx"], True, "Max temperature (°C)"),
    "weather_precip": ("IDAHO_EPSCOR/GRIDMET", ["pr"], True, "Precipitation (mm)"),
    "population": ("CIESIN/GPWv411/GPW_Population_Density", ["population_density"], False, "Population density"),
}


def _get_ee_scale_for_layer(layer_key: str) -> int:
    """Nominal sampling scale for each EE layer."""
    return EE_RASTER_SCALE_BY_LAYER.get(layer_key, EE_RASTER_SCALE)


def _get_time_window(
    date_str: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = 30,
) -> tuple[str | None, str | None]:
    """Return an EE date window covering the fire period or a fallback lookback."""
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days)
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    if date_str:
        end_dt = pd.to_datetime(date_str) + pd.Timedelta(days=1)
        start_dt = pd.to_datetime(date_str) - pd.Timedelta(days=lookback_days)
        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    end_dt = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    start_dt = end_dt - pd.Timedelta(days=lookback_days)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def _get_ee_image_for_layer(
    layer_key: str,
    date_str: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> ee.Image | None:
    """Get EE Image for a raster layer. Returns None if EE not available."""
    if not HAS_EARTH_ENGINE or layer_key not in EE_RASTER_LAYER_CONFIG:
        return None
    dataset, bands, is_time_varying, _ = EE_RASTER_LAYER_CONFIG[layer_key]
    if layer_key == "population":
        # CIESIN GPW is ImageCollection (years 2000–2020), not a single Image
        img = ee.ImageCollection(dataset).select(bands).filterDate("2020-01-01", "2020-12-31").first()
    elif layer_key == "landcover":
        img = ee.ImageCollection(dataset).select(bands).first()
    elif is_time_varying and date_str:
        window_start, window_end = _get_time_window(date_str=date_str, start_date=start_date, end_date=end_date)
        coll = ee.ImageCollection(dataset).filterDate(window_start, window_end).select(bands)
        if layer_key == "weather_precip":
            img = coll.sum()
        elif layer_key == "weather_temp":
            img = coll.mean()
        else:
            img = coll.median()
    elif is_time_varying:
        window_start, window_end = _get_time_window(date_str=date_str, start_date=start_date, end_date=end_date)
        coll = ee.ImageCollection(dataset).filterDate(window_start, window_end).select(bands)
        if layer_key == "weather_precip":
            img = coll.sum()
        elif layer_key == "weather_temp":
            img = coll.mean()
        else:
            img = coll.median()
    else:
        img = ee.Image(dataset).select(bands)
    # Apply scale factors (VIIRS NDVI: 0.0001)
    if layer_key == "vegetation":
        img = img.multiply(0.0001)
    if img is None:
        return None
    return img.rename(layer_key)


def raster_array_to_dataframe(
    arr: np.ndarray,
    bbox: tuple[float, float, float, float],
    value_col: str,
) -> pd.DataFrame:
    """Flatten a raster into a standalone lon/lat/value dataset."""
    if arr.size == 0:
        return pd.DataFrame(columns=["longitude", "latitude", value_col])

    w, s, e, n = bbox
    nrows, ncols = arr.shape
    lon_step = (e - w) / max(ncols, 1)
    lat_step = (n - s) / max(nrows, 1)
    lon_vals = w + lon_step * (np.arange(ncols) + 0.5)
    lat_vals = n - lat_step * (np.arange(nrows) + 0.5)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

    out = pd.DataFrame(
        {
            "longitude": lon_grid.ravel(),
            "latitude": lat_grid.ravel(),
            value_col: arr.ravel(),
        }
    )
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    return out.dropna(subset=[value_col]).reset_index(drop=True)


def get_fire_centroids_dataframe(
    df: pd.DataFrame,
    cluster_eps: float = 0.02,
    cluster_min_samples: int = 2,
) -> pd.DataFrame:
    """Return centroids for the fire detections to support centroid-based EE sampling."""
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return pd.DataFrame(columns=["cluster_id", "centroid_lat", "centroid_lon", "n_detections"])

    _, centroids_df = cluster_fire_points_to_polygons(
        df, eps_deg=cluster_eps, min_samples=cluster_min_samples
    )
    if centroids_df.empty:
        centroids_df = df[["latitude", "longitude"]].copy()
        centroids_df = centroids_df.rename(
            columns={"latitude": "centroid_lat", "longitude": "centroid_lon"}
        )
        centroids_df.insert(0, "cluster_id", np.arange(len(centroids_df)))
        centroids_df["n_detections"] = 1
    return centroids_df.reset_index(drop=True)


def extract_ee_raster(
    bbox: tuple[float, float, float, float],
    layer_key: str,
    date_str: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    scale: int = EE_RASTER_SCALE,
    ee_project: str | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    """
    Extract Earth Engine raster as numpy array for a bbox.
    Returns (array, (w,s,e,n)) or None on failure.
    """
    if not HAS_EARTH_ENGINE or not _init_earth_engine(ee_project):
        return None
    img = _get_ee_image_for_layer(layer_key, date_str, start_date=start_date, end_date=end_date)
    if img is None:
        return None
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n])
    # Ensure we stay under sampleRectangle limit (262144 pixels)
    width_m = abs(e - w) * 111000 * max(0.7, np.cos(np.radians((s + n) / 2)))
    height_m = abs(n - s) * 111000
    max_dim = int(np.sqrt(EE_RASTER_MAX_PIXELS))
    scale_x = width_m / max_dim
    scale_y = height_m / max_dim
    layer_scale = _get_ee_scale_for_layer(layer_key)
    area_scale = np.sqrt((width_m * height_m) / max(1, int(EE_RASTER_MAX_PIXELS * 0.8)))
    actual_scale = max(scale, layer_scale, int(np.ceil(max(scale_x, scale_y, area_scale))))
    crs = "EPSG:4326"
    for _ in range(5):
        try:
            img_for_sample = img.reproject(crs=crs, scale=actual_scale).clip(region).unmask(-9999)
            result = img_for_sample.sampleRectangle(region=region, defaultValue=-9999)
            info = result.getInfo()
            if not info or "properties" not in info:
                return None
            props = info["properties"]
            raw = props.get(layer_key)
            if raw is None:
                return None
            arr = np.asarray(raw, dtype=float)
            if arr.ndim < 2:
                arr = np.atleast_2d(arr)
            arr[arr == -9999] = np.nan
            return arr, bbox
        except Exception as ex:
            if "Too many pixels" in str(ex):
                actual_scale = int(np.ceil(actual_scale * 1.5))
                continue
            print(f"  EE raster extract failed ({layer_key}): {ex}")
            return None
    print(f"  EE raster extract failed ({layer_key}): exceeded pixel limit after rescaling")
    return None


def sample_ee_at_points(
    points_df: pd.DataFrame,
    layer_key: str,
    date_str: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    ee_project: str | None = None,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> pd.DataFrame:
    """Sample one EE layer at point locations and return a standalone dataset."""
    if (
        points_df.empty
        or lon_col not in points_df.columns
        or lat_col not in points_df.columns
        or not HAS_EARTH_ENGINE
        or not _init_earth_engine(ee_project)
    ):
        return pd.DataFrame()

    img = _get_ee_image_for_layer(layer_key, date_str, start_date=start_date, end_date=end_date)
    if img is None:
        return pd.DataFrame()
    img = img.unmask(-9999)

    features = []
    rows = points_df.reset_index(drop=True)
    for idx, row in rows.iterrows():
        features.append(
            ee.Feature(
                ee.Geometry.Point([row[lon_col], row[lat_col]]),
                {"row_id": int(idx)},
            )
        )

    try:
        sampled = img.sampleRegions(
            collection=ee.FeatureCollection(features),
            properties=["row_id"],
            scale=_get_ee_scale_for_layer(layer_key),
            tileScale=4,
        ).getInfo()
    except Exception as ex:
        print(f"  EE point sampling failed ({layer_key}): {ex}")
        return pd.DataFrame()

    out = rows.copy()
    out[layer_key] = np.nan
    features_out = sampled.get("features", []) if sampled else []
    for feature in features_out:
        props = feature.get("properties", {})
        row_id = props.get("row_id")
        if row_id is None or row_id >= len(out):
            continue
        value = props.get(layer_key)
        if value == -9999:
            value = np.nan
        out.at[row_id, layer_key] = value
    return out


def plot_raster_layer(
    arr: np.ndarray,
    bbox: tuple[float, float, float, float],
    title: str,
    path: Path,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    add_basemap: bool = False,
) -> None:
    """Plot raster array as map and save PNG (same style as fire mask)."""
    w, s, e, n = bbox
    masked = np.ma.masked_invalid(arr.astype(float))
    if masked.size == 0 or np.all(masked.mask):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_title(f"{title} (no data)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path} (no data)")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        masked,
        extent=[w, e, s, n],
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label=title)
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # Note: basemap would require reprojecting raster to Web Mercator for alignment; skip for now

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# Colormaps and ranges for each layer
EE_RASTER_VIS = {
    "elevation": ("terrain", None, None),
    "vegetation": ("RdYlGn", -0.2, 1.0),
    "drought": ("BrBG", -4, 4),
    "weather_temp": ("YlOrRd", 0, 45),
    "weather_precip": ("Blues", 0, 50),
    "population": ("Purples", 0, None),
}


def extract_and_plot_ee_raster_layers(
    bbox: tuple[float, float, float, float],
    layers: list[str],
    data_dir: Path,
    visual_dir: Path,
    base_name: str,
    date_str: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    add_basemap: bool = False,
    ee_project: str | None = None,
) -> None:
    """
    Extract and visualize EE raster layers for a bbox (same workflow as fire mask).
    Saves GeoTIFF-style numpy (.npy) and PNG for each layer.
    """
    if not HAS_EARTH_ENGINE:
        print("  EE raster layers require earthengine-api. Install: pip install earthengine-api")
        return
    if not _init_earth_engine(ee_project):
        return

    valid_layers = [l for l in layers if l in EE_RASTER_LAYER_CONFIG]
    if not valid_layers:
        print(f"  No valid layers. Options: {list(EE_RASTER_LAYER_CONFIG.keys())}")
        return

    print(f"Extracting and visualizing EE raster layers: {', '.join(valid_layers)}")
    data_dir = Path(data_dir)
    visual_dir = Path(visual_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    for layer_key in valid_layers:
        result = extract_ee_raster(
            bbox,
            layer_key,
            date_str,
            start_date=start_date,
            end_date=end_date,
            ee_project=ee_project,
        )
        if result is None:
            continue
        arr, bounds = result
        _, _, _, title = EE_RASTER_LAYER_CONFIG[layer_key]
        vis = EE_RASTER_VIS.get(layer_key, ("viridis", None, None))
        cmap, vmin, vmax = vis

        # Save array for reuse (like fire footprint CSV)
        npy_path = data_dir / f"{base_name}_{layer_key}.npy"
        meta_path = data_dir / f"{base_name}_{layer_key}_bounds.txt"
        np.save(npy_path, arr)
        with open(meta_path, "w") as f:
            f.write(f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}\n")
        print(f"  Extracted: {npy_path}")

        # Visualize (like fire mask PNG)
        png_path = visual_dir / f"{base_name}_{layer_key}.png"
        plot_raster_layer(arr, bounds, title, png_path, cmap=cmap, vmin=vmin, vmax=vmax, add_basemap=add_basemap)


def plot_ee_raster_grid(
    layer_results: list[tuple[str, np.ndarray, tuple[float, float, float, float]]],
    path: Path,
    title: str,
) -> None:
    """Save one figure containing all standalone EE raster feature maps."""
    if not layer_results:
        return

    n_panels = len(layer_results)
    ncols = 2 if n_panels <= 4 else 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (layer_key, arr, bounds) in zip(axes, layer_results):
        w, s, e, n = bounds
        masked = np.ma.masked_invalid(arr.astype(float))
        cmap, vis_title = EE_FEATURE_VIS.get(layer_key, ("viridis", layer_key))
        vis_meta = EE_RASTER_VIS.get(layer_key, (cmap, None, None))
        cmap = vis_meta[0]
        vmin = vis_meta[1]
        vmax = vis_meta[2]

        if masked.size == 0 or np.all(masked.mask):
            ax.set_title(f"{vis_title} (no data)")
            ax.set_xlim(w, e)
            ax.set_ylim(s, n)
            ax.grid(True, alpha=0.3)
            continue

        im = ax.imshow(
            masked,
            extent=[w, e, s, n],
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label=vis_title, shrink=0.85)
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(vis_title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def export_standalone_ee_feature_datasets(
    bbox: tuple[float, float, float, float],
    data_dir: Path,
    visual_dir: Path,
    base_name: str,
    date_str: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    add_basemap: bool = False,
    ee_project: str | None = None,
    layers: tuple[str, ...] = EE_STANDALONE_FEATURE_LAYERS,
    fire_df: pd.DataFrame | None = None,
    cluster_eps: float = 0.02,
    cluster_min_samples: int = 2,
) -> None:
    """
    Export Earth Engine features as standalone raster datasets, not FIRMS-row enrichments.
    Saves bbox rasters and centroid-based samples so features can be reviewed either way.
    """
    valid_layers = [l for l in layers if l in EE_RASTER_LAYER_CONFIG]
    if not valid_layers:
        return

    layer_results: list[tuple[str, np.ndarray, tuple[float, float, float, float]]] = []
    combined_rows: list[pd.DataFrame] = []
    centroid_wide_df = pd.DataFrame()
    centroid_long_rows: list[pd.DataFrame] = []
    centroids_df = (
        get_fire_centroids_dataframe(
            fire_df, cluster_eps=cluster_eps, cluster_min_samples=cluster_min_samples
        )
        if fire_df is not None and not fire_df.empty
        else pd.DataFrame()
    )
    if not centroids_df.empty:
        centroid_wide_df = centroids_df.rename(
            columns={"centroid_lon": "longitude", "centroid_lat": "latitude"}
        ).copy()
    print(f"Exporting standalone EE feature datasets: {', '.join(valid_layers)}")

    for layer_key in valid_layers:
        result = extract_ee_raster(
            bbox,
            layer_key,
            date_str,
            start_date=start_date,
            end_date=end_date,
            ee_project=ee_project,
        )
        if result is not None:
            arr, bounds = result
            layer_results.append((layer_key, arr, bounds))
            _, _, _, layer_title = EE_RASTER_LAYER_CONFIG[layer_key]
            cmap, vmin, vmax = EE_RASTER_VIS.get(layer_key, ("viridis", None, None))

            npy_path = data_dir / f"{base_name}_{layer_key}.npy"
            meta_path = data_dir / f"{base_name}_{layer_key}_bounds.txt"
            csv_path = data_dir / f"{base_name}_{layer_key}_grid.csv"
            png_path = visual_dir / f"{base_name}_{layer_key}.png"

            np.save(npy_path, arr)
            with open(meta_path, "w") as f:
                f.write(f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}\n")
            print(f"  Saved: {npy_path}")

            layer_df = raster_array_to_dataframe(arr, bounds, layer_key)
            layer_df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path} ({len(layer_df)} rows)")

            combined_rows.append(layer_df.assign(layer=layer_key).rename(columns={layer_key: "value"}))
            plot_raster_layer(arr, bounds, layer_title, png_path, cmap=cmap, vmin=vmin, vmax=vmax, add_basemap=add_basemap)

        if not centroids_df.empty:
            centroid_layer_df = sample_ee_at_points(
                centroid_wide_df[["cluster_id", "n_detections", "longitude", "latitude"]],
                layer_key,
                date_str=date_str,
                start_date=start_date,
                end_date=end_date,
                ee_project=ee_project,
            )
            if not centroid_layer_df.empty:
                centroid_csv_path = data_dir / f"{base_name}_{layer_key}_centroids.csv"
                centroid_png_path = visual_dir / f"{base_name}_{layer_key}_centroids.png"
                centroid_layer_df.to_csv(centroid_csv_path, index=False)
                print(f"  Saved: {centroid_csv_path} ({len(centroid_layer_df)} rows)")
                plot_ee_feature_map(
                    centroid_layer_df,
                    bbox,
                    layer_key,
                    centroid_png_path,
                    title=f"{base_name} – {_get_ee_feature_label(layer_key)} centroids",
                )
                centroid_wide_df[layer_key] = centroid_layer_df[layer_key].values
                centroid_long_rows.append(
                    centroid_layer_df.assign(layer=layer_key).rename(columns={layer_key: "value"})
                )

    if combined_rows:
        combined_path = data_dir / f"{base_name}_ee_features_standalone.csv"
        pd.concat(combined_rows, ignore_index=True).to_csv(combined_path, index=False)
        print(f"  Saved: {combined_path}")

    grid_path = visual_dir / f"{base_name}_ee_features_grid.png"
    plot_ee_raster_grid(layer_results, grid_path, title=f"{base_name} – Google Earth Engine features")
    if centroid_long_rows:
        centroid_combined_path = data_dir / f"{base_name}_ee_features_centroids.csv"
        pd.concat(centroid_long_rows, ignore_index=True).to_csv(centroid_combined_path, index=False)
        print(f"  Saved: {centroid_combined_path}")
        centroid_grid_path = visual_dir / f"{base_name}_ee_features_centroids_grid.png"
        plot_ee_feature_grid(
            centroid_wide_df,
            bbox,
            centroid_grid_path,
            title=f"{base_name} – Google Earth Engine features at fire centroids",
            feature_cols=valid_layers,
        )


def export_daily_visual_samples(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    dataset_root: Path,
    visual_root: Path,
    base_name: str,
    plot_points: bool = True,
    fire_mask: bool = False,
    time_plot: bool = False,
    add_basemap: bool = False,
    enrich_earth_engine: bool = False,
    ee_raster_layers: list[str] | None = None,
    ee_project: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    cluster_eps: float = 0.02,
    cluster_min_samples: int = 2,
) -> None:
    """
    Export one day-by-day sample folder with fire and EE visualizations.
    Each day gets its own subdirectory so the fire signal and same-day features
    can be reviewed side by side across the requested time range.
    """
    df = add_acq_datetime(df)
    if "acq_date" in df.columns:
        df = df.copy()
        df["_acq_date_str"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if start_date or end_date:
        start_str = start_date or end_date
        end_str = end_date or start_date or start_str
        day_list = [d.strftime("%Y-%m-%d") for d in pd.date_range(start=start_str, end=end_str, freq="D")]
    elif not df.empty and "acq_date" in df.columns:
        day_list = sorted(x for x in df["_acq_date_str"].dropna().unique().tolist() if x)
    else:
        return

    if not day_list:
        return

    daily_data_root = dataset_root / "daily_samples"
    daily_visual_root = visual_root / "daily_samples"
    daily_data_root.mkdir(parents=True, exist_ok=True)
    daily_visual_root.mkdir(parents=True, exist_ok=True)
    color_col = "frp" if "frp" in df.columns else "bright_ti4"
    print(f"Exporting daily visual samples to: {daily_visual_root}")

    for day_str in day_list:
        day_data_dir = daily_data_root / day_str
        day_visual_dir = daily_visual_root / day_str
        day_data_dir.mkdir(parents=True, exist_ok=True)
        day_visual_dir.mkdir(parents=True, exist_ok=True)
        if not df.empty and "_acq_date_str" in df.columns:
            day_df = df[df["_acq_date_str"] == day_str].copy()
        else:
            day_df = pd.DataFrame()

        day_base = f"{base_name}_{day_str}"

        save_outputs(
            day_df,
            day_data_dir,
            day_base,
            save_csv=True,
            save_shp=False,
            save_kml=False,
            save_hdf5=False,
        )

        if plot_points or (not fire_mask and not time_plot):
            plot_fire_map(
                day_df,
                f"FIRMS – {base_name} {day_str}",
                bbox,
                day_visual_dir / f"{day_base}_fire.png",
                color_col,
            )
        if fire_mask:
            plot_fire_mask(
                day_df,
                f"FIRMS – {base_name} {day_str} fire mask",
                bbox,
                day_visual_dir / f"{day_base}_mask.png",
                add_basemap=add_basemap,
                cluster_eps=cluster_eps,
                cluster_min_samples=cluster_min_samples,
                centroids_dir=day_data_dir,
            )
        if time_plot:
            plot_fire_map_time_based(
                day_df,
                f"FIRMS – {base_name} {day_str} time-based",
                bbox,
                day_visual_dir / f"{day_base}_time.png",
                add_basemap=add_basemap,
            )

        if enrich_earth_engine and HAS_EARTH_ENGINE:
            export_standalone_ee_feature_datasets(
                bbox,
                day_data_dir,
                day_visual_dir,
                day_base,
                date_str=day_str,
                start_date=day_str,
                end_date=day_str,
                add_basemap=add_basemap,
                ee_project=ee_project,
                fire_df=day_df,
                cluster_eps=cluster_eps,
                cluster_min_samples=cluster_min_samples,
            )
        elif enrich_earth_engine:
            print("  WARNING: daily EE export ignored. Install: pip install earthengine-api")

        if ee_raster_layers:
            extract_and_plot_ee_raster_layers(
                bbox,
                ee_raster_layers,
                day_data_dir,
                day_visual_dir,
                day_base,
                date_str=day_str,
                start_date=day_str,
                end_date=day_str,
                add_basemap=add_basemap,
                ee_project=ee_project,
            )


# ---------------------------------------------------------------------------
# Earth Engine feature enrichment
# ---------------------------------------------------------------------------
# EE project/bucket/folder come from UI (--ee-project, etc.) or interactive prompts.
# Uses your start_date, end_date, region/bbox from the same run.
EE_SAMPLE_SCALE = 1000  # meters; used for reduceRegions
EE_BATCH_SIZE = 1000   # max points per EE call to avoid timeouts


def _init_earth_engine(project: str | None = None) -> bool:
    """Initialize Earth Engine; authenticate if needed. Returns True if ready.
    Project comes from --ee-project or interactive prompt (see run_interactive).
    """
    if not HAS_EARTH_ENGINE:
        return False
    proj = (project or os.environ.get("EE_PROJECT", "")).strip()
    try:
        if proj:
            ee.Initialize(project=proj)
        else:
            ee.Initialize()
        return True
    except Exception as init_err:
        err_msg = str(init_err).lower()
        if "no project found" in err_msg:
            print("  Earth Engine requires a Google Cloud project.")
            print("  Provide --ee-project or enter it when prompted:")
            print("    set EE_PROJECT=your-gcp-project-id     (Windows CMD)")
            print("    $env:EE_PROJECT=\"your-gcp-project-id\"  (PowerShell)")
            print("  Or run: earthengine set_project your-gcp-project-id")
            print("  Register at: https://code.earthengine.google.com")
            return False
        try:
            ee.Authenticate()
            if proj:
                ee.Initialize(project=proj)
            else:
                ee.Initialize()
            return True
        except Exception as e:
            print(f"  Earth Engine auth failed: {e}")
            return False


# Drought (PDSI) and weather bands from ee_utils - aligned with export_ee_data pipeline
EE_DROUGHT_BANDS = ["pdsi"]  # Palmer Drought Severity Index
EE_WEATHER_BANDS = ["pr", "sph", "th", "tmmn", "tmmx", "vs", "erc"]  # precip, humidity, temp, wind, ERC


def enrich_with_earth_engine(
    df: pd.DataFrame,
    features: tuple[str, ...] = ("elevation", "ndvi", "population", "landcover", "drought", "weather"),
    ee_project: str | None = None,
) -> pd.DataFrame:
    """
    Sample elevation, vegetation (NDVI), population, land cover, drought (PDSI), and weather
    at each fire point using Google Earth Engine. Requires earthengine-api and prior
    `earthengine authenticate`.

    Datasets:
    - elevation: USGS SRTM 30m DEM
    - ndvi: MODIS MOD13A2 (16-day composite, uses acq_date when available)
    - population: CIESIN GPW v4 population density (~1km)
    - landcover: ESA WorldCover 2020 (10m, 11 classes)
    - drought: GRIDMET PDSI (Palmer Drought Severity Index) - requires acq_date
    - weather: GRIDMET (pr, sph, th, tmmn, tmmx, vs, erc) - requires acq_date
    """
    if not HAS_EARTH_ENGINE or df.empty:
        return df
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df
    if not _init_earth_engine(ee_project):
        print("  WARNING: Earth Engine enrichment skipped (init failed)")
        return df

    feat_list = [f for f in features if f in ("elevation", "ndvi", "population", "landcover", "drought", "weather")]
    print(f"Enriching with Earth Engine ({', '.join(feat_list)})...")
    df = df.copy()

    # Static layers: elevation, population, land cover
    # Note: CIESIN GPW is ImageCollection (years 2000–2020), not a single Image
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    pop = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density").select("population_density").filterDate("2020-01-01", "2020-12-31").first()
    # ESA WorldCover is ImageCollection (single year 2020), not a single Image
    lc = ee.ImageCollection("ESA/WorldCover/v100").select("Map").first().rename("landcover")

    # NDVI: use date closest to fire if acq_date available
    ndvi_img = None
    if "ndvi" in features and "acq_date" in df.columns:
        max_date = pd.to_datetime(df["acq_date"]).max()
        start = (max_date - pd.Timedelta(days=16)).strftime("%Y-%m-%d")
        end = (max_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        ndvi_img = (
            ee.ImageCollection("MODIS/061/MOD13A2")
            .filterDate(start, end)
            .select("NDVI")
            .mean()
            .multiply(0.0001)
            .rename("ndvi")
        )
    elif "ndvi" in features:
        ndvi_img = (
            ee.ImageCollection("MODIS/061/MOD13A2")
            .filter(ee.Filter.date("2020-01-01", "2020-12-31"))
            .select("NDVI")
            .mean()
            .multiply(0.0001)
            .rename("ndvi")
        )

    # Build base composite (static + NDVI)
    layers = [dem.rename("ee_elevation"), pop.rename("ee_population_density"), lc.rename("ee_landcover")]
    if ndvi_img is not None:
        layers.append(ndvi_img.rename("ee_ndvi"))
    base_composite = ee.Image.cat(layers)

    has_acq_date = "acq_date" in df.columns
    add_drought = "drought" in features and has_acq_date
    add_weather = "weather" in features and has_acq_date

    if (add_drought or add_weather) and not has_acq_date:
        print("  NOTE: drought and weather require acq_date; skipping those features.")

    def _get_composite_for_date(d: str) -> ee.Image:
        """Build composite with drought + weather for a given acq_date.
        Uses a 30-day lookback to handle GRIDMET/DROUGHT data lag (often 2–4 weeks)."""
        date_str = pd.to_datetime(d).strftime("%Y-%m-%d")
        # Look back 30 days to find most recent available data (avoids "Empty date ranges" error)
        start = pd.to_datetime(d) - pd.Timedelta(days=30)
        start_str = start.strftime("%Y-%m-%d")
        imgs = [base_composite]
        if add_drought:
            drought_coll = ee.ImageCollection("GRIDMET/DROUGHT").filterDate(start_str, date_str).select(EE_DROUGHT_BANDS)
            drought_img = drought_coll.sort("system:time_start", False).first()
            imgs.append(drought_img.rename([f"ee_{b}" for b in EE_DROUGHT_BANDS]))
        if add_weather:
            weather_coll = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(start_str, date_str).select(EE_WEATHER_BANDS)
            weather_img = weather_coll.sort("system:time_start", False).first()
            imgs.append(weather_img.rename([f"ee_{b}" for b in EE_WEATHER_BANDS]))
        return ee.Image.cat(imgs)

    all_props: list[dict] = []
    ee_feature_keys = ["ee_elevation", "ee_ndvi", "ee_population_density", "ee_landcover"]
    if add_drought:
        ee_feature_keys.extend([f"ee_{b}" for b in EE_DROUGHT_BANDS])
    if add_weather:
        ee_feature_keys.extend([f"ee_{b}" for b in EE_WEATHER_BANDS])

    if add_drought or add_weather:
        # Process by acq_date to use correct daily drought/weather; preserve original row order
        all_props = [{}] * len(df)  # placeholder, fill by index
        for acq_date, group in df.groupby("acq_date"):
            orig_indices = group.index.tolist()
            chunk = group.reset_index(drop=True)
            composite = _get_composite_for_date(str(acq_date))
            points_list = [
                ee.Feature(ee.Geometry.Point([row["longitude"], row["latitude"]]))
                for _, row in chunk.iterrows()
            ]
            fc_batch = ee.FeatureCollection(points_list)
            sampled = composite.reduceRegions(
                collection=fc_batch,
                reducer=ee.Reducer.first(),
                scale=EE_SAMPLE_SCALE,
                tileScale=4,
            )
            try:
                batch_list = sampled.getInfo()
                if batch_list and "features" in batch_list:
                    for i, f in enumerate(batch_list["features"]):
                        if i < len(orig_indices):
                            all_props[orig_indices[i]] = f.get("properties", {})
                else:
                    for i in orig_indices:
                        all_props[i] = {}
            except Exception as e:
                print(f"  EE sampling error for date {acq_date}: {e}")
                for i in orig_indices:
                    all_props[i] = {}
    else:
        # Single composite for all (no date-varying layers)
        n = len(df)
        for start_idx in range(0, n, EE_BATCH_SIZE):
            end_idx = min(start_idx + EE_BATCH_SIZE, n)
            chunk = df.iloc[start_idx:end_idx]
            points_list = [
                ee.Feature(ee.Geometry.Point([row["longitude"], row["latitude"]]))
                for _, row in chunk.iterrows()
            ]
            fc_batch = ee.FeatureCollection(points_list)
            sampled = base_composite.reduceRegions(
                collection=fc_batch,
                reducer=ee.Reducer.first(),
                scale=EE_SAMPLE_SCALE,
                tileScale=4,
            )
            try:
                batch_list = sampled.getInfo()
                if batch_list and "features" in batch_list:
                    for f in batch_list["features"]:
                        all_props.append(f.get("properties", {}))
                else:
                    for _ in range(end_idx - start_idx):
                        all_props.append({})
            except Exception as e:
                print(f"  EE sampling error (indices {start_idx}-{end_idx}): {e}")
                for _ in range(end_idx - start_idx):
                    all_props.append({})

    # Merge into dataframe (preserve order)
    for key in ee_feature_keys:
        vals = [p.get(key) for p in all_props]
        if any(v is not None for v in vals):
            df[key] = vals
    print(f"  Added Earth Engine features to {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------
def _prompt(msg: str, default: str = "") -> str:
    """Prompt user; return stripped input or default if empty."""
    if default:
        out = input(f"  {msg} [{default}]: ").strip() or default
    else:
        out = input(f"  {msg}: ").strip()
    return out


def run_interactive() -> argparse.Namespace:
    """Interactive terminal UI; prompts for options and returns args-like namespace."""
    from datetime import date, timedelta

    args = argparse.Namespace()
    args.no_verify_key = False

    print("\n" + "=" * 60)
    print("  FIRMS Fire Data – Interactive Mode")
    print("  (Press Enter to use default where shown)")
    print("=" * 60 + "\n")

    # 1. Date range
    today = date.today()
    default_end = today.strftime("%Y-%m-%d")
    default_start = (today - timedelta(days=4)).strftime("%Y-%m-%d")
    print("1. DATE RANGE")
    use_range = _prompt("Use date range? (y/n; n = last 5 days)", "n").lower()
    if use_range.startswith("y"):
        args.start_date = _prompt("Start date (YYYY-MM-DD)", default_start)
        args.end_date = _prompt("End date (YYYY-MM-DD)", args.start_date)
        args.days = 5  # unused when both set
    else:
        args.start_date = None
        args.end_date = None
        args.days = int(_prompt("Days of data (1-10)", "5") or 5)

    # 2. Area (multi-select: comma-separated, e.g. 1,2,3 or 'all')
    print("\n2. AREA (comma for multiple: 1,2 = California and Eaton; 'all' = all areas)")
    print("   1 = California  2 = Eaton  3 = Palisades  4 = Custom bbox")
    area_choice = _prompt("Area(s)", "1").strip().lower()
    args.areas = []
    args.bbox = None
    args.shapefile = None
    args.state = None
    args.wkt_polygon = None
    if area_choice == "all":
        args.areas = ["california", "eaton", "palisades"]
    else:
        for a in area_choice.replace(" ", "").split(","):
            if a == "2":
                args.areas.append("eaton")
            elif a == "3":
                args.areas.append("palisades")
            elif a == "4":
                args.areas.append("custom")
                if not args.bbox:
                    args.bbox = _prompt("Bbox (west,south,east,north)", "-124.5,32.5,-114.1,42.0")
            elif a == "1":
                args.areas.append("california")
    if not args.areas:
        args.areas = ["california"]

    # 3. Data source
    print("\n3. DATA SOURCE")
    print("   1 = Auto (NRT + archive)  2 = Archive only  3 = NRT only")
    src_choice = _prompt("Source type", "1").strip()
    args.source = None
    if src_choice == "2":
        args.use_archive = True
        args.nrt_only = False
    elif src_choice == "3":
        args.use_archive = False
        args.nrt_only = True
    else:
        args.use_archive = False
        args.nrt_only = False

    # 4. Visualization (multi-select: comma-separated, e.g. 1,2,3 or 'all')
    print("\n4. VISUALIZATION (comma for multiple: 1,2,3 or 'all' = all types)")
    print("   1 = Points (FRP)  2 = Fire mask  3 = Time-based")
    viz = _prompt("Plot type(s)", "1").strip().lower()
    args.plot_points = False
    args.fire_mask = False
    args.time_plot = False
    if viz == "all":
        args.plot_points = args.fire_mask = args.time_plot = True
    else:
        for v in viz.replace(" ", "").split(","):
            if v == "1":
                args.plot_points = True
            elif v == "2":
                args.fire_mask = True
            elif v == "3":
                args.time_plot = True
    if not (args.plot_points or args.fire_mask or args.time_plot):
        args.plot_points = True
    args.basemap = _prompt("Add basemap? (y/n)", "n").lower().startswith("y")
    args.cluster_eps = 0.02
    args.cluster_min_samples = 2

    # 5. Filters
    print("\n5. FILTERS (optional)")
    args.confidence = _prompt("Confidence (n,h,l or Enter for all)", "") or None
    args.min_frp = None
    frp_in = _prompt("Min FRP (Enter for none)", "")
    if frp_in:
        try:
            args.min_frp = float(frp_in)
        except ValueError:
            pass
    args.daynight = _prompt("Day/night (D/N or Enter for both)", "") or None

    # 6. Outputs
    print("\n6. OUTPUTS")
    args.save_shapefile = _prompt("Save Shapefile? (y/n)", "n").lower().startswith("y")
    args.save_kml = _prompt("Save KML? (y/n)", "n").lower().startswith("y")
    args.save_hdf5 = _prompt("Save HDF5? (y/n)", "n").lower().startswith("y")
    args.enrich_earth_engine = _prompt("Export standalone Earth Engine features? (y/n)", "n").lower().startswith("y")
    args.daily_visual_samples = _prompt("Export daily fire + EE visual samples? (y/n)", "n").lower().startswith("y")
    ee_in = _prompt("EE raster layers (comma-separated or 'all'; Enter for none)", "")
    if ee_in and ee_in.strip().lower() == "all":
        args.ee_raster_layers = "elevation,vegetation,landcover,drought,weather_temp,weather_precip,population"
    else:
        args.ee_raster_layers = ee_in.strip() if ee_in else None
    # EE config (uses your start_date, end_date, region/bbox from above)
    if args.enrich_earth_engine or args.ee_raster_layers:
        print("\n7. EARTH ENGINE CONFIG (uses your dates & region from above)")
        args.ee_project = _prompt("GCP/EE project ID", os.environ.get("EE_PROJECT", "")).strip() or None
        args.ee_bucket = _prompt("GCS bucket (for export; Enter to skip)", "").strip() or None
        args.ee_folder = _prompt("Folder in bucket (for export; Enter for 'exports')", "exports").strip() or "exports"
    else:
        args.ee_project = args.ee_bucket = args.ee_folder = None
    args.output_dir = None

    print("\n" + "-" * 60 + "\n")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # When run with no args, use interactive mode
    if len(sys.argv) == 1:
        sys.argv.append("-i")
    parser = argparse.ArgumentParser(
        description="Fetch, subset, and visualize FIRMS fire data for analysis and ML.",
        epilog="Run with -i for interactive mode. Use --help to see all flags.",
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode: prompt for options (default when no flags given)")
    parser.add_argument("--start-date", type=str, default=None, metavar="2025-01-07", help="Start date")
    parser.add_argument("--end-date", type=str, default=None, metavar="2025-01-31", help="End date (with --start-date)")
    parser.add_argument("--days", type=int, default=5, choices=range(1, 11), help="Days of data if no date range (1-10)")
    parser.add_argument("--source", type=str, default=None, help="FIRMS source (e.g. VIIRS_SNPP_NRT, VIIRS_SNPP_SP)")
    parser.add_argument("--use-archive", action="store_true", help="Use archive/SP sources only (for older data)")
    parser.add_argument("--nrt-only", action="store_true", help="Use NRT sources only (recent data, skip archive)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--no-verify-key", action="store_true", help="Skip MAP_KEY status check")

    # Data ingest filters (firms_data_ingest)
    parser.add_argument("--confidence", type=str, default=None, help="Confidence filter: n,h or n,h,l")
    parser.add_argument("--min-frp", type=float, default=None, help="Minimum Fire Radiative Power")
    parser.add_argument("--daynight", type=str, default=None, choices=("D", "N"), help="Day (D) or Night (N) only")

    # Shapefile / polygon (subSetDataFromShapeFileOrPolygon)
    parser.add_argument("--shapefile", type=Path, default=None, help="Path to shapefile for subsetting")
    parser.add_argument("--state", type=str, default=None, help="State name when using --shapefile")
    parser.add_argument("--wkt-polygon", type=str, default=None, help="WKT polygon for subsetting, e.g. POLYGON((-123 38,-121 38,-121 40,-123 40,-123 38))")

    # Bbox / area override (--area can be repeated for multiple)
    parser.add_argument("--bbox", type=str, default=None, help="Bbox: west,south,east,north")
    parser.add_argument("--area", action="append", default=None, dest="areas", metavar="AREA",
                        choices=("eaton", "california", "palisades", "all"), help="Area (repeat for multiple, or 'all' for all areas)")

    # Output options
    parser.add_argument("--save-shapefile", action="store_true", help="Save output as Shapefile")
    parser.add_argument("--save-kml", action="store_true", help="Save output as KML")
    parser.add_argument("--save-hdf5", action="store_true", help="Save output as HDF5 (.h5)")
    parser.add_argument("--points", action="store_true", help="Plot points (FRP-colored)")
    parser.add_argument("--time-plot", action="store_true", help="Create time-based colored plot")
    parser.add_argument("--fire-mask", action="store_true", help="Plot fire footprint polygons (detected area mask)")
    parser.add_argument("--cluster-eps", type=float, default=0.02, help="Fire cluster max distance in degrees (~0.02≈2km). Used with --fire-mask.")
    parser.add_argument("--cluster-min-samples", type=int, default=2, help="Min detections per cluster. Used with --fire-mask.")
    parser.add_argument("--all-viz", action="store_true", help="All visualization types: points, fire mask, time-based")
    parser.add_argument("--basemap", action="store_true", help="Add basemap to time plot / fire mask (requires contextily)")
    parser.add_argument("--enrich-earth-engine", action="store_true", help="Export standalone EE feature datasets for the fire-time window: elevation, NDVI, population, land cover, drought (PDSI), weather. Requires earthengine authenticate.")
    parser.add_argument("--daily-visual-samples", action="store_true", help="Create per-day folders containing daily fire visualizations and same-day EE feature outputs across the selected date range.")
    parser.add_argument("--ee-raster-layers", type=str, default=None, metavar="Layers",
                        help="Extract and visualize EE raster layers. Comma-separated or 'all' for all layers")
    parser.add_argument("--ee-project", type=str, default=None, help="GCP/EE project ID (for EE init; uses your start_date, end_date, region)")
    parser.add_argument("--ee-bucket", type=str, default=None, help="GCS bucket for export (optional)")
    parser.add_argument("--ee-folder", type=str, default="exports", help="Folder in bucket (default: exports)")
    args = parser.parse_args()

    if args.interactive:
        args = run_interactive()
    else:
        # Normalize for CLI: areas list, plot flags
        if args.areas is None:
            args.areas = ["custom"] if args.bbox else ["california"]
        elif "all" in args.areas:
            args.areas = ["california", "eaton", "palisades"]
        if not hasattr(args, "plot_points"):
            args.plot_points = args.points
        if getattr(args, "all_viz", False):
            args.plot_points = args.fire_mask = args.time_plot = True
        elif not (args.plot_points or args.fire_mask or args.time_plot):
            args.plot_points = True

    if args.source:
        sources = (args.source,)
    elif args.use_archive:
        sources = SOURCES_ARCHIVE
    elif args.nrt_only:
        sources = SOURCES_NRT
    else:
        sources = SOURCES  # NRT first, then archive fallback
    out_dir = args.output_dir or (Path(__file__).resolve().parent / "data output")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}\n")

    def _area_bbox_name(area: str):
        """Return (bbox, base_name) for area."""
        if area == "eaton":
            return EATON_BBOX, "firms_eaton"
        if area == "palisades":
            return PALISADES_BBOX, "firms_palisades"
        if area == "california":
            return CALIFORNIA_BBOX, "firms_california"
        if area == "custom" and args.bbox:
            parts = [float(x.strip()) for x in args.bbox.split(",")]
            if len(parts) == 4:
                return tuple(parts), "firms_custom"
        return CALIFORNIA_BBOX, "firms_california"

    def _get_fire_date_window(df_subset: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
        """Return (min_date, max_date, reference_date) for EE queries."""
        if not df_subset.empty and "acq_date" in df_subset.columns:
            dates = pd.to_datetime(df_subset["acq_date"], errors="coerce").dropna()
            if not dates.empty:
                start_date = dates.min().strftime("%Y-%m-%d")
                end_date = dates.max().strftime("%Y-%m-%d")
                return start_date, end_date, end_date
        start_date = args.start_date
        end_date = args.end_date or args.start_date
        ref_date = end_date or start_date or datetime.now().strftime("%Y-%m-%d")
        return start_date, end_date, ref_date

    # Shapefile/WKT path: single output, no multi-area
    use_areas = not args.wkt_polygon and not args.shapefile

    if use_areas:
        # Compute union bbox for fetch (covers all selected areas)
        bboxes = [_area_bbox_name(a)[0] for a in args.areas]
        w = min(b[0] for b in bboxes)
        s = min(b[1] for b in bboxes)
        e = max(b[2] for b in bboxes)
        n = max(b[3] for b in bboxes)
        fetch_bbox = (w, s, e, n)
    else:
        fetch_bbox = CALIFORNIA_BBOX
        if args.bbox:
            parts = [float(x.strip()) for x in args.bbox.split(",")]
            if len(parts) == 4:
                fetch_bbox = tuple(parts)

    map_key = get_map_key()
    if map_key and not args.no_verify_key:
        check_map_key(map_key)

    df = fetch_firms_data(
        map_key=map_key or None,
        bbox=fetch_bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        days=args.days,
        sources=sources,
    )

    ee_raster_layers = None
    if getattr(args, "ee_raster_layers", None):
        raw = [x.strip() for x in args.ee_raster_layers.split(",") if x.strip()]
        if raw and raw[0].lower() == "all" and len(raw) == 1:
            ee_raster_layers = list(EE_RASTER_LAYER_CONFIG.keys())
        else:
            ee_raster_layers = raw

    if df.empty and not ee_raster_layers:
        print("No FIRMS data retrieved. Exiting.")
        return

    # Apply data ingest filters
    conf_list = [c.strip() for c in args.confidence.split(",")] if args.confidence else None
    df = apply_confidence_frp_daynight_filters(
        df,
        confidence=tuple(conf_list) if conf_list else None,
        min_frp=args.min_frp,
        daynight=args.daynight,
    )

    if not use_areas:
        # Shapefile or WKT polygon (single output)
        if args.wkt_polygon:
            df = subset_by_wkt_polygon(df, args.wkt_polygon)
            base_name = "firms_polygon"
        else:
            df = subset_by_shapefile(df, args.shapefile, state_name=args.state)
            base_name = "firms_shapefile" + (f"_{args.state}" if args.state else "")
        bbox = fetch_bbox
        dataset_dir, visual_dir = get_output_dirs(out_dir, base_name, bbox)
        if not df.empty:
            df = add_acq_datetime(df)
            df = add_footprint_columns(df)
            save_outputs(df, dataset_dir, base_name, save_csv=True, save_shp=args.save_shapefile, save_kml=args.save_kml, save_hdf5=args.save_hdf5)
            color_col = "frp" if "frp" in df.columns else "bright_ti4"
            if args.fire_mask:
                plot_fire_mask(df, "FIRMS – Fire detection mask", bbox, visual_dir / f"{base_name}_mask.png",
                              add_basemap=args.basemap, cluster_eps=args.cluster_eps,
                              cluster_min_samples=args.cluster_min_samples, centroids_dir=dataset_dir)
            if args.time_plot:
                plot_fire_map_time_based(df, "FIRMS – Time-based", bbox, visual_dir / f"{base_name}_time.png", add_basemap=args.basemap)
            if args.plot_points:
                plot_fire_map(df, "FIRMS – Fire detections", bbox, visual_dir / f"{base_name}.png", color_col)
            if args.enrich_earth_engine and HAS_EARTH_ENGINE:
                ee_start_date, ee_end_date, date_str = _get_fire_date_window(df)
                export_standalone_ee_feature_datasets(
                    bbox,
                    dataset_dir,
                    visual_dir,
                    base_name,
                    date_str=date_str,
                    start_date=ee_start_date,
                    end_date=ee_end_date,
                    add_basemap=args.basemap,
                    ee_project=getattr(args, "ee_project", None),
                    fire_df=df,
                    cluster_eps=args.cluster_eps,
                    cluster_min_samples=args.cluster_min_samples,
                )
            elif args.enrich_earth_engine:
                print("  WARNING: --enrich-earth-engine ignored. Install: pip install earthengine-api")
            if getattr(args, "daily_visual_samples", False):
                export_daily_visual_samples(
                    df,
                    bbox,
                    dataset_dir,
                    visual_dir,
                    base_name,
                    plot_points=args.plot_points,
                    fire_mask=args.fire_mask,
                    time_plot=args.time_plot,
                    add_basemap=args.basemap,
                    enrich_earth_engine=args.enrich_earth_engine,
                    ee_raster_layers=ee_raster_layers,
                    ee_project=getattr(args, "ee_project", None),
                    start_date=ee_start_date if args.enrich_earth_engine else args.start_date,
                    end_date=ee_end_date if args.enrich_earth_engine else (args.end_date or args.start_date),
                    cluster_eps=args.cluster_eps,
                    cluster_min_samples=args.cluster_min_samples,
                )
        # EE raster layers (same bbox as fire mask; works with or without FIRMS data)
        if ee_raster_layers:
            ee_start_date, ee_end_date, date_str = _get_fire_date_window(df)
            extract_and_plot_ee_raster_layers(
                bbox,
                ee_raster_layers,
                dataset_dir,
                visual_dir,
                base_name,
                date_str=date_str,
                start_date=ee_start_date,
                end_date=ee_end_date,
                add_basemap=args.basemap,
                ee_project=getattr(args, "ee_project", None),
            )
    else:
        # Multi-area path
        if not df.empty:
            df = add_acq_datetime(df)
            df = add_footprint_columns(df)

        color_col = "frp" if "frp" in df.columns else "bright_ti4"
        for area in args.areas:
            bbox, base_name = _area_bbox_name(area)
            dataset_dir, visual_dir = get_output_dirs(out_dir, base_name, bbox)
            area_df = subset_bbox(df, bbox) if not df.empty else pd.DataFrame()
            if not area_df.empty:
                save_outputs(area_df, dataset_dir, base_name, save_csv=True, save_shp=args.save_shapefile, save_kml=args.save_kml, save_hdf5=args.save_hdf5)
                if args.fire_mask:
                    plot_fire_mask(area_df, f"FIRMS – {area.title()} fire mask", bbox, visual_dir / f"{base_name}_mask.png",
                                  add_basemap=args.basemap, cluster_eps=args.cluster_eps,
                                  cluster_min_samples=args.cluster_min_samples, centroids_dir=dataset_dir)
                if args.time_plot:
                    plot_fire_map_time_based(area_df, f"FIRMS – {area.title()} time-based", bbox, visual_dir / f"{base_name}_time.png", add_basemap=args.basemap)
                if args.plot_points:
                    plot_fire_map(area_df, f"FIRMS – {area.title()}", bbox, visual_dir / f"{base_name}.png", color_col)
            if args.enrich_earth_engine and HAS_EARTH_ENGINE:
                ee_start_date, ee_end_date, date_str = _get_fire_date_window(area_df)
                export_standalone_ee_feature_datasets(
                    bbox,
                    dataset_dir,
                    visual_dir,
                    base_name,
                    date_str=date_str,
                    start_date=ee_start_date,
                    end_date=ee_end_date,
                    add_basemap=args.basemap,
                    ee_project=getattr(args, "ee_project", None),
                    fire_df=area_df,
                    cluster_eps=args.cluster_eps,
                    cluster_min_samples=args.cluster_min_samples,
                )
            elif args.enrich_earth_engine:
                print("  WARNING: --enrich-earth-engine ignored. Install: pip install earthengine-api")
            if getattr(args, "daily_visual_samples", False):
                export_daily_visual_samples(
                    area_df,
                    bbox,
                    dataset_dir,
                    visual_dir,
                    base_name,
                    plot_points=args.plot_points,
                    fire_mask=args.fire_mask,
                    time_plot=args.time_plot,
                    add_basemap=args.basemap,
                    enrich_earth_engine=args.enrich_earth_engine,
                    ee_raster_layers=ee_raster_layers,
                    ee_project=getattr(args, "ee_project", None),
                    start_date=ee_start_date if args.enrich_earth_engine else args.start_date,
                    end_date=ee_end_date if args.enrich_earth_engine else (args.end_date or args.start_date),
                    cluster_eps=args.cluster_eps,
                    cluster_min_samples=args.cluster_min_samples,
                )
            if area_df.empty and not args.enrich_earth_engine and not ee_raster_layers:
                print(f"  No data for {area}, skipping.")
                continue
            # EE raster layers (same bbox as fire mask per area)
            if ee_raster_layers:
                ee_start_date, ee_end_date, date_str = _get_fire_date_window(area_df)
                extract_and_plot_ee_raster_layers(
                    bbox,
                    ee_raster_layers,
                    dataset_dir,
                    visual_dir,
                    base_name,
                    date_str=date_str,
                    start_date=ee_start_date,
                    end_date=ee_end_date,
                    add_basemap=args.basemap,
                    ee_project=getattr(args, "ee_project", None),
                )

    print("\nDone. Use the CSV/shapefile outputs for ML feature extraction.")
    print("To export EE training data to GCS: python data_export/export_ee_training_data_main.py -i")


if __name__ == "__main__":
    main()
