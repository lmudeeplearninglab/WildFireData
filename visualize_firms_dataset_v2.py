# coding=utf-8
from __future__ import annotations

import argparse
import builtins
import io
import os
import platform
import re
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

try:
    import rasterio
    from rasterio.transform import from_bounds as _rio_from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from joblib import Memory as _JoblibMemory
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


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
EARTH_RADIUS_KM = 6371.0088

FOOTPRINT_COLUMNS = (
    "footprint_nw_lat", "footprint_nw_lon", "footprint_ne_lat", "footprint_ne_lon",
    "footprint_se_lat", "footprint_se_lon", "footprint_sw_lat", "footprint_sw_lon",
)

# Terrain-derived layers share the DEM, so they count as static.
STATIC_EE_LAYERS = {"elevation", "slope", "aspect", "population", "landcover"}
_EE_INIT_CACHE: dict[str, bool] = {}
_EE_IMAGE_CACHE: dict[tuple, object] = {}
_EE_RASTER_CACHE: dict[tuple, tuple[np.ndarray, tuple[float, float, float, float]]] = {}
_EE_CACHE_MAX_ENTRIES = 64  # bound the in-process raster cache

# Timezone used to convert UTC acquisition times to the local day that
# daily gridded products (GRIDMET etc.) are indexed by.
LOCAL_TZ = os.environ.get("FIRMS_LOCAL_TZ", "America/Los_Angeles")

# Feature/label separation. Features must come strictly from before the
# label day or the model can read the fire out of its own burn scar.
DEFAULT_FEATURE_LAG_DAYS = 1


def _cache_put(cache: dict, key, value) -> None:
    """Insert into a bounded FIFO cache."""
    if len(cache) >= _EE_CACHE_MAX_ENTRIES:
        cache.pop(next(iter(cache)), None)
    cache[key] = value


def _disk_cache():
    """Return a joblib Memory for cross-run EE caching, or None."""
    if not HAS_JOBLIB:
        return None
    root = os.environ.get("FIRMS_CACHE_DIR", str(Path.home() / ".cache" / "firms_ee"))
    try:
        return _JoblibMemory(root, verbose=0)
    except Exception:
        return None


_DISK_CACHE = _disk_cache()


# ---------------------------------------------------------------------------
# Run transcript (report.txt)
# ---------------------------------------------------------------------------
# Mirrors everything printed to stdout/stderr, plus every interactive answer
# typed at a prompt, into a single plain-text file. On by default; the point
# is that a failed run leaves behind a complete, pasteable record without
# anyone having to remember to ask for one.
#
# Disable with --no-report, redirect with --report-file PATH or the
# FIRMS_REPORT_FILE environment variable, accumulate runs with --report-append.
DEFAULT_REPORT_NAME = "report.txt"

# The transcript exists to be handed to someone else, so a live credential
# must not ride along in it. A FIRMS MAP_KEY is 32 hex characters, and it
# appears in every FIRMS request URL -- which means it lands in the text of
# any requests exception that gets printed. See KEY_ROTATION.md.
_HEX32_RE = re.compile(r"\b[0-9a-fA-F]{32}\b")
_MAPKEY_QS_RE = re.compile(r"(?i)\b(MAP_KEY=)[^&\s\"'>]+")
_APIKEY_RE = re.compile(r"(?i)\b(api[_-]?key['\"]?\s*[:=]\s*['\"]?)([A-Za-z0-9_\-]{12,})")
# Prompts matching this never have their answer recorded, only the prompt.
_SECRET_PROMPT_RE = re.compile(r"(?i)key|token|secret|password|credential")

REDACTED = "<redacted>"


def redact_secrets(text: str) -> str:
    """Mask anything that looks like a credential before it reaches the log."""
    out = _HEX32_RE.sub(REDACTED, text)
    out = _MAPKEY_QS_RE.sub(r"\1" + REDACTED, out)
    out = _APIKEY_RE.sub(r"\1" + REDACTED, out)
    return out


class _TeeStream(io.TextIOBase):
    """Write to the real stream and to the transcript file at the same time.

    isatty() and fileno() delegate to the wrapped stream so that code testing
    for a terminal keeps behaving the way it does without the transcript.
    """

    def __init__(self, stream, log):
        self._stream = stream
        self._log = log

    def write(self, s: str) -> int:
        n = self._stream.write(s)
        try:
            self._log.write(redact_secrets(s))
        except Exception:
            pass
        return n

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._log.flush()
        except Exception:
            pass

    def close(self) -> None:
        # Never close the wrapped stream or the log; RunTranscript owns both.
        self.flush()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


_ACTIVE_REPORT_PATH: Path | None = None


def _package_versions() -> list[str]:
    """Versions of the packages whose absence or version changes behaviour."""
    rows: list[str] = []
    for name, mod in (
        ("numpy", np), ("pandas", pd), ("requests", requests),
        ("matplotlib", matplotlib),
        ("geopandas", gpd if HAS_GEOPANDAS else None),
        ("shapely", __import__("shapely") if HAS_SHAPELY else None),
        ("contextily", cx if HAS_CONTEXTILY else None),
        ("earthengine-api", ee if HAS_EARTH_ENGINE else None),
        ("rasterio", rasterio if HAS_RASTERIO else None),
    ):
        if mod is None:
            rows.append(f"  {name:<16} not installed")
        else:
            rows.append(f"  {name:<16} {getattr(mod, '__version__', 'unknown')}")
    try:
        import sklearn
        rows.append(f"  {'scikit-learn':<16} {sklearn.__version__}")
    except ImportError:
        rows.append(f"  {'scikit-learn':<16} not installed")
    return rows


class RunTranscript:
    """Context manager that mirrors a whole run into a text file.

    Captures, in order: a header describing the environment, every byte
    written to stdout and stderr, every prompt shown and answer typed in
    interactive mode, any traceback, and a footer with the exit status and
    wall time. Credentials are masked on the way in.

    Passing path=None makes the whole thing a no-op, so callers do not need
    to branch on whether reporting is enabled.
    """

    def __init__(self, path: Path | None, append: bool = False):
        self.path = Path(path) if path else None
        self.append = append
        self._log = None
        self._stdout = None
        self._stderr = None
        self._input = None
        self._started = None

    # -- prompt/answer capture ------------------------------------------
    def _logging_input(self, prompt: str = "") -> str:
        """Replacement for builtins.input that records both sides.

        The prompt is written through the tee rather than handed to input(),
        because CPython passes a prompt straight to the C-level readline on a
        tty, which would bypass sys.stdout and never reach the log. The typed
        answer is written only to the log -- the terminal already echoed it.
        """
        text = "" if prompt is None else str(prompt)
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()
        try:
            reply = self._input("")
        except EOFError:
            self._write_raw("\n[transcript] stdin closed (EOF) at prompt\n")
            raise
        if reply and _SECRET_PROMPT_RE.search(text):
            self._write_raw(REDACTED + "\n")
        else:
            self._write_raw(redact_secrets(reply) + "\n")
        return reply

    def _write_raw(self, text: str) -> None:
        if self._log is not None:
            try:
                self._log.write(text)
                self._log.flush()
            except Exception:
                pass

    # -- context manager ------------------------------------------------
    def __enter__(self) -> "RunTranscript":
        global _ACTIVE_REPORT_PATH
        if self.path is None:
            return self
        self._started = datetime.now()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(
                self.path,
                "a" if self.append else "w",
                encoding="utf-8",
                errors="replace",
                buffering=1,          # line buffered: survives a hard kill
            )
        except OSError as err:
            print(f"WARNING: could not open transcript {self.path}: {err}")
            self.path = None
            return self

        _ACTIVE_REPORT_PATH = self.path
        started_utc = datetime.now(tz=None).astimezone()
        self._write_raw(
            "=" * 72 + "\n"
            f"FIRMS run transcript\n"
            f"  started    {started_utc.isoformat(timespec='seconds')}\n"
            f"  command    {redact_secrets(' '.join([Path(sys.argv[0]).name] + sys.argv[1:]))}\n"
            f"  cwd        {Path.cwd()}\n"
            f"  script     {Path(__file__).resolve()}\n"
            f"  python     {sys.version.split()[0]} ({platform.platform()})\n"
            f"  local tz   {LOCAL_TZ}\n"
            + "\n".join(_package_versions()) + "\n"
            + "=" * 72 + "\n\n"
        )

        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(self._stdout, self._log)
        sys.stderr = _TeeStream(self._stderr, self._log)
        self._input = builtins.input
        builtins.input = self._logging_input
        print(f"Transcript: {self.path.resolve()}\n")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        global _ACTIVE_REPORT_PATH
        if self._log is None:
            return False
        if exc_type is SystemExit:
            code = exc.code if exc is not None else 0
            if isinstance(code, (int, type(None))):
                status = f"exited: {code or 0}"
            else:
                # SystemExit("message") -- the interpreter prints this to
                # stderr after __exit__ has already restored the streams,
                # so it has to be recorded here or it never reaches the file.
                self._write_raw("\n" + redact_secrets(str(code)) + "\n")
                status = "exited: 1"
        elif exc_type is not None:
            status = f"FAILED: {exc_type.__name__}"
            self._write_raw(
                "\n" + "-" * 72 + "\nTraceback\n" + "-" * 72 + "\n"
                + redact_secrets("".join(traceback.format_exception(exc_type, exc, tb)))
            )
        else:
            status = "completed"
        elapsed = (datetime.now() - self._started).total_seconds()
        self._write_raw(
            "\n" + "=" * 72 + "\n"
            f"  {status} after {elapsed:.1f}s at "
            f"{datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            + "=" * 72 + "\n"
        )

        sys.stdout, sys.stderr = self._stdout, self._stderr
        builtins.input = self._input
        try:
            self._log.close()
        except Exception:
            pass
        self._log = None
        _ACTIVE_REPORT_PATH = None
        return False  # never swallow the exception


def resolve_report_target(argv: list[str]) -> tuple[Path | None, bool]:
    """Work out the transcript path before argparse runs.

    The transcript has to be open before argument parsing so that a usage
    error, a --help dump, or an interactive session all land in it. That
    means reading these three flags directly off argv; they are also
    registered on the real parser so --help documents them.
    """
    if "--no-report" in argv:
        return None, False
    append = "--report-append" in argv
    raw = os.environ.get("FIRMS_REPORT_FILE", "").strip() or None
    for i, token in enumerate(argv):
        if token == "--report-file" and i + 1 < len(argv):
            raw = argv[i + 1]
        elif token.startswith("--report-file="):
            raw = token.split("=", 1)[1]
    if raw:
        path = Path(raw).expanduser()
        if path.is_dir():
            path = path / DEFAULT_REPORT_NAME
    else:
        path = Path(__file__).resolve().parent / DEFAULT_REPORT_NAME
    return path, append


# ---------------------------------------------------------------------------
# Data Ingest (from firms_data_ingest.ipynb)                                -
# ---------------------------------------------------------------------------
def add_acq_datetime(df: pd.DataFrame, local_tz: str = LOCAL_TZ) -> pd.DataFrame:
    """
    Combine acq_date and acq_time into a timezone-aware UTC timestamp, and
    derive the *local* calendar date.

    FIRMS reports acq_time in UTC. Daily gridded products (GRIDMET, drought)
    are indexed by local day. Joining on the raw UTC acq_date misattributes
    every night overpass by one day -- roughly half of all VIIRS detections
    in the western US. Downstream joins should use `local_date`.

    acq_time is parsed numerically because concatenating sources with
    differing column sets promotes the column to float, and str(1234.0)
    is not fixed by zfill.
    """
    if df.empty or "acq_date" not in df.columns or "acq_time" not in df.columns:
        return df
    df = df.copy()
    minutes = pd.to_numeric(df["acq_time"], errors="coerce")
    hhmm = minutes.fillna(0).astype("int64").astype(str).str.zfill(4)
    time_str = df["acq_date"].astype(str) + " " + hhmm
    df["acq_datetime"] = pd.to_datetime(
        time_str, format="%Y-%m-%d %H%M", errors="coerce", utc=True
    )
    try:
        df["local_date"] = (
            df["acq_datetime"].dt.tz_convert(local_tz).dt.strftime("%Y-%m-%d")
        )
    except Exception:
        df["local_date"] = df["acq_datetime"].dt.strftime("%Y-%m-%d")
    return df


# MODIS reports confidence as an integer 0-100; VIIRS uses l/n/h.
# These are the bands MODIS documentation maps the letter classes onto.
MODIS_CONFIDENCE_BANDS = {"l": (0, 30), "n": (30, 80), "h": (80, 101)}


def apply_confidence_frp_daynight_filters(
    df: pd.DataFrame,
    confidence: tuple[str, ...] | None = None,
    min_frp: float | None = None,
    daynight: str | None = None,
) -> pd.DataFrame:
    """
    Subset by confidence (n/normal, h/high, l/low), min FRP, daynight (D/N).

    Handles both confidence encodings. Passing letter classes against a
    numeric (MODIS) column previously dropped every row silently; a missing
    confidence column previously raised KeyError.
    """
    if df.empty:
        return df
    out = df.copy()
    if confidence:
        if "confidence" not in out.columns:
            print("  WARNING: no 'confidence' column; confidence filter skipped.")
        else:
            wanted = [str(c).strip().lower() for c in confidence]
            col = out["confidence"]
            if pd.api.types.is_numeric_dtype(col):
                mask = pd.Series(False, index=out.index)
                for c in wanted:
                    if c not in MODIS_CONFIDENCE_BANDS:
                        print(f"  WARNING: unknown confidence class {c!r}; ignored.")
                        continue
                    lo, hi = MODIS_CONFIDENCE_BANDS[c]
                    mask |= col.between(lo, hi, inclusive="left")
                out = out[mask]
            else:
                out = out[col.astype(str).str.strip().str.lower().isin(wanted)]
    if min_frp is not None and "frp" in out.columns:
        out = out[pd.to_numeric(out["frp"], errors="coerce") >= min_frp]
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
    # FIRMS coordinates are always EPSG:4326. Tagging them with the
    # shapefile's CRS silently produces a wrong join whenever that
    # shapefile is projected (Albers, State Plane, ...).
    fire_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    if states_gdf.crs is not None:
        fire_gdf = fire_gdf.to_crs(states_gdf.crs)
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


def add_footprint_columns(df: pd.DataFrame, use_scan_track: bool = True) -> pd.DataFrame:
    """
    Add Fire Footprint Area columns (dx, dy, corners, area_km2).

    Prefers the per-detection `scan` and `track` dimensions that FIRMS
    supplies, which capture the growth of the pixel toward the swath edge
    (VIIRS spans roughly 0.375 km at nadir to ~0.8 km at scan edge -- a
    factor of ~4 in area). Falls back to the nominal nadir coefficient
    only when those columns are absent.
    """
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return df
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)

    if "instrument" in df.columns:
        instruments = df["instrument"].fillna("VIIRS")
        coefs = np.array([get_footprint_coef(inst) for inst in instruments])
    else:
        coefs = np.full(len(df), 0.375)

    have_scan_track = use_scan_track and {"scan", "track"} <= set(df.columns)
    if have_scan_track:
        scan = pd.to_numeric(df["scan"], errors="coerce").to_numpy(dtype=float)
        track = pd.to_numeric(df["track"], errors="coerce").to_numpy(dtype=float)
        # Fall back per-row where FIRMS omitted a value.
        scan = np.where(np.isfinite(scan) & (scan > 0), scan, coefs)
        track = np.where(np.isfinite(track) & (track > 0), track, coefs)
    else:
        scan = coefs
        track = coefs

    # Evaluate cos() on clipped latitudes: np.where computes both branches,
    # so an unclipped cos(90 deg) raises a divide-by-zero warning.
    cos_lat = np.cos(np.radians(np.clip(lat, -89.99, 89.99)))
    dx = scan / (cos_lat * KM_PER_DEG) / 2.0
    dy = (track / KM_PER_DEG) / 2.0
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
    df["footprint_area_km2"] = scan * track
    df["footprint_source"] = "scan_track" if have_scan_track else "nominal"
    return df


def _footprint_corner_array(df: pd.DataFrame) -> np.ndarray:
    """Stack footprint corners into an (n, 4, 2) array of (lon, lat) rings."""
    return np.stack(
        [
            np.column_stack([df["footprint_nw_lon"].to_numpy(), df["footprint_nw_lat"].to_numpy()]),
            np.column_stack([df["footprint_ne_lon"].to_numpy(), df["footprint_ne_lat"].to_numpy()]),
            np.column_stack([df["footprint_se_lon"].to_numpy(), df["footprint_se_lat"].to_numpy()]),
            np.column_stack([df["footprint_sw_lon"].to_numpy(), df["footprint_sw_lat"].to_numpy()]),
        ],
        axis=1,
    )


def footprint_polygons(df: pd.DataFrame) -> list:
    """
    Build one Shapely polygon per detection footprint.

    Uses Shapely 2.0's vectorized constructor when available; iterrows on
    100k+ California detections dominates all non-network runtime.
    """
    if df.empty or not HAS_SHAPELY:
        return []
    req = FOOTPRINT_COLUMNS
    if not all(c in df.columns for c in req):
        return []
    corners = _footprint_corner_array(df)
    try:
        import shapely
        if hasattr(shapely, "polygons"):
            polys = shapely.polygons(corners)
            return [p for p in polys if p is not None and not p.is_empty]
    except Exception:
        pass
    from shapely.geometry import Polygon
    out = []
    for ring in corners:
        poly = Polygon(ring)
        if not poly.is_empty:
            out.append(poly)
    return out


def _cluster_hull(points_xy: np.ndarray, concave_ratio: float | None = 0.3):
    """
    Boundary polygon for one cluster's footprint corners.

    Prefers a concave hull: real fire perimeters run up canyons and stretch
    downwind, so a convex hull over an L- or crescent-shaped fire can easily
    double the reported area. Falls back to convex hull on older Shapely.
    """
    from shapely.geometry import MultiPoint

    if len(points_xy) < 3:
        return None
    mp = MultiPoint([tuple(p) for p in points_xy])
    hull = None
    if concave_ratio is not None:
        try:
            import shapely
            if hasattr(shapely, "concave_hull"):
                hull = shapely.concave_hull(mp, ratio=concave_ratio)
        except Exception:
            hull = None
    if hull is None or hull.is_empty or hull.geom_type not in ("Polygon", "MultiPolygon"):
        hull = mp.convex_hull
    if hull.is_empty or hull.geom_type == "LineString" or hull.area == 0:
        return None
    if hull.geom_type == "MultiPolygon":
        return max(hull.geoms, key=lambda g: g.area)
    return hull


def _cluster_feature_matrix(
    df: pd.DataFrame,
    eps_km: float,
    time_days: float | None,
    spread_km_per_day: float,
) -> tuple[np.ndarray, float, str]:
    """
    Build the DBSCAN input.

    Space-only clustering runs on radians with the haversine metric so the
    neighbourhood is a true circle. Degrees with a euclidean metric are
    anisotropic: 0.02 deg of longitude at 34N is 1.84 km against 2.22 km of
    latitude, a ~17% distortion that grows across California's span.

    When a time window is given, points are projected to a local km plane and
    a scaled time axis appended, so detections at the same place months apart
    are no longer merged into a single fire.
    """
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)

    if time_days is None or "acq_datetime" not in df.columns:
        coords = np.radians(np.column_stack([lat, lon]))
        return coords, eps_km / EARTH_RADIUS_KM, "haversine"

    t = pd.to_datetime(df["acq_datetime"], errors="coerce", utc=True)
    if t.isna().all():
        coords = np.radians(np.column_stack([lat, lon]))
        return coords, eps_km / EARTH_RADIUS_KM, "haversine"

    t_days = (t - t.min()).dt.total_seconds().to_numpy() / 86400.0
    t_days = np.nan_to_num(t_days, nan=0.0)

    lat0 = float(np.nanmean(lat))
    y_km = (lat - lat0) * KM_PER_DEG
    x_km = (lon - float(np.nanmean(lon))) * KM_PER_DEG * np.cos(np.radians(lat0))
    # Scale time so `time_days` of separation costs the same as eps_km of
    # distance; spread_km_per_day is the linking aggressiveness knob.
    t_scaled = t_days * spread_km_per_day
    return np.column_stack([x_km, y_km, t_scaled]), eps_km, "euclidean"


def cluster_fire_points_to_polygons(
    df: pd.DataFrame,
    eps_km: float = 2.0,
    min_samples: int = 2,
    time_days: float | None = None,
    spread_km_per_day: float = 15.0,
    keep_singletons: bool = True,
    concave_ratio: float | None = 0.3,
) -> tuple[list, pd.DataFrame]:
    """
    Cluster fire detections and return one boundary polygon per cluster
    plus cluster centroids.

    Args:
        df: DataFrame with footprint corner columns (and latitude, longitude).
        eps_km: Neighbourhood radius in kilometres.
        min_samples: Minimum detections to form a dense cluster.
        time_days: If set, cluster in space-time over this window rather than
            space alone. Recommended for any multi-week archive pull.
        spread_km_per_day: Time-axis scaling for space-time clustering.
        keep_singletons: Retain DBSCAN noise points as one-detection clusters.
            An isolated detection is a *new ignition* -- the most valuable
            positive sample for a real-time model. Dropping it (the old
            behaviour with min_samples=2) throws away exactly the cases the
            model is meant to catch.
        concave_ratio: Concave hull tightness, or None to force convex hulls.

    Returns:
        (polygons, centroids_df) where centroids_df has cluster_id,
        centroid_lat, centroid_lon, n_detections, is_singleton, and summary
        statistics used downstream as labels.
    """
    req = FOOTPRINT_COLUMNS
    if not all(c in df.columns for c in req) or df.empty:
        return [], pd.DataFrame()

    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return [], pd.DataFrame()  # Caller will fall back to individual footprints

    X, eps, metric = _cluster_feature_matrix(df, eps_km, time_days, spread_km_per_day)
    algorithm = "ball_tree" if metric == "haversine" else "auto"
    labels = DBSCAN(
        eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm
    ).fit_predict(X)

    df = df.reset_index(drop=True)
    labels = np.asarray(labels)

    groups: list[tuple[int, np.ndarray, bool]] = []
    for label in sorted(set(labels[labels >= 0])):
        groups.append((int(label), np.flatnonzero(labels == label), False))
    if keep_singletons:
        noise_idx = np.flatnonzero(labels < 0)
        for i, row_idx in enumerate(noise_idx):
            groups.append((-(i + 1), np.array([row_idx]), True))

    polygons = []
    centroid_rows = []
    for label, idx, is_singleton in groups:
        sub = df.iloc[idx]
        points = _footprint_corner_array(sub).reshape(-1, 2)
        hull = _cluster_hull(points, concave_ratio=concave_ratio)
        if hull is None:
            continue

        row = {
            "cluster_id": int(label),
            "centroid_lat": float(sub["latitude"].mean()),
            "centroid_lon": float(sub["longitude"].mean()),
            "n_detections": int(len(sub)),
            "is_singleton": bool(is_singleton),
            "hull_area_km2": float(_polygon_area_km2(hull)),
        }
        if "frp" in sub.columns:
            frp = pd.to_numeric(sub["frp"], errors="coerce")
            row["frp_sum"] = float(frp.sum())
            row["frp_max"] = float(frp.max()) if frp.notna().any() else np.nan
        if "footprint_area_km2" in sub.columns:
            row["detected_area_km2"] = float(sub["footprint_area_km2"].sum())
        if "acq_datetime" in sub.columns:
            t = pd.to_datetime(sub["acq_datetime"], errors="coerce", utc=True)
            if t.notna().any():
                row["first_detection"] = t.min().strftime("%Y-%m-%d %H:%M")
                row["last_detection"] = t.max().strftime("%Y-%m-%d %H:%M")
                row["duration_hours"] = float(
                    (t.max() - t.min()).total_seconds() / 3600.0
                )
        if "local_date" in sub.columns:
            row["first_local_date"] = str(sub["local_date"].min())

        polygons.append(hull)
        centroid_rows.append(row)

    centroids_df = pd.DataFrame(centroid_rows) if centroid_rows else pd.DataFrame()
    return polygons, centroids_df


def _polygon_area_km2(poly) -> float:
    """
    Approximate polygon area in km2 using a local equal-area scaling.

    Adequate for the sub-100 km fire extents here; use an equal-area
    projection if you need this to be defensible at state scale.
    """
    try:
        lat0 = np.radians(poly.centroid.y)
        return float(poly.area * (KM_PER_DEG ** 2) * np.cos(lat0))
    except Exception:
        return float("nan")


FIRMS_MAP_KEY_URL = "https://firms.modaps.eosdis.nasa.gov/api/map_key"


def get_map_key() -> str:
    """
    FIRMS MAP_KEY from the FIRMS_MAP_KEY environment variable.

    Never hardcode a key here: it is a live credential, it ends up in git
    history, and the NASA quota is per-key. Prompting is only attempted on
    an interactive terminal so cron and CI fail loudly instead of hanging
    on stdin.
    """
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if key:
        return key
    if not sys.stdin.isatty():
        raise SystemExit(
            "FIRMS_MAP_KEY is not set. Export it before running:\n"
            "    export FIRMS_MAP_KEY=your_key\n"
            f"Free keys: {FIRMS_MAP_KEY_URL}"
        )
    print(f"FIRMS MAP_KEY not set. Get a free key at {FIRMS_MAP_KEY_URL}")
    key = input("Paste MAP_KEY: ").strip()
    if not key:
        raise SystemExit("No MAP_KEY provided.")
    return key


def _firms_session(retries: int = 4) -> requests.Session:
    """Session with backoff on FIRMS throttling (429) and transient 5xx."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_FIRMS_SESSION = _firms_session()


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


class FirmsRequestError(RuntimeError):
    """FIRMS rejected the request (bad key, quota exhausted, bad params)."""


def fetch_area_csv(
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    day_range: int = 5,
    date: str | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Fetch one FIRMS area CSV.

    Uses requests rather than pd.read_csv(url) so that an invalid key, an
    exhausted quota and a genuinely fire-free window are distinguishable.
    Previously all three returned an empty DataFrame identically, which made
    a broken run look like a quiet fire season.
    """
    w, s, e, n = bbox
    area = f"{w},{s},{e},{n}"
    url = f"{FIRMS_AREA_BASE}/{map_key}/{source}/{area}/{day_range}"
    if date:
        url = f"{url}/{date}"
    try:
        resp = _FIRMS_SESSION.get(url, timeout=timeout)
    except requests.RequestException as err:
        print(f"  Network error ({source}): {err}")
        return pd.DataFrame()

    if resp.status_code == 429:
        raise FirmsRequestError(
            "FIRMS rate limit hit (429). The quota is per-key on a rolling "
            "10-minute window; wait and retry, or reduce the date range."
        )
    if resp.status_code >= 400:
        raise FirmsRequestError(f"FIRMS returned HTTP {resp.status_code}: {resp.text[:200]}")

    head = resp.text[:500]
    lowered = head.lower()
    for marker in ("invalid map_key", "invalid mapkey", "transaction limit"):
        if marker in lowered:
            raise FirmsRequestError(f"FIRMS rejected the request: {head.strip()[:200]}")
    if "<html" in lowered[:200]:
        raise FirmsRequestError(f"FIRMS returned HTML, not CSV: {head.strip()[:200]}")

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as err:
        print(f"  Could not parse CSV ({source}): {err}")
        return pd.DataFrame()
    if "latitude" not in df.columns:
        return pd.DataFrame()
    return df


def subset_bbox(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Subset to bbox (firms_data_ingest style)."""
    w, s, e, n = bbox
    return df[
        (df["longitude"] >= w) & (df["latitude"] >= s)
        & (df["longitude"] <= e) & (df["latitude"] <= n)
    ].copy()


DEDUP_KEYS = ("latitude", "longitude", "acq_date", "acq_time", "satellite")


def _dedup_detections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate detections.

    `satellite` is part of the key: without it, two satellites detecting the
    same pixel in the same minute collapse into one row, silently discarding
    a genuine independent observation.
    """
    keys = [c for c in DEDUP_KEYS if c in df.columns]
    if not keys:
        return df
    return df.drop_duplicates(subset=keys, keep="first")


def _fetch_window(
    map_key: str,
    bbox: tuple[float, float, float, float],
    sources: tuple[str, ...],
    day_range: int,
    date: str | None,
    all_sources: bool = True,
) -> list[pd.DataFrame]:
    """
    Fetch one time window across sources.

    With all_sources=True (default) every requested source is queried and
    tagged. The old behaviour stopped at the first source returning any rows,
    so one chunk could be VIIRS_SNPP and the next NOAA-20 or MODIS -- making
    detection density a function of satellite availability rather than fire
    activity, which a model will happily learn as signal.
    """
    out = []
    for src_name in sources:
        chunk = fetch_area_csv(map_key, src_name, bbox, day_range=day_range, date=date)
        if chunk.empty:
            continue
        out.append(chunk.assign(firms_source=src_name))
        if not all_sources:
            break
    return out


def _enforce_requested_window(
    df: pd.DataFrame, start_date: str | None, end_date: str | None
) -> pd.DataFrame:
    """Drop detections outside the requested date range, loudly.

    The FIRMS area endpoint does not reject a date it cannot serve. Asking an
    NRT source (roughly a 10-day rolling window) for a date two years back can
    come back populated with *recent* detections instead of empty. Silently
    concatenated with archive rows, that puts today's fires in a historical
    training set under historical dates' neighbours -- contamination that is
    invisible in every plot and fatal to the labels.

    Anything outside the window is dropped here and reported per source, so a
    misbehaving source shows up as a number rather than as a mystery.
    """
    if df.empty or not start_date or "acq_date" not in df.columns:
        return df
    lo = pd.to_datetime(start_date, errors="coerce")
    hi = pd.to_datetime(end_date or start_date, errors="coerce")
    if pd.isna(lo) or pd.isna(hi):
        return df
    dates = pd.to_datetime(df["acq_date"], errors="coerce")
    keep = dates.between(lo, hi)
    dropped = int((~keep).sum())
    if dropped:
        print(f"  WARNING: {dropped} detections outside the requested window "
              f"{lo.date()}..{hi.date()} were returned and have been dropped.")
        if "firms_source" in df.columns:
            offenders = df.loc[~keep, "firms_source"].value_counts().to_dict()
            print(f"           by source: {offenders}")
            out_range = dates[~keep].dropna()
            if not out_range.empty:
                print(f"           their dates ran {out_range.min().date()} "
                      f"to {out_range.max().date()}")
        print("           A source returning out-of-window rows is serving a "
              "different period than asked for; prefer --use-archive for "
              "historical pulls.")
    return df.loc[keep].copy()


def fetch_firms_data(
    map_key: str | None = None,
    bbox: tuple[float, float, float, float] = CALIFORNIA_BBOX,
    start_date: str | None = None,
    end_date: str | None = None,
    days: int = 5,
    sources: tuple[str, ...] = SOURCES,
    use_sample_fallback: bool = True,
    all_sources: bool = True,
) -> pd.DataFrame:
    """
    Fetch FIRMS fire data for a bbox.

    1) If map_key: call the FIRMS area API, chunked to the endpoint's 10-day
       cap (5-day chunks here).
    2) If nothing came back and no dates were requested, fall back to the
       public sample CSV so the script is runnable without a key.
    """
    df = pd.DataFrame()
    if map_key:
        print("Fetching from FIRMS API...")
        chunks: list[pd.DataFrame] = []
        if start_date and end_date:
            start_d = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_d = datetime.strptime(end_date, "%Y-%m-%d").date()
            if start_d > end_d:
                raise SystemExit(f"start_date {start_date} is after end_date {end_date}")
            d = start_d
            while d <= end_d:
                chunk_end = min(d + timedelta(days=4), end_d)
                n_days = (chunk_end - d).days + 1
                chunks.extend(
                    _fetch_window(
                        map_key, bbox, sources, n_days,
                        d.strftime("%Y-%m-%d"), all_sources=all_sources,
                    )
                )
                d = chunk_end + timedelta(days=1)
        else:
            chunks.extend(
                _fetch_window(
                    map_key, bbox, sources, days, start_date, all_sources=all_sources
                )
            )
        if chunks:
            df = _dedup_detections(pd.concat(chunks, ignore_index=True))
        if not df.empty:
            df = subset_bbox(df, bbox)
            df = _enforce_requested_window(df, start_date, end_date)
            if "firms_source" in df.columns:
                counts = df["firms_source"].value_counts().to_dict()
                print(f"  API: {len(df)} detections {counts}")
            else:
                print(f"  API: {len(df)} detections")

    if df.empty and use_sample_fallback:
        # Don't fall back to sample when the user specified dates: it would
        # silently return 2023 data for a 2025 request.
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
    # add_acq_datetime returns the original object when the source columns
    # are absent, so copy before assigning _day or we write through to the
    # caller's frame.
    df = add_acq_datetime(df).copy()
    date_col = (
        "local_date" if "local_date" in df.columns
        else "acq_date" if "acq_date" in df.columns
        else "acq_datetime" if "acq_datetime" in df.columns
        else None
    )
    if not date_col:
        plot_fire_map(df, title, bbox, path)
        return

    has_footprints = all(c in df.columns for c in FOOTPRINT_COLUMNS)
    if not has_footprints or not HAS_GEOPANDAS:
        df["_day"] = pd.to_datetime(df[date_col]).dt.date
        days = sorted(df["_day"].unique())
        if not days:
            plot_fire_map(df, title, bbox, path)
            return
        cmap = plt.get_cmap("YlOrRd", len(days) + 1)
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

    df["_day"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    days = sorted(d for d in df["_day"].unique() if pd.notna(d))
    cmap = plt.get_cmap("YlOrRd", len(days) + 1)

    gdf_by_day = []
    for day in days:
        sub = df[df["_day"] == day]
        polygons = footprint_polygons(sub)
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

    from matplotlib.patches import Patch

    handles = []
    for i, (day, gdf_day) in enumerate(gdf_by_day):
        color = cmap((i + 1) / (len(days) + 1))
        to_plot = gdf_day.to_crs(epsg=3857) if basemap_ok else gdf_day
        to_plot.plot(ax=ax, facecolor=color, edgecolor="black", alpha=0.5, linewidth=0.3)
        # GeoDataFrame.plot draws a PatchCollection, which matplotlib will not
        # build legend handles from; without explicit proxies the day legend
        # renders empty, which defeats the point of the plot.
        handles.append(Patch(facecolor=color, edgecolor="black", alpha=0.5, label=str(day)))

    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=2)
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
    cluster_eps_km: float = 2.0,
    cluster_min_samples: int = 2,
    cluster_time_days: float | None = None,
    keep_singletons: bool = True,
    concave_ratio: float | None = 0.3,
    centroids_dir: Path | None = None,
    save_vector: bool = True,
) -> None:
    """
    Plot the fire detection mask as clustered polygons and export those
    polygons as vector data.

    The polygons -- not the PNG -- are the actual ML label, so they are
    written to GeoJSON alongside the centroid table whenever geopandas is
    available.

    Note: FIRMS provides active fire detections, not post-fire burned
    perimeters. Absence of a detection is missing data (cloud, smoke,
    overpass gap), not evidence of no fire.
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

    if not all(c in df.columns for c in FOOTPRINT_COLUMNS):
        print("  WARNING: Footprint columns missing; falling back to point plot.")
        plot_fire_map(df, title, bbox, path)
        return

    if not HAS_GEOPANDAS:
        print("  WARNING: Fire mask requires geopandas. Falling back to point plot.")
        plot_fire_map(df, title, bbox, path)
        return

    polygons, centroids_df = cluster_fire_points_to_polygons(
        df,
        eps_km=cluster_eps_km,
        min_samples=cluster_min_samples,
        time_days=cluster_time_days,
        keep_singletons=keep_singletons,
        concave_ratio=concave_ratio,
    )

    if not polygons:
        # Fallback: individual footprints (e.g. if sklearn is not installed)
        polygons = footprint_polygons(df)
        if not polygons:
            plot_fire_map(df, title, bbox, path)
            return
        print("  Note: Install scikit-learn for fire clustering. Using individual footprints.")
        centroids_df = df[["latitude", "longitude"]].copy()
        centroids_df = centroids_df.rename(
            columns={"latitude": "centroid_lat", "longitude": "centroid_lon"}
        )
        centroids_df.insert(0, "cluster_id", np.arange(len(centroids_df)))
        centroids_df["n_detections"] = 1
        centroids_df["is_singleton"] = True

    gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    if not centroids_df.empty and len(centroids_df) == len(gdf):
        gdf = gpd.GeoDataFrame(
            centroids_df.reset_index(drop=True), geometry=list(polygons), crs="EPSG:4326"
        )

    out_root = centroids_dir or path.parent
    out_root.mkdir(parents=True, exist_ok=True)
    stem = path.stem.replace("_mask", "")

    if not centroids_df.empty:
        centroids_path = out_root / f"{stem}_centroids.csv"
        centroids_df.to_csv(centroids_path, index=False)
        n_single = int(centroids_df.get("is_singleton", pd.Series(dtype=bool)).sum())
        print(
            f"  Saved: {centroids_path} ({len(centroids_df)} clusters, "
            f"{n_single} singleton)"
        )

    # The polygons are the label. Persist them, not just the picture.
    if save_vector:
        vector_path = out_root / f"{stem}_clusters.geojson"
        try:
            gdf.to_file(vector_path, driver="GeoJSON")
            print(f"  Saved: {vector_path} ({len(gdf)} polygons)")
        except Exception as ex:
            print(f"  GeoJSON save failed: {ex}")

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
            # PyTables cannot serialize a mixed-type object column, and
            # `confidence` is exactly that once VIIRS (l/n/h) and MODIS
            # (0-100 int) rows are concatenated. Cast object columns to str
            # for the HDF5 copy only -- the CSV keeps the native values, and
            # apply_confidence_frp_daynight_filters still sees the original
            # dtypes upstream of this call.
            h5_out = out.copy()
            for col in h5_out.columns[h5_out.dtypes == object]:
                if not pd.api.types.is_string_dtype(h5_out[col]):
                    h5_out[col] = h5_out[col].astype(str)
            h5_out.to_hdf(h5_path, key="firms", mode="w", format="table")
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
    "humidity": ("GnBu", "Specific humidity (g/kg)"),
    "weather_temp": ("YlOrRd", "Max temperature (C)"),
    "weather_precip": ("Blues", "Precipitation (mm)"),
    "wind_speed": ("cividis", "Wind speed (m/s)"),
    "wind_direction": ("twilight", "Wind direction (deg)"),
    "wind_u": ("coolwarm", "Wind u (east, m/s)"),
    "wind_v": ("coolwarm", "Wind v (north, m/s)"),
    "slope": ("magma", "Slope (deg)"),
    "aspect": ("twilight", "Aspect (deg)"),
    "erc": ("inferno", "Energy Release Component"),
    "vpd": ("YlOrRd", "Vapour pressure deficit (kPa)"),
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


# ---------------------------------------------------------------------------
# Earth Engine raster extraction and visualization (like fire footprints)
# ---------------------------------------------------------------------------
# sampleRectangle caps at 262144 values. Requesting lon/lat alongside the
# data band (see _sample_ee_rectangle_array) triples the payload, so the
# per-band budget is a third of the raw cap.
EE_RASTER_SCALE = 500   # meters per pixel (min); increased for large regions
EE_RASTER_MAX_PIXELS = 262144 // 3

EE_RASTER_SCALE_BY_LAYER = {
    # SRTM is 30 m natively. Slope and aspect are meaningless if sampled at
    # 500 m -- fire responds to terrain at ridge-and-canyon scale.
    "elevation": 90,
    "slope": 90,
    "aspect": 90,
    "vegetation": 500,
    "landcover": 250,
    "population": 1000,
    "drought": 4000,
    "humidity": 4000,
    "weather_temp": 4000,
    "weather_precip": 4000,
    "wind_speed": 4000,
    "wind_direction": 4000,
    "wind_u": 4000,
    "wind_v": 4000,
    "erc": 4000,
    "vpd": 4000,
}

EE_STANDALONE_FEATURE_LAYERS = (
    "elevation",
    "slope",
    "aspect",
    "vegetation",
    "population",
    "landcover",
    "drought",
    "humidity",
    "vpd",
    "erc",
    "weather_temp",
    "weather_precip",
    "wind_speed",
    "wind_u",
    "wind_v",
)

# Layer config: (EE dataset, band(s), is_time_varying, title)
EE_RASTER_LAYER_CONFIG = {
    "elevation": ("USGS/SRTMGL1_003", ["elevation"], False, "Elevation (m)"),
    "slope": ("USGS/SRTMGL1_003", ["elevation"], False, "Slope (deg)"),
    "aspect": ("USGS/SRTMGL1_003", ["elevation"], False, "Aspect (deg)"),
    "vegetation": ("NASA/VIIRS/002/VNP13A1", ["NDVI"], True, "Vegetation (NDVI)"),
    "landcover": ("ESA/WorldCover/v100", ["Map"], False, "Land cover"),
    "drought": ("GRIDMET/DROUGHT", ["pdsi"], True, "Drought (PDSI)"),
    "humidity": ("IDAHO_EPSCOR/GRIDMET", ["sph"], True, "Specific humidity (g/kg)"),
    "weather_temp": ("IDAHO_EPSCOR/GRIDMET", ["tmmx"], True, "Max temperature (°C)"),
    "weather_precip": ("IDAHO_EPSCOR/GRIDMET", ["pr"], True, "Precipitation (mm)"),
    "wind_speed": ("IDAHO_EPSCOR/GRIDMET", ["vs"], True, "Wind speed (m/s)"),
    "wind_direction": ("IDAHO_EPSCOR/GRIDMET", ["th"], True, "Wind direction (deg)"),
    # Wind decomposed into components. Feed these to the model rather than
    # raw degrees: 359 and 1 are adjacent directions but maximally distant
    # numbers, and no tree split or linear weight can express that.
    "wind_u": ("IDAHO_EPSCOR/GRIDMET", ["vs", "th"], True, "Wind u (east, m/s)"),
    "wind_v": ("IDAHO_EPSCOR/GRIDMET", ["vs", "th"], True, "Wind v (north, m/s)"),
    # ERC is the best-established US fire danger index; VPD tracks fire
    # activity better than raw humidity.
    "erc": ("IDAHO_EPSCOR/GRIDMET", ["erc"], True, "Energy Release Component"),
    "vpd": ("IDAHO_EPSCOR/GRIDMET", ["tmmx", "tmmn", "sph"], True, "VPD (kPa)"),
    "population": ("CIESIN/GPWv411/GPW_Population_Density", ["population_density"], False, "Population density"),
}

# Scale factors applied to raw band values (band value * factor = physical).
EE_SCALE_FACTORS = {
    "vegetation": 1e-4,   # VNP13A1 NDVI is stored scaled by 1e-4
}

# How a time-varying layer is reduced across its window.
EE_TEMPORAL_REDUCER = {
    "weather_precip": "sum",
    "weather_temp": "mean",
    "humidity": "mean",
    "vpd": "mean",
    "erc": "mean",
    "wind_speed": "mean",
    "wind_u": "mean",
    "wind_v": "mean",
}


def _get_ee_scale_for_layer(layer_key: str) -> int:
    """Nominal sampling scale for each EE layer."""
    return EE_RASTER_SCALE_BY_LAYER.get(layer_key, EE_RASTER_SCALE)


# Global feature lag, set once from --feature-lag-days.
_FEATURE_LAG_DAYS = DEFAULT_FEATURE_LAG_DAYS


def set_feature_lag_days(days: int) -> None:
    """Set the feature/label separation used by every EE time window."""
    global _FEATURE_LAG_DAYS
    _FEATURE_LAG_DAYS = max(0, int(days))
    if _FEATURE_LAG_DAYS == 0:
        print(
            "  WARNING: --feature-lag-days 0 lets the fire day into the feature\n"
            "  window. Post-fire NDVI collapse and LST spike are then visible to\n"
            "  the model, which can identify fires from their own burn scar."
        )


def _get_time_window(
    date_str: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = 30,
    lag_days: int | None = None,
) -> tuple[str | None, str | None]:
    """
    EE date window ending `lag_days` before the label day.

    The window is [end - lookback, end - lag), i.e. end-exclusive. The
    previous version ended at end_date + 1 day, so the fire day itself was
    inside the feature window -- the model could read the fire out of its
    own burn signature and test accuracy looked excellent while real-time
    accuracy did not.
    """
    lag = _FEATURE_LAG_DAYS if lag_days is None else max(0, int(lag_days))
    anchor = end_date or date_str
    if anchor:
        end_dt = pd.to_datetime(anchor) - pd.Timedelta(days=lag - 1 if lag else -1)
        base = pd.to_datetime(start_date or anchor)
        start_dt = base - pd.Timedelta(days=lookback_days)
    else:
        end_dt = pd.Timestamp.now().normalize() + pd.Timedelta(days=1 - lag)
        start_dt = end_dt - pd.Timedelta(days=lookback_days)
    if end_dt <= start_dt:
        end_dt = start_dt + pd.Timedelta(days=1)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def _get_exact_day_window(
    date_str: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lag_days: int | None = None,
) -> tuple[str | None, str | None]:
    """
    Single-day EE window for the day `lag_days` before the label day.

    With the default lag of 1, features for a fire on 2025-01-08 come from
    2025-01-07: what a real-time system would actually have had available.
    """
    lag = _FEATURE_LAG_DAYS if lag_days is None else max(0, int(lag_days))
    day = end_date or start_date or date_str
    if not day:
        day = pd.Timestamp.now().strftime("%Y-%m-%d")
    day_dt = pd.to_datetime(day) - pd.Timedelta(days=lag)
    next_dt = day_dt + pd.Timedelta(days=1)
    return day_dt.strftime("%Y-%m-%d"), next_dt.strftime("%Y-%m-%d")


def _normalize_ee_time_args(
    layer_key: str,
    date_str: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    time_mode: str = "period",
) -> tuple[str | None, str | None, str | None, str]:
    """Normalize temporal args so caches can reuse equivalent EE requests."""
    if layer_key in STATIC_EE_LAYERS:
        return None, None, None, "static"
    if time_mode == "daily":
        day = end_date or start_date or date_str
        return day, day, day, "daily"
    return date_str, start_date, end_date, time_mode


def _reduce_collection(coll, layer_key: str):
    """Reduce an ImageCollection across its window using the layer's reducer."""
    how = EE_TEMPORAL_REDUCER.get(layer_key, "median")
    if how == "sum":
        return coll.sum()
    if how == "mean":
        return coll.mean()
    return coll.median()


def _gridmet_wind_components(coll, want: str):
    """
    Mean wind as u/v components.

    Wind direction is circular, so reducing `th` with median or mean is
    wrong: the median of 350, 355, 5, 10 degrees is 180 -- the exact
    opposite of the true mean. Decompose each image to vector components,
    average those, and return the requested component.

    GRIDMET `th` is the direction the wind blows *from* (meteorological
    convention), hence the negation.
    """
    def to_components(img):
        th = img.select("th").multiply(np.pi / 180.0)
        vs = img.select("vs")
        u = vs.multiply(th.sin()).multiply(-1).rename("wind_u")
        v = vs.multiply(th.cos()).multiply(-1).rename("wind_v")
        return u.addBands(v).copyProperties(img, ["system:time_start"])

    components = coll.map(to_components).mean()
    return components.select([want])


def _gridmet_vpd(coll):
    """
    Vapour pressure deficit in kPa from GRIDMET tmmx/tmmn/sph.

    Saturation vapour pressure via the Tetens equation at mean daily air
    temperature; actual vapour pressure from specific humidity at a nominal
    101.3 kPa surface pressure. VPD tracks fuel dryness and fire activity
    more directly than raw specific humidity.
    """
    mean_img = coll.mean()
    tmean_c = mean_img.select("tmmx").add(mean_img.select("tmmn")).divide(2).subtract(273.15)
    es = tmean_c.multiply(17.27).divide(tmean_c.add(237.3)).exp().multiply(0.6108)
    sph = mean_img.select("sph")
    ea = sph.multiply(101.3).divide(sph.multiply(0.378).add(0.622))
    return es.subtract(ea).max(0).rename("vpd")


def _get_ee_image_for_layer(
    layer_key: str,
    date_str: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    time_mode: str = "period",
):
    """Get the EE Image for a raster layer. Returns None if EE is unavailable."""
    if not HAS_EARTH_ENGINE or layer_key not in EE_RASTER_LAYER_CONFIG:
        return None
    date_str, start_date, end_date, time_mode = _normalize_ee_time_args(
        layer_key, date_str=date_str, start_date=start_date, end_date=end_date, time_mode=time_mode
    )
    cache_key = (layer_key, date_str, start_date, end_date, time_mode, _FEATURE_LAG_DAYS)
    if cache_key in _EE_IMAGE_CACHE:
        return _EE_IMAGE_CACHE[cache_key]
    dataset, bands, is_time_varying, _ = EE_RASTER_LAYER_CONFIG[layer_key]

    # -------------------------------------------------- static layers
    if layer_key in ("slope", "aspect"):
        # Terrain derivatives come free from the DEM already being loaded.
        # Slope drives spread rate; aspect drives fuel curing.
        dem = ee.Image(dataset).select(bands)
        img = ee.Terrain.slope(dem) if layer_key == "slope" else ee.Terrain.aspect(dem)
    elif layer_key == "population":
        # CIESIN GPW is an ImageCollection (2000-2020), not a single Image.
        img = ee.ImageCollection(dataset).select(bands).filterDate(
            "2020-01-01", "2020-12-31"
        ).first()
    elif layer_key == "landcover":
        img = ee.ImageCollection(dataset).select(bands).first()
    elif not is_time_varying:
        img = ee.Image(dataset).select(bands)

    # -------------------------------------------------- time-varying layers
    else:
        if time_mode == "daily":
            if layer_key == "vegetation":
                # VNP13A1 is a 16-day composite, so a single day is usually
                # empty; take a short window ending before the label day.
                anchor = end_date or start_date or date_str
                window_start, window_end = _get_time_window(date_str=anchor, lookback_days=8)
            else:
                window_start, window_end = _get_exact_day_window(
                    date_str=date_str, start_date=start_date, end_date=end_date
                )
        else:
            window_start, window_end = _get_time_window(
                date_str=date_str, start_date=start_date, end_date=end_date
            )

        coll = ee.ImageCollection(dataset).filterDate(window_start, window_end)
        if layer_key in ("wind_u", "wind_v"):
            img = _gridmet_wind_components(coll.select(bands), layer_key)
        elif layer_key == "vpd":
            img = _gridmet_vpd(coll.select(bands))
        else:
            img = _reduce_collection(coll.select(bands), layer_key)

    if img is None:
        return None

    # -------------------------------------------------- units
    # VNP13A1 NDVI is stored scaled by 1e-4. Skipping this factor put NDVI
    # in the thousands while EE_RASTER_VIS clamps the colour range to
    # [-1, 1], so every pixel saturated and the map read as uniformly green.
    factor = EE_SCALE_FACTORS.get(layer_key)
    if factor is not None:
        img = img.multiply(factor)
    if layer_key == "weather_temp":
        img = img.subtract(273.15)          # K -> C
    elif layer_key == "humidity":
        img = img.multiply(1000.0)          # kg/kg -> g/kg

    img = img.rename(layer_key)
    _cache_put(_EE_IMAGE_CACHE, cache_key, img)
    return img


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


def dataframe_to_raster_array(df: pd.DataFrame, value_col: str) -> np.ndarray:
    """Convert a lon/lat/value dataframe back into a raster array."""
    if df.empty or value_col not in df.columns:
        return np.empty((0, 0), dtype=float)

    keyed = df.copy()
    keyed["_lon_key"] = keyed["longitude"].round(8)
    keyed["_lat_key"] = keyed["latitude"].round(8)
    lon_values = sorted(keyed["_lon_key"].unique())
    lat_values = sorted(keyed["_lat_key"].unique(), reverse=True)
    grid = keyed.pivot_table(
        index="_lat_key",
        columns="_lon_key",
        values=value_col,
        aggfunc="first",
    )
    grid = grid.reindex(index=lat_values, columns=lon_values)
    return grid.to_numpy(dtype=float)


def _sample_ee_rectangle_array(
    img,
    bbox: tuple[float, float, float, float],
    layer_key: str,
    scale: int,
    with_bounds: bool = True,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | np.ndarray | None:
    """
    Sample one EE raster rectangle at a fixed scale.

    Returns the array *and its true bounds*. sampleRectangle snaps the
    region outward to the projection's pixel grid, so the returned array
    does not span the requested bbox exactly. Inferring pixel centres from
    the request (the old behaviour) misregistered every pixel by up to a
    full pixel -- 90 m for terrain, 4 km for GRIDMET -- which silently
    offsets features from labels and is invisible in the PNGs.

    Requesting ee.Image.pixelLonLat alongside the data band gives the exact
    coordinate of every returned cell.
    """
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n])
    stacked = img
    if with_bounds:
        stacked = img.addBands(ee.Image.pixelLonLat())
    img_for_sample = (
        stacked.reproject(crs="EPSG:4326", scale=scale).clip(region).unmask(-9999)
    )
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

    if not with_bounds:
        return arr

    bounds = bbox
    lon_raw = props.get("longitude")
    lat_raw = props.get("latitude")
    if lon_raw is not None and lat_raw is not None:
        lon = np.asarray(lon_raw, dtype=float)
        lat = np.asarray(lat_raw, dtype=float)
        lon[lon == -9999] = np.nan
        lat[lat == -9999] = np.nan
        if np.isfinite(lon).any() and np.isfinite(lat).any() and lon.shape == arr.shape:
            nrows, ncols = arr.shape
            # pixelLonLat gives centres; expand by half a cell for extent.
            half_x = (np.nanmax(lon) - np.nanmin(lon)) / max(2 * (ncols - 1), 1)
            half_y = (np.nanmax(lat) - np.nanmin(lat)) / max(2 * (nrows - 1), 1)
            bounds = (
                float(np.nanmin(lon) - half_x),
                float(np.nanmin(lat) - half_y),
                float(np.nanmax(lon) + half_x),
                float(np.nanmax(lat) + half_y),
            )
    return arr, bounds


def _sample_array_only(img, bbox, layer_key, scale) -> np.ndarray | None:
    """Array without bounds, for the tiling path where bounds are known."""
    out = _sample_ee_rectangle_array(img, bbox, layer_key, scale, with_bounds=False)
    return out if isinstance(out, np.ndarray) else None


def _split_bbox_into_tiles(
    bbox: tuple[float, float, float, float],
    x_tiles: int,
    y_tiles: int,
) -> list[tuple[float, float, float, float]]:
    """Split a bbox into an x by y grid of smaller bboxes."""
    w, s, e, n = bbox
    tiles = []
    for y_idx in range(y_tiles):
        tile_s = s + (n - s) * (y_idx / y_tiles)
        tile_n = s + (n - s) * ((y_idx + 1) / y_tiles)
        for x_idx in range(x_tiles):
            tile_w = w + (e - w) * (x_idx / x_tiles)
            tile_e = w + (e - w) * ((x_idx + 1) / x_tiles)
            tiles.append((tile_w, tile_s, tile_e, tile_n))
    return tiles


def extract_ee_raster(
    bbox: tuple[float, float, float, float],
    layer_key: str,
    date_str: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    time_mode: str = "period",
    scale: int = EE_RASTER_SCALE,
    ee_project: str | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    """
    Extract Earth Engine raster as numpy array for a bbox.
    Returns (array, (w,s,e,n)) or None on failure.
    """
    if not HAS_EARTH_ENGINE or not _init_earth_engine(ee_project):
        return None
    date_str, start_date, end_date, time_mode = _normalize_ee_time_args(
        layer_key, date_str=date_str, start_date=start_date, end_date=end_date, time_mode=time_mode
    )
    bbox_key = tuple(round(v, 6) for v in bbox)
    cache_key = (bbox_key, layer_key, date_str, start_date, end_date, time_mode, scale)
    if cache_key in _EE_RASTER_CACHE:
        cached_arr, cached_bbox = _EE_RASTER_CACHE[cache_key]
        return cached_arr.copy(), cached_bbox
    img = _get_ee_image_for_layer(
        layer_key,
        date_str,
        start_date=start_date,
        end_date=end_date,
        time_mode=time_mode,
    )
    if img is None:
        return None
    w, s, e, n = bbox
    # Ensure we stay under the sampleRectangle value limit
    width_m = abs(e - w) * 111000 * max(0.7, np.cos(np.radians((s + n) / 2)))
    height_m = abs(n - s) * 111000
    max_dim = int(np.sqrt(EE_RASTER_MAX_PIXELS))
    scale_x = width_m / max_dim
    scale_y = height_m / max_dim
    layer_scale = _get_ee_scale_for_layer(layer_key)
    max_pixels = max(1, int(EE_RASTER_MAX_PIXELS * 0.8))

    # Keep population at a consistent nominal scale by tiling large bboxes
    # rather than coarsening the raster to fit the request.
    if layer_key == "population":
        total_cols = max(1, int(np.ceil(width_m / layer_scale)))
        total_rows = max(1, int(np.ceil(height_m / layer_scale)))
        if total_cols * total_rows > max_pixels:
            tile_dim = max(1, int(np.sqrt(max_pixels)))
            x_tiles = max(1, int(np.ceil(total_cols / tile_dim)))
            y_tiles = max(1, int(np.ceil(total_rows / tile_dim)))
            tile_frames = []
            for tile_bbox in _split_bbox_into_tiles(bbox, x_tiles, y_tiles):
                try:
                    tile_arr = _sample_array_only(img, tile_bbox, layer_key, layer_scale)
                except Exception as ex:
                    print(f"  EE raster extract failed ({layer_key} tile): {ex}")
                    return None
                if tile_arr is None:
                    continue
                tile_frames.append(raster_array_to_dataframe(tile_arr, tile_bbox, layer_key))

            if not tile_frames:
                return None

            merged = pd.concat(tile_frames, ignore_index=True)
            merged["_lon_key"] = merged["longitude"].round(8)
            merged["_lat_key"] = merged["latitude"].round(8)
            merged = merged.drop_duplicates(subset=["_lon_key", "_lat_key"], keep="first")
            merged = merged.drop(columns=["_lon_key", "_lat_key"])
            arr = dataframe_to_raster_array(merged, layer_key)
            _cache_put(_EE_RASTER_CACHE, cache_key, (arr.copy(), bbox))
            return arr, bbox

        try:
            sampled = _sample_ee_rectangle_array(img, bbox, layer_key, layer_scale)
            if sampled is None:
                return None
            arr, bounds = sampled
            _cache_put(_EE_RASTER_CACHE, cache_key, (arr.copy(), bounds))
            return arr, bounds
        except Exception as ex:
            print(f"  EE raster extract failed ({layer_key}): {ex}")
            return None

    area_scale = np.sqrt((width_m * height_m) / max(1, int(EE_RASTER_MAX_PIXELS * 0.8)))
    actual_scale = max(scale, layer_scale, int(np.ceil(max(scale_x, scale_y, area_scale))))
    for _ in range(5):
        try:
            sampled = _sample_ee_rectangle_array(img, bbox, layer_key, actual_scale)
            if sampled is None:
                return None
            arr, bounds = sampled
            _cache_put(_EE_RASTER_CACHE, cache_key, (arr.copy(), bounds))
            return arr, bounds
        except Exception as ex:
            if "Too many pixels" in str(ex):
                actual_scale = int(np.ceil(actual_scale * 1.5))
                continue
            print(f"  EE raster extract failed ({layer_key}): {ex}")
            return None
    print(f"  EE raster extract failed ({layer_key}): exceeded pixel limit after rescaling")
    return None


def plot_raster_layer(
    arr: np.ndarray,
    bbox: tuple[float, float, float, float],
    title: str,
    path: Path,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    add_basemap: bool = False,
    auto_scale: bool = False,
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

    if auto_scale:
        vals = np.asarray(masked.compressed(), dtype=float)
        if vals.size:
            local_vmin = float(np.nanpercentile(vals, 2))
            local_vmax = float(np.nanpercentile(vals, 98))
            if np.isfinite(local_vmin) and np.isfinite(local_vmax) and local_vmax > local_vmin:
                vmin = local_vmin
                vmax = local_vmax

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        masked,
        extent=[w, e, s, n],
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
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


def save_raster(
    arr: np.ndarray,
    bounds: tuple[float, float, float, float],
    path: Path,
    layer_key: str,
    write_npy: bool = False,
) -> Path:
    """
    Persist a raster, preferring GeoTIFF.

    The previous .npy + bounds.txt sidecar pair was a private format: no CRS
    travelled with the data, QGIS could not open it, and every consumer had
    to know the row-order convention. GeoTIFF is readable by rioxarray in
    one call, which is also what makes resampling every layer onto a common
    grid tractable.
    """
    w, s, e, n = bounds
    if HAS_RASTERIO:
        tif_path = path.with_suffix(".tif")
        nrows, ncols = arr.shape
        transform = _rio_from_bounds(w, s, e, n, ncols, nrows)
        with rasterio.open(
            tif_path, "w", driver="GTiff", height=nrows, width=ncols, count=1,
            dtype="float32", crs="EPSG:4326", transform=transform,
            nodata=np.nan, compress="deflate",
        ) as dst:
            dst.write(arr.astype("float32"), 1)
            dst.set_band_description(1, layer_key)
        if not write_npy:
            return tif_path

    npy_path = path.with_suffix(".npy")
    np.save(npy_path, arr)
    with open(npy_path.with_name(npy_path.stem + "_bounds.txt"), "w") as f:
        f.write(f"{w},{s},{e},{n}\n")
    if not HAS_RASTERIO:
        print("  NOTE: install rasterio for GeoTIFF output (pip install rasterio)")
    return npy_path


def extract_ee_rasters_parallel(
    bbox: tuple[float, float, float, float],
    layers: list[str],
    max_workers: int = 8,
    **kwargs,
) -> dict[str, tuple[np.ndarray, tuple[float, float, float, float]]]:
    """
    Extract several EE layers concurrently.

    This pipeline is network-latency bound: essentially all wall time sits
    inside blocking getInfo() calls. The EE client is thread-safe and the
    backend parallelizes well, so a modest pool is the single largest
    speedup available short of moving to batch exports.
    """
    results: dict[str, tuple[np.ndarray, tuple[float, float, float, float]]] = {}
    if not layers:
        return results
    workers = max(1, min(max_workers, len(layers)))
    if workers == 1:
        for key in layers:
            out = extract_ee_raster(bbox, key, **kwargs)
            if out is not None:
                results[key] = out
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_ee_raster, bbox, key, **kwargs): key for key in layers}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                out = fut.result()
            except Exception as ex:
                print(f"  EE extract failed ({key}): {ex}")
                continue
            if out is not None:
                results[key] = out
    return results


# Colormaps and ranges for each layer
EE_RASTER_VIS = {
    "elevation": ("terrain", None, None),
    "vegetation": ("RdYlGn", -1.0, 1.0),
    "drought": ("BrBG", -4, 4),
    "humidity": ("GnBu", 0, 20),
    "weather_temp": ("YlOrRd", 0, 45),
    "weather_precip": ("Blues", 0, 50),
    "wind_speed": ("cividis", 0, 25),
    "wind_direction": ("twilight", 0, 360),
    "wind_u": ("coolwarm", -15, 15),
    "wind_v": ("coolwarm", -15, 15),
    "slope": ("magma", 0, 45),
    "aspect": ("twilight", 0, 360),
    "erc": ("inferno", 0, 100),
    "vpd": ("YlOrRd", 0, 6),
    "population": ("Purples", 0, 10000),
}

EE_DAILY_AUTO_SCALE_LAYERS = {
    "humidity",
    "wind_speed",
    "wind_direction",
    "wind_u",
    "wind_v",
    "weather_temp",
    "weather_precip",
    "vpd",
    "erc",
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
    time_mode: str = "period",
    add_basemap: bool = False,
    ee_project: str | None = None,
    max_workers: int = 8,
) -> None:
    """
    Extract and visualize EE raster layers for a bbox (same workflow as the
    fire mask). Saves a GeoTIFF (or .npy fallback) and a PNG per layer.
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

    results = extract_ee_rasters_parallel(
        bbox,
        valid_layers,
        max_workers=max_workers,
        date_str=date_str,
        start_date=start_date,
        end_date=end_date,
        time_mode=time_mode,
        ee_project=ee_project,
    )

    for layer_key in valid_layers:
        result = results.get(layer_key)
        if result is None:
            continue
        arr, bounds = result
        _, _, _, title = EE_RASTER_LAYER_CONFIG[layer_key]
        cmap, vmin, vmax = EE_RASTER_VIS.get(layer_key, ("viridis", None, None))

        raster_path = save_raster(
            arr, bounds, data_dir / f"{base_name}_{layer_key}", layer_key
        )
        print(f"  Extracted: {raster_path}")

        # Visualize (like fire mask PNG)
        png_path = visual_dir / f"{base_name}_{layer_key}.png"
        plot_raster_layer(
            arr,
            bounds,
            title,
            png_path,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_basemap=add_basemap,
            auto_scale=(time_mode == "daily" and layer_key in EE_DAILY_AUTO_SCALE_LAYERS),
        )


def plot_ee_raster_grid(
    layer_results: list[tuple[str, np.ndarray, tuple[float, float, float, float]]],
    path: Path,
    title: str,
    fire_df: pd.DataFrame | None = None,
    cluster_eps_km: float = 2.0,
    cluster_min_samples: int = 2,
    expected_layers: list[str] | None = None,
) -> None:
    """Save one figure containing all standalone EE raster feature maps."""
    layer_map = {layer_key: (arr, bounds) for layer_key, arr, bounds in layer_results}
    ordered_layers = expected_layers or [layer_key for layer_key, _, _ in layer_results]
    total_panels = len(ordered_layers) + (1 if fire_df is not None else 0)
    if total_panels == 0:
        return

    n_panels = total_panels
    ncols = 3 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    ax_idx = 0

    def _clean_grid_axis(ax) -> None:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_box_aspect(1)

    if fire_df is not None:
        ax = axes[ax_idx]
        ax_idx += 1
        first_bounds = next(iter(layer_map.values()))[1] if layer_map else None
        if first_bounds is None:
            w = fire_df["longitude"].min() if "longitude" in fire_df.columns and not fire_df.empty else 0
            e = fire_df["longitude"].max() if "longitude" in fire_df.columns and not fire_df.empty else 1
            s = fire_df["latitude"].min() if "latitude" in fire_df.columns and not fire_df.empty else 0
            n = fire_df["latitude"].max() if "latitude" in fire_df.columns and not fire_df.empty else 1
        else:
            w, s, e, n = first_bounds

        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_aspect("equal", adjustable="box")
        _clean_grid_axis(ax)

        can_plot_mask = (
            fire_df is not None
            and not fire_df.empty
            and all(c in fire_df.columns for c in FOOTPRINT_COLUMNS)
            and HAS_GEOPANDAS
        )
        if can_plot_mask:
            polygons, _ = cluster_fire_points_to_polygons(
                fire_df, eps_km=cluster_eps_km, min_samples=cluster_min_samples
            )
            if polygons:
                gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
                gdf.plot(ax=ax, facecolor="red", edgecolor="darkred", alpha=0.6, linewidth=0.5)
            else:
                ax.scatter(fire_df["longitude"], fire_df["latitude"], c="red", s=8, alpha=0.8)
        elif fire_df is not None and not fire_df.empty and "longitude" in fire_df.columns and "latitude" in fire_df.columns:
            ax.scatter(fire_df["longitude"], fire_df["latitude"], c="red", s=8, alpha=0.8)
        ax.set_title("Fire mask")

    for ax, layer_key in zip(axes[ax_idx:], ordered_layers):
        cmap, vis_title = EE_FEATURE_VIS.get(layer_key, ("viridis", layer_key))
        vis_meta = EE_RASTER_VIS.get(layer_key, (cmap, None, None))
        cmap = vis_meta[0]
        vmin = vis_meta[1]
        vmax = vis_meta[2]
        result = layer_map.get(layer_key)
        if result is None:
            if layer_map:
                bounds = next(iter(layer_map.values()))[1]
            elif fire_df is not None and not fire_df.empty and "longitude" in fire_df.columns and "latitude" in fire_df.columns:
                bounds = (
                    float(fire_df["longitude"].min()),
                    float(fire_df["latitude"].min()),
                    float(fire_df["longitude"].max()),
                    float(fire_df["latitude"].max()),
                )
            else:
                bounds = (0.0, 0.0, 1.0, 1.0)
            w, s, e, n = bounds
            ax.set_title(f"{vis_title} (no data)")
            ax.set_xlim(w, e)
            ax.set_ylim(s, n)
            _clean_grid_axis(ax)
            continue

        arr, bounds = result
        w, s, e, n = bounds
        masked = np.ma.masked_invalid(arr.astype(float))

        if masked.size == 0 or np.all(masked.mask):
            ax.set_title(f"{vis_title} (no data)")
            ax.set_xlim(w, e)
            ax.set_ylim(s, n)
            _clean_grid_axis(ax)
            continue

        ax.imshow(
            masked,
            extent=[w, e, s, n],
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_title(vis_title)
        ax.set_aspect("equal", adjustable="box")
        _clean_grid_axis(ax)

    for ax in axes[n_panels:]:
        ax.axis("off")

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
    time_mode: str = "period",
    add_basemap: bool = False,
    ee_project: str | None = None,
    layers: tuple[str, ...] = EE_STANDALONE_FEATURE_LAYERS,
    fire_df: pd.DataFrame | None = None,
    cluster_eps_km: float = 2.0,
    cluster_min_samples: int = 2,
    max_workers: int = 8,
    write_grid_csv: bool = False,
) -> None:
    """
    Export Earth Engine features as standalone raster datasets, not FIRMS-row enrichments.
    Saves standalone bbox rasters and related visualizations.
    """
    valid_layers = [l for l in layers if l in EE_RASTER_LAYER_CONFIG]
    if not valid_layers:
        return

    layer_results: list[tuple[str, np.ndarray, tuple[float, float, float, float]]] = []
    combined_rows: list[pd.DataFrame] = []
    print(f"Exporting standalone EE feature datasets: {', '.join(valid_layers)}")

    results = extract_ee_rasters_parallel(
        bbox,
        valid_layers,
        max_workers=max_workers,
        date_str=date_str,
        start_date=start_date,
        end_date=end_date,
        time_mode=time_mode,
        ee_project=ee_project,
    )

    for layer_key in valid_layers:
        result = results.get(layer_key)
        if result is None:
            continue
        arr, bounds = result
        layer_results.append((layer_key, arr, bounds))
        _, _, _, layer_title = EE_RASTER_LAYER_CONFIG[layer_key]
        cmap, vmin, vmax = EE_RASTER_VIS.get(layer_key, ("viridis", None, None))

        raster_path = save_raster(
            arr, bounds, data_dir / f"{base_name}_{layer_key}", layer_key
        )
        print(f"  Saved: {raster_path}")

        layer_df = raster_array_to_dataframe(arr, bounds, layer_key)
        if write_grid_csv:
            csv_path = data_dir / f"{base_name}_{layer_key}_grid.csv"
            layer_df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path} ({len(layer_df)} rows)")

        combined_rows.append(
            layer_df.assign(layer=layer_key).rename(columns={layer_key: "value"})
        )
        plot_raster_layer(
            arr,
            bounds,
            layer_title,
            visual_dir / f"{base_name}_{layer_key}.png",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_basemap=add_basemap,
            auto_scale=(time_mode == "daily" and layer_key in EE_DAILY_AUTO_SCALE_LAYERS),
        )

    if combined_rows and write_grid_csv:
        combined_path = data_dir / f"{base_name}_ee_features_standalone.csv"
        pd.concat(combined_rows, ignore_index=True).to_csv(combined_path, index=False)
        print(f"  Saved: {combined_path}")

    grid_path = visual_dir / f"{base_name}_ee_features_grid.png"
    plot_ee_raster_grid(
        layer_results,
        grid_path,
        title=f"{base_name} – Google Earth Engine features",
        fire_df=fire_df,
        cluster_eps_km=cluster_eps_km,
        cluster_min_samples=cluster_min_samples,
        expected_layers=valid_layers,
    )


def sample_background_points(
    fire_df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    n_samples: int,
    exclusion_km: float = 5.0,
    date_pool: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Draw no-fire (negative) samples matched to the fire samples in space and
    time.

    A classifier trained only on fire locations has no decision boundary to
    learn. This produces the complement: points inside the same bbox and
    drawn from the same set of dates as the positives, excluding anything
    within `exclusion_km` of a real detection.

    THIS IS A RESEARCH DESIGN DECISION, NOT A SOLVED PROBLEM. The defaults
    here are a starting point, not a recommendation:

    * Uniform sampling over a bbox biases toward whatever dominates it by
      area -- ocean and desert for California -- so the model can score well
      by learning geography rather than fire risk. Consider restricting
      draws to burnable land cover, or stratifying by ecoregion.
    * Hard negatives teach the boundary: cells adjacent to fires, or cells
      with high fire weather that did not ignite. Uniform draws are mostly
      trivially negative.
    * The positive:negative ratio you choose becomes a prior baked into the
      model. Record it and report PR-AUC, not accuracy or ROC-AUC -- fire
      cells are on the order of 0.01-0.1% of all cells.
    * Absence of a FIRMS detection is not evidence of no fire. Cloud, smoke
      and overpass gaps all cause misses, so some "negatives" are
      mislabelled positives.

    Returns a DataFrame with latitude, longitude, local_date and label=0.
    """
    if n_samples <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    w, s, e, n = bbox

    if date_pool is None:
        if "local_date" in fire_df.columns and not fire_df.empty:
            date_pool = sorted(fire_df["local_date"].dropna().unique().tolist())
        elif "acq_date" in fire_df.columns and not fire_df.empty:
            date_pool = sorted(fire_df["acq_date"].dropna().astype(str).unique().tolist())
    if not date_pool:
        date_pool = [pd.Timestamp.now().strftime("%Y-%m-%d")]

    have_fires = (
        not fire_df.empty
        and "latitude" in fire_df.columns
        and "longitude" in fire_df.columns
    )
    if have_fires:
        fire_pts = np.radians(fire_df[["latitude", "longitude"]].to_numpy(dtype=float))
        try:
            from sklearn.neighbors import BallTree
            tree = BallTree(fire_pts, metric="haversine")
        except ImportError:
            tree = None
    else:
        tree = None

    exclusion_rad = exclusion_km / EARTH_RADIUS_KM
    kept_lat: list[float] = []
    kept_lon: list[float] = []
    attempts = 0
    max_attempts = 40

    while len(kept_lat) < n_samples and attempts < max_attempts:
        attempts += 1
        draw = max(n_samples - len(kept_lat), 1) * 3
        lat = rng.uniform(s, n, draw)
        lon = rng.uniform(w, e, draw)
        if tree is not None:
            cand = np.radians(np.column_stack([lat, lon]))
            dist, _ = tree.query(cand, k=1)
            ok = dist[:, 0] > exclusion_rad
            lat, lon = lat[ok], lon[ok]
        elif have_fires:
            # Fallback without sklearn: coarse degree-box rejection.
            keep = np.ones(len(lat), dtype=bool)
            deg = exclusion_km / KM_PER_DEG
            for flat, flon in fire_df[["latitude", "longitude"]].to_numpy():
                keep &= ~((np.abs(lat - flat) < deg) & (np.abs(lon - flon) < deg))
            lat, lon = lat[keep], lon[keep]
        kept_lat.extend(lat.tolist())
        kept_lon.extend(lon.tolist())

    if not kept_lat:
        print("  WARNING: no background points survived the exclusion buffer.")
        return pd.DataFrame()

    k = min(n_samples, len(kept_lat))
    out = pd.DataFrame(
        {
            "latitude": kept_lat[:k],
            "longitude": kept_lon[:k],
            "local_date": rng.choice(date_pool, size=k),
            "label": 0,
        }
    )
    if k < n_samples:
        print(f"  NOTE: requested {n_samples} background points, kept {k}.")
    return out


def export_background_samples(
    fire_df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    data_dir: Path,
    base_name: str,
    n_samples: int,
    exclusion_km: float = 5.0,
    seed: int = 0,
) -> None:
    """Write matched negative samples next to the positive detections."""
    neg = sample_background_points(
        fire_df, bbox, n_samples, exclusion_km=exclusion_km, seed=seed
    )
    if neg.empty:
        return
    path = data_dir / f"{base_name}_background.csv"
    neg.to_csv(path, index=False)
    n_pos = len(fire_df)
    ratio = f"1:{len(neg) / n_pos:.1f}" if n_pos else "n/a"
    print(f"  Saved: {path} ({len(neg)} negatives, positive:negative {ratio})")
    print("         Review sample_background_points() docstring before training.")


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
    cluster_eps_km: float = 2.0,
    cluster_min_samples: int = 2,
    cluster_time_days: float | None = None,
    max_workers: int = 8,
    write_grid_csv: bool = False,
) -> None:
    """
    Export one day-by-day sample folder with fire and EE visualizations.
    Each day gets its own subdirectory so the fire signal and same-day features
    can be reviewed side by side across the requested time range.
    """
    df = add_acq_datetime(df)
    # Group by *local* date: GRIDMET and the drought products are indexed by
    # local day, so grouping on the UTC acq_date misfiles every night pass.
    if "local_date" in df.columns:
        df = df.copy()
        df["_acq_date_str"] = df["local_date"].astype(str)
    elif "acq_date" in df.columns:
        df = df.copy()
        df["_acq_date_str"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if start_date or end_date:
        start_str = start_date or end_date
        end_str = end_date or start_date or start_str
        day_list = [d.strftime("%Y-%m-%d") for d in pd.date_range(start=start_str, end=end_str, freq="D")]
    elif not df.empty and "_acq_date_str" in df.columns:
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
                cluster_eps_km=cluster_eps_km,
                cluster_min_samples=cluster_min_samples,
                cluster_time_days=cluster_time_days,
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
                time_mode="daily",
                add_basemap=add_basemap,
                ee_project=ee_project,
                fire_df=day_df,
                cluster_eps_km=cluster_eps_km,
                cluster_min_samples=cluster_min_samples,
                max_workers=max_workers,
                write_grid_csv=write_grid_csv,
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
                time_mode="daily",
                add_basemap=add_basemap,
                ee_project=ee_project,
                max_workers=max_workers,
            )


# ---------------------------------------------------------------------------
# Earth Engine feature enrichment
# ---------------------------------------------------------------------------
# EE project/bucket/folder come from UI (--ee-project, etc.) or interactive prompts.
# Uses your start_date, end_date, region/bbox from the same run.
EE_SAMPLE_SCALE = 1000  # meters; used for reduceRegions
EE_BATCH_SIZE = 1000   # max points per EE call to avoid timeouts


_EE_INIT_LOCK = threading.Lock()


def _init_earth_engine(project: str | None = None) -> bool:
    """Initialize Earth Engine; authenticate if needed. Returns True if ready.
    Project comes from --ee-project or interactive prompt (see run_interactive).

    Serialized behind a lock, and the interactive OAuth flow is only ever run
    on the main thread. Both matter: this is called from inside the extraction
    thread pool, so an unauthenticated machine used to start one browser
    OAuth flow per worker, all of them binding the same localhost callback
    port. They collide, every one of them fails with
    ``KeyError: 'code'`` / ``Missing required parameter: code``, and every
    layer in that batch is silently dropped. Call
    ``ensure_earth_engine_ready()`` once from the main thread before any
    parallel extraction so the credential exists by the time workers start.
    """
    if not HAS_EARTH_ENGINE:
        return False
    proj = (project or os.environ.get("EE_PROJECT", "")).strip()
    with _EE_INIT_LOCK:
        if proj in _EE_INIT_CACHE:
            return _EE_INIT_CACHE[proj]
        try:
            if proj:
                ee.Initialize(project=proj)
            else:
                ee.Initialize()
            _EE_INIT_CACHE[proj] = True
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
                _EE_INIT_CACHE[proj] = False
                return False
            if threading.current_thread() is not threading.main_thread():
                # A browser OAuth flow from a worker thread cannot succeed:
                # nothing is reading stdin for the verification code and the
                # callback port is contended. Fail with instructions instead.
                print("  Earth Engine is not authenticated. Run this once, "
                      "then re-run:\n      earthengine authenticate")
                _EE_INIT_CACHE[proj] = False
                return False
            try:
                ee.Authenticate()
                if proj:
                    ee.Initialize(project=proj)
                else:
                    ee.Initialize()
                _EE_INIT_CACHE[proj] = True
                return True
            except Exception as e:
                print(f"  Earth Engine auth failed: {e}")
                print("  Run 'earthengine authenticate' in a terminal, then re-run.")
                _EE_INIT_CACHE[proj] = False
                return False


def ensure_earth_engine_ready(project: str | None = None) -> bool:
    """Authenticate and initialize EE up front, on the main thread.

    Called from main() before any extraction so the one-time browser flow
    happens once, in a predictable place, rather than N times in parallel
    from inside a thread pool ten minutes into a run.
    """
    if not HAS_EARTH_ENGINE:
        print("  EE features requested but earthengine-api is not installed.")
        return False
    print("Checking Earth Engine credentials...")
    ok = _init_earth_engine(project)
    if ok:
        print("  Earth Engine ready.")
    else:
        print("  Earth Engine unavailable; EE layers will be skipped.")
    return ok


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
    args.cluster_eps_km = 2.0
    args.cluster_min_samples = 2
    args.cluster_time_days = None
    args.keep_singletons = True
    args.concave_ratio = 0.3

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
        args.ee_raster_layers = "elevation,vegetation,landcover,drought,humidity,weather_temp,weather_precip,wind_speed,wind_direction,population"
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
    args.feature_lag_days = DEFAULT_FEATURE_LAG_DAYS
    args.ee_workers = 8
    args.write_grid_csv = False
    args.all_sources = True
    args.background_samples = 0
    args.background_exclusion_km = 5.0
    args.seed = 0

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
    parser.add_argument("--cluster-eps-km", type=float, default=2.0,
                        help="Fire cluster radius in KM (haversine). Used with --fire-mask.")
    parser.add_argument("--cluster-min-samples", type=int, default=2,
                        help="Min detections per dense cluster. Used with --fire-mask.")
    parser.add_argument("--cluster-time-days", type=float, default=None,
                        help="Cluster in space-TIME over this window. Strongly recommended "
                             "for multi-week pulls: without it, fires months apart at the "
                             "same location merge into one cluster.")
    parser.add_argument("--drop-singletons", action="store_true",
                        help="Discard isolated detections as DBSCAN noise. Off by default: "
                             "an isolated detection is a new ignition, which is the most "
                             "valuable positive sample for a real-time model.")
    parser.add_argument("--convex-hull", action="store_true",
                        help="Force convex hulls instead of concave. Convex hulls over-estimate "
                             "area for elongated wind-driven fires.")
    parser.add_argument("--all-viz", action="store_true", help="All visualization types: points, fire mask, time-based")
    parser.add_argument("--basemap", action="store_true", help="Add basemap to time plot / fire mask (requires contextily)")
    parser.add_argument("--enrich-earth-engine", action="store_true", help="Export standalone EE feature datasets for the fire-time window: elevation, NDVI, population, land cover, drought, humidity, weather temperature/precipitation, and wind speed/direction. Requires earthengine authenticate.")
    parser.add_argument("--daily-visual-samples", action="store_true", help="Create per-day folders containing daily fire visualizations and same-day EE feature outputs across the selected date range.")
    parser.add_argument("--ee-raster-layers", type=str, default=None, metavar="Layers",
                        help="Extract and visualize EE raster layers. Comma-separated or 'all' for all layers")
    parser.add_argument("--ee-project", type=str, default=None, help="GCP/EE project ID (for EE init; uses your start_date, end_date, region)")
    parser.add_argument("--ee-bucket", type=str, default=None, help="GCS bucket for export (optional)")
    parser.add_argument("--ee-folder", type=str, default="exports", help="Folder in bucket (default: exports)")
    parser.add_argument("--ee-workers", type=int, default=8,
                        help="Concurrent Earth Engine requests (default 8). This pipeline is "
                             "network-latency bound; raise for more speed until EE throttles.")
    parser.add_argument("--feature-lag-days", type=int, default=DEFAULT_FEATURE_LAG_DAYS,
                        help="Days between the end of the feature window and the label day "
                             "(default 1). Setting 0 admits the fire day into the features and "
                             "lets the model read the fire from its own burn scar.")
    parser.add_argument("--write-grid-csv", action="store_true",
                        help="Also write flattened lon/lat/value CSVs per raster (large and slow; "
                             "the GeoTIFF carries the same data).")
    parser.add_argument("--first-source-only", action="store_true",
                        help="Stop at the first FIRMS source returning data. Faster, but makes "
                             "detection density depend on satellite availability.")
    parser.add_argument("--background-samples", type=int, default=0,
                        help="Draw N matched no-fire samples per area. Required for supervised "
                             "training; read sample_background_points() before relying on it.")
    parser.add_argument("--background-exclusion-km", type=float, default=5.0,
                        help="Exclusion radius around real detections for background samples.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for background sampling")

    # Run transcript. Read off argv before parsing (see resolve_report_target)
    # so that usage errors and interactive prompts are captured too; declared
    # here so --help documents them and they are not rejected as unknown.
    parser.add_argument("--report-file", type=str, default=None, metavar="PATH",
                        help=f"Where to write the run transcript (default: {DEFAULT_REPORT_NAME} "
                             "next to this script). A directory is accepted. Also settable "
                             "via FIRMS_REPORT_FILE.")
    parser.add_argument("--report-append", action="store_true",
                        help="Append to the transcript instead of overwriting it, so several "
                             "runs accumulate in one file.")
    parser.add_argument("--no-report", action="store_true",
                        help="Do not write a run transcript.")
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
        args.keep_singletons = not args.drop_singletons
        args.concave_ratio = None if args.convex_hull else 0.3
        args.all_sources = not args.first_source_only

    # Applies to every EE time window; set once, before any extraction.
    set_feature_lag_days(getattr(args, "feature_lag_days", DEFAULT_FEATURE_LAG_DAYS))

    # Authenticate Earth Engine before anything expensive happens. The browser
    # flow needs the main thread and a free localhost callback port; running it
    # lazily from inside the extraction pool starts one flow per worker, and
    # they knock each other over. Doing it here also means a credential problem
    # surfaces in the first few seconds rather than after the FIRMS fetch.
    wants_ee = bool(
        getattr(args, "enrich_earth_engine", False)
        or getattr(args, "ee_raster_layers", None)
    )
    if wants_ee:
        if not ensure_earth_engine_ready(getattr(args, "ee_project", None)):
            print("  Continuing without Earth Engine: FIRMS outputs only.\n")

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
        all_sources=getattr(args, "all_sources", True),
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
        # Bound unconditionally: these were previously assigned only inside
        # `if args.enrich_earth_engine and HAS_EARTH_ENGINE`, then read by the
        # daily-samples block -- an UnboundLocalError whenever EE was requested
        # but earthengine-api was not installed, after all fetching was done.
        ee_start_date, ee_end_date, date_str = _get_fire_date_window(df)
        if not df.empty:
            df = add_acq_datetime(df)
            df = add_footprint_columns(df)
            save_outputs(df, dataset_dir, base_name, save_csv=True, save_shp=args.save_shapefile, save_kml=args.save_kml, save_hdf5=args.save_hdf5)
            if getattr(args, "background_samples", 0):
                export_background_samples(
                    df, bbox, dataset_dir, base_name,
                    n_samples=args.background_samples,
                    exclusion_km=args.background_exclusion_km,
                    seed=args.seed,
                )
            color_col = "frp" if "frp" in df.columns else "bright_ti4"
            if args.fire_mask:
                plot_fire_mask(df, "FIRMS – Fire detection mask", bbox, visual_dir / f"{base_name}_mask.png",
                              add_basemap=args.basemap, cluster_eps_km=args.cluster_eps_km,
                              cluster_min_samples=args.cluster_min_samples,
                              cluster_time_days=args.cluster_time_days,
                              keep_singletons=args.keep_singletons,
                              concave_ratio=args.concave_ratio, centroids_dir=dataset_dir)
            if args.time_plot:
                plot_fire_map_time_based(df, "FIRMS – Time-based", bbox, visual_dir / f"{base_name}_time.png", add_basemap=args.basemap)
            if args.plot_points:
                plot_fire_map(df, "FIRMS – Fire detections", bbox, visual_dir / f"{base_name}.png", color_col)
            if args.enrich_earth_engine and HAS_EARTH_ENGINE:
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
                    cluster_eps_km=args.cluster_eps_km,
                    cluster_min_samples=args.cluster_min_samples,
                    max_workers=args.ee_workers,
                    write_grid_csv=args.write_grid_csv,
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
                    cluster_eps_km=args.cluster_eps_km,
                    cluster_min_samples=args.cluster_min_samples,
                    max_workers=args.ee_workers,
                    write_grid_csv=args.write_grid_csv,
                )
        # EE raster layers (same bbox as fire mask; works with or without FIRMS data)
        if ee_raster_layers:
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
                max_workers=args.ee_workers,
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
            ee_start_date, ee_end_date, date_str = _get_fire_date_window(area_df)
            if not area_df.empty:
                save_outputs(area_df, dataset_dir, base_name, save_csv=True, save_shp=args.save_shapefile, save_kml=args.save_kml, save_hdf5=args.save_hdf5)
                if getattr(args, "background_samples", 0):
                    export_background_samples(
                        area_df, bbox, dataset_dir, base_name,
                        n_samples=args.background_samples,
                        exclusion_km=args.background_exclusion_km,
                        seed=args.seed,
                    )
                if args.fire_mask:
                    plot_fire_mask(area_df, f"FIRMS – {area.title()} fire mask", bbox, visual_dir / f"{base_name}_mask.png",
                                  add_basemap=args.basemap, cluster_eps_km=args.cluster_eps_km,
                                  cluster_min_samples=args.cluster_min_samples,
                                  cluster_time_days=args.cluster_time_days,
                                  keep_singletons=args.keep_singletons,
                                  concave_ratio=args.concave_ratio, centroids_dir=dataset_dir)
                if args.time_plot:
                    plot_fire_map_time_based(area_df, f"FIRMS – {area.title()} time-based", bbox, visual_dir / f"{base_name}_time.png", add_basemap=args.basemap)
                if args.plot_points:
                    plot_fire_map(area_df, f"FIRMS – {area.title()}", bbox, visual_dir / f"{base_name}.png", color_col)
            if args.enrich_earth_engine and HAS_EARTH_ENGINE:
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
                    cluster_eps_km=args.cluster_eps_km,
                    cluster_min_samples=args.cluster_min_samples,
                    max_workers=args.ee_workers,
                    write_grid_csv=args.write_grid_csv,
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
                    cluster_eps_km=args.cluster_eps_km,
                    cluster_min_samples=args.cluster_min_samples,
                    max_workers=args.ee_workers,
                    write_grid_csv=args.write_grid_csv,
                )
            if area_df.empty and not args.enrich_earth_engine and not ee_raster_layers:
                print(f"  No data for {area}, skipping.")
                continue
            # EE raster layers (same bbox as fire mask per area)
            if ee_raster_layers:
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

    print("\nDone. Outputs are in:", out_dir.resolve())
    print(f"Feature/label separation: {_FEATURE_LAG_DAYS} day(s) "
          "(features end before the label day; see --feature-lag-days).")
    if not getattr(args, "background_samples", 0):
        print("NOTE: only fire (positive) samples were exported. Supervised training "
              "needs matched negatives -- see --background-samples.")
    if _ACTIVE_REPORT_PATH is not None:
        print(f"Full run transcript: {_ACTIVE_REPORT_PATH.resolve()}")


if __name__ == "__main__":
    _report_path, _report_append = resolve_report_target(sys.argv[1:])
    with RunTranscript(_report_path, append=_report_append):
        try:
            main()
        except FirmsRequestError as err:
            raise SystemExit(f"FIRMS request failed: {err}")
        except KeyboardInterrupt:
            raise SystemExit("\nInterrupted.")
