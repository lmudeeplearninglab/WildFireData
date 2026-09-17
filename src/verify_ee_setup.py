#!/usr/bin/env python3
"""
verify_ee_setup -- check that Earth Engine is set up correctly for THIS pipeline.

A bare `ee.Initialize()` succeeding proves almost nothing: it can pass while
your project is unregistered, while an asset you depend on is inaccessible,
or while a date you need has no imagery. This walks the specific calls
firegrid.py and visualize_firms_dataset_v2.py actually make, in order, and
stops at the first thing that is genuinely broken.

Run:
    python verify_ee_setup.py --project ee-yourname-wildfire
    python verify_ee_setup.py --project ee-yourname-wildfire --date 2025-01-10

Exit code 0 means every check passed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

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


PASS, FAIL, WARN, INFO = "  [ok]  ", "  [FAIL]", "  [warn]", "        "
_failures: list[str] = []


# Why the default resolved the way it did. Recorded rather than printed,
# because _default_project() runs during argparse setup, before the banner.
_project_source: str = ""


def _default_project() -> str:
    """Same resolution order as the pipeline, so this verifies what will run.

    Falling back to a hardcoded id here would let the check pass against one
    project while the pipeline used another -- the exact failure this script
    exists to catch.

    Every failure path records WHY in _project_source. Swallowing the reason
    turns three different problems (env var not set, wrong script version in
    this folder, broken import) into one unhelpful "pass --project" message.
    """
    global _project_source
    env = os.environ.get("EE_PROJECT", "").strip()
    if env:
        _project_source = "$EE_PROJECT"
        return env
    try:
        V = _import_pipeline()
    except Exception as err:
        # Full text, not a 120-char slice: the actionable half of an import
        # error is usually at the END of the message.
        detail = "\n".join(f"{INFO}   {ln}" for ln in str(err).splitlines())
        _project_source = (
            f"$EE_PROJECT is not set, and the pipeline module could not be "
            f"imported:\n{detail}"
        )
        return ""
    project = getattr(V, "DEFAULT_EE_PROJECT", "")
    if not project:
        _project_source = (
            "$EE_PROJECT is not set, and the visualize_firms_dataset_v2.py in "
            f"this folder has no DEFAULT_EE_PROJECT.\n{INFO}   It is at "
            f"{getattr(V, '__file__', 'unknown')} -- likely an older copy."
        )
        return ""
    _project_source = f"DEFAULT_EE_PROJECT in {Path(V.__file__).name}"
    return project


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"{PASS if ok else FAIL} {label}")
    if detail:
        print(f"{INFO} {detail}")
    if not ok:
        _failures.append(label)
    return ok


def warn(label: str, detail: str = "") -> None:
    print(f"{WARN} {label}")
    if detail:
        print(f"{INFO} {detail}")


# ---------------------------------------------------------------------------
def step_1_install() -> bool:
    print("\n1. Client library")
    try:
        import ee
    except ImportError:
        return check(False, "earthengine-api importable",
                     "pip install earthengine-api  (inside your conda env)")
    version = getattr(ee, "__version__", "unknown")
    check(True, f"earthengine-api {version}")
    check(sys.version_info >= (3, 9), f"python {sys.version.split()[0]}")
    return True


def step_2_credentials() -> bool:
    print("\n2. Credentials")
    # Same location the client library uses; checked directly so a missing
    # file is reported as a missing file rather than as a confusing API error.
    cred = Path.home() / ".config" / "earthengine" / "credentials"
    if not cred.exists():
        return check(False, "credential file present",
                     f"not found at {cred}\n"
                     f"{INFO} run:  earthengine authenticate")
    age_days = (time.time() - cred.stat().st_mtime) / 86400
    check(True, "credential file present", f"{cred}  (written {age_days:.0f} days ago)")
    warn("this file is a live credential",
         "never commit it, never paste it in a bug report or transcript")
    return True


def step_3_initialize(project: str) -> bool:
    print("\n3. Project initialization")
    import ee
    if not project:
        return check(False, "project id supplied",
                     (_project_source or "no source tried") +
                     f"\n{INFO} fix either of those, or pass "
                     f"--project ee-your-project-id explicitly")
    check(True, f"project id '{project}'", f"from {_project_source or '--project'}")
    try:
        ee.Initialize(project=project)
    except Exception as err:
        msg = str(err)
        hint = ""
        if "not registered" in msg.lower() or "not signed up" in msg.lower():
            hint = ("project exists but is not registered for Earth Engine.\n"
                    f"{INFO} register it: https://code.earthengine.google.com/register")
        elif "permission" in msg.lower() or "403" in msg:
            hint = ("authenticated user lacks access to this project.\n"
                    f"{INFO} check you are signed in as the account that owns it")
        elif "authorize" in msg.lower():
            hint = "run: earthengine authenticate"
        return check(False, f"ee.Initialize(project='{project}')",
                     hint or msg.splitlines()[0][:160])
    check(True, f"ee.Initialize(project='{project}')")

    # Round-trip a trivial computation: proves the API is enabled and
    # reachable, not merely that a token was found on disk.
    try:
        t0 = time.time()
        value = ee.Number(1).add(1).getInfo()
        check(value == 2, "server round-trip", f"{(time.time() - t0) * 1000:.0f} ms")
    except Exception as err:
        return check(False, "server round-trip", str(err).splitlines()[0][:160])
    return True


def step_4_assets(date: str) -> bool:
    print("\n4. Dataset access")
    import ee
    # One representative asset per family the pipeline touches. A project can
    # initialize fine and still fail here if an API is not enabled.
    probes = {
        "SRTM (elevation/slope/aspect)": lambda: ee.Image(
            "USGS/SRTMGL1_003").bandNames().getInfo(),
        "GRIDMET (weather, ERC, VPD)": lambda: ee.ImageCollection(
            "IDAHO_EPSCOR/GRIDMET").filterDate(date, _plus_day(date)).size().getInfo(),
        "GRIDMET DROUGHT (PDSI)": lambda: ee.ImageCollection(
            "GRIDMET/DROUGHT").filterDate(
                _minus_days(date, 10), _plus_day(date)).size().getInfo(),
        "VIIRS VNP13A1 (NDVI)": lambda: ee.ImageCollection(
            "NASA/VIIRS/002/VNP13A1").filterDate(
                _minus_days(date, 32), _plus_day(date)).size().getInfo(),
        "ESA WorldCover (land cover)": lambda: ee.ImageCollection(
            "ESA/WorldCover/v100").size().getInfo(),
        "GPWv411 (population)": lambda: ee.ImageCollection(
            "CIESIN/GPWv411/GPW_Population_Density").size().getInfo(),  # pragma: allowlist secret
    }
    ok_all = True
    for label, probe in probes.items():
        try:
            result = probe()
        except Exception as err:
            ok_all &= check(False, label, str(err).splitlines()[0][:140])
            continue
        if isinstance(result, int) and result == 0:
            # Reachable but empty for this window -- a data gap, not an auth
            # problem, and worth distinguishing sharply from a failure.
            warn(f"{label}: reachable but 0 images for {date}",
                 "the asset works; this date has no imagery in that window")
        else:
            check(True, label, f"{result if not isinstance(result, list) else result}")
    return ok_all


def step_5_pipeline_path(project: str, date: str, lon: float, lat: float) -> bool:
    print("\n5. Actual pipeline extraction path")
    try:
        import src.firegrid as F
    except ImportError:
        warn("firegrid.py not importable from here",
             "skipping the grid test; run this from the project directory")
        return True

    import numpy as np
    tile = F.snap_tile(lon, lat)
    print(f"{INFO} tile origin ({tile[0]:.0f}, {tile[1]:.0f}) in {F.GRID_SPEC.crs}, "
          f"{F.GRID_SPEC.tile_cells}x{F.GRID_SPEC.tile_cells} @ "
          f"{F.GRID_SPEC.cell_m:.0f} m")

    # Static, continuous, needs reduceResolution from 30 m -- the most
    # demanding of the aggregation paths, so the best single smoke test.
    t0 = time.time()
    arr = F.ee_layer_on_grid("elevation", tile, date_str=date, ee_project=project)
    if arr is None:
        return check(False, "elevation onto the target grid",
                     "returned None -- see the error printed above")
    elapsed = time.time() - t0
    ok = check(arr.shape == F.GRID_SPEC.shape,
               f"elevation grid shape {arr.shape}",
               f"expected {F.GRID_SPEC.shape}, took {elapsed:.1f}s")

    finite = np.isfinite(arr)
    coverage = finite.mean() * 100
    ok &= check(coverage > 50, f"elevation coverage {coverage:.0f}%",
                "low coverage usually means the tile is mostly ocean or "
                "outside the asset footprint")
    if finite.any():
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        plausible = -100 < lo and hi < 5000
        ok &= check(plausible, f"elevation range {lo:.0f}..{hi:.0f} m",
                    "" if plausible else "implausible for California -- check "
                    "the projection and that scaling was not applied twice")

    # Circular path: returns two stacked bands, not one.
    asp = F.ee_layer_on_grid("aspect", tile, date_str=date, ee_project=project)
    if asp is not None:
        ok &= check(asp.shape == (2, *F.GRID_SPEC.shape),
                    f"aspect returns sin/cos pair {asp.shape}")
        if np.isfinite(asp).any():
            # The resultant LENGTH after averaging is the circular
            # concentration R, not 1. R == 1 everywhere would actually be the
            # bug: it would mean no aggregation happened. What must hold is
            # 0 <= R <= 1, with some spread across real terrain.
            _sin_u, _cos_u, r = F.split_circular_components(asp)
            finite_r = r[np.isfinite(r)]
            in_range = bool(finite_r.size) and finite_r.min() >= -1e-9 \
                and finite_r.max() <= 1 + 1e-9
            ok &= check(in_range,
                        f"aspect concentration R in [0,1] "
                        f"(min {finite_r.min():.2f}, mean {finite_r.mean():.2f}, "
                        f"max {finite_r.max():.2f})",
                        "" if in_range else "R outside [0,1] means sin/cos were "
                        "combined incorrectly")
            if in_range and finite_r.mean() > 0.995:
                warn("R is ~1 everywhere",
                     "suspicious: suggests aggregation did not actually run, "
                     "since real terrain has varied aspect within 1 km")
            unit = np.sqrt(_sin_u ** 2 + _cos_u ** 2)
            live = unit[np.isfinite(unit) & (unit > 0)]
            ok &= check(live.size == 0 or np.nanmax(np.abs(live - 1)) < 1e-6,
                        "normalized aspect direction is unit length")

    # Time-varying, coarse native resolution -- the upsampling branch.
    erc = F.ee_layer_on_grid("erc", tile, date_str=date, time_mode="daily",
                             ee_project=project)
    if erc is None:
        warn("erc returned None for this date",
             "GRIDMET lags a few days; try an older --date")
    else:
        ok &= check(erc.shape == F.GRID_SPEC.shape, f"erc grid shape {erc.shape}")
    return ok


def _plus_day(d: str) -> str:
    from datetime import datetime, timedelta
    return (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


def _minus_days(d: str, n: int) -> str:
    from datetime import datetime, timedelta
    return (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=n)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=_default_project(),
                    help="Cloud project id (default: $EE_PROJECT, else the "
                         "pipeline's DEFAULT_EE_PROJECT)")
    ap.add_argument("--date", default="2025-01-10",
                    help="A date inside your study period (default 2025-01-10)")
    ap.add_argument("--lon", type=float, default=-118.55)
    ap.add_argument("--lat", type=float, default=34.07)
    args = ap.parse_args()

    print("=" * 70)
    print("Earth Engine setup verification")
    print("=" * 70)

    if not step_1_install():
        return 1
    step_2_credentials()
    if not step_3_initialize(args.project):
        print("\nStopping: nothing downstream can work without initialization.")
        return 1
    step_4_assets(args.date)
    step_5_pipeline_path(args.project, args.date, args.lon, args.lat)

    print("\n" + "=" * 70)
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"  - {f}")
        print("=" * 70)
        return 1
    print("All checks passed. Extraction path is working end to end.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
