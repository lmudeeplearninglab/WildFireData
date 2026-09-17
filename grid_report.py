#!/usr/bin/env python3
"""
Grid audit for the FIRMS/EE pipeline.

Answers three questions empirically, without needing EE credentials or a
FIRMS key, by replicating the scale-selection arithmetic in
extract_ee_raster():

  1. How big is each area, in km and km^2?
  2. What array shape does each EE layer actually come back as?
  3. Are those shapes consistent with each other (spatially normalized)?

Run:  python grid_report.py
      python grid_report.py --area palisades --target-scale 1000 --tile-km 64
"""
from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

def _import_pipeline():
    """Import the main pipeline module regardless of its version suffix.

    The file gets renamed as it is revised (v2 -> v3 -> ...), and hardcoding
    one name here means every rename silently breaks this script.
    """
    import importlib
    import re
    here = Path(__file__).resolve().parent
    found = sorted(
        (p.stem for p in here.glob("visualize_firms_dataset_v*.py")),
        key=lambda n: int(m.group(1)) if (m := re.search(r"_v(\d+)$", n)) else 0,
        reverse=True,
    )
    for name in found + ["visualize_firms_dataset"]:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as err:
            if err.name != name:
                raise
    raise ImportError(f"no visualize_firms_dataset*.py found in {here}")


V = _import_pipeline()

AREAS = {
    "california": V.CALIFORNIA_BBOX,
    "palisades": V.PALISADES_BBOX,
    "eaton": V.EATON_BBOX,
}


def extent_m(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Width and height in metres, using the same approximation as the pipeline."""
    w, s, e, n = bbox
    width_m = abs(e - w) * 111000 * max(0.7, np.cos(np.radians((s + n) / 2)))
    height_m = abs(n - s) * 111000
    return width_m, height_m


def effective_grid(bbox, layer: str) -> tuple[int, int, tuple[int, int]]:
    """Reproduce extract_ee_raster()'s scale escalation.

    Returns (nominal_scale, actual_scale, (rows, cols)). The gap between
    nominal and actual is the silent coarsening imposed by the
    sampleRectangle value cap.
    """
    width_m, height_m = extent_m(bbox)
    max_dim = int(np.sqrt(V.EE_RASTER_MAX_PIXELS))
    scale_x, scale_y = width_m / max_dim, height_m / max_dim
    layer_scale = V._get_ee_scale_for_layer(layer)
    area_scale = np.sqrt(
        (width_m * height_m) / max(1, int(V.EE_RASTER_MAX_PIXELS * 0.8))
    )
    actual = max(
        V.EE_RASTER_SCALE, layer_scale,
        int(np.ceil(max(scale_x, scale_y, area_scale))),
    )
    return layer_scale, actual, (
        int(round(height_m / actual)), int(round(width_m / actual))
    )


def report(area: str, bbox, target_scale: int, tile_km: int) -> None:
    width_m, height_m = extent_m(bbox)
    kx, ky = width_m / 1000, height_m / 1000
    print(f"\n{'=' * 78}\n{area.upper()}  {bbox}\n{'=' * 78}")
    print(f"  extent      {kx:,.1f} km x {ky:,.1f} km = {kx * ky:,.0f} km^2")
    print(f"  at {target_scale} m    a common grid would be "
          f"{int(round(height_m / target_scale))} x {int(round(width_m / target_scale))} cells")
    print(f"  as tiles    {int(np.ceil(kx / tile_km))} x {int(np.ceil(ky / tile_km))} "
          f"= {int(np.ceil(kx / tile_km)) * int(np.ceil(ky / tile_km))} "
          f"tiles of {tile_km} km")

    print(f"\n  {'layer':<14} {'nominal':>9} {'actual':>9} {'shape':>14} {'cells':>9}  note")
    print(f"  {'-' * 72}")
    shapes: dict[tuple[int, int], list[str]] = {}
    for layer in V.EE_RASTER_LAYER_CONFIG:
        nominal, actual, shape = effective_grid(bbox, layer)
        shapes.setdefault(shape, []).append(layer)
        factor = actual / nominal
        note = ""
        if factor >= 2:
            note = f"COARSENED {factor:.0f}x"
        if shape[0] * shape[1] < 100:
            note = (note + "  " if note else "") + "near-degenerate"
        print(f"  {layer:<14} {nominal:>7d} m {actual:>7d} m {str(shape):>14} "
              f"{shape[0] * shape[1]:>9,}  {note}")

    print(f"\n  distinct grids: {len(shapes)}")
    if len(shapes) > 1:
        for shape, layers in sorted(shapes.items(), key=lambda kv: -kv[0][0] * kv[0][1]):
            print(f"    {str(shape):>14}  {', '.join(layers)}")
        print("  -> layers are NOT co-registered; they cannot be stacked as "
              "channels without resampling to a common grid.")
    else:
        print("  -> all layers share one grid.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", choices=[*AREAS, "all"], default="all")
    ap.add_argument("--target-scale", type=int, default=1000,
                    help="Candidate common-grid resolution in metres (default 1000)")
    ap.add_argument("--tile-km", type=int, default=64,
                    help="Candidate tile edge in km (default 64)")
    args = ap.parse_args()

    print(f"sampleRectangle budget: {V.EE_RASTER_MAX_PIXELS:,} values per band "
          f"(max square {int(np.sqrt(V.EE_RASTER_MAX_PIXELS))} x "
          f"{int(np.sqrt(V.EE_RASTER_MAX_PIXELS))})")
    print("Nominal = the scale configured for the layer. Actual = what the "
          "pixel cap forces.")

    targets = AREAS if args.area == "all" else {args.area: AREAS[args.area]}
    for name, bbox in targets.items():
        report(name, bbox, args.target_scale, args.tile_km)

    print(f"\n{'=' * 78}")
    print("Fire mask: exported as GeoJSON polygons (DBSCAN clusters), not a "
          "raster.\nThere is no fire-mask array whose dimensions can be "
          "compared to the above.")
    print("=" * 78)


if __name__ == "__main__":
    main()
