#!/usr/bin/env python3
"""
fireattrib -- which fire is each detection?

A 64 km tile holds more than the fire you asked for. Around Palisades in
January 2025 it also held Hurst and Kenneth, and a few small unnamed fires.
They are real fire, so they stay fire in the label, but until now nothing said
WHICH fire a cell belonged to. This module answers that, per detection, so
that a build can

  - draw every fire in its own colour, with its name, and
  - save a fire-ID map beside each label, so training can decide per
    experiment whether other fires count (see swin_data.ignore_other_fires).

HOW A DETECTION GETS A FIRE
---------------------------
1. MAPPED: inside a fire's agency perimeter, grown by `buffer_m` (2 km covers
   the footprint, geolocation error and spotting past the final line), and
   detected between alarm - 1 day and containment + tail. The target fire is
   one of these; the others come from firelookup.perimeters_in_area.
   Inside two buffers -> the nearer perimeter wins; ties go to the target.
2. CONNECTED: detections within `eps_km` of each other form chains (DBSCAN,
   single linkage). An unassigned detection chained to mapped ones joins the
   fire most of its chain belongs to -- a spot fire 3 km past the line is
   still that fire.
3. UNMAPPED: whatever is left, one fire per chain, named "Unmapped #k". Small
   fires under the agency mapping threshold, or fires not yet in FRAP.

IDs: 1 is always the target. Mapped fires follow, by detection count, then
unmapped ones. IDs are per build: fire 2 in one fire's folder is not fire 2
in another's. fires.geojson in each fire folder is the key.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_ID = 1
# Fire 1 is always red; the rest cycle through colours that stay distinct
# from it and from the grey used for unobserved cells.
TARGET_COLOR = "#d62728"
OTHER_COLORS = ("#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
                "#8c564b", "#e377c2", "#bcbd22", "#393b79", "#637939")


def fire_color(fire_id: int) -> str:
    if fire_id == TARGET_ID:
        return TARGET_COLOR
    return OTHER_COLORS[(fire_id - 2) % len(OTHER_COLORS)]


@dataclass
class FireTag:
    fire_id: int
    name: str
    kind: str                         # target | mapped | unmapped
    source: str = ""
    alarm_date: str | None = None
    contain_date: str | None = None
    acres: float | None = None
    geometry: dict | None = field(default=None, repr=False)
    n_detections: int = 0
    first_seen: str | None = None
    last_seen: str | None = None
    centroid: tuple | None = None     # lon, lat of its detections

    def label(self) -> str:
        if self.kind == "target":
            return f"{self.name} (target)"
        return self.name

    def properties(self) -> dict:
        return {"fire_id": self.fire_id, "name": self.name, "kind": self.kind,
                "source": self.source, "alarm_date": self.alarm_date,
                "contain_date": self.contain_date, "acres": self.acres,
                "n_detections": self.n_detections, "first_seen": self.first_seen,
                "last_seen": self.last_seen,
                "centroid": list(self.centroid) if self.centroid else None,
                "color": fire_color(self.fire_id)}


# ---------------------------------------------------------------------------
def _same_fire(a, b) -> bool:
    """The area query returns the target too; recognise it by name and year."""
    return (a.name.strip().upper() == b.name.strip().upper()
            and (a.year is None or b.year is None or a.year == b.year))


def _project(geometry: dict, fwd):
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform
    return shp_transform(lambda x, y, z=None: fwd.transform(x, y), shape(geometry))


def _shift(day: str | None, days: int) -> str | None:
    if not day:
        return None
    return (pd.Timestamp(day) + pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def attribute(used: pd.DataFrame, target, neighbours=(), crs: str = "EPSG:3310",
              buffer_m: float = 2000.0, eps_km: float = 2.0,
              tail_days: int = 3) -> tuple[pd.DataFrame, list[FireTag]]:
    """Add `fire_id` and `fire_name` to every detection. Returns (df, tags).

    `used` keeps its index, so the result can be joined back onto the audit
    table. `target` and `neighbours` are firelookup.FireRecord objects.
    """
    import shapely
    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    out = used.copy()
    ids = np.zeros(len(out), dtype=int)
    if out.empty:
        out["fire_id"], out["fire_name"] = ids, ""
        return out, []

    x, y = fwd.transform(out["longitude"].to_numpy(float),
                         out["latitude"].to_numpy(float))
    x, y = np.asarray(x), np.asarray(y)
    day_col = "local_date" if "local_date" in out.columns else "acq_date"
    days = out[day_col].astype(str).to_numpy()

    # 1. Mapped fires: target first, then neighbours that are not the target.
    mapped = [target] + [r for r in neighbours if not _same_fire(r, target)]
    best = np.full(len(out), np.inf)
    for k, rec in enumerate(mapped):
        geom = rec.geometry
        if not geom and rec.bbox:            # no polygon published: its box
            bw, bs, be, bn = rec.bbox
            geom = {"type": "Polygon", "coordinates": [[[bw, bs], [be, bs],
                    [be, bn], [bw, bn], [bw, bs]]]}
        if not geom or geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        poly = _project(geom, fwd)
        dist = shapely.distance(poly, shapely.points(x, y))
        ok = dist <= buffer_m
        lo, hi = _shift(rec.alarm_date, -1), _shift(rec.contain_date, tail_days)
        if lo:
            ok &= days >= lo
        if hi:
            ok &= days <= hi
        better = ok & (dist < best)          # strict: ties keep the earlier
        ids[better] = k + 1                  # (target) assignment
        best[better] = dist[better]

    # 2-3. Chains of detections within eps_km of each other.
    from sklearn.cluster import DBSCAN
    chain = DBSCAN(eps=eps_km * 1000.0, min_samples=1).fit_predict(
        np.column_stack([x, y]))
    unmapped_chains = []
    for c in np.unique(chain):
        members = chain == c
        known = ids[members][ids[members] > 0]
        if known.size:
            ids[members & (ids == 0)] = np.bincount(known).argmax()
        else:
            unmapped_chains.append((int(members.sum()), c))

    # Final numbering: target 1, mapped by size, then unmapped by size.
    counts = {k: int((ids == k + 1).sum()) for k in range(len(mapped))}
    order = [0] + sorted((k for k in counts if k and counts[k]),
                         key=lambda k: -counts[k])
    final = np.zeros_like(ids)
    tags: list[FireTag] = []

    def _tag(mask, name, kind, rec=None):
        fid = len(tags) + 1
        final[mask] = fid
        sel = out[mask]
        d = sel[day_col].astype(str)
        cen = ((float(sel["longitude"].mean()), float(sel["latitude"].mean()))
               if len(sel) else (rec.centroid if rec else None))
        tags.append(FireTag(
            fire_id=fid, name=name, kind=kind,
            source=rec.source if rec else "detections only",
            alarm_date=rec.alarm_date if rec else None,
            contain_date=rec.contain_date if rec else None,
            acres=rec.acres if rec else None,
            geometry=rec.geometry if rec else None,
            n_detections=int(mask.sum()),
            first_seen=d.min() if len(d) else None,
            last_seen=d.max() if len(d) else None,
            centroid=cen))

    for k in order:
        rec = mapped[k]
        _tag(ids == k + 1, rec.name, "target" if k == 0 else "mapped", rec)
    for i, (_, c) in enumerate(sorted(unmapped_chains, key=lambda t: -t[0])):
        _tag(chain == c, f"Unmapped #{i + 1}", "unmapped")

    out["fire_id"] = final
    names = {t.fire_id: t.name for t in tags}
    out["fire_name"] = [names[i] for i in final]
    return out, tags


# ---------------------------------------------------------------------------
def fire_id_map(step_df: pd.DataFrame, tile, spec) -> np.ndarray:
    """Per-cell fire ID for one window: 0 = no fire, else the fire's ID.

    Drawn with the same footprint rasterizer as the label, one fire at a time,
    target LAST so it wins a cell two fires both touch. The union of the
    per-fire masks is exactly the label's fire cells, which verify_dataset
    checks.
    """
    import firegrid as F
    out = np.zeros(spec.shape, dtype=np.uint16)
    if step_df.empty or "fire_id" not in step_df.columns:
        return out
    for fid in sorted(step_df["fire_id"].unique(), reverse=True):
        if fid <= 0:
            continue
        hit = F.rasterize_fire_mask(step_df[step_df["fire_id"] == fid], tile, spec) > 0
        out[hit] = fid
    return out


def cells_by_fire(fire_ids: np.ndarray, tags: list[FireTag]) -> list[dict]:
    names = {t.fire_id: t.name for t in tags}
    vals, n = np.unique(fire_ids[fire_ids > 0], return_counts=True)
    return [{"id": int(v), "name": names.get(int(v), f"#{v}"), "cells": int(c)}
            for v, c in zip(vals, n)]


def _agency(t: FireTag, sep: str = "") -> str:
    """Official dates, kept apart from the satellite ones: containment means a
    control line around the fire, not that it stopped burning or showing."""
    if t.kind == "unmapped" or not t.alarm_date:
        return ""
    return f"{sep}agency {t.alarm_date[5:]} to {(t.contain_date or '?')[5:]}"


def summarize(tags: list[FireTag]) -> str:
    if not tags:
        return "  fires       none"
    head = f"  fires       {len(tags)} in the tile"
    lines = [head]
    for t in tags:
        span = (f"{t.first_seen} .. {t.last_seen}" if t.first_seen != t.last_seen
                else f"{t.first_seen}")
        src = {"target": f"target, {_agency(t)}", "mapped": f"{t.source}, {_agency(t)}",
               "unmapped": "no agency perimeter"}[t.kind]
        lines.append(f"    {t.fire_id:>2}  {t.name:<18} {t.n_detections:>6,} det  "
                     f"seen {span:<24} {src}")
    return "\n".join(lines)


def save_fires(out_dir: Path, tags: list[FireTag]) -> Path:
    """fires.geojson: the key to every fire_id in this folder.

    Mapped fires carry their perimeter; unmapped ones a point at the centre
    of their detections.
    """
    feats = []
    for t in tags:
        geom = t.geometry or ({"type": "Point", "coordinates": list(t.centroid)}
                              if t.centroid else None)
        feats.append({"type": "Feature", "properties": t.properties(),
                      "geometry": geom})
    path = Path(out_dir) / "fires.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return path


def load_fires(fire_dir: Path) -> list[dict]:
    """Properties and geometry of each fire, by ID order. [] if absent."""
    p = Path(fire_dir) / "fires.geojson"
    if not p.exists():
        return []
    feats = json.loads(p.read_text()).get("features", [])
    return sorted(({**f["properties"], "geometry": f.get("geometry")} for f in feats),
                  key=lambda f: f["fire_id"])


# ---------------------------------------------------------------------------
def plot_fires(df: pd.DataFrame, tags: list[FireTag], envelope, path: Path,
               title: str, basemap: bool = False) -> Path:
    """Every fire in the tile, in its own colour, with its name.

    Detections are drawn as their footprints (what the label is built from),
    mapped perimeters as outlines, and each fire is named at its detections.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    w, s, e, n = envelope
    fig, ax = plt.subplots(figsize=(13.5, 10 * (n - s) / max(e - w, 1e-9) * 1.2))
    corners = ["nw", "ne", "se", "sw"]
    have_fp = all(f"footprint_{c}_lon" in df.columns for c in corners)
    handles = []
    for t in tags:
        sel = df[df["fire_id"] == t.fire_id]
        col = fire_color(t.fire_id)
        if have_fp and len(sel):
            polys = np.stack([np.column_stack([sel[f"footprint_{c}_lon"],
                                               sel[f"footprint_{c}_lat"]])
                              for c in corners], axis=1)
            ax.add_collection(PolyCollection(polys, facecolors=col, edgecolors=col,
                                             alpha=0.35, linewidths=0.3))
        elif len(sel):
            ax.scatter(sel["longitude"], sel["latitude"], s=6, color=col)
        if t.geometry and t.geometry.get("type") in ("Polygon", "MultiPolygon"):
            polys = (t.geometry["coordinates"] if t.geometry["type"] == "MultiPolygon"
                     else [t.geometry["coordinates"]])
            for poly in polys:
                ring = np.asarray(poly[0])
                ax.plot(ring[:, 0], ring[:, 1], color=col,
                        lw=1.6 if t.kind == "target" else 1.0,
                        ls="-" if t.kind == "target" else "--")
        if t.centroid and t.kind != "target":
            # A ring, so a two-detection fire is findable at tile scale.
            ax.scatter([t.centroid[0]], [t.centroid[1]], s=160, facecolors="none",
                       edgecolors=col, linewidths=1.4, zorder=3)
        if t.centroid:
            span = (t.first_seen if t.first_seen == t.last_seen
                    else f"{t.first_seen[5:]} to {t.last_seen[5:]}") if t.first_seen else ""
            ax.annotate(f"{t.name}\n{t.n_detections:,} det, seen {span}{_agency(t, chr(10))}",
                        t.centroid, xytext=(8, 8), textcoords="offset points",
                        fontsize=8 if t.kind != "target" else 9,
                        fontweight="bold" if t.kind == "target" else "normal",
                        color="black",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=col, lw=1, alpha=0.85))
        kind = {"target": "target", "mapped": "agency perimeter",
                "unmapped": "no perimeter"}[t.kind]
        handles.append(Line2D([0], [0], marker="s", ls="", color=col, markersize=9,
                              label=f"{t.fire_id}  {t.name} ({kind})"))
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect(1 / np.cos(np.radians((s + n) / 2)))
    if basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(ax, crs="EPSG:4326", attribution_size=6,
                            source=ctx.providers.OpenStreetMap.HOT)
        except Exception as err:
            print(f"    fires plot: basemap unavailable ({str(err)[:80]})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.3)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=8, framealpha=0.9, title="fire_id  name", title_fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
