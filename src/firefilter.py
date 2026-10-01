#!/usr/bin/env python3
"""
firefilter -- decide which FIRMS detections are allowed to become labels.

firelookup decides where and when. firegrid decides what shape. This module
decides WHICH DETECTIONS COUNT, which is its own question and was previously
answered by accident -- every detection in the tile became fire.

Four steps, applied in order by annotate():

  1. CONFIDENCE, normalized across sensors
     VIIRS reports 'l' / 'n' / 'h'. MODIS reports an integer 0-100. Mixed in
     one column they cannot be compared, sorted or thresholded, and the
     column cannot even be written to HDF5. Mapped onto one scale using the
     FIRMS thresholds for MODIS: low < 30, nominal 30-79, high >= 80.

  2. STATIC SOURCES, removed
     Refineries, power plants and gas flares produce persistent thermal
     anomalies that FIRMS reports every day. In the Eaton pull they were
     83% of all detections on 2025-01-13. Left in, they become "fire" in
     the label on exactly the late days where a model learns burnout.

     Two signals, because either alone is insufficient:
       - FIRMS 'type' == 2 ("other static land source"), 1 (volcano), 3
         (offshore). Only the standard-processing products carry this
         field. VIIRS_NOAA21_NRT does not, so its detections at a known
         refinery cannot be filtered on type.
       - Location. Any cell where a typed source has ever appeared is a
         static site, and every detection there -- from any satellite -- is
         static. This is what catches the NOAA-21 rows.
     Plus persistence: a cell lit on most days of the window at low FRP is
     static whether or not anything typed it.

     Guarded by FRP: a detection above `keep_frp_above` MW is kept even at
     a static site, because a real fire burning past a refinery is exactly
     the case where dropping data would be wrong.

  3. TARGET FIRE, optionally isolated
     A 64 km tile often contains more than the named fire. Whether the
     others belong in the label is a modelling choice -- they are real fire
     -- so this is off by default.

  4. STOP RULE (see blank_run_stop_date)
     After N consecutive days with no surviving detections, the fire is
     treated as out and later days are not written. Only meaningful AFTER
     static removal: a refinery lights up every day, so without step 2 a
     run of blank days never occurs and the rule never fires.

Every detection is kept in the output frame with its verdict, so the CSV
is an audit trail for the label rather than a copy of it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CONF_ORDER = {"low": 0, "nominal": 1, "high": 2}

# FIRMS 'type' codes (standard-processing products only)
TYPE_VEGETATION = 0
TYPE_VOLCANO = 1
TYPE_STATIC_LAND = 2
TYPE_OFFSHORE = 3
NON_FIRE_TYPES = {TYPE_VOLCANO, TYPE_STATIC_LAND, TYPE_OFFSHORE}


@dataclass
class FilterConfig:
    min_confidence: str = "nominal"     # low | nominal | high
    remove_static: bool = True
    static_cell_deg: float = 0.01       # ~1 km grouping for static sites
    persist_frac: float = 0.5           # lit on >= this share of days ...
    persist_max_frp: float = 5.0        # ... at median FRP below this (MW)
    keep_frp_above: float = 10.0        # never drop a detection this hot
    target_only: bool = False
    target_pad_km: float = 5.0

    def describe(self) -> str:
        parts = [f"confidence >= {self.min_confidence}"]
        parts.append("static sources removed" if self.remove_static
                     else "static sources KEPT")
        if self.target_only:
            parts.append(f"target fire only (+{self.target_pad_km:g} km)")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
def normalize_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Add `confidence_class` (low/nominal/high) valid across VIIRS and MODIS.

    The original `confidence` column is left untouched.
    """
    out = df.copy()
    if "confidence" not in out.columns:
        out["confidence_class"] = "nominal"
        return out

    raw = out["confidence"].astype(str).str.strip().str.lower()
    viirs = {"l": "low", "n": "nominal", "h": "high",
             "low": "low", "nominal": "nominal", "high": "high"}
    cls = raw.map(viirs)

    numeric = pd.to_numeric(out["confidence"], errors="coerce")
    is_num = cls.isna() & numeric.notna()
    cls.loc[is_num] = np.where(numeric[is_num] >= 80, "high",
                               np.where(numeric[is_num] >= 30, "nominal", "low"))
    out["confidence_class"] = cls.fillna("low")
    return out


def _cells(df: pd.DataFrame, deg: float) -> pd.Series:
    """Snap coordinates to a coarse grid for site matching."""
    return list(zip((df["latitude"] / deg).round().astype(int),
                    (df["longitude"] / deg).round().astype(int)))


def _inside(df: pd.DataFrame, bbox) -> pd.Series:
    w, s, e, n = bbox
    return df["longitude"].between(w, e) & df["latitude"].between(s, n)


def find_static_cells(df: pd.DataFrame, cfg: FilterConfig,
                      protect_bbox: tuple | None = None,
                      alarm_date: str | None = None) -> dict:
    """Identify static heat sources. Returns {cell: reason}.

    FIRMS 'type' is NOT trusted on its own. In the Eaton pull it marked 44
    detections inside the burn perimeter -- MODIS pixels at up to 800 MW on
    ignition night -- as "other static land source", most likely because the
    fire burned into urban Altadena, where the static-source mask applies.
    Propagating that label by location then dropped 176 real detections on
    the peak day.

    So a cell is static only if it shows the signature a fire does not:
    persistent across many days, at low power. Type evidence lowers the bar
    for how many days are needed, but never substitutes for it. And inside
    the named fire's own perimeter, a cell must additionally have been lit
    BEFORE ignition -- activity that predates the fire cannot be the fire.
    """
    work = df.copy()
    work["_cell"] = _cells(work, cfg.static_cell_deg)
    date_col = "local_date" if "local_date" in work.columns else "acq_date"
    n_days = work[date_col].nunique()
    if "frp" not in work.columns or n_days < 3:
        return {}

    work["_frp"] = pd.to_numeric(work["frp"], errors="coerce")
    typed_cells = set()
    if "type" in work.columns:
        typed_cells = set(work.loc[pd.to_numeric(work["type"], errors="coerce")
                                   .isin(NON_FIRE_TYPES), "_cell"])

    per = work.groupby("_cell").agg(days=(date_col, "nunique"),
                                    frp=("_frp", "median"))
    low_power = per["frp"] < cfg.persist_max_frp

    # Persistent on its own: lit on a large share of the window.
    persistent = per.index[(per["days"] >= max(3, cfg.persist_frac * n_days))
                           & low_power]
    # Typed AND low-power AND recurring: a lower bar, but still a bar.
    typed_recurring = per.index[per.index.isin(typed_cells)
                                & (per["days"] >= 3) & low_power]

    pre_ignition = set()
    if alarm_date:
        pre_ignition = set(work.loc[work[date_col].astype(str) < alarm_date,
                                    "_cell"])

    def inside_protected(cell) -> bool:
        if protect_bbox is None:
            return False
        w, s, e, n = protect_bbox
        lat, lon = cell[0] * cfg.static_cell_deg, cell[1] * cfg.static_cell_deg
        return w <= lon <= e and s <= lat <= n

    static: dict = {}
    for cell, why in [(c, "persistent low-FRP site") for c in persistent] + \
                     [(c, "recurring typed static source") for c in typed_recurring]:
        if inside_protected(cell) and cell not in pre_ignition:
            continue          # in the burn area and not active before it
        static.setdefault(cell, why)
    return static


def annotate(df: pd.DataFrame, cfg: FilterConfig,
             target_bbox: tuple | None = None,
             alarm_date: str | None = None) -> pd.DataFrame:
    """Attach a verdict to every detection.

    Adds:
      confidence_class   low / nominal / high
      drop_reason        '' when kept, otherwise why it was excluded
      used_in_label      True for detections that become fire in the label
    """
    out = normalize_confidence(df)
    reason = pd.Series("", index=out.index, dtype=object)

    # confidence
    rank = out["confidence_class"].map(CONF_ORDER).fillna(0)
    too_low = rank < CONF_ORDER[cfg.min_confidence]
    reason[too_low] = "below confidence threshold"

    # static sources
    if cfg.remove_static:
        static = find_static_cells(out, cfg, protect_bbox=target_bbox,
                                   alarm_date=alarm_date)
        cells = pd.Series(_cells(out, cfg.static_cell_deg), index=out.index)
        hot = pd.to_numeric(out.get("frp"), errors="coerce").fillna(0) \
            >= cfg.keep_frp_above
        # A site straddles cell edges: 84 of 85 Torrance refinery rows fell in
        # flagged cells, and the one that landed next door became "fire".
        # Outside the named fire, a static cell's 8 neighbours count too, but
        # only at the site's own power (<= 2x its median FRP): a 5 MW Hurst
        # Fire pixel beside a landfill is fire, not the landfill.
        frp = pd.to_numeric(out.get("frp"), errors="coerce")
        site_frp = frp.groupby(cells).median()
        ring: dict = {}
        for (r, c), why in static.items():
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    lim = 2 * site_frp.get((r, c), 0)
                    if lim > ring.get((r + dr, c + dc), ("", -1))[1]:
                        ring[(r + dr, c + dc)] = (why, lim)
        outside = ~_inside(out, target_bbox) if target_bbox else True
        lim = cells.map(lambda k: ring.get(k, ("", -1))[1])
        match = cells.isin(static.keys()) | ((frp <= lim) & outside)
        for idx in out.index[(reason == "") & match & ~hot]:
            reason[idx] = static.get(cells[idx]) or ring[cells[idx]][0]

    # target fire isolation
    if cfg.target_only and target_bbox is not None:
        w, s, e, n = target_bbox
        dlat = cfg.target_pad_km / 111.0
        dlon = cfg.target_pad_km / (111.0 * max(0.2, np.cos(np.radians((s + n) / 2))))
        inside = (out["longitude"].between(w - dlon, e + dlon)
                  & out["latitude"].between(s - dlat, n + dlat))
        reason[(reason == "") & ~inside] = "outside target fire"

    out["drop_reason"] = reason
    out["used_in_label"] = reason == ""
    return out


def summarize(annotated: pd.DataFrame) -> str:
    """One line per drop reason, for the run transcript."""
    total = len(annotated)
    kept = int(annotated["used_in_label"].sum())
    lines = [f"  kept {kept:,} of {total:,} detections "
             f"({kept / total * 100:.0f}%)" if total else "  no detections"]
    for why, n in annotated.loc[~annotated["used_in_label"], "drop_reason"] \
            .value_counts().items():
        lines.append(f"    dropped {n:>5,}  {why}")
    return "\n".join(lines)


def blank_run_stop_date(used: pd.DataFrame, days: list[str],
                        blank_days: int, date_col: str | None = None) -> str | None:
    """First day after which extraction should stop, or None.

    Counts consecutive days with no surviving detections, starting only once
    the fire has been seen -- blank days before ignition are the lead-in, not
    a burnout. Returns the last day to KEEP.
    """
    if blank_days <= 0 or used.empty:
        return None
    if date_col is None:
        date_col = "local_date" if "local_date" in used.columns else "acq_date"
    active = set(used[date_col].astype(str))
    seen, run = False, 0
    for i, day in enumerate(days):
        if day in active:
            seen, run = True, 0
            continue
        if seen:
            run += 1
            if run >= blank_days:
                return days[i - blank_days]    # last active day
    return None
