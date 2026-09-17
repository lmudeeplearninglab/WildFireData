#!/usr/bin/env python3
"""
firelookup -- turn a fire NAME into a bounding box, a tile and a date window.

Replaces hand-typed bounding boxes. Ask for "Palisades" and get back the
actual burned perimeter, its alarm and containment dates, and a ready-made
command line for the extraction pipeline.

Two sources, tried in order:

  1. CAL FIRE FRAP historic perimeters (authoritative polygon)
     services1.arcgis.com/jUJYIo9tSA7EHvfZ/.../California_Historic_Fire_Perimeters
     FRAP is compiled by CAL FIRE with USFS Region 5, BLM, NPS and FWS, and
     is republished annually after the calendar year closes. It gives a true
     footprint, so the bbox is measured rather than guessed.

  2. CAL FIRE incidents feed (point + acreage)
     incidents.fire.ca.gov/umbraco/api/IncidentApi/List
     Near-real-time, so it covers fires FRAP has not published yet, but it
     carries only a centroid. The bbox is then INFERRED from acreage by
     assuming a circular burn -- a crude approximation that is wrong for any
     wind-driven fire, and is labelled as such in the output.

FRAP itself warns that the perimeter record is incomplete, particularly for
older and smaller fires. A name that returns nothing has not necessarily
been proven not to exist.

Run:
    python firelookup.py "Palisades" --year 2025
    python firelookup.py "Eaton" --year 2025 --emit-command
    python firelookup.py --list-year 2025 --min-acres 1000
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

FRAP_URL = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
            "California_Historic_Fire_Perimeters/FeatureServer/2/query")
INCIDENTS_URL = "https://incidents.fire.ca.gov/umbraco/api/IncidentApi/List"

# National sources. NIFC's Interagency Fire Perimeter History conglomerates
# the authoritative perimeters of USFS, BLM, BIA, FWS, NPS, the Alaska
# Interagency Fire Center, CAL FIRE and WFIGS -- so it covers every state,
# not just California.
NIFC_HISTORY_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0/query")
# WFIGS current-year perimeters, updated continuously. Carries real discovery
# and containment timestamps, which the history layer often does not.
WFIGS_CURRENT_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query")

# Field names vary between services and between vintages within a service, so
# dates are read by trying candidates in order rather than assuming one name.
DISCOVERY_FIELDS = ("attr_FireDiscoveryDateTime", "FireDiscoveryDateTime",
                    "ALARM_DATE", "DISCOVERY_DATE", "attr_CreatedOnDateTime_dt")
CONTAIN_FIELDS = ("attr_ContainmentDateTime", "ContainmentDateTime",
                  "CONT_DATE", "attr_FireOutDateTime", "DATE_CUR")

ACRE_KM2 = 0.00404685642
TIMEOUT = 30


# ---------------------------------------------------------------------------
@dataclass
class FireRecord:
    name: str
    year: int | None
    alarm_date: str | None          # YYYY-MM-DD
    contain_date: str | None
    acres: float | None
    bbox: tuple[float, float, float, float] | None   # w, s, e, n
    centroid: tuple[float, float] | None             # lon, lat
    source: str
    exact_footprint: bool           # False when the bbox was inferred
    state: str = ""                 # e.g. "US-CA", when the source reports it
    dates_uncertain: bool = False   # True when the alarm date was derived
                                    # from the fire year rather than recorded

    def date_window(self, lead_in_days: int = 2, tail_days: int = 3
                    ) -> tuple[str | None, str | None]:
        """Fetch window: a lead-in before ignition, a tail after containment.

        The lead-in exists because prev_fire_mask needs day t-1 to build a
        sample for day t; without it the first labelled day is unusable. The
        tail exists so burnout is observed rather than truncated at the edge
        of the request.
        """
        if not self.alarm_date:
            return (None, None)
        start = datetime.strptime(self.alarm_date, "%Y-%m-%d") - timedelta(days=lead_in_days)
        end_src = self.contain_date or self.alarm_date
        end = datetime.strptime(end_src, "%Y-%m-%d") + timedelta(days=tail_days)
        return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    def describe(self) -> str:
        acres = f"{self.acres:,.0f} acres" if self.acres else "acreage unknown"
        span = f"{self.alarm_date or '?'} .. {self.contain_date or '?'}"
        if self.dates_uncertain:
            span += " (dates approximate)"
        kind = "perimeter" if self.exact_footprint else "INFERRED from acreage"
        where = f", {self.state}" if self.state else ""
        return (f"{self.name} ({self.year or '?'}{where})  {acres}  {span}\n"
                f"    bbox {self._bbox_str()}  [{kind}, {self.source}]")

    def _bbox_str(self) -> str:
        if not self.bbox:
            return "unavailable"
        w, s, e, n = self.bbox
        return f"({w:.4f}, {s:.4f}, {e:.4f}, {n:.4f})"


# ---------------------------------------------------------------------------
def _epoch_to_date(value) -> str | None:
    """ArcGIS returns dates as epoch MILLISECONDS; pass through ISO strings."""
    if value in (None, "", 0):
        return None
    if isinstance(value, str):
        return value[:10] if len(value) >= 10 else None
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc) \
            .strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return None


def _ring_bounds(geometry: dict) -> tuple[float, float, float, float] | None:
    """Bounds of a GeoJSON Polygon or MultiPolygon.

    Walked manually rather than via shapely so a name lookup does not require
    the geo stack to be installed.
    """
    if not geometry:
        return None
    coords = geometry.get("coordinates")
    gtype = geometry.get("type")
    if not coords:
        return None
    rings = coords if gtype == "Polygon" else \
        [ring for poly in coords for ring in poly] if gtype == "MultiPolygon" else []
    xs, ys = [], []
    for ring in rings:
        for point in ring:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def pad_bbox(bbox, km: float) -> tuple[float, float, float, float]:
    """Grow a bbox by km on every side.

    Padding matters: FIRMS detects heat outside the final mapped perimeter
    (spot fires, the active front ahead of the burn), so a bbox clipped to
    the perimeter would cut off exactly the detections a spread model needs.
    """
    import math
    w, s, e, n = bbox
    dlat = km / 111.0
    mid = math.radians((s + n) / 2)
    dlon = km / (111.0 * max(0.2, math.cos(mid)))
    return (w - dlon, s - dlat, e + dlon, n + dlat)


# ---------------------------------------------------------------------------
def search_perimeters(name: str, year: int | None = None,
                      limit: int = 10) -> list[FireRecord]:
    """Query FRAP for perimeters whose FIRE_NAME contains `name`."""
    where = f"UPPER(FIRE_NAME) LIKE '%{name.upper().strip()}%'"
    if year:
        where += f" AND YEAR_ = {int(year)}"
    params = {
        "where": where,
        "outFields": "FIRE_NAME,YEAR_,ALARM_DATE,CONT_DATE,GIS_ACRES,AGENCY,UNIT_ID",
        "returnGeometry": "true",
        "outSR": "4326",          # ask for lon/lat; the service stores 3310
        "f": "geojson",
        "resultRecordCount": limit,
    }
    resp = requests.get(FRAP_URL, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"FRAP query failed: {data['error']}")

    out: list[FireRecord] = []
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        bounds = _ring_bounds(feat.get("geometry") or {})
        centroid = None
        if bounds:
            centroid = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        out.append(FireRecord(
            name=str(props.get("FIRE_NAME", "")).strip(),
            year=int(props["YEAR_"]) if props.get("YEAR_") else None,
            alarm_date=_epoch_to_date(props.get("ALARM_DATE")),
            contain_date=_epoch_to_date(props.get("CONT_DATE")),
            acres=float(props["GIS_ACRES"]) if props.get("GIS_ACRES") else None,
            bbox=bounds,
            centroid=centroid,
            source="FRAP perimeter",
            exact_footprint=True,
        ))
    return out


def search_incidents(name: str, year: int | None = None,
                     limit: int = 10) -> list[FireRecord]:
    """Query the CAL FIRE incidents feed. Point only -- bbox is inferred."""
    import math
    params = {"inactive": "true"}
    if year:
        params["year"] = int(year)
    resp = requests.get(INCIDENTS_URL, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload if isinstance(payload, list) else payload.get("Incidents", [])

    needle = name.upper().strip()
    out: list[FireRecord] = []
    for row in rows:
        label = str(row.get("Name", ""))
        if needle not in label.upper():
            continue
        lat, lon = row.get("Latitude"), row.get("Longitude")
        acres = row.get("AcresBurned")
        bbox = centroid = None
        if lat and lon:
            centroid = (float(lon), float(lat))
            # Circular-burn approximation. Wrong for wind-driven fires, which
            # is most of them -- flagged via exact_footprint=False.
            radius_km = math.sqrt(float(acres or 0) * ACRE_KM2 / math.pi) or 5.0
            bbox = pad_bbox((centroid[0], centroid[1], centroid[0], centroid[1]),
                            max(radius_km, 5.0))
        started = str(row.get("Started", ""))[:10] or None
        out.append(FireRecord(
            name=label.strip(),
            year=int(started[:4]) if started else None,
            alarm_date=started,
            contain_date=str(row.get("Extinguished", ""))[:10] or None,
            acres=float(acres) if acres else None,
            bbox=bbox,
            centroid=centroid,
            source="CAL FIRE incidents",
            exact_footprint=False,
        ))
        if len(out) >= limit:
            break
    return out



def _first_date(props: dict, candidates) -> str | None:
    """First parseable date among several possible field names."""
    for field in candidates:
        if field in props:
            got = _epoch_to_date(props[field])
            if got:
                return got
    return None


def _arcgis_perimeters(url: str, where: str, out_fields: str, limit: int,
                       name_field: str, year_field: str, acres_field: str,
                       state_field: str | None, source_label: str
                       ) -> list[FireRecord]:
    """Shared ArcGIS perimeter query. All these services speak the same REST."""
    params = {
        "where": where, "outFields": out_fields, "returnGeometry": "true",
        "outSR": "4326", "f": "geojson", "resultRecordCount": limit,
    }
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"{source_label} query failed: {data['error']}")

    out: list[FireRecord] = []
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        bounds = _ring_bounds(feat.get("geometry") or {})
        if not bounds:
            continue
        alarm = _first_date(props, DISCOVERY_FIELDS)
        contain = _first_date(props, CONTAIN_FIELDS)
        # year_field is None for services that publish no fire-year column
        # (WFIGS); the year is then taken from the discovery date. Reading a
        # timestamp field as a year yields nonsense like 1769040000000.
        year = props.get(year_field) if year_field else None
        try:
            year = int(year) if year not in (None, "") else None
        except (TypeError, ValueError):
            year = None
        if year is not None and not (1800 <= year <= 2200):
            year = None
        # The history layer frequently lacks a discovery date, carrying only
        # the fire year. Falling back to Jan 1 of that year would silently
        # produce a wildly wrong fetch window, so the record is flagged and
        # the caller decides what to do.
        uncertain = False
        if not alarm and year:
            alarm = f"{year}-01-01"
            uncertain = True
        out.append(FireRecord(
            name=str(props.get(name_field, "")).strip(),
            year=year or (int(alarm[:4]) if alarm else None),
            alarm_date=alarm,
            contain_date=contain,
            acres=float(props[acres_field]) if props.get(acres_field) else None,
            bbox=bounds,
            centroid=((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2),
            source=source_label,
            exact_footprint=True,
            state=str(props.get(state_field, "") or "") if state_field else "",
            dates_uncertain=uncertain,
        ))
    return out


def search_nifc_history(name: str, year: int | None = None,
                        state: str | None = None,
                        limit: int = 10) -> list[FireRecord]:
    """NIFC Interagency Fire Perimeter History -- national, all years."""
    where = f"UPPER(INCIDENT) LIKE '%{name.upper().strip()}%'"
    if year:
        where += f" AND FIRE_YEAR = '{int(year)}'"
    return _arcgis_perimeters(
        NIFC_HISTORY_URL, where,
        "INCIDENT,FIRE_YEAR,GIS_ACRES,AGENCY,FEATURE_CA,DATE_CUR",
        limit, "INCIDENT", "FIRE_YEAR", "GIS_ACRES", None,
        "NIFC perimeter history")


def search_wfigs_current(name: str, state: str | None = None,
                         limit: int = 10) -> list[FireRecord]:
    """WFIGS current-year perimeters -- national, with real timestamps."""
    where = f"UPPER(poly_IncidentName) LIKE '%{name.upper().strip()}%'"
    if state:
        where += f" AND attr_POOState = '{_state_code(state)}'"
    return _arcgis_perimeters(
        WFIGS_CURRENT_URL, where,
        "poly_IncidentName,attr_FireDiscoveryDateTime,attr_ContainmentDateTime,"
        "poly_GISAcres,attr_POOState,attr_IncidentTypeCategory",
        limit, "poly_IncidentName", None,
        "poly_GISAcres", "attr_POOState", "WFIGS current")


def _state_code(state: str) -> str:
    """Normalize 'CA' or 'California' to the WFIGS form 'US-CA'."""
    st = state.strip().upper()
    if st.startswith("US-"):
        return st
    if len(st) == 2:
        return f"US-{st}"
    return f"US-{st[:2]}"


# California first, deliberately. FRAP carries real alarm and containment
# dates, while NIFC's history layer often has only a fire year -- which turns
# a two-week fire into an eight-month fetch window. While the project is
# California-only, the better dates are worth more than national coverage.
# Set --national (or pass sources=NATIONAL_CHAIN) when that changes.
SOURCE_CHAIN = ("frap", "incidents", "nifc", "wfigs")
NATIONAL_CHAIN = ("nifc", "wfigs", "frap", "incidents")


def resolve_fire(name: str, year: int | None = None,
                 state: str | None = None,
                 sources: tuple[str, ...] = SOURCE_CHAIN) -> list[FireRecord]:
    """Try each source in turn and return the first that knows this fire.

    Order matters. NIFC history is national and authoritative but its dates
    are patchy; WFIGS covers the current season with real timestamps; FRAP is
    California-only with good dates; the incidents feed has no polygon at all.
    A source that errors is skipped rather than aborting the lookup, so one
    service being down does not end the search.
    """
    lookups = {
        "nifc": lambda: search_nifc_history(name, year, state),
        "wfigs": lambda: search_wfigs_current(name, state),
        "frap": lambda: search_perimeters(name, year),
        "incidents": lambda: search_incidents(name, year),
    }
    for key in sources:
        if key not in lookups:
            continue
        try:
            hits = lookups[key]()
        except Exception as err:
            print(f"  {key} lookup failed ({err}); trying the next source")
            continue
        if state and key in ("nifc", "frap", "incidents"):
            # Those services do not all expose a state field, so filter
            # geographically instead of trusting an attribute that may be absent.
            hits = [h for h in hits if _in_state(h, state)]
        if hits:
            if key == "incidents":
                print("  NOTE: no perimeter found; bbox inferred from acreage.")
            if any(h.dates_uncertain for h in hits):
                print("  NOTE: some records carry only a fire year, not a "
                      "discovery date. Check the window before building.")
            return sorted(hits, key=lambda r: -(r.acres or 0))
    return []


# Rough bounding boxes, used only to filter lookups by state when the service
# does not report one. Deliberately generous: excluding a real fire is worse
# than showing one extra.
_STATE_BOXES = {
    "CA": (-124.6, 32.4, -114.0, 42.1), "OR": (-124.7, 41.9, -116.4, 46.3),
    "WA": (-124.9, 45.5, -116.9, 49.1), "NV": (-120.1, 34.9, -113.9, 42.1),
    "AZ": (-115.0, 31.3, -108.9, 37.1), "ID": (-117.3, 41.9, -110.9, 49.1),
    "MT": (-116.1, 44.3, -103.9, 49.1), "CO": (-109.1, 36.9, -101.9, 41.1),
    "NM": (-109.1, 31.3, -102.9, 37.1), "UT": (-114.1, 36.9, -108.9, 42.1),
    "WY": (-111.1, 40.9, -103.9, 45.1), "TX": (-106.7, 25.8, -93.5, 36.6),
    "AK": (-179.9, 51.0, -129.0, 71.5), "FL": (-87.7, 24.4, -79.9, 31.1),
}


def _in_state(record: FireRecord, state: str) -> bool:
    code = _state_code(state)[3:]
    if record.state:
        return record.state.upper().endswith(code)
    box = _STATE_BOXES.get(code)
    if not box or not record.centroid:
        return True          # unknown state: do not filter it out
    lon, lat = record.centroid
    return box[0] <= lon <= box[2] and box[1] <= lat <= box[3]


# ---------------------------------------------------------------------------
def _covers(outer, inner) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


def _area_ratio(a, b) -> float:
    def area(x):
        return abs(x[2] - x[0]) * abs(x[3] - x[1])
    return area(a) / area(b) if area(b) else 0.0


def to_tile(record: FireRecord):
    """Snapped grid tile for this fire, if firegrid is importable."""
    if not record.centroid:
        return None
    try:
        import src.firegrid as F
    except ImportError:
        return None
    return F.snap_tile(record.centroid[0], record.centroid[1])


def tile_envelope(record: FireRecord) -> tuple[float, float, float, float] | None:
    """Lon/lat envelope of this fire's snapped grid tile, if firegrid is present.

    The projected tile is a rectangle in EPSG:3310 and therefore NOT one in
    lon/lat, so the envelope is slightly larger. That is the right direction
    to err: a fetch must cover the whole tile, never merely most of it.
    """
    tile = to_tile(record)
    if tile is None:
        return None
    import src.firegrid as F
    return F.tile_bounds_lonlat(tile)


def emit_command(record: FireRecord, pad_km: float, lead_in: int,
                 tail: int, match_tile: bool = False) -> str:
    """The pipeline invocation for this fire.

    match_tile widens the fetch to the full 64 km grid tile. Without it the
    perimeter-plus-pad bbox covers only ~10-15% of the tile, and every cell
    outside it would be written as no-fire in the label -- indistinguishable,
    to a model, from ground that was observed and did not burn. That teaches
    the rectangle, not the fire.
    """
    if not record.bbox:
        return "# no bbox available for this record"
    envelope = tile_envelope(record) if match_tile else None
    w, s, e, n = envelope if envelope else pad_bbox(record.bbox, pad_km)
    # Round OUTWARD. Printing at 4 decimals rounds to the nearest ~11 m, which
    # can land inside the tile and leave a sliver of it unfetched -- the exact
    # gap this option exists to close.
    import math
    q = 10_000
    w, s = math.floor(w * q) / q, math.floor(s * q) / q
    e, n = math.ceil(e * q) / q, math.ceil(n * q) / q
    start, end = record.date_window(lead_in, tail)
    parts = ["python visualize_firms_dataset_v3.py",
             f'--bbox "{w:.4f},{s:.4f},{e:.4f},{n:.4f}"']
    if start and end:
        parts.append(f"--start-date {start} --end-date {end}")
    parts.append("--fire-mask --daily-visual-samples")
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help='Fire name, e.g. "Palisades"')
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--state", default=None,
                    help="Restrict to a state, e.g. CA, OR, MT. Useful because "
                         "incident names repeat across the country.")
    ap.add_argument("--source", default=None,
                    choices=["nifc", "wfigs", "frap", "incidents"],
                    help="Force one source instead of the fallback chain")
    ap.add_argument("--national", action="store_true",
                    help="Search the national NIFC/WFIGS records first. "
                         "Default is California-first (FRAP), which has better "
                         "ignition and containment dates.")
    ap.add_argument("--pad-km", type=float, default=5.0,
                    help="Grow the bbox by this much on each side (default 5)")
    ap.add_argument("--lead-in-days", type=int, default=2,
                    help="Days fetched before ignition, for prev_fire_mask")
    ap.add_argument("--tail-days", type=int, default=3)
    ap.add_argument("--emit-command", action="store_true",
                    help="Print the pipeline command for the best match")
    ap.add_argument("--match-tile", action="store_true",
                    help="Fetch the full 64 km grid tile instead of the padded "
                         "perimeter, so every cell in the tile is actually "
                         "queried. Use this when the output feeds firegrid.")
    ap.add_argument("--json", type=str, default=None,
                    help="Write all matches to this JSON file")
    args = ap.parse_args()

    if not args.name:
        ap.error("give a fire name")

    chain = ((args.source,) if args.source
             else NATIONAL_CHAIN if args.national else SOURCE_CHAIN)
    records = resolve_fire(args.name, args.year, args.state, chain)
    if not records:
        print(f"No fire matching '{args.name}'"
              + (f" in {args.year}" if args.year else "") + ".")
        print("Tried: " + ", ".join(chain) + ".")
        print("Perimeter archives are republished annually, so a fire from "
              "the current season may only\nbe in 'wfigs'. Try without "
              "--year, a shorter name fragment, or --source wfigs.")
        return 1

    print(f"{len(records)} match(es) for '{args.name}':\n")
    for i, rec in enumerate(records):
        print(f"  [{i}] {rec.describe()}")
        start, end = rec.date_window(args.lead_in_days, args.tail_days)
        if start:
            print(f"        fetch window {start} .. {end} "
                  f"({args.lead_in_days}d lead-in, {args.tail_days}d tail)")
        tile = to_tile(rec)
        if tile:
            print(f"        snapped tile origin ({tile[0]:.0f}, {tile[1]:.0f})")
            env = tile_envelope(rec)
            padded = pad_bbox(rec.bbox, args.pad_km) if rec.bbox else None
            if env and padded and not _covers(padded, env):
                pct = _area_ratio(padded, env) * 100
                print(f"        WARNING: the padded bbox covers only {pct:.0f}% "
                      f"of that tile.\n"
                      f"                 Unqueried cells become no-fire in the "
                      f"label. Use --match-tile.")
        print()

    best = records[0]
    if args.emit_command:
        print("Pipeline command for the largest match:\n")
        print("  " + emit_command(best, args.pad_km, args.lead_in_days,
                                  args.tail_days, args.match_tile) + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(
            [{**r.__dict__, "bbox": list(r.bbox) if r.bbox else None,
              "centroid": list(r.centroid) if r.centroid else None}
             for r in records], indent=2))
        print(f"Saved: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
