#!/usr/bin/env python3
"""
verify_dataset -- check that a built dataset is actually normalized.

Everything in this pipeline guarantees uniform geometry *by construction*.
This script tests that claim against the files on disk, which is a different
thing: a partial run, a spec change halfway through, or a silently degraded
Earth Engine fetch all produce a directory that looks fine and is not.

Run:
    python verify_dataset.py ml_dataset
    python verify_dataset.py ml_dataset --json report.json
    python verify_dataset.py ml_dataset --strict     # warnings become failures

Exit code 0 means every check passed.

WHAT IT CHECKS
--------------
Spatial normalization
  - every sample has the same tensor and label shape, and the same dtypes
  - every sample declares the same CRS and cell size
  - tile origins are exact multiples of the cell size (global-grid alignment)
  - all samples of one fire share one tile
  - channel names are present, complete and in the canonical order

Temporal normalization
  - dates parse, are unique, and are consecutive with no gaps
  - the per-channel temporal footprint is reported, because it is NOT uniform:
    static layers have no time dimension, weather is daily, drought and
    vegetation summarize multi-day windows

Content
  - labels contain only {0, 1, 2}
  - per-channel coverage: a channel that is entirely fill is reported, since
    that is what a silent EE failure looks like
  - class balance across the dataset
  - prev_fire_mask on day t matches the fire cells of day t-1, which is the
    only end-to-end test that the temporal chain was built correctly
  - tile overlap between different fires, which is a train/test leakage risk
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import reality_checks as RC


def _import_firegrid():
    import importlib
    return importlib.import_module("firegrid")


PASS, FAIL, WARN, INFO = "  [ok]  ", "  [FAIL]", "  [warn]", "        "


class Report:
    """Collects results so the summary can be printed and exported."""

    def __init__(self, strict: bool = False):
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.facts: dict = {}
        self.strict = strict

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        print(f"{PASS if ok else FAIL} {label}")
        if detail:
            print(f"{INFO} {detail}")
        if not ok:
            self.failures.append(label)
        return ok

    def warn(self, label: str, detail: str = "") -> None:
        print(f"{WARN} {label}")
        if detail:
            print(f"{INFO} {detail}")
        (self.failures if self.strict else self.warnings).append(label)

    def note(self, text: str) -> None:
        print(f"{INFO} {text}")


# ---------------------------------------------------------------------------
def load_all(root: Path, F) -> dict[str, list[dict]]:
    """Read every sample, grouped by fire directory."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(root.rglob("*.npz")):
        try:
            features, label, channels, meta = F.load_sample(path)
        except Exception as err:
            print(f"{FAIL} unreadable: {path.name} ({err})")
            continue
        groups[path.parent.name].append({
            "path": path, "features": features, "label": label,
            "channels": list(channels), "meta": meta,
        })
    return groups


def check_spatial(samples: list[dict], rep: Report, F) -> None:
    spec = F.GRID_SPEC
    shapes = {s["features"].shape for s in samples}
    rep.check(len(shapes) == 1, f"tensor shape uniform: {shapes.pop() if len(shapes)==1 else shapes}",
              "" if len(shapes) <= 1 else "samples disagree -- they cannot be batched")

    label_shapes = {s["label"].shape for s in samples}
    rep.check(len(label_shapes) == 1 and label_shapes == {spec.shape},
              f"label shape uniform and matches the spec: {spec.shape}")

    dtypes = {(str(s["features"].dtype), str(s["label"].dtype)) for s in samples}
    rep.check(dtypes == {("float32", "uint8")},
              f"dtypes consistent: {dtypes}")

    crs = {s["meta"].get("crs") for s in samples}
    cells = {s["meta"].get("cell_m") for s in samples}
    rep.check(len(crs) == 1 and crs == {spec.crs}, f"CRS uniform: {crs}")
    rep.check(len(cells) == 1 and cells == {spec.cell_m},
              f"cell size uniform: {cells} m")

    # Global-grid alignment. Origins must be exact multiples of the cell size,
    # or the same ground falls in different cells in different samples.
    misaligned = []
    for s in samples:
        tile = s["meta"].get("tile")
        if not tile:
            misaligned.append(f"{s['path'].name}: no tile in meta")
            continue
        if tile[0] % spec.cell_m or tile[1] % spec.cell_m:
            misaligned.append(f"{s['path'].name}: origin ({tile[0]}, {tile[1]})")
    rep.check(not misaligned, "tile origins snapped to the global grid",
              "" if not misaligned else "; ".join(misaligned[:3]))

    tiles = {tuple(s["meta"]["tile"]) for s in samples if s["meta"].get("tile")}
    rep.check(len(tiles) == 1, "all samples of this fire share one tile",
              "" if len(tiles) == 1 else f"{len(tiles)} distinct tiles found")

    orders = {tuple(s["channels"]) for s in samples}
    rep.check(len(orders) == 1, "channel order identical across samples")
    if len(orders) == 1:
        got = list(orders.pop())
        rep.check(got == list(F.CHANNELS),
                  f"channel order matches firegrid.CHANNELS ({len(got)} channels)",
                  "" if got == list(F.CHANNELS) else
                  f"first mismatch at index {next(i for i,(a,b) in enumerate(zip(got, F.CHANNELS)) if a != b)}")



def _window(meta: dict):
    """(start datetime, step hours, key) for a sample, daily or sub-daily.

    Samples built before sub-daily support carry only `date`; they are
    treated as 24-hour steps so old datasets still verify.
    """
    step = int(meta.get("step_hours", 24))
    key = meta.get("window_start") or meta.get("date")
    for fmt in ("%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            return datetime.strptime(key, fmt), step, key
        except (TypeError, ValueError):
            continue
    return None, step, key

def check_temporal(samples: list[dict], rep: Report) -> list[str]:
    parsed = []
    for smp in samples:
        t, step, key = _window(smp["meta"])
        if t is None:
            rep.warn(f"unparseable window in {smp['path'].name}: {key!r}")
        else:
            parsed.append((t, step, key))
    if not parsed:
        return []
    parsed.sort()
    steps = {st for _, st, _ in parsed}
    rep.check(len(steps) == 1, f"time step uniform: {sorted(steps)} h",
              "" if len(steps) == 1 else "mixed step sizes in one fire")
    step = parsed[0][1]
    times = [t for t, _, _ in parsed]

    rep.check(len(set(times)) == len(times), "no duplicate windows",
              "" if len(set(times)) == len(times) else
              f"{len(times) - len(set(times))} duplicates")
    gaps = [(times[i], times[i + 1]) for i in range(len(times) - 1)
            if times[i + 1] - times[i] != timedelta(hours=step)]
    unit = "day" if step == 24 else f"{step}-hour step"
    rep.check(not gaps,
              f"windows consecutive: {parsed[0][2]} .. {parsed[-1][2]} "
              f"({len(times)} samples, one per {unit})",
              "" if not gaps else
              "; ".join(f"{a:%Y-%m-%dT%H} -> {b:%Y-%m-%dT%H}" for a, b in gaps[:3]))
    return [k for _, _, k in parsed]


def check_content(samples: list[dict], rep: Report, F) -> dict:
    labels = np.stack([s["label"] for s in samples])
    values = set(np.unique(labels).tolist())
    rep.check(values <= {F.LABEL_NO_FIRE, F.LABEL_FIRE, F.LABEL_UNOBSERVED},
              f"label values within {{0, 1, 2}}: {sorted(values)}")

    counts = Counter(labels.ravel().tolist())
    total = labels.size
    rep.note(f"class balance: fire {counts.get(1,0):,} "
             f"({counts.get(1,0)/total*100:.2f}%), "
             f"unobserved {counts.get(2,0):,} "
             f"({counts.get(2,0)/total*100:.2f}%), "
             f"no-fire {counts.get(0,0):,}")

    # Per-channel coverage. An entirely-fill channel is exactly what a silent
    # Earth Engine failure produces, and it is invisible in the file listing.
    channels = samples[0]["channels"]
    stack = np.stack([s["features"] for s in samples])
    empty, sparse = [], []
    for i, name in enumerate(channels):
        finite = np.isfinite(stack[:, i]).mean()
        if finite == 0:
            empty.append(name)
        elif finite < 0.5:
            sparse.append(f"{name} {finite*100:.0f}%")
    rep.check(not empty, f"every channel carries data ({len(channels)} channels)",
              "" if not empty else f"ENTIRELY EMPTY: {', '.join(empty)}")
    if sparse:
        rep.warn("channels under 50% coverage", ", ".join(sparse))

    return {"class_counts": {str(k): int(v) for k, v in counts.items()},
            "empty_channels": empty}


def check_prev_mask_chain(samples: list[dict], rep: Report, F) -> None:
    """The one end-to-end test that the temporal chain is correct.

    prev_fire_mask at step t must equal the fire cells of step t-1. If it
    does not, either the ordering is wrong or the carry-forward logic
    misfired -- and nothing else in the pipeline would reveal that.
    """
    idx = list(samples[0]["channels"]).index("prev_fire_mask")
    by_time = {}
    for smp in samples:
        t, step, _ = _window(smp["meta"])
        if t is not None:
            by_time[t] = (smp, step)
    checked, mismatched, carried = 0, [], 0

    for t, (smp, step) in sorted(by_time.items()):
        prev = by_time.get(t - timedelta(hours=step))
        if prev is None:
            continue          # first labelled step; its prev came from lead-in
        prev_label = prev[0]["label"]
        declared = np.nan_to_num(smp["features"][idx]) > 0
        checked += 1
        if not np.array_equal(declared, prev_label == F.LABEL_FIRE):
            # A carry across a coverage gap holds the older mask on purpose.
            if (prev_label == F.LABEL_UNOBSERVED).any() or not (prev_label == F.LABEL_FIRE).any():
                carried += 1
            else:
                mismatched.append(smp["meta"].get("window_start") or smp["meta"].get("date"))

    rep.check(not mismatched,
              f"prev_fire_mask matches step t-1 ({checked} transitions checked)",
              "" if not mismatched else
              f"mismatched at {', '.join(mismatched[:4])}")
    if carried:
        rep.note(f"{carried} transition(s) carried across a coverage gap "
                 f"(expected; see --max-carry-days)")


def check_csv_correspondence(samples: list[dict], rep: Report, F) -> None:
    """Do the CSV and the .npz describe the same detections?

    The CSV written by --visualize carries every detection with a verdict.
    The rows with used_in_label=True must, for each day:
      - number exactly the `detections` recorded in that sample's meta, and
      - rasterize to exactly the label's fire cells.

    The second test is the strong one. It rebuilds each label from the CSV
    and compares cell by cell, so any drift between what the tables say and
    what the model trains on is caught -- a filter applied to one but not the
    other, a date-grouping mismatch, a stale CSV from an earlier run.
    """
    import pandas as pd
    fire_dir = samples[0]["path"].parent
    csvs = sorted((fire_dir / "tables").glob("*.csv")) if (fire_dir / "tables").is_dir() else []
    csvs = [c for c in csvs if not c.stem.endswith(("_centroids", "_clusters"))]
    if not csvs:
        rep.note("no tables/*.csv beside these samples (built without "
                 "--visualize); CSV correspondence not checked")
        return

    df = pd.read_csv(csvs[0])
    if "used_in_label" in df.columns:
        used = df[df["used_in_label"].astype(str).str.lower().isin(["true", "1"])]
        rep.note(f"CSV {csvs[0].name}: {len(df):,} rows, "
                 f"{len(used):,} used in labels")
    else:
        used = df
        rep.warn("CSV has no used_in_label column",
                 "built before detection filtering; comparing all rows")

    sub_daily = int(samples[0]["meta"].get("step_hours", 24)) < 24
    if sub_daily and "window" not in used.columns:
        rep.warn("sub-daily samples but the CSV has no window column",
                 "rebuild with --visualize so the audit trail carries window keys")
        return
    date_col = ("window" if "window" in used.columns
                else "local_date" if "local_date" in used.columns else "acq_date")
    count_bad, cell_bad, checked = [], [], 0
    for smp in samples:
        date = (smp["meta"].get("window_start") if date_col == "window"
                else smp["meta"].get("date"))
        tile = smp["meta"].get("tile")
        if not date or not tile:
            continue
        day = used[used[date_col].astype(str) == date]
        expected = smp["meta"].get("detections")
        if expected is not None and len(day) != int(expected):
            count_bad.append(f"{date}: csv {len(day)} vs npz {expected}")
        rebuilt = F.rasterize_fire_mask(day, tuple(tile)) > 0
        actual = smp["label"] == F.LABEL_FIRE
        if not np.array_equal(rebuilt, actual):
            diff = int((rebuilt != actual).sum())
            cell_bad.append(f"{date}: {diff} cells differ")
        checked += 1

    rep.check(not count_bad,
              f"CSV detection counts match .npz meta ({checked} windows)",
              "; ".join(count_bad[:3]))
    rep.check(not cell_bad,
              f"CSV rows rasterize to exactly each label's fire cells ({checked} windows)",
              "; ".join(cell_bad[:3]))

def check_tile_overlap(groups: dict[str, list[dict]], rep: Report) -> None:
    """Tiles from different fires that overlap are a leakage risk."""
    tiles = {}
    for fire, samples in groups.items():
        t = samples[0]["meta"].get("tile")
        if t:
            tiles[fire] = tuple(t)
    names = list(tiles)
    overlaps = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = tiles[names[i]], tiles[names[j]]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                frac = (ox * oy) / ((a[2]-a[0]) * (a[3]-a[1]))
                overlaps.append(f"{names[i]} / {names[j]}: {frac*100:.0f}%")
    if overlaps:
        rep.warn("tiles from different fires overlap",
                 "; ".join(overlaps) + "\n" + INFO +
                 "   split train/test by geography or date, NEVER by fire name")
    else:
        rep.check(True, "no tile overlap between fires")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--json", default=None, help="Write the results to this file")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip the QA sheets (qa_labels.png, qa_channels.png)")
    ap.add_argument("--strict", action="store_true",
                    help="Treat warnings as failures")
    args = ap.parse_args()

    root = Path(args.dataset_dir).expanduser()
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    F = _import_firegrid()
    rep = Report(args.strict)

    print("=" * 72)
    print(f"Dataset verification: {root.resolve()}")
    print("=" * 72)

    groups = load_all(root, F)
    if not groups:
        print(f"{FAIL} no .npz samples found")
        return 1

    total = sum(len(v) for v in groups.values())
    print(f"\n{len(groups)} fire(s), {total} samples\n")

    per_fire = {}
    for fire, samples in sorted(groups.items()):
        print("-" * 72)
        print(f"{fire}  ({len(samples)} samples)")
        print("-" * 72)
        check_spatial(samples, rep, F)
        dates = check_temporal(samples, rep)
        stats = check_content(samples, rep, F)
        check_prev_mask_chain(samples, rep, F)
        check_csv_correspondence(samples, rep, F)
        print(f"\n{INFO} -- against the real world --")
        truth = RC.run(samples, samples[0]["path"].parent, rep, F,
                       plots=not args.no_plots)
        per_fire[fire] = {"samples": len(samples), "dates": dates, **stats,
                          "ground_truth": truth}
        print()

    print("-" * 72)
    print("Across fires")
    print("-" * 72)
    check_tile_overlap(groups, rep)

    # The temporal footprint is reported rather than asserted, because it is
    # deliberately not uniform: the products have different cadences.
    print(f"\n{INFO} temporal footprint per channel (not uniform by design):")
    for ch in F.CHANNELS:
        if ch == "prev_fire_mask":
            window = "1 day (t-1)"
        elif ch in ("elevation", "slope", "landcover", "population") or ch.startswith("aspect"):
            window = "static"
        else:
            lb = F.LAYER_LOOKBACK_DAYS.get(ch)
            window = f"{lb}-day window" if lb else "1 day"
        print(f"{INFO}   {ch:<20} {window}")
    print(f"{INFO} labels are exactly one local calendar day; state this in "
          f"any methods write-up.")

    print("\n" + "=" * 72)
    if rep.failures:
        print(f"{len(rep.failures)} check(s) FAILED:")
        for f in rep.failures:
            print(f"  - {f}")
    else:
        print("All checks passed.")
    if rep.warnings:
        print(f"{len(rep.warnings)} warning(s):")
        for w in rep.warnings:
            print(f"  - {w}")
    print("=" * 72)

    if args.json:
        Path(args.json).write_text(json.dumps({
            "dataset": str(root.resolve()),
            "fires": per_fire,
            "failures": rep.failures,
            "warnings": rep.warnings,
        }, indent=2))
        print(f"Saved: {args.json}")

    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
