#!/usr/bin/env python3
"""
export_hdf5 -- pack a built dataset into one HDF5 file for training.

    python export_hdf5.py ml_dataset -o wildfire.h5

WHY
---
build_dataset writes one .npz per fire-day, which is easy to inspect and
append to but awkward to train from: a data loader has to open hundreds of
files per epoch. HDF5 stores the whole dataset as a few large arrays with
random access, so a loader reads sample i without touching the rest, and
nothing needs to fit in memory.

LAYOUT
------
    /features            (N, 18, 64, 64) float32   chunked one sample per chunk
    /labels              (N, 64, 64)     uint8     0 no-fire, 1 fire, 2 unobserved
    /meta/fire           (N,)            str
    /meta/date           (N,)            str       YYYY-MM-DD, local
    /meta/window_start   (N,)            str       YYYY-MM-DDTHH for sub-daily steps
    /meta/step_hours     (N,)            int16
    /meta/tile           (N, 4)          float64   x0, y0, x1, y1 in the CRS
    /meta/detections     (N,)            int32
    /fire_id             (N, 64, 64)     uint16    0 none, 1 the named fire, 2.. others
    /meta/fire_names     (N,)            str       JSON {id: name} for that sample
    attrs                channels, crs, cell_m, tile_cells, label_classes,
                         source_dir, created, filter

fire_id numbers are per build folder: fire 2 under palisades_2025 is not
fire 2 under eaton_2025. /meta/fire_names says which fire each id is.

Samples are ordered by fire, then date, so consecutive indices are
consecutive days of the same fire.

Chunking is one sample per chunk: a loader fetching sample i decompresses
exactly one chunk. Chunking across samples would make random access -- the
normal training pattern -- decompress neighbours it immediately discards.

LOADING
-------
    import h5py, torch
    class FireDataset(torch.utils.data.Dataset):
        def __init__(self, path):
            self.path = path
            with h5py.File(path, "r") as f:
                self.n = f["labels"].shape[0]
        def __len__(self):
            return self.n
        def __getitem__(self, i):
            # open per-worker: h5py handles must not cross fork boundaries
            if not hasattr(self, "f"):
                self.f = h5py.File(self.path, "r")
            x = torch.from_numpy(self.f["features"][i])
            y = torch.from_numpy(self.f["labels"][i].astype("int64"))
            return x, y

Mask class 2 (unobserved) out of the loss: CrossEntropyLoss(ignore_index=2).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("-o", "--output", default=None,
                    help="Output file (default: <dataset_dir>/dataset.h5)")
    ap.add_argument("--no-compress", action="store_true",
                    help="Skip gzip. Larger file, faster reads.")
    args = ap.parse_args()

    try:
        import h5py
    except ImportError:
        print("h5py is required:  pip install h5py")
        return 1
    import firegrid as F

    root = Path(args.dataset_dir).expanduser()
    paths = sorted(root.rglob("*.npz"), key=lambda p: (p.parent.name, p.name))
    if not paths:
        print(f"No .npz samples under {root}")
        return 1
    out = Path(args.output) if args.output else root / "dataset.h5"

    # First pass: validate shape and channels before allocating anything.
    first_f, first_l, channels, first_meta = F.load_sample(paths[0])
    n = len(paths)
    print(f"{n} samples, features {first_f.shape}, labels {first_l.shape}")

    comp = None if args.no_compress else "gzip"
    with h5py.File(out, "w") as h5:
        feats = h5.create_dataset(
            "features", shape=(n, *first_f.shape), dtype="float32",
            chunks=(1, *first_f.shape), compression=comp,
            fillvalue=np.nan)
        labs = h5.create_dataset(
            "labels", shape=(n, *first_l.shape), dtype="uint8",
            chunks=(1, *first_l.shape), compression=comp)
        str_t = h5py.string_dtype("utf-8")
        m_fire = h5.create_dataset("meta/fire", shape=(n,), dtype=str_t)
        m_date = h5.create_dataset("meta/date", shape=(n,), dtype=str_t)
        m_win = h5.create_dataset("meta/window_start", shape=(n,), dtype=str_t)
        m_step = h5.create_dataset("meta/step_hours", shape=(n,), dtype="int16")
        m_tile = h5.create_dataset("meta/tile", shape=(n, 4), dtype="float64")
        m_det = h5.create_dataset("meta/detections", shape=(n,), dtype="int32")
        fids = h5.create_dataset(
            "fire_id", shape=(n, *first_l.shape), dtype="uint16",
            chunks=(1, *first_l.shape), compression=comp)
        m_names = h5.create_dataset("meta/fire_names", shape=(n,), dtype=str_t)

        crs_seen, filters_seen = set(), set()
        for i, path in enumerate(paths):
            f, l, ch, meta = F.load_sample(path)
            # Refuse to mix incompatible samples into one training file.
            if f.shape != first_f.shape or list(ch) != list(channels):
                print(f"ABORT: {path.name} has shape {f.shape} / different "
                      f"channels. Run verify_dataset.py first.")
                h5.close()
                out.unlink(missing_ok=True)
                return 1
            feats[i] = f
            labs[i] = l
            m_fire[i] = meta.get("fire", path.parent.name)
            m_date[i] = meta.get("date", "")
            m_win[i] = meta.get("window_start", meta.get("date", ""))
            m_step[i] = int(meta.get("step_hours", 24))
            m_tile[i] = meta.get("tile", [np.nan] * 4)
            m_det[i] = int(meta.get("detections", -1))
            fid = F.load_fire_id(path)
            if fid is not None:          # older samples: zeros, names "{}"
                fids[i] = fid
            m_names[i] = json.dumps({str(f["id"]): f["name"]
                                     for f in meta.get("fires", [])})
            crs_seen.add(meta.get("crs"))
            filters_seen.add(meta.get("filter", "unrecorded"))
            if (i + 1) % 50 == 0 or i + 1 == n:
                print(f"  {i + 1}/{n}")

        if len(crs_seen) != 1:
            print(f"WARNING: samples declare {len(crs_seen)} different CRSs: "
                  f"{crs_seen}")
        h5.attrs["channels"] = json.dumps(list(channels))
        h5.attrs["crs"] = next(iter(crs_seen)) or ""
        h5.attrs["cell_m"] = float(first_meta.get("cell_m", F.GRID_SPEC.cell_m))
        h5.attrs["tile_cells"] = int(F.GRID_SPEC.tile_cells)
        h5.attrs["label_classes"] = json.dumps(
            {"0": "no fire", "1": "fire", "2": "unobserved (ignore in loss)"})
        h5.attrs["filter"] = json.dumps(sorted(filters_seen))
        h5.attrs["source_dir"] = str(root.resolve())
        h5.attrs["created"] = datetime.now().isoformat(timespec="seconds")

    size = out.stat().st_size / 1e6
    raw = sum(p.stat().st_size for p in paths) / 1e6
    print(f"\nSaved: {out}  ({size:.1f} MB; the .npz files total {raw:.1f} MB)")
    if len(filters_seen) > 1:
        print(f"NOTE: samples were built with {len(filters_seen)} different "
              f"filter settings -- recorded in attrs['filter'].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
