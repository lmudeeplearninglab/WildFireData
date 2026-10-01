#!/usr/bin/env python3
"""
swin_data -- prepare the extracted dataset for Swin-UNet.

The extraction pipeline produces physically meaningful tensors. A transformer
needs them in a different form, and four things go wrong silently if the
conversion is done naively.

1. GEOMETRY
   Swin-UNet was published at 224 x 224 inputs, 4 x 4 patches and a 7 x 7
   attention window. On a 64 x 64 tile that gives a 16 x 16 token grid,
   which a window of 7 cannot divide -- the reference implementation crashes.
   Patch 2 / window 4 gives token grids of 32, 16, 8, 4 across four stages,
   all divisible. It also keeps the finest token at 2 x 2 km instead of
   4 x 4 km, which matters when an hour of spread is one or two cells.
   `check_geometry()` validates any configuration before you train.

2. MISSING VALUES
   The tensors use NaN for "no data". A transformer propagates NaN through
   every attention product and the loss becomes NaN on the first step.
   `normalize()` standardizes each channel with TRAINING-split statistics
   and fills missing values with the channel mean (0 after standardization).

3. CHANNEL TYPES -- not everything should be standardized
   - scalar      (elevation, erc, ...)       mean-centred, divided by std
   - vector      (wind_u/v, aspect_sin/cos)  scaled by a shared magnitude,
                                             NEVER mean-centred: subtracting
                                             a mean wind moves the vector
                                             origin, so rotating the tile no
                                             longer rotates the wind
   - bounded     (aspect_consistency)        left in [0, 1]
   - binary      (prev_fire_mask)            left as 0/1
   - categorical (landcover)                 left as class codes; one-hot
                                             or embed it in the model. As a
                                             float it implies class 90 is
                                             "more" than class 10.

4. AUGMENTATION MUST ROTATE THE VECTORS
   The Swin-UNet paper augments with flips and rotations. Rotating the image
   without rotating the wind produces a sample where the wind blows one way
   and the terrain says another -- a contradictory example the model will
   try to learn. `augment()` rotates the (east, north) vector channels
   together with the grid. `test_augmentation()` proves it: after any
   rotation or flip, wind that blew uphill still blows uphill.

Usage:
    python swin_data.py check --tile 64 --patch 2 --window 4
    python swin_data.py stats dataset.h5 --train-fires palisades_2025
    python swin_data.py selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Channel roles. Names must match firegrid.CHANNELS.
VECTOR_PAIRS = [("wind_u", "wind_v"), ("aspect_sin", "aspect_cos")]   # (east, north)
BINARY = {"prev_fire_mask"}
CATEGORICAL = {"landcover"}
BOUNDED = {"aspect_consistency"}
IGNORE_INDEX = 2        # label class 2 = unobserved; exclude from the loss


# ---------------------------------------------------------------------------
# 1. Geometry
# ---------------------------------------------------------------------------
def check_geometry(tile: int = 64, patch: int = 2, window: int = 4,
                   stages: int = 4) -> tuple[bool, str]:
    """Validate a Swin-UNet configuration against the tile size."""
    if tile % patch:
        return False, f"tile {tile} is not divisible by patch {patch}"
    sizes = [tile // patch // (2 ** i) for i in range(stages)]
    if sizes[-1] < 1:
        return False, f"too many stages: token grids {sizes}"
    bad = [s for s in sizes if s >= window and s % window]
    small = [s for s in sizes if s < window]
    km = tile // (tile // patch)
    msg = (f"token grids per stage {sizes}; finest token = {patch} x {patch} "
           f"cells" + (f" (~{km} km at 1 km cells)" if km else ""))
    if bad:
        return False, f"{msg}\n  FAILS: {bad} not divisible by window {window}"
    if small:
        return True, (f"{msg}\n  runs, but stage(s) {small} are smaller than the "
                      f"window, so attention there degenerates to global")
    return True, f"{msg}\n  clean: every stage divides by window {window}"


# ---------------------------------------------------------------------------
# 2-3. Normalization
# ---------------------------------------------------------------------------
def compute_norm_stats(features: np.ndarray, channels: list[str]) -> dict:
    """Per-channel statistics from TRAINING samples only.

    features: (N, C, H, W). Computing these over the whole dataset leaks the
    test split into training through the normalization -- small, but real,
    and exactly the kind of thing reviewers check.
    """
    stats: dict = {"channels": list(channels), "per_channel": {}}
    idx = {c: i for i, c in enumerate(channels)}
    in_pair = {c for pair in VECTOR_PAIRS for c in pair}

    for name, i in idx.items():
        x = features[:, i].astype("float64")
        finite = x[np.isfinite(x)]
        if name in in_pair:
            continue                         # handled per pair below
        if name in BINARY | CATEGORICAL | BOUNDED or finite.size == 0:
            role = ("binary" if name in BINARY else
                    "categorical" if name in CATEGORICAL else "bounded")
            stats["per_channel"][name] = {"role": role, "mean": 0.0, "std": 1.0,
                                          "fill": 0.0}
            continue
        mean, std = float(finite.mean()), float(finite.std())
        stats["per_channel"][name] = {"role": "scalar", "mean": mean,
                                      "std": std if std > 1e-9 else 1.0,
                                      "fill": mean}

    for a, b in VECTOR_PAIRS:
        if a not in idx or b not in idx:
            continue
        u = features[:, idx[a]].astype("float64")
        v = features[:, idx[b]].astype("float64")
        mag = np.sqrt(u ** 2 + v ** 2)
        mag = mag[np.isfinite(mag)]
        scale = float(np.sqrt((mag ** 2).mean())) if mag.size else 1.0
        for name in (a, b):
            stats["per_channel"][name] = {"role": "vector", "mean": 0.0,
                                          "std": scale if scale > 1e-9 else 1.0,
                                          "fill": 0.0}
    return stats


def normalize(x: np.ndarray, stats: dict) -> np.ndarray:
    """Standardize one sample (C, H, W) and fill missing values.

    Missing values become the channel mean, which is 0 after centring, so the
    model sees "average" rather than NaN. Vector channels are only scaled.
    """
    out = np.empty_like(x, dtype="float32")
    for i, name in enumerate(stats["channels"]):
        s = stats["per_channel"][name]
        ch = x[i].astype("float32")
        ch = np.where(np.isfinite(ch), ch, s["fill"])
        out[i] = (ch - s["mean"]) / s["std"]
    return out


def ignore_other_fires(y: np.ndarray, fire_id: np.ndarray,
                       keep: tuple[int, ...] = (1,)) -> np.ndarray:
    """Label with other fires' cells set to IGNORE_INDEX.

    Labels keep every fire as fire, because spread is spread. This is the
    switch for an experiment that should learn from the named fire only:
    cells of other fires (fire_id not in `keep`) drop out of the loss instead
    of being called no-fire, which would be false. Apply BEFORE augment(),
    so fire_id never needs rotating.
    """
    out = np.array(y, copy=True)
    out[(fire_id > 0) & ~np.isin(fire_id, keep)] = IGNORE_INDEX
    return out


# ---------------------------------------------------------------------------
# 4. Direction-aware augmentation
# ---------------------------------------------------------------------------
def _rotate_vec(e: np.ndarray, n: np.ndarray, k: int):
    """Rotate (east, north) components counter-clockwise by k * 90 deg."""
    for _ in range(k % 4):
        e, n = -n, e
    return e, n


def augment(x: np.ndarray, y: np.ndarray, channels: list[str],
            k: int = 0, flip_lr: bool = False, flip_ud: bool = False):
    """Rotate/flip a sample, rotating vector channels to match.

    Grid convention: row 0 is north, column 0 is west (north-up, as written
    by firegrid). np.rot90 with k=1 turns the map counter-clockwise, so a
    vector pointing east must end up pointing north.
    """
    idx = {c: i for i, c in enumerate(channels)}
    x = np.rot90(x, k=k, axes=(-2, -1))
    y = np.rot90(y, k=k, axes=(-2, -1))
    if flip_lr:
        x, y = x[..., ::-1], y[..., ::-1]
    if flip_ud:
        x, y = x[..., ::-1, :], y[..., ::-1, :]
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    for a, b in VECTOR_PAIRS:
        if a not in idx or b not in idx:
            continue
        # Copy first. For k=1 the rotated north component IS the original
        # east array (no negation creates a new one), so writing east back
        # before north would overwrite the value north is about to take --
        # an aliasing bug the self-test caught at k=1 only.
        e, n = _rotate_vec(x[idx[a]].copy(), x[idx[b]].copy(), k)
        if flip_lr:
            e = -e                 # mirroring east-west reverses east
        if flip_ud:
            n = -n                 # mirroring north-south reverses north
        x[idx[a]], x[idx[b]] = e, n
    return x, y


def random_augment(x, y, channels, rng: np.random.Generator):
    """One of the 8 dihedral transforms, vectors handled."""
    return augment(x, y, channels, k=int(rng.integers(4)),
                   flip_lr=bool(rng.integers(2)))


# ---------------------------------------------------------------------------
# Self-test: the invariant that matters
# ---------------------------------------------------------------------------
def _uphill_alignment(x: np.ndarray, channels: list[str]) -> float:
    """Cosine between the wind vector and the uphill direction of elevation."""
    idx = {c: i for i, c in enumerate(channels)}
    elev = x[idx["elevation"]]
    d_row, d_col = np.gradient(elev)
    up_e, up_n = d_col, -d_row                 # row index grows southward
    u, v = x[idx["wind_u"]], x[idx["wind_v"]]
    num = (up_e * u + up_n * v).mean()
    den = np.sqrt((up_e ** 2 + up_n ** 2).mean() * (u ** 2 + v ** 2).mean())
    return float(num / den)


def test_augmentation() -> None:
    """Wind that blows uphill must still blow uphill after any transform."""
    channels = ["elevation", "wind_u", "wind_v", "aspect_sin", "aspect_cos"]
    H = 64
    rows, cols = np.mgrid[0:H, 0:H].astype(float)
    # Terrain rising toward the north-east, wind blowing north-east (uphill).
    elev = 1000 - 5 * rows + 3 * cols
    x = np.stack([elev,
                  np.full((H, H), 3.0), np.full((H, H), 5.0),     # wind e, n
                  np.full((H, H), 0.6), np.full((H, H), 0.8)])    # aspect
    y = np.zeros((H, H), np.uint8)
    base = _uphill_alignment(x, channels)
    assert base > 0.99, base

    worst_fixed, worst_naive = 1.0, 1.0
    for k in range(4):
        for flr in (False, True):
            for fud in (False, True):
                xa, _ = augment(x, y, channels, k, flr, fud)
                worst_fixed = min(worst_fixed, _uphill_alignment(xa, channels))
                # Naive: move the pixels, leave the vector values alone.
                xn = np.rot90(x, k=k, axes=(-2, -1))
                if flr:
                    xn = xn[..., ::-1]
                if fud:
                    xn = xn[..., ::-1, :]
                worst_naive = min(worst_naive, _uphill_alignment(
                    np.ascontiguousarray(xn), channels))
    assert worst_fixed > 0.99, worst_fixed
    print(f"  augmentation: wind stays aligned uphill under all 16 transforms "
          f"(worst cosine {worst_fixed:.3f})")
    print(f"  naive rotation, vectors untouched: worst cosine {worst_naive:.3f} "
          f"-- the wind ends up blowing downhill")


def test_ignore_other_fires() -> None:
    y = np.array([[1, 1, 0], [0, 2, 1]], dtype=np.uint8)
    fid = np.array([[1, 3, 0], [0, 0, 2]], dtype=np.uint16)
    got = ignore_other_fires(y, fid)
    assert got.tolist() == [[1, 2, 0], [0, 2, 2]], got
    assert ignore_other_fires(y, fid, keep=(1, 2, 3)).tolist() == y.tolist()
    print("  ignore_other_fires: other fires leave the loss, the named fire stays")


def test_normalization() -> None:
    channels = ["elevation", "wind_u", "wind_v", "landcover", "prev_fire_mask"]
    rng = np.random.default_rng(0)
    f = np.stack([rng.normal(500, 200, (8, 16, 16)),
                  rng.normal(4, 2, (8, 16, 16)), rng.normal(-3, 2, (8, 16, 16)),
                  rng.choice([10., 20., 30.], (8, 16, 16)),
                  rng.integers(0, 2, (8, 16, 16)).astype(float)], axis=1)
    f[0, 0, :4, :4] = np.nan
    st = compute_norm_stats(f, channels)
    z = normalize(f[0], st)
    assert np.isfinite(z).all(), "NaN survived normalization"
    assert abs(normalize(f[1], st)[0].mean()) < 0.5
    assert st["per_channel"]["wind_u"]["mean"] == 0.0, "vectors must not be centred"
    assert np.array_equal(z[3], f[0, 3]), "categorical must be untouched"
    assert np.array_equal(z[4], f[0, 4]), "binary must be untouched"
    print("  normalization: no NaN survives, vectors scaled not centred, "
          "categorical and binary untouched")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("check", help="Validate a Swin-UNet geometry")
    g.add_argument("--tile", type=int, default=64)
    g.add_argument("--patch", type=int, default=2)
    g.add_argument("--window", type=int, default=4)
    g.add_argument("--stages", type=int, default=4)
    s = sub.add_parser("stats", help="Normalization stats from the training fires")
    s.add_argument("h5")
    s.add_argument("--train-fires", nargs="+", required=True,
                   help="Fire names as stored in meta/fire, e.g. PALISADES")
    s.add_argument("-o", "--output", default="norm_stats.json")
    sub.add_parser("selftest", help="Prove augmentation and normalization")
    args = ap.parse_args()

    if args.cmd == "check":
        ok, msg = check_geometry(args.tile, args.patch, args.window, args.stages)
        print(msg)
        return 0 if ok else 1

    if args.cmd == "selftest":
        test_augmentation()
        test_normalization()
        test_ignore_other_fires()
        return 0

    import h5py
    with h5py.File(args.h5, "r") as f:
        fires = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in f["meta/fire"][:]])
        wanted = {w.upper() for w in args.train_fires}
        rows = np.where(np.isin(np.char.upper(fires.astype(str)), list(wanted)))[0]
        if rows.size == 0:
            print(f"No samples for {sorted(wanted)}. Fires present: "
                  f"{sorted(set(fires))}")
            return 1
        channels = json.loads(f.attrs["channels"])
        feats = f["features"][rows.tolist()]
    stats = compute_norm_stats(feats, channels)
    stats["train_fires"] = sorted(wanted)
    stats["n_samples"] = int(rows.size)
    Path(args.output).write_text(json.dumps(stats, indent=2))
    print(f"Stats from {rows.size} training samples -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
