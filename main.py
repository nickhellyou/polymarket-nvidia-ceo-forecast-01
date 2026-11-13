# Polymarket "Next CEO of NVIDIA?" forecaster (program/v1, Python 3.13 stdlib only).
#
# Reads one case at a time from /input (<id>.in, canonical JSON
# makefun/polymarket-case/v1: every trading market's Yes price on a 1-minute
# grid for the 72 hours up to the checkpoint) and writes /output/<id>.out with
# five quantiles (5/25/50/75/95%) of each market's price for each of the 240
# minutes after the checkpoint. Offline, deterministic, no model calls.
#
# Model: persistence median with an adaptive random-walk spread, per market.
#   q_tau(h) = clamp(x + z_tau * sigma * sqrt(h), 0, 1), rounded to 6 decimals
# where x is the market's last price and sigma is that market's own recent
# realized 1-minute volatility, estimated from its last 120 minutes as the
# average of the sample standard deviation and a MAD-based robust deviation
# of the 1-minute price changes, clipped to [0.0005, 0.005] per sqrt-minute.
# The house baseline uses a fixed sigma of 0.0025 for every market; the
# adaptive sigma is the whole edge: most minutes these prices barely move
# (realized sigma well under 0.0025), while a few markets in a news burst
# move several times faster, and a fixed spread is too wide for the former
# and too narrow for the latter. The inner quantiles are narrowed slightly
# (x0.85) and the outer ones widened slightly (x1.15): minute changes are
# mostly tiny with occasional jumps, so the middle of the distribution is
# tighter than a normal's and the tails fatter.
#
# What was tried and rejected on the six practice cases (backtest: forecast
# case p0k, score against the next case p0k+1's grid for the 240 minutes,
# mean pinball loss in percentage points; house baseline = 0.398 on the
# same 5 transitions):
#   - drift from the recent trend (any window 60-1440 min, any scale,
#     including contrarian): always worse (0.39-0.74). Trends reverse.
#   - mean reversion to a trailing mean: worse (0.44).
#   - long-window volatility (1440 min) or multi-hour change volatility:
#     much worse (up to 1.03). Volatility regimes change fast here.
#   - sigma scaled by the price level sqrt(4x(1-x)): worse (0.32).
#   - horizon exponent other than sqrt(h): worse (0.32).
# Chosen config backtests at 0.310 mean pinball loss vs 0.398 baseline
# (per-transition: 0.469 / 0.311 / 0.333 / 0.237 / 0.201). The house reports
# the baseline at 0.363589 on the full 6-case practice; the backtest is not
# directly comparable (different case set/weighting) but the relative gain
# is consistent in every transition.

import json
import math
import os
import statistics
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at each level, inner pair narrowed x0.85 and
# outer pair widened x1.15 (see header).
Z = [-1.644854 * 1.15, -0.67449 * 0.85, 0.0, 0.67449 * 0.85, 1.644854 * 1.15]

VOL_WINDOW = 120      # minutes of 1-minute changes used for sigma
SIGMA_FLOOR = 0.0005  # per sqrt-minute
SIGMA_CAP = 0.005     # per sqrt-minute
SIGMA_FALLBACK = 0.0025  # house baseline value, used when too little history


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def estimate_sigma(prices):
    """Per-market 1-minute volatility from the last VOL_WINDOW changes."""
    diffs = [b - a for a, b in zip(prices, prices[1:])][-VOL_WINDOW:]
    if len(diffs) < 5:
        return SIGMA_FALLBACK
    std = statistics.pstdev(diffs)
    med = statistics.median(diffs)
    mad = statistics.median([abs(d - med) for d in diffs]) * 1.4826
    sigma = 0.5 * std + 0.5 * mad
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


def forecast(prices, horizon):
    """240 (or horizon) rows of five quantiles for one market."""
    if not prices:
        return [[0.5] * 5 for _ in range(horizon)]
    x = prices[-1]
    sigma = estimate_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        row = [price6(x + z * spread) for z in Z]
        # enforce non-decreasing quantiles after rounding/clamping
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(Path(INPUT_DIR).glob("*.in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        with path.open("r", encoding="utf-8") as f:
            case = json.load(f)
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(
                f"{path.name} asks for quantiles {case['quantiles']}, not {QUANTILES}"
            )
        horizon = int(case["horizon_minutes"])
        forecasts = [
            {"id": m["id"], "q": forecast(m["prices"], horizon)}
            for m in case["markets"]
        ]
        out = out_dir / f"{case_id}.out"
        with out.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
