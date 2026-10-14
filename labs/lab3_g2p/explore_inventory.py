"""Explore the phone inventory and durations — lab 3, stage 2, question 1.

Reads every TextGrid, collects the duration of each phone, aggregates per phone type,
and writes `data/phone_stats.csv` plus a small console summary.

Run from the lab directory::

    python explore_inventory.py
"""
import glob
import os
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import tqdm

from praatio import textgrid

ALIGN_DIR = '../../data/RUSLAN_align_v2/'
PHONES_MFA = '../../common/phones_mfa.txt'
OUT_PATH = 'data/phone_stats.csv'

SIL = '<SIL>'
LENGTH_MARK = 'ː'


def load_target_inventory(path: str) -> list[str]:
    """Read the target phone list (one per line, minus trailing whitespace)."""
    with open(path, encoding='utf-8') as fh:
        return [line.strip() for line in fh if line.strip()]


def collect_durations(align_dir: str) -> dict[str, list[float]]:
    """Iterate every TextGrid and collect phone durations per phone label.

    Returns:
        `phone -> list of durations (seconds)`. An empty label becomes `<SIL>`,
        the length mark `ː` is kept (it is part of the target inventory), NFC applied.
    """
    durations: dict[str, list[float]] = defaultdict(list)
    paths = sorted(glob.glob(os.path.join(align_dir, '*.TextGrid')))
    for path in tqdm.tqdm(paths):
        try:
            tg = textgrid.openTextgrid(path, True)
        except Exception as exc:
            print(f'failed {os.path.basename(path)}: {exc}')
            continue
        _, phone_tier = tg.tiers
        for entry in phone_tier.entries:
            label = unicodedata.normalize('NFC', entry.label) if entry.label else SIL
            durations[label].append(entry.end - entry.start)
    return durations


def main() -> None:
    target = load_target_inventory(PHONES_MFA)
    print(f'target inventory: {len(target)} phones')
    print(f'target: {" ".join(target)}')

    durations = collect_durations(ALIGN_DIR)
    print(f'\nphones observed in corpus: {len(durations)}')

    rows = []
    for phone, dur in durations.items():
        arr = np.asarray(dur, dtype=float)
        rows.append({
            'phone': phone,
            'count': len(arr),
            'total_s': float(arr.sum()),
            'mean_s': float(arr.mean()),
            'std_s': float(arr.std()),
            'min_s': float(arr.min()),
            'max_s': float(arr.max()),
            'in_target': phone in target,
        })
    df = pd.DataFrame(rows).sort_values('count', ascending=False)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f'\nwrote {OUT_PATH} ({len(df)} phones)')

    # Missing from corpus but present in target
    observed = set(durations) | {SIL}
    missing = [p for p in target if p not in observed]
    print(f'\ntarget phones missing from corpus: {missing}')

    # Observed but not in target
    extra = sorted(p for p in durations if p not in target and p != SIL and p != 'spn')
    print(f'observed phones not in target (excluding SIL, spn): {extra}')

    # Summary stats
    print('\ntotal phone instances:', int(df["count"].sum()))
    print(f'mean phone duration: {np.average(df["mean_s"], weights=df["count"]):.4f} s')

    print('\n--- top-10 by count ---')
    print(df.head(10).to_string(index=False))
    print('\n--- 10 rarest by count ---')
    print(df.tail(10).to_string(index=False))


if __name__ == '__main__':
    main()