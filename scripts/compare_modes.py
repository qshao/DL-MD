#!/usr/bin/env python3
"""Compare validation reports across sampling modes.

Usage:
    python scripts/compare_modes.py baseline.json modeA.json [modeB.json ...]
"""
import argparse
import json


_METRICS = [
    ("relax_ratio", "kinetic",       "relax_ratio"),
    ("fes_js",      "thermodynamic", "fes_js"),
    ("pop_tv",      "thermodynamic", "pop_tv"),
    ("rmsf_corr",   "structural",    "rmsf_corr"),
    ("dist_js",     "structural",    "dist_js"),
]


def _mean(proteins, section, key):
    vals = [p[section][key] for p in proteins.values()
            if p[section].get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _pct(base, new):
    if base is None or new is None or base == 0:
        return "n/a"
    return f"{100 * (new - base) / abs(base):+.0f}%"


def _fmt(v):
    return "null" if v is None else f"{v:.3g}"


def main():
    ap = argparse.ArgumentParser(description="Compare validation mode reports")
    ap.add_argument("reports", nargs="+", help="JSON report files; first = baseline")
    args = ap.parse_args()

    loaded = []
    for path in args.reports:
        with open(path) as fh:
            loaded.append((path, json.load(fh)))

    names = [p.replace("validation_", "").replace(".json", "")
             for p, _ in loaded]
    W = max(12, max(len(n) for n in names) + 2)

    header = f"{'Metric':<14}" + "".join(f"{n:>{W}}" for n in names)
    if len(loaded) > 1:
        header += "".join(f"{'Δvs-' + names[0]:>{W}}" for n in names[1:])
    print(header)
    print("-" * len(header))

    base_proteins = loaded[0][1]["proteins"]
    for label, section, key in _METRICS:
        base_val = _mean(base_proteins, section, key)
        row = f"{label:<14}" + f"{_fmt(base_val):>{W}}"
        for _, rep in loaded[1:]:
            row += f"{_fmt(_mean(rep['proteins'], section, key)):>{W}}"
        for _, rep in loaded[1:]:
            row += f"{_pct(base_val, _mean(rep['proteins'], section, key)):>{W}}"
        print(row)


if __name__ == "__main__":
    main()
