#!/usr/bin/env python3
"""
Six scatter plots relating the fitted stream-power exponents (m, c, n) from
unweightedflowplots.py / weightedflowplots.py's Plot 1/2 fits to age and to
each fit's sample size (n points):

  1. age vs m      4. n_points vs m
  2. age vs c      5. n_points vs c
  3. age vs n      6. n_points vs n

One point per VOLCANO per formulation (not per island -- each of the 7
islands is broken into its constituent named volcanoes via
usefulfunctions.nearest_volcano_age's VOLCANO tag, e.g. Hawaii splits into
Kohala/Mauna Kea/Hualalai/Mauna Loa/Kilauea, Oahu into Waianae/Koolau, etc.
-- giving ~14 points per formulation instead of 7). Each plot overlays both
formulations so they're directly comparable:

  - unweighted (A = raw D8 cell-count drainage area proxy, from
    unweightedflowplots.py's included points in
    all_islands_unweighted_stream_points.shp)
  - weighted   (Q = maxflux precipitation-weighted discharge proxy, from
    weightedflowplots.py's included points in
    all_islands_maxflux_stream_points.shp)

m is the single shared exponent from the d vs AS / d vs QS fit (Plot 1):
    d = k*(AS)^m   or   d = k*(QS)^m
c and n are the separate exponents from the multivariate d vs A^c*S^n /
d vs Q^c*S^n fit (Plot 2):
    d = k*A^c*S^n   or   d = k*Q^c*S^n

This script recomputes both fits itself, per volcano (rather than calling
unweightedflowplots.py's / weightedflowplots.py's own make_plot_* functions,
which fit per island and also save PNGs), so it can label both formulations'
multivariate X exponent 'c' uniformly -- unweightedflowplots.py's own copy
of that fit still names it 'm' internally (only weightedflowplots.py's was
renamed).

age is each volcano's own AVG_AGE (kyr) from ageprocessing.py's
volcano_avg_age_regions.shp (a single fixed value per named volcano -- every
point assigned to that volcano by nearest_volcano_age shares it, so no
median/aggregation is needed the way a whole-island age would). n_points is
that volcano's own fit sample size. Volcanoes with fewer than
MIN_POINTS_PER_VOLCANO points are skipped (too few to fit 3 free parameters
-- b, c, n -- in the multivariate fit).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from usefulfunctions import nearest_volcano_age
from unweightedflowplots import island_files as UNWEIGHTED_ISLAND_FILES, get_island_A_S_d
from weightedflowplots import island_files as WEIGHTED_ISLAND_FILES, get_island_Q_S_d

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = f'{base_dir}/Temp Output Placements/flow_accumulation_plots/exponent_vs_age_and_n'
os.makedirs(output_dir, exist_ok=True)

ISLANDS = list(UNWEIGHTED_ISLAND_FILES.keys())
assert ISLANDS == list(WEIGHTED_ISLAND_FILES.keys()), \
    "unweightedflowplots.py and weightedflowplots.py must cover the same islands"

MIN_POINTS_PER_VOLCANO = 30


def single_exponent_fit(X, S, d):
    """Same fit as make_plot_d_AS / make_plot_d_QS (Plot 1), minus the
    plotting -- d = k*(XS)^m. Returns (m, n_points)."""
    XS = X * S
    log_XS, log_d = np.log10(XS), np.log10(d)
    m, b, r, p, se = stats.linregress(log_XS, log_d)
    return m, len(d)


def multivariate_fit(X, S, d):
    """Same fit as make_plot_d_AmSn / make_plot_d_QcSn (Plot 2), minus the
    plotting -- d = k*X^c*S^n. Returns (c, n_exp, n_points)."""
    log_X, log_S, log_d = np.log10(X), np.log10(S), np.log10(d)
    n_pts = len(log_d)
    M = np.column_stack([np.ones(n_pts), log_X, log_S])
    beta, _, _, _ = np.linalg.lstsq(M, log_d, rcond=None)
    b, c, n_exp = beta
    return c, n_exp, n_pts


def collect_results(get_island_X_S_d, island_files, label):
    """Returns a list of per-volcano dicts with m, c, n, n_pts (that
    volcano's own fit sample size), age_kyr (that volcano's own AVG_AGE),
    volcano (the VOLCANO tag), and island (which island it belongs to, for
    reference). Skips any volcano with fewer than MIN_POINTS_PER_VOLCANO
    valid, age-tagged points."""
    results = []
    for island in island_files:
        X, S, d, xs, ys, crs = get_island_X_S_d(island)
        if len(X) < 10:
            print(f"[{label}/{island}] too few valid points ({len(X)}), skipping")
            continue

        age, volcano_tag = nearest_volcano_age(xs, ys, crs, return_tags=True)
        valid = np.isfinite(age) & (age > 0) & (volcano_tag != '')
        X, S, d, age, volcano_tag = X[valid], S[valid], d[valid], age[valid], volcano_tag[valid]

        for tag in np.unique(volcano_tag):
            mask = volcano_tag == tag
            n_here = int(mask.sum())
            if n_here < MIN_POINTS_PER_VOLCANO:
                print(f"[{label}/{island}/{tag}] too few points ({n_here}), skipping")
                continue

            X_v, S_v, d_v, age_v = X[mask], S[mask], d[mask], age[mask]
            m, n_pts = single_exponent_fit(X_v, S_v, d_v)
            c, n_exp, _ = multivariate_fit(X_v, S_v, d_v)
            age_kyr = age_v[0] / 1000.0  # fixed per volcano -- identical for every point in the group

            results.append({
                'volcano': tag, 'island': island.capitalize(), 'm': m, 'c': c,
                'n': n_exp, 'n_pts': n_pts, 'age_kyr': age_kyr,
            })
            print(f"[{label}/{island}/{tag}] m={m:.4f}  c={c:.4f}  n={n_exp:.4f}  "
                  f"n_pts={n_pts:,}  age={age_kyr:,.1f} kyr")
    return results


def _print_table(label, rows):
    print(f"\n--- {label} ---")
    header = f"{'Volcano':<10} {'Island':<12} {'m':>8} {'c':>8} {'n':>8} {'n_pts':>10} {'age (kyr)':>10}"
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['volcano']:<10} {r['island']:<12} {r['m']:>8.4f} {r['c']:>8.4f} {r['n']:>8.4f} "
              f"{r['n_pts']:>10,} {r['age_kyr']:>10,.1f}")


def _scatter_plot(x_key, y_key, x_label, y_label, title, out_name,
                   unweighted_rows, weighted_rows, log_x=True):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    ux = [r[x_key] for r in unweighted_rows]
    uy = [r[y_key] for r in unweighted_rows]
    wx = [r[x_key] for r in weighted_rows]
    wy = [r[y_key] for r in weighted_rows]

    ax.scatter(ux, uy, s=70, marker='s', color='#c96a3a', edgecolors='black',
               linewidths=0.6, label='Unweighted (A)', zorder=3)
    ax.scatter(wx, wy, s=70, marker='o', color='#3a6ea5', edgecolors='black',
               linewidths=0.6, label='Weighted (Q)', zorder=3)

    for r in unweighted_rows:
        ax.annotate(r['volcano'], (r[x_key], r[y_key]), fontsize=7,
                    color='#c96a3a', xytext=(5, 4), textcoords='offset points')
    for r in weighted_rows:
        ax.annotate(r['volcano'], (r[x_key], r[y_key]), fontsize=7,
                    color='#3a6ea5', xytext=(5, -10), textcoords='offset points')

    if log_x:
        ax.set_xscale('log')
    ax.minorticks_on()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    out_path = f'{output_dir}/{out_name}.png'
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    unweighted_rows = collect_results(get_island_A_S_d, UNWEIGHTED_ISLAND_FILES, 'unweighted')
    weighted_rows = collect_results(get_island_Q_S_d, WEIGHTED_ISLAND_FILES, 'weighted')

    _print_table('Unweighted (A) per-volcano exponents', unweighted_rows)
    _print_table('Weighted (Q) per-volcano exponents', weighted_rows)

    plots = [
        ('age_kyr', 'm', 'Volcano Representative Age (kyr)', 'm (single exponent, d vs XS)',
         'Age vs m', 'age_vs_m'),
        ('age_kyr', 'c', 'Volcano Representative Age (kyr)', 'c (X exponent, d vs X^c S^n)',
         'Age vs c', 'age_vs_c'),
        ('age_kyr', 'n', 'Volcano Representative Age (kyr)', 'n (S exponent, d vs X^c S^n)',
         'Age vs n', 'age_vs_n'),
        ('n_pts', 'm', 'Fit sample size (n points)', 'm (single exponent, d vs XS)',
         '#Points vs m', 'npts_vs_m'),
        ('n_pts', 'c', 'Fit sample size (n points)', 'c (X exponent, d vs X^c S^n)',
         '#Points vs c', 'npts_vs_c'),
        ('n_pts', 'n', 'Fit sample size (n points)', 'n (S exponent, d vs X^c S^n)',
         '#Points vs n', 'npts_vs_n'),
    ]
    for x_key, y_key, x_label, y_label, title, out_name in plots:
        _scatter_plot(x_key, y_key, x_label, y_label, title, out_name,
                      unweighted_rows, weighted_rows)


if __name__ == '__main__':
    main()
