"""
Plot single-peakness of the ISO PSD per frequency band, using the
DTWDistToSinglePeak column (dynamic-time-warping distance to an ideal single
peak; smaller = more single-peaked).

Reads iso_results_all_subjects_bands_with_DTWDist_<dataset>.csv, averages runs
within each subject (one value per SID per band), and draws one boxplot per
band, overlaid with the individual subjects (jittered strip) to show the
distribution. Outliers are not drawn separately (the strip shows every point).

Style mimics GraphPad Prism: clean sans-serif font, outward ticks, thick
spines, no grid.
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


DATASETS = ["MSSV", "OxfordBenchmark", "ShenLab"]
CSV_TEMPLATE = "iso_results_all_subjects_bands_with_DTWDist_{dataset}.csv"
OUT_TEMPLATE = "DTW_singlepeak_{dataset}.png"

VALUE_COL = "DTWDistToSinglePeak"
PLOT_COL = "single_peakness"  # = -log10(VALUE_COL), see plot_dataset()

# Band order (low -> high frequency) and frequency ranges for the x labels.
BAND_ORDER = ["delta", "theta", "alpha", "sigma", "beta", "gamma"]
BAND_FREQ = {
    "gamma": (30, 45),
    "beta": (15, 30),
    "sigma": (11, 15),
    "alpha": (8, 12),
    "theta": (5, 10),  # for mice
    "delta": (1, 4),
}
BAND_LABELS = [f"{b}\n({BAND_FREQ[b][0]}-{BAND_FREQ[b][1]}Hz)" for b in BAND_ORDER]

BOX_COLOR = "0.85"
MEDIAN_COLOR = "red"


def set_prism_style():
    """Approximate the GraphPad Prism look with matplotlib rcParams."""
    sns.set_style("ticks")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 13,
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "savefig.dpi": 300,
    })


def make_boxplot(ax, df):
    """Draw per-band boxplots of single-peakness, overlaid with individual points."""
    sns.boxplot(
        ax=ax, data=df, x="band_name", y=PLOT_COL, order=BAND_ORDER,
        color=BOX_COLOR, width=0.6, showfliers=False, linewidth=1.5,
        medianprops=dict(color=MEDIAN_COLOR, linewidth=2.5),
        whiskerprops=dict(color="0.3", linewidth=1.5),
        capprops=dict(color="0.3", linewidth=1.5),
        boxprops=dict(edgecolor="0.3", linewidth=1.5),
    )
    sns.stripplot(ax=ax, data=df, x="band_name", y=PLOT_COL,
                  order=BAND_ORDER, color="0.4", size=3, alpha=0.5,
                  jitter=0.2, zorder=2)

    ax.set_xticks(range(len(BAND_ORDER)))
    ax.set_xticklabels(BAND_LABELS)
    ax.set_xlabel("")
    ax.set_ylabel("Single-peakness (a.u.)\n(negative distance to a single-peak template)")
    # no y ticks / tick labels (arbitrary units)
    ax.set_yticks([])
    ax.tick_params(left=False, labelleft=False)
    sns.despine(ax=ax)


def plot_dataset(dataset):
    csv_path = CSV_TEMPLATE.format(dataset=dataset)
    out_path = OUT_TEMPLATE.format(dataset=dataset)

    df = pd.read_csv(csv_path)
    df = df[df["band_name"].isin(BAND_ORDER)].dropna(subset=[VALUE_COL])
    # average runs within each subject, so each SID contributes one value/band
    df = (df.groupby(["sid", "band_name"], as_index=False)[VALUE_COL].mean())
    # single-peakness = negative distance, on a log scale so gamma's large
    # spread doesn't swamp the other bands (arbitrary units; higher = more
    # single-peaked).
    df[PLOT_COL] = -np.log10(df[VALUE_COL])

    set_prism_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    make_boxplot(ax, df)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {os.path.abspath(out_path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*",
                        help=f"Dataset(s) to plot (choices: {', '.join(DATASETS)}; "
                             "default: all).")
    args = parser.parse_args()

    datasets = args.datasets or DATASETS
    for dataset in datasets:
        if dataset not in DATASETS:
            parser.error(f"invalid dataset {dataset!r}; choose from {DATASETS}")
        plot_dataset(dataset)


if __name__ == "__main__":
    main()
