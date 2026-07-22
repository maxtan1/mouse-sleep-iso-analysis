"""
Plot ISO (infraslow oscillation) PSD per frequency band as panels.

Each panel overlays one group-mean curve per series defined in ``SERIES``,
where a series selects a subset of subjects from one dataset:
  - "MSSV: WT, All Ages, Male"     -> MSSV subjects with sex == Male
  - "MSSV: WT, All Ages, Female"   -> MSSV subjects with sex == Female
  - "OxfordBenchmark: WT, Adult, Male" -> all OxfordBenchmark subjects
  - "Oscar: TG, Adult, Male"       -> all Oscar subjects
Sex for the MSSV series is read from ``mastersheet_MSSV.csv``.

Averaging scheme:
  1. Convert each ISO PSD to relative power: after interpolation onto the
     common frequency grid, divide by the sum across grid points and scale by
     100, so every curve sums to 100% over the grid (plain % of total power).
     This removes per-recording amplitude differences before averaging.
  2. Within each subject, average the relative-power curves across runs.
  3. Average the per-subject mean curves across subjects.

Because non-REM bouts (and hence the resulting FFTs) have different frequency
resolutions across runs/subjects, every PSD is first interpolated onto a common
frequency grid (NaN outside each PSD's own support) before averaging.

Style mimics GraphPad Prism: clean sans-serif font, outward ticks, thick
spines, no grid, individual subjects in light grey with the group mean in red.
"""

import os
import csv
import argparse
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns  # optional, used for the despine/ticks aesthetic


# Datasets that provide the ISO PSD pickles; the pickle file names are derived
# from the dataset name.
DATASETS = ["MSSV", "OxfordBenchmark", "Oscar"]#, "ShenLab"
iso_psd_type = 'raw'
PICKLE_TEMPLATE = "data_all_subjects_bands_{dataset}.pickle"
OUT_PATH = f"ISO_PSD_per_band.png"

# Per-dataset mastersheet; provides the canonical SID (used to average within a
# subject and to count sample size) and, for MSSV, the sex column.
MASTERSHEET_TEMPLATE = "mastersheet_{dataset}.csv"
MSSV_MASTERSHEET = MASTERSHEET_TEMPLATE.format(dataset="MSSV")

# One overlaid curve per series in every panel. "sex" filters MSSV subjects by
# the sex column of the mastersheet; None means "use all subjects" (the
# OxfordBenchmark / Oscar cohorts already satisfy the stated criteria).
# Nature Publishing Group (NPG) palette from ggsci. The two MSSV sexes are the
# paired NPG reds (deep red / salmon); the other datasets get the NPG teal and
# navy for strong, warm-vs-cool contrast.
SERIES = [
    {"label": "MSSV: WT, All Ages, M",        "dataset": "MSSV",            "sex": "Male",   "color": "#DC0000"},
    {"label": "MSSV: WT, All Ages, F",        "dataset": "MSSV",            "sex": "Female", "color": "#F39B7F"},
    {"label": "OxfordBenchmark: WT, Adult, M","dataset": "OxfordBenchmark", "sex": None,     "color": "#00A087"},
    {"label": "Oscar: TG (LepR-Cre), Adult, M",          "dataset": "Oscar",           "sex": None,     "color": "#3C5488"},
]

# Band order for the panels (rows x cols), low -> high frequency.
BAND_ORDER = ["delta", "theta", "alpha", "sigma", "beta", "gamma"]

# Greek symbol for each band name, shown next to the band label.
BAND_GREEK = {
    "delta": "δ",
    "theta": "θ",
    "alpha": "α",
    "sigma": "σ",
    "beta": "β",
    "gamma": "γ",
}

# Frequency range (Hz) of each band, for the panel labels.
BAND_FREQ = {
    "gamma": (30, 45),
    "beta": (15, 30),
    "sigma": (11, 15),
    "alpha": (8, 12),
    "theta": (5, 10),  # for mice
    "delta": (1, 4),
}

# Common frequency grid (Hz) onto which every PSD is interpolated before
# averaging, and the x-range that is actually displayed.
COMMON_FREQ = np.linspace(0.005, 0.1, 200)


def set_prism_style():
    """Approximate the GraphPad Prism look with matplotlib rcParams."""
    sns.set_style("ticks")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 17,
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


def to_percent(psd):
    """Normalize a (common-grid) PSD to % of total power.

    Divides by the sum across grid points and scales by 100, so the curve sums
    to 100%, removing per-recording amplitude differences. NaNs (grid points
    outside the recording's support) are ignored in the total.
    """
    psd = np.asarray(psd, dtype=float)
    total = np.nansum(psd)
    if not np.isfinite(total) or total <= 0:
        return np.full_like(psd, np.nan)
    return 100.0 * psd / total


def load_sex_map(mastersheet_path):
    """Return {subject ID: sex} from the mastersheet CSV."""
    sex_map = {}
    with open(mastersheet_path, newline="") as f:
        for row in csv.DictReader(f):
            sex_map[row["SID"]] = row["sex"]
    return sex_map


def load_canonical_sid_map(mastersheet_path):
    """Return a function mapping a pickle subject-ID to its canonical SID.

    The pickle's subject-ID is either the mastersheet SID itself (MSSV,
    OxfordBenchmark) or a recording/segment string that begins with the EDF
    basename stem (Oscar, where one animal SID such as ``ORP575`` yields several
    keys like ``ORP575_Inh-sham_6days_h0-23``). Collapsing to the mastersheet
    SID ensures runs are averaged within a subject and sample size counts unique
    subjects.
    """
    sids = set()
    stems = []  # (edf basename stem, canonical SID)
    with open(mastersheet_path, newline="") as f:
        for row in csv.DictReader(f):
            sids.add(row["SID"])
            stem = os.path.splitext(os.path.basename(row["EDFPath"]))[0]
            stems.append((stem, row["SID"]))
    # match the most specific (longest) stem first
    stems.sort(key=lambda s: len(s[0]), reverse=True)

    def canonical(pickle_sid):
        if pickle_sid in sids:
            return pickle_sid
        for stem, sid in stems:
            if pickle_sid.startswith(stem):
                return sid
        return pickle_sid  # fall back to the raw id if unmatched

    return canonical


def load_band_data(pickle_path, canonical=None):
    """Return {band: {subject: [(freq, psd), ...]}} from the pickle.

    ``canonical`` optionally maps each pickle subject-ID to a canonical SID so
    that multiple recordings/segments of the same subject are grouped together.
    """
    with open(pickle_path, "rb") as f:
        d = pickle.load(f)
    iso_freq = d["iso_freq"]
    avg_iso_psd = d["avg_iso_psd"]

    band_data = defaultdict(lambda: defaultdict(list))
    for key in iso_freq:
        sid, _, band = key  # key = (subject ID, edf path / run, band name)
        if canonical is not None:
            sid = canonical(sid)
        freq = np.asarray(iso_freq[key], dtype=float)
        psd = np.asarray(avg_iso_psd[key], dtype=float)
        band_data[band][sid].append((freq, psd))
    return band_data


def subject_mean_psd(runs):
    """Average a subject's relative-power runs onto the common frequency grid."""
    interped = []
    for freq, psd in runs:
        # Interpolate onto the common grid (NaN outside this run's support),
        # then normalize to % of total power on that shared grid.
        on_grid = np.interp(COMMON_FREQ, freq, psd, left=np.nan, right=np.nan)
        interped.append(to_percent(on_grid))
    return np.nanmean(interped, axis=0)  # mean across runs


def series_subjects(band_data, band, series, sex_map):
    """Yield the runs of every subject in ``band`` that matches ``series``.

    A ``series["sex"]`` of None keeps all subjects; otherwise only subjects whose
    mastersheet sex matches are kept.
    """
    want_sex = series["sex"]
    for sid, runs in band_data[band].items():
        if want_sex is not None and sex_map.get(sid) != want_sex:
            continue
        yield runs


def series_n(band_data, series, sex_map):
    """Number of unique subjects contributing to a series (across all bands)."""
    want_sex = series["sex"]
    subs = set()
    for subjects in band_data.values():
        for sid in subjects:
            if want_sex is None or sex_map.get(sid) == want_sex:
                subs.add(sid)
    return len(subs)


def series_group_mean(band_data, band, series, sex_map):
    """Group-mean relative-power curve for one series in one band (or None)."""
    subj_means = [subject_mean_psd(runs)
                  for runs in series_subjects(band_data, band, series, sex_map)]
    if not subj_means:
        return None
    return np.nanmean(np.array(subj_means), axis=0)


def plot_all():
    """Build and save the per-band ISO PSD figure overlaying all series."""
    set_prism_style()

    # Load each dataset's band data once, collapsing pickle subject-IDs to their
    # canonical mastersheet SID so runs are averaged within a subject. Also load
    # the MSSV sex lookup.
    band_data = {
        ds: load_band_data(
            PICKLE_TEMPLATE.format(dataset=ds),
            canonical=load_canonical_sid_map(MASTERSHEET_TEMPLATE.format(dataset=ds)),
        )
        for ds in DATASETS
    }
    sex_map = load_sex_map(MSSV_MASTERSHEET)

    # Legend label per series with its sample size.
    series_labels = {series["label"]: f"{series['label']} (n={series_n(band_data[series['dataset']], series, sex_map)})"
                     for series in SERIES}

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 7.5),
                             sharex=True, squeeze=False)

    for idx, (ax, band) in enumerate(zip(axes.ravel(), BAND_ORDER)):
        for series in SERIES:
            group_mean = series_group_mean(band_data[series["dataset"]], band,
                                           series, sex_map)
            if group_mean is None:
                continue
            ax.plot(COMMON_FREQ, group_mean, color=series["color"], lw=2.5,
                    label=series_labels[series["label"]], zorder=3)

        ax.set_xlim(0.005,0.05)
        ax.set_xticks([0.01,0.02,0.03,0.04])
        ax.set_ylim(0.3, 1.4)
        # show x tick labels on every panel, including the top row
        ax.tick_params(labelbottom=True)
        # band name + greek symbol + frequency range as text inside the panel (lower-left)
        lb, ub = BAND_FREQ[band]
        ax.text(0.04, 0.06, f"{band} {BAND_GREEK[band]} ({lb}-{ub}Hz)",
                transform=ax.transAxes, fontsize=17, fontweight="bold",
                va="bottom", ha="left")
        # panel label (a, b, c, ...) in the upper-left corner
        ax.text(-0.15, 0.95, chr(ord("a") + idx), transform=ax.transAxes,
                fontsize=20, fontweight="bold", va="bottom", ha="left")
        sns.despine(ax=ax)

    # Legend in the upper-right panel (top row, last column).
    axes[0, -1].legend(loc="upper right", fontsize=11.4, frameon=False,
                       handlelength=1.15, handletextpad=0.7)

    # Shared axis labels.
    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative power (%)")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {os.path.abspath(OUT_PATH)}")


def main():
    plot_all()


if __name__ == "__main__":
    main()
