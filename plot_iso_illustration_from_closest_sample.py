"""
Band-power infraslow oscillations (ISO) from the most representative *actual*
recording, drawn in the same stacked style as plot_iso_illustration.py.

Instead of synthesizing waveforms, this script:
  1. Builds a per-recording cross-frequency profile (Spearman correlation and
     PLV for every band pair) from cross_freq_corr_results_<dataset>.csv.
  2. Z-scores those features across recordings and picks the recording closest
     to the average profile (smallest distance to the mean, i.e. to the origin
     in z-scored space).
  3. Plots that recording's real band-power time courses (its longest NREM bout)
     from bout_band_powers in the pickle, one per band, stacked.

Each band's trace is band-pass filtered to the ISO band (0.01-0.03 Hz, as in the
analysis) so the infraslow oscillation and the cross-band phase relationships
are visible, then z-scored for display (so the stacking is comparable). This
keeps its ISO shape and phase but not its absolute amplitude. Labels show the
chosen recording's own PLV-to-delta and Spearman-correlation-to-delta.

Two figures are written per dataset: the band-only version over the full window,
and a separate "_with_eeg" version over a shorter window (EEG_SEGMENT_S) that adds
a bottom panel of the broadband (0.1-45 Hz) raw EEG for the exact same NREM bout
and time window, reconstructed from the source EDF/.mat by replaying get_iso.py's
bout detection (so the raw trace lines up with the band powers it was derived from).

Reuses styling/constants from plot_iso_illustration.py and plot_cross_freq.py.
"""

import os
import pickle
import argparse
from itertools import groupby

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns

from plot_cross_freq import BANDS, GREEK, PICKLE_TEMPLATE, CSV_TEMPLATE
from plot_iso_psd import BAND_FREQ, set_prism_style
import get_iso as gi

REF = "sigma"      # reference band whose correlations are reported (was in plot_iso_illustration)


DATASETS = ['MSSV']#, 'OxfordBenchmark']
OUT_TEMPLATE = "iso_illustration_from_closest_sample_{dataset}.png"
OUT_EEG_TEMPLATE = "iso_illustration_from_closest_sample_with_eeg_{dataset}.png"
EPOCH_S = 4.0      # band-power sampling step (s); ISO is sampled once per epoch
EEG_BAND = (0.1, 45.0)    # broadband EEG band-pass (Hz) for the raw-trace panel
SEGMENT_S = 400.0  # length (s) of the random segment shown (band-only figure)
EEG_SEGMENT_S = 100.0  # shorter window (s) for the figure that also shows raw EEG
# raw-EEG zoom windows (s), drawn top-to-bottom, each with its own highlight colour
ZOOMS = [((20.0, 40.0), "#e07a3f"),
         ((60.0, 80.0), "#f2c14e")]
AMP = 0.8          # vertical scale of each waveform
OFFSET_STEP = 2.6  # vertical spacing between stacked bands
SEED = 1990        # for the random segment choice (reproducible)

# Map each plot-level dataset tag back to get_iso's (dataset, oscar_eeg,
# include_hours) so the raw EEG and its NREM bouts can be reconstructed exactly
# as they were when the pickle was built.
DATASET_META = {
    "MSSV": ("MSSV", "EEG1", None),
    "OxfordBenchmark": ("OxfordBenchmark", "EEG1", None),
    #"ShenLab": ("ShenLab", "EEG1", None),
    "Oscar": ("Oscar", "EEG1", None),
}


def closest_recording(dataset):
    """(sid, edf) of the recording whose cross-frequency profile is most average.

    Features = Spearman correlation and PLV for every band pair, z-scored across
    recordings; the closest-to-average recording minimizes the distance to the
    mean (the origin after z-scoring).
    """
    df = pd.read_csv(CSV_TEMPLATE.format(dataset=dataset))
    feat = df.pivot_table(index=["sid", "edf"], columns=["band1", "band2"],
                          values=["spearman_corr", "plv"])
    feat = feat.dropna(axis=0)  # keep recordings with a complete profile

    z = (feat - feat.mean()) / (feat.std(ddof=0) + 1e-12)
    dist = np.linalg.norm(z.values, axis=1)
    sid, edf = feat.index[int(np.argmin(dist))]
    return sid, edf


def recording_ref_stats(dataset, sid, edf):
    """PLV-to-REF and Spearman-corr-to-REF for the chosen recording, per band."""
    df = pd.read_csv(CSV_TEMPLATE.format(dataset=dataset))
    rec = df[(df.sid == sid) & (df.edf == edf)]
    plv, corr = {}, {}
    for b in BANDS:
        if b == REF:
            continue
        row = rec[((rec.band1 == b) & (rec.band2 == REF)) |
                  ((rec.band1 == REF) & (rec.band2 == b))]
        plv[b] = float(row["plv"].iloc[0]) if len(row) else np.nan
        corr[b] = float(row["spearman_corr"].iloc[0]) if len(row) else np.nan
    return plv, corr


def segment_signals(dataset, sid, edf, segment_s=SEGMENT_S):
    """A random ``segment_s``-long, ISO-filtered segment of the recording.

    A bout long enough to contain the segment is chosen at random, each band's
    full bout is ISO-band filtered, and the same random window is taken from all
    bands (so they stay aligned in time). Also returns ``(bi, sl)`` so the same
    bout/window can be located in the raw EEG.
    """
    with open(PICKLE_TEMPLATE.format(dataset=dataset), "rb") as f:
        bout_band_powers = pickle.load(f)["bout_band_powers_f"]

    req = int(round(segment_s / EPOCH_S)) + 1        # samples (inclusive of endpoint)
    lengths = [len(b) for b in bout_band_powers[(sid, edf, REF)]]
    rng = np.random.default_rng(SEED)
    candidates = [i for i, L in enumerate(lengths) if L >= req]
    if candidates:
        bi = int(rng.choice(candidates))
        start = int(rng.integers(0, lengths[bi] - req + 1))
        sl = slice(start, start + req)
    else:                                            # no long-enough bout
        bi = int(np.argmax(lengths))
        sl = slice(0, lengths[bi])

    signals = {b: np.asarray(bout_band_powers[(sid, edf, b)][bi], float)[sl]
               for b in BANDS}
    t = np.arange(len(signals[REF])) * EPOCH_S
    return t, signals, (bi, sl)


def nrem_bouts(dataset, sid, edf):
    """Reconstruct the (start, end) NREM epochs and raw EEG for one recording.

    Replays the exact bout-detection used in get_iso.py (sleep-stage smoothing,
    optional include-hours masking, the >=96 s NREM rule) so the bouts come out
    in the same order as those stored in the pickle. Returns
    ``(bouts, eeg, fs, epoch_size)`` where ``bouts[i]`` is the (start, end) epoch
    index of the i-th band-power bout.
    """
    gi_ds, oscar_eeg, include_hours = DATASET_META[dataset]
    sids, edf_paths, annot_paths, eeg_chs, _ = gi.get_file_list(gi_ds)#, oscar_eeg=oscar_eeg)
    base = os.path.basename(edf)
    j = next(i for i, p in enumerate(edf_paths) if os.path.basename(p) == base)
    annot_path, eeg_ch = annot_paths[j], eeg_chs[j]

    eeg, fs = gi.load_signal(edf_paths[j], gi_ds, eeg_ch)
    epoch_duration = 4
    epoch_size = int(round(epoch_duration * fs))

    df_annot = gi.load_annot(annot_path, gi_ds)
    sleep_stages_ = df_annot.stage.values
    sleep_stages = np.array(sleep_stages_)
    for i in range(len(sleep_stages_) - 4):
        if (sleep_stages_[i + 2] != 2) and (sleep_stages_[i] == sleep_stages_[i + 1] ==
                                            sleep_stages_[i + 3] == sleep_stages_[i + 4] == 2):
            sleep_stages[i + 2] = 2
    L = len(eeg) // epoch_size
    assert L==len(sleep_stages)

    if include_hours is not None:
        epoch_hours = np.arange(len(sleep_stages)) * epoch_duration / 3600
        included = np.zeros(len(sleep_stages), dtype=bool)
        for start_h, end_h in include_hours:
            included |= (epoch_hours >= start_h) & (epoch_hours < end_h)
        sleep_stages[~included] = 0

    bouts, counter = [], 0
    for k, l in groupby(sleep_stages):
        n = len(list(l))
        if k == 2 and n * epoch_duration >= 96:
            bouts.append((counter, counter + n))
        counter += n
    return bouts, eeg, fs, epoch_size


def eeg_filter(x, fs):
    """Band-pass the raw EEG to EEG_BAND (0.1-45 Hz)."""
    y = mne.filter.filter_data(x, fs, EEG_BAND[0], EEG_BAND[1], verbose=False)
    return y


def raw_eeg_segment(dataset, sid, edf, bi, sl):
    """0.1-45 Hz filtered raw EEG for the exact bout/window shown above.

    Returns ``(t, eeg_uV)`` on the same time origin as the band-power traces.
    """
    bouts, eeg, fs, epoch_size = nrem_bouts(dataset, sid, edf)
    start_ep, _ = bouts[bi]
    a = (start_ep + sl.start) * epoch_size
    b = (start_ep + sl.stop) * epoch_size
    seg = np.asarray(eeg[a:b], dtype=float)
    seg = eeg_filter(seg, fs)
    t = np.arange(len(seg)) / fs
    return t, seg


def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def draw_bands(ax, t, signals, plv, corr):
    """Draw the stacked, ISO-filtered band traces with their per-band labels."""
    colors = sns.color_palette("husl", len(BANDS))
    ref_sym = GREEK[REF].strip("$")

    for i, b in enumerate(BANDS):
        y = AMP * zscore(signals[b]) + i * OFFSET_STEP
        ax.plot(t, y, color=colors[i], lw=2.0)
        lb, ub = BAND_FREQ[b]
        if b == REF:
            ptext = f"PLV$_{{{ref_sym}}}$=ref"
            rtext = f"r$_{{{ref_sym}}}$=ref"
        else:
            sign = "+" if corr[b] >= 0 else "−"
            ptext = f"PLV$_{{{ref_sym}}}$={plv[b]:.2f}"
            rtext = f"r$_{{{ref_sym}}}$={sign}{abs(corr[b]):.2f}"
        label = f"{GREEK[b]}  {b} ({lb}-{ub} Hz)\n{ptext},  {rtext}"
        ax.text(-0.012 * t[-1], i * OFFSET_STEP, label,
                ha="right", va="center", fontsize=14, color=colors[i])

    ax.set_yticks([])
    ax.margins(y=0.02)
    ax.grid(axis="x", color="0.85", linewidth=0.8)   # vertical gridlines at the ticks
    ax.set_axisbelow(True)
    ax.set_title("Illustration of infraslow oscillations in multiple bands",
                 fontsize=15, pad=10)


def plot_dataset(dataset):
    """Original stacked-band figure (no EEG), full SEGMENT_S window."""
    sid, edf = closest_recording(dataset)
    t, signals, _ = segment_signals(dataset, sid, edf)   # already ISO-filtered
    plv, corr = recording_ref_stats(dataset, sid, edf)

    set_prism_style()
    fig, ax = plt.subplots(figsize=(11, 5.7))
    draw_bands(ax, t, signals, plv, corr)
    ax.set_xlim(0, t[-1])
    ax.set_xticks(np.arange(0, t[-1] + 1, 50))       # ticks every 50 s incl. last
    ax.set_xlabel("Time (s)")
    sns.despine(ax=ax, left=True)

    fig.tight_layout()
    out_path = OUT_TEMPLATE.format(dataset=dataset)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {os.path.abspath(out_path)} (closest: {sid})")


def _draw_eeg_zoom(ax_zoom, t_eeg, eeg_uV, window, color):
    """Draw one zoomed-in raw-EEG panel, colour-keyed to its highlight box.

    ``ax_zoom`` shows ``t_eeg``/``eeg_uV`` restricted to ``window`` (blown up to
    full width). Its frame and label are drawn in ``color`` so it reads against
    the matching translucent highlight box in the full EEG panel above.
    """
    z0, z1 = window
    ax_zoom.plot(t_eeg, eeg_uV, color="0.2", lw=0.7)
    ax_zoom.text(z0 - 0.012 * (z1 - z0), 0, f"EEG\n({z0:g}-{z1:g} s zoom)",
                 ha="right", va="center", fontsize=14, color=color)
    ax_zoom.set_xlim(z0, z1)
    ax_zoom.set_xticks(np.arange(z0, z1 + 1, 5))     # ticks every 5 s incl. last
    ax_zoom.set_yticks([])
    ax_zoom.margins(y=0.05)
    #ax_zoom.grid(axis="x", color="0.85", linewidth=0.8)
    ax_zoom.set_axisbelow(True)

    for sp in ax_zoom.spines.values():               # colour-keyed frame
        sp.set_visible(True)
        sp.set_color(color)
        sp.set_linewidth(1.8)


def plot_dataset_with_eeg(dataset):
    """Stacked-band figure plus a multi-level raw-EEG zoom.

    Below the band panel sits the full EEG_SEGMENT_S raw-EEG window with each
    ZOOMS region highlighted by a translucent colour box, and beneath it one
    blown-up panel per zoom window. Each zoom panel is colour-keyed (frame and
    label) to its highlight box, so the correspondence is clear without the
    expander lines crossing between panels.
    """
    sid, edf = closest_recording(dataset)
    t, signals, (bi, sl) = segment_signals(dataset, sid, edf, segment_s=EEG_SEGMENT_S)
    plv, corr = recording_ref_stats(dataset, sid, edf)
    t_eeg, eeg_uV = raw_eeg_segment(dataset, sid, edf, bi, sl)   # 0.1-45 Hz raw EEG

    eeg_lb, eeg_ub = EEG_BAND

    set_prism_style()
    # rows: stacked bands, full EEG, then (spacer, zoom) per zoom window
    heights = [4.0, 1.0]
    for _ in ZOOMS:
        heights += [0.5, 1.2]
    fig = plt.figure(figsize=(11, 1.35 * sum(heights)))
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.0)
    ax = fig.add_subplot(gs[0])
    ax_eeg = fig.add_subplot(gs[1], sharex=ax)

    draw_bands(ax, t, signals, plv, corr)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(axis="x", length=0, labelbottom=False)  # x labels live on the EEG panels

    # full panel: broadband raw EEG (EEG_BAND) over the whole window, zooms marked
    ax_eeg.plot(t_eeg, eeg_uV, color="0.2", lw=0.5)
    for (z0, z1), color in ZOOMS:
        ax_eeg.axvspan(z0, z1, color=color, alpha=0.25, lw=0)
    ax_eeg.text(-0.012 * t[-1], 0, f"EEG\n({eeg_lb:g}-{eeg_ub:g} Hz)",
                ha="right", va="center", fontsize=14, color="0.2")
    ax_eeg.set_xlim(0, t[-1])
    ax_eeg.set_xticks(np.arange(0, t[-1] + 1, 50))   # gridlines only at 0, 50, 100 s
    ax_eeg.set_yticks([])                             # amplitude axis hidden, as above
    ax_eeg.margins(y=0.05)
    ax_eeg.grid(axis="x", color="0.85", linewidth=0.8)
    ax_eeg.set_axisbelow(True)
    sns.despine(ax=ax_eeg, left=True)

    # one blown-up panel per zoom window, each linked back to the full panel
    zoom_axes = []
    for i, (window, color) in enumerate(ZOOMS):
        ax_zoom = fig.add_subplot(gs[3 + 2 * i])
        _draw_eeg_zoom(ax_zoom, t_eeg, eeg_uV, window, color)
        zoom_axes.append(ax_zoom)
    zoom_axes[-1].set_xlabel("Time (s)")             # units on the bottom-most panel

    out_path = OUT_EEG_TEMPLATE.format(dataset=dataset)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {os.path.abspath(out_path)} (closest: {sid})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*",
                        help=f"Dataset(s) to plot (choices: {', '.join(DATASETS)}; "
                             "default: all).")
    args = parser.parse_args()
    for dataset in (args.datasets or DATASETS):
        if dataset not in DATASETS:
            parser.error(f"invalid dataset {dataset!r}; choose from {DATASETS}")
        plot_dataset(dataset)            # original, band-only, 400 s
        plot_dataset_with_eeg(dataset)   # separate version with raw EEG, 100 s


if __name__ == "__main__":
    main()
