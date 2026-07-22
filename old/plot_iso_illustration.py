"""
Illustrative time-domain band-power infraslow oscillations (ISO).

Draws one synthetic ISO waveform per frequency band, stacked in a single figure,
constructed so the inter-band relationships mirror the measured cross-frequency
statistics:

  - Phase offset of each band is set so its Spearman power correlation with the
    reference band (delta) is reproduced exactly: for two oscillations,
    corr = cos(delta_phase), so phase offset = +/- arccos(r_to_delta). The sign
    (lead vs lag) is taken from the measured ISO phase difference (leading
    eigenvector of <exp(i(phase_x - phase_y))>). delta is negatively correlated
    with every other band, so the others appear in (near) antiphase to it.
  - Phase-locking value (PLV) with the reference band sets how tightly each
    waveform is locked: bands with high PLV-to-reference oscillate cleanly,
    bands with low PLV wobble (phase jitter ~ 1 - normalized PLV). The reference
    band has no self-PLV and is drawn as the cleanest (jitter 0).
  - ISO-peak clarity (normalized ISO band power, iso_bp_n) sets how much
    broadband (slow) noise is added on top: bands whose ISO peak is less clear
    (lower iso_bp_n, e.g. theta most, beta less so) get more additive noise, so
    their oscillation looks less well-defined. This is separate from the PLV
    phase jitter (a band can be phase-locked yet have a weak/broad ISO peak).

This is an illustration (one synthetic realization), not a reconstruction of
any recording. Reuses the data-loading/phase helpers from plot_cross_freq.py.

Style mimics GraphPad Prism: clean sans-serif font, no grid.
"""

import os
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from plot_cross_freq import (
    DATASETS, BANDS, GREEK, P_THRESH, load_per_sid,
    phase_coupling_matrix, phases_from_Z,
)


OUT_TEMPLATE = "iso_illustration_{dataset}.png"
ISO_RESULTS_TEMPLATE = "iso_results_all_subjects_bands_with_DTWDist_{dataset}.csv"

# ISO band ranges (Hz) for the labels.
BAND_FREQ = {
    "gamma": (30, 45), "beta": (15, 30), "sigma": (11, 15),
    "alpha": (8, 12), "theta": (5, 10), "delta": (1, 4),
}

REF = "delta"      # reference band whose correlations are reproduced exactly
F0 = 0.02          # representative infraslow frequency (Hz), ~50 s period
DURATION = 200.0   # seconds shown
DT = 0.5           # time step (s)
MAX_JITTER = 1.1   # max phase-jitter amplitude (rad) for the least-locked band
NOISE_GAIN = 0.6   # max broadband-noise amplitude for the least clear ISO peak
NOISE_FLOOR = 0.13  # baseline broadband noise so no band (incl. delta) is perfectly clean
NOISE_SMOOTH_S = 12.0  # smoothing window (s) for the broadband noise
SEED = 0


def set_prism_style():
    sns.set_style("ticks")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 16,
        "axes.linewidth": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "xtick.major.width": 1.5,
        "savefig.dpi": 300,
    })


def band_plv_with(df, ref=REF):
    """PLV of each band with the reference band (NaN for the reference itself)."""
    pv = df.groupby(["band1", "band2"])["plv"].mean()
    out = {}
    for b in BANDS:
        if b == ref:
            out[b] = np.nan
        elif (b, ref) in pv.index:
            out[b] = pv.loc[(b, ref)]
        elif (ref, b) in pv.index:
            out[b] = pv.loc[(ref, b)]
        else:
            out[b] = np.nan
    return out


def band_corr_with(df, ref="delta"):
    """Mean Spearman correlation of each band with the reference band."""
    sp = df.groupby(["band1", "band2"])["spearman_corr"].mean()
    out = {}
    for b in BANDS:
        if b == ref:
            out[b] = 1.0
        elif (b, ref) in sp.index:
            out[b] = sp.loc[(b, ref)]
        elif (ref, b) in sp.index:
            out[b] = sp.loc[(ref, b)]
        else:
            out[b] = np.nan
    return out


def band_iso_clarity(dataset):
    """ISO-peak clarity per band = normalized ISO band power (iso_bp_n).

    Higher iso_bp_n means power is more concentrated in the ISO band, i.e. a
    clearer ISO peak. Averaged within subject, then across subjects.
    """
    df = pd.read_csv(ISO_RESULTS_TEMPLATE.format(dataset=dataset))
    df = df.groupby(["band_name", "sid"], as_index=False)["iso_bp_n"].mean()
    m = df.groupby("band_name")["iso_bp_n"].mean()
    return {b: float(m.get(b, np.nan)) for b in BANDS}


def band_quantities(dataset):
    """Per-band quantities (PLV-to-REF, corr-to-REF, phase-coupling Z, clarity)."""
    df = load_per_sid(dataset)
    return dict(plv=band_plv_with(df, ref=REF),
                corr=band_corr_with(df, ref=REF),
                Z=phase_coupling_matrix(dataset),
                clarity=band_iso_clarity(dataset))


def combine_quantities(qs):
    """Average the per-band quantities across datasets (consensus)."""
    def avg(key, b):
        vals = [q[key][b] for q in qs if np.isfinite(q[key][b])]
        return float(np.mean(vals)) if vals else np.nan
    return dict(plv={b: (np.nan if b == REF else avg("plv", b)) for b in BANDS},
                corr={b: avg("corr", b) for b in BANDS},
                clarity={b: avg("clarity", b) for b in BANDS},
                Z=np.mean([q["Z"] for q in qs], axis=0))


def make_signals(q):
    """Return time vector and {band: waveform}, plus the offsets/PLV/corr used.

    `q` is a quantities dict (from band_quantities or combine_quantities).
    """
    rel_phases = phases_from_Z(q["Z"])             # ISO phase (rad), for sign
    plv = q["plv"]                                 # PLV with the reference band
    corr_ref = q["corr"]                           # Spearman correlation to REF

    # phase offset reproduces corr-to-REF exactly: corr = cos(offset);
    # magnitude = arccos(r), sign (lead/lag) from the measured phase difference.
    phi = {}
    for b in BANDS:
        mag = np.arccos(np.clip(corr_ref[b], -1.0, 1.0))
        d = (rel_phases[b] - rel_phases[REF] + np.pi) % (2 * np.pi) - np.pi
        sign = 1.0 if d >= 0 else -1.0
        phi[b] = sign * mag

    # locking fraction in [0, 1] from PLV; least-locked band gets most jitter.
    # the reference band has no self-PLV; treat it as the cleanest (jitter 0).
    pmax = np.nanmax([plv[b] for b in BANDS])
    pv = np.array([plv[b] if np.isfinite(plv[b]) else pmax for b in BANDS])
    lock = (pv - pv.min()) / (pv.max() - pv.min()) if pv.max() > pv.min() else np.ones_like(pv)
    jitter_amp = {b: MAX_JITTER * (1.0 - lock[i]) for i, b in enumerate(BANDS)}

    # ISO-peak clarity (iso_bp_n) in [0, 1]; the least clear peak (e.g. theta)
    # gets the most broadband noise added on top of the oscillation.
    cl = np.array([q["clarity"][b] for b in BANDS], dtype=float)
    clarity = (cl - np.nanmin(cl)) / (np.nanmax(cl) - np.nanmin(cl)) \
        if np.nanmax(cl) > np.nanmin(cl) else np.ones_like(cl)
    noise_amp = {b: NOISE_FLOOR + NOISE_GAIN * (1.0 - clarity[i])
                 for i, b in enumerate(BANDS)}

    t = np.arange(0, DURATION, DT)
    win = max(1, int(round(NOISE_SMOOTH_S / DT)))
    kernel = np.ones(win) / win
    rng = np.random.default_rng(SEED)
    signals = {}
    for b in BANDS:
        # smooth random-walk phase jitter, normalized to unit std over the window
        walk = np.cumsum(rng.standard_normal(len(t)))
        walk = (walk - walk.mean()) / (walk.std() + 1e-9)
        phase = 2 * np.pi * F0 * t + phi[b] + jitter_amp[b] * walk
        # broadband (slow) noise: smoothed white noise, unit std, scaled by unclarity
        noise = np.convolve(rng.standard_normal(len(t)), kernel, mode="same")
        noise = (noise - noise.mean()) / (noise.std() + 1e-9)
        signals[b] = np.cos(phase) + noise_amp[b] * noise
    return t, signals, plv, corr_ref


def plot_quantities(q, out_path):
    t, signals, plv, corr_delta = make_signals(q)

    set_prism_style()
    fig, ax = plt.subplots(figsize=(11, 5.7))
    colors = sns.color_palette("husl", len(BANDS))

    offset_step = 2.6
    amp = 0.8  # vertical scale of each waveform (smaller -> shorter curves)
    ref_sym = GREEK[REF].strip("$")
    # stack low frequency at the bottom -> high at the top
    for i, b in enumerate(BANDS):
        y = amp * signals[b] + i * offset_step
        ax.plot(t, y, color=colors[i], lw=2.0)
        lb, ub = BAND_FREQ[b]
        if b == REF:
            ptext = f"PLV$_{{{ref_sym}}}$=ref"
            rtext = f"r$_{{{ref_sym}}}$=ref"
        else:
            sign = "+" if corr_delta[b] >= 0 else "−"
            ptext = f"PLV$_{{{ref_sym}}}$={plv[b]:.2f}"
            rtext = f"r$_{{{ref_sym}}}$={sign}{abs(corr_delta[b]):.2f}"
        label = f"{GREEK[b]}  {b} ({lb}-{ub} Hz)\n{ptext},  {rtext}"
        ax.text(-0.012 * DURATION, i * offset_step, label,
                ha="right", va="center", fontsize=14, color=colors[i])

    ax.set_xlim(0, DURATION)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.margins(y=0.02)
    ax.set_title("Illustration of infraslow oscillations in multiple bands",
                 fontsize=16, pad=10)
    sns.despine(ax=ax, left=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {os.path.abspath(out_path)}")


def plot_dataset(dataset):
    plot_quantities(band_quantities(dataset), OUT_TEMPLATE.format(dataset=dataset))


def plot_common(datasets):
    q = combine_quantities([band_quantities(d) for d in datasets])
    out_path = "iso_illustration_common_" + "_".join(datasets) + ".png"
    plot_quantities(q, out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*",
                        help=f"Dataset(s) to plot (choices: {', '.join(DATASETS)}; "
                             "default: all).")
    parser.add_argument("--common", nargs="+", metavar="DATASET",
                        help="Produce a single illustration from the consensus "
                             "of the given datasets (e.g. --common MSSV OxfordBenchmark).")
    args = parser.parse_args()

    if args.common:
        for d in args.common:
            if d not in DATASETS:
                parser.error(f"invalid dataset {d!r}; choose from {DATASETS}")
        plot_common(args.common)
        return

    for dataset in (args.datasets or DATASETS):
        if dataset not in DATASETS:
            parser.error(f"invalid dataset {dataset!r}; choose from {DATASETS}")
        plot_dataset(dataset)


if __name__ == "__main__":
    main()
