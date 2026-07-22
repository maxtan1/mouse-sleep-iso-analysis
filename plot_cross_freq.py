"""
Cross-frequency coupling network plot (built once, pooled across datasets).

Draws panels whose six vertices are the frequency bands (Greek letters).
Edges between vertices encode a pairwise band-band measure:

  - Spearman correlation : un-arrowed lines, coolwarm colormap (blue<0, red>0)
  - PLV (phase locking)  : un-arrowed lines, coolwarm colormap (blue<0, red>0)

(Granger causality is intentionally excluded -- its panel/code is commented out.)

Significance is decided by the *external validation* dataset (OxfordBenchmark):
an edge is drawn for a band pair only where that measure's FDR-adjusted p-value
in cross_freq_corr_results_across_subjects_fdr_OxfordBenchmark.csv is < 0.05.

The plotted effect sizes, however, pool the discovery + validation recordings:
the per-recording tables cross_freq_corr_results_{MSSV,OxfordBenchmark}.csv are
concatenated and reduced to one across-subject value per band pair via
`across_subjects` (imported from stepb_get_fdr_across_subjects_cross_band_coupling).

Vertex positions:
  - Spearman: a 2D MDS embedding, so strongly coupled bands (high |value|) are
    placed close together.
  - PLV: a "phase clock". The angular position of each band is its relative ISO
    phase (from the leading eigenvector of the complex phase-coupling matrix
    z(x, y) = <exp(i(phase_x - phase_y))>, recovered from bout_phases in the
    pickles, pooled over datasets). Counter-clockwise = phase lead, so a band
    sitting counter-clockwise of another leads it. Near-overlapping bands are
    spread apart for legibility, preserving their circular order.

The MDS embedding distances use the pooled effect sizes for all pairs (no
p-filter), so every band pair contributes a distance.

Style mimics GraphPad Prism: clean sans-serif font, no grid.
"""

import os
import pickle
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
import networkx as nx
import seaborn as sns

from stepb_get_fdr_across_subjects_cross_band_coupling import across_subjects


# The coupling figure is built once, not per dataset: significant band pairs come
# from the external validation dataset's FDR-corrected results, while the plotted
# effect sizes pool the discovery + validation recordings.
SIG_DATASET = "OxfordBenchmark"              # external validation: decides significance
PLOT_DATASETS = ["MSSV", "OxfordBenchmark"]  # pooled for the plotted effect sizes
ACROSS_TEMPLATE = "cross_freq_corr_results_across_subjects_fdr_{dataset}.csv"
CSV_TEMPLATE = "cross_freq_corr_results_{dataset}.csv"
PICKLE_TEMPLATE = "data_all_subjects_bands_{dataset}.pickle"
MASTERSHEET_TEMPLATE = "mastersheet_{dataset}.csv"
OUT_PATH = "cross_freq_across_subjects.png"

P_THRESH = 0.05

BANDS = ["delta", "theta", "alpha", "sigma", "beta", "gamma"]
GREEK = {
    "delta": r"$\delta$", "theta": r"$\theta$", "alpha": r"$\alpha$",
    "sigma": r"$\sigma$", "beta": r"$\beta$", "gamma": r"$\gamma$",
}

LW_MIN, LW_MAX = 1.5, 6.0     # edge line widths scaled by |value|
NODE_SIZE = 550               # scatter marker area for vertices
SHRINK = 20                   # points to shrink edges away from node markers
DIST_FLOOR = 0.35             # min embedding distance so close nodes don't overlap


def set_prism_style():
    sns.set_style("ticks")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 15,
        "axes.linewidth": 1.5,
        "savefig.dpi": 300,
    })


def pkey(a, b):
    """Order-independent key for a band pair."""
    return frozenset((a, b))


def lw_for(frac):
    """Map a 0..1 magnitude fraction to a line width."""
    return LW_MIN + (LW_MAX - LW_MIN) * float(np.clip(frac, 0, 1))


def sid_from_edf(datasets):
    """Map each recording's EDF basename to its true subject SID (mastersheet).

    The `sid` column in the data files is per-recording for some datasets (e.g.
    Oscar, where one subject has several recordings), so the subject grouping
    must come from mastersheet_<dataset>.csv, keyed by the EDF file name. Accepts
    one dataset name or a list; the maps are pooled (EDF basenames are unique
    across datasets).
    """
    if isinstance(datasets, str):
        datasets = [datasets]
    smap = {}
    for dataset in datasets:
        ms = pd.read_csv(MASTERSHEET_TEMPLATE.format(dataset=dataset))
        smap.update({os.path.basename(str(p)): sid
                     for sid, p in zip(ms["SID"], ms["EDFPath"])})
    return smap


def significant_pairs(dataset):
    """Band pairs that are FDR-significant per measure in the validation dataset.

    Reads cross_freq_corr_results_across_subjects_fdr_<dataset>.csv (produced by
    stepb) and returns {measure: set of (band1, band2)} for the rows whose
    across-subject FDR-adjusted p-value is < P_THRESH. Only the x -> y row carries
    the undirected Spearman/PLV value, so these keys match the pooled across-
    subjects table directly. Granger is intentionally omitted.
    """
    fdr = pd.read_csv(ACROSS_TEMPLATE.format(dataset=dataset))
    sig = {"spearman": set(), "plv": set()}
    for r in fdr.itertuples():
        if np.isfinite(r.spearman_p_value_fdr) and r.spearman_p_value_fdr < P_THRESH:
            sig["spearman"].add((r.band1, r.band2))
        if np.isfinite(r.plv_p_value_fdr) and r.plv_p_value_fdr < P_THRESH:
            sig["plv"].add((r.band1, r.band2))
    return sig


def pooled_across(datasets):
    """Across-subject effect sizes from the pooled per-recording tables.

    The per-recording files cross_freq_corr_results_<dataset>.csv are concatenated
    and collapsed to one row per band pair by `across_subjects` (from stepb),
    subject resolved from the pooled mastersheets. Returns the effect sizes only
    (no FDR); significance comes separately from `significant_pairs`.
    """
    df_long = pd.concat(
        [pd.read_csv(CSV_TEMPLATE.format(dataset=d)) for d in datasets],
        ignore_index=True)
    ms = pd.concat(
        [pd.read_csv(MASTERSHEET_TEMPLATE.format(dataset=d))[["SID", "EDFPath"]]
         for d in datasets],
        ignore_index=True)
    return across_subjects(df_long, ms)


def embed_positions(similarity, dist_floor=DIST_FLOOR):
    """2D Kamada-Kawai layout from a band-pair similarity dict.

    `similarity` maps pkey(a, b) -> non-negative coupling strength. It is
    min-max normalized to [0, 1]; distance ranges from `dist_floor` (strongest
    coupling) to 1 (weakest), so strongly coupled bands sit close together
    without overlapping. A larger floor spreads the nodes more evenly.

    Kamada-Kawai places the nodes by minimizing a spring energy whose rest
    lengths are these pairwise distances (deterministic given the same
    distances).
    """
    vals = np.array(list(similarity.values()), dtype=float)
    smin, smax = np.nanmin(vals), np.nanmax(vals)
    span = (smax - smin) if smax > smin else 1.0

    # nested {band: {band: distance}} for networkx's Kamada-Kawai solver
    dist = {a: {} for a in BANDS}
    for a in BANDS:
        for b in BANDS:
            if a == b:
                continue
            s = similarity[pkey(a, b)]
            norm = (s - smin) / span
            dist[a][b] = dist_floor + (1.0 - dist_floor) * (1.0 - norm)

    G = nx.complete_graph(BANDS)
    layout = nx.kamada_kawai_layout(G, dist=dist)
    xy = np.array([layout[band] for band in BANDS], dtype=float)

    # center and scale to roughly unit radius for consistent axis limits
    xy = xy - xy.mean(axis=0)
    scale = np.abs(xy).max()
    if scale > 0:
        xy = xy / scale
    return {band: xy[i] for i, band in enumerate(BANDS)}


def phase_coupling_matrix(datasets):
    """Hermitian phase-coupling matrix Z[x, y] = <exp(i(phase_x - phase_y))>.

    Averaged within subject, then across subjects, from bout_phases pooled over
    the datasets' pickles. Z carries per-pair magnitude (PLV) and angle (mean
    phase difference).
    """
    if isinstance(datasets, str):
        datasets = [datasets]
    bout_phases = {}
    for dataset in datasets:
        with open(PICKLE_TEMPLATE.format(dataset=dataset), "rb") as f:
            bout_phases.update(pickle.load(f)["bout_phases"])

    smap = sid_from_edf(datasets)
    recordings = sorted({(sid, edf) for (sid, edf, _band) in bout_phases})
    idx = {b: i for i, b in enumerate(BANDS)}

    per_sid = defaultdict(lambda: defaultdict(list))  # pair -> subject -> [complex z]
    for sid, edf in recordings:
        # resolve the true subject from the mastersheet (pickle sid is per-recording)
        subject = smap.get(os.path.basename(str(edf)), sid)
        for a, b in combinations(BANDS, 2):
            if (sid, edf, a) not in bout_phases or (sid, edf, b) not in bout_phases:
                continue
            pa = np.concatenate([np.asarray(v, float) for v in bout_phases[(sid, edf, a)]])
            pb = np.concatenate([np.asarray(v, float) for v in bout_phases[(sid, edf, b)]])
            ok = (~np.isnan(pa)) & (~np.isnan(pb))
            if ok.sum() == 0:
                continue
            per_sid[(a, b)][subject].append(np.mean(np.exp(1j * (pa[ok] - pb[ok]))))

    Z = np.eye(len(BANDS), dtype=complex)
    for (a, b), by_sid in per_sid.items():
        zbar = np.mean([np.mean(v) for v in by_sid.values()])  # avg within sid, then across
        Z[idx[a], idx[b]] = zbar
        Z[idx[b], idx[a]] = np.conj(zbar)
    return Z


def phases_from_Z(Z):
    """Globally consistent per-band phase from the leading eigenvector of Z."""
    idx = {b: i for i, b in enumerate(BANDS)}
    _, eigvecs = np.linalg.eigh(Z)
    phases = np.angle(eigvecs[:, -1])   # eigenvector of largest eigenvalue
    return {b: phases[idx[b]] for b in BANDS}


def spread_on_circle(phases_rad, pad):
    """Add a fixed angular padding to every gap, preserving relative spacing.

    Rather than clamping each gap to a hard minimum (which collapses all the
    tiny within-cluster gaps to one equal value), a constant `pad` is added to
    every consecutive gap and the result is renormalized to sum to 2*pi. This
    guarantees a minimum separation (no overlap) while keeping the bands in
    their true phase order AND keeping larger real gaps proportionally larger,
    so the spacing still reflects the actual phase differences (only the global
    scale is inflated for legibility).
    """
    bands = list(phases_rad)
    ang = np.array([phases_rad[b] % (2 * np.pi) for b in bands], dtype=float)
    order = np.argsort(ang)
    a = ang[order]

    ext = np.concatenate([a, [a[0] + 2 * np.pi]])
    gaps = np.diff(ext) + pad
    gaps = gaps / gaps.sum() * (2 * np.pi)

    new = a[0] + np.concatenate([[0.0], np.cumsum(gaps[:-1])])
    return {bands[order[i]]: new[i] for i in range(len(new))}


def phase_clock_positions(phases, radius=1.0, pad_deg=30.0):
    """Vertex positions on a circle, angle = relative ISO phase (CCW = leads)."""
    phases = spread_on_circle(phases, np.deg2rad(pad_deg))
    # rotate so delta sits at the top (90 deg); rotation preserves lead/lag order
    rot = np.pi / 2 - phases["delta"]
    return {b: radius * np.array([np.cos(phases[b] + rot), np.sin(phases[b] + rot)])
            for b in BANDS}


def draw_nodes(ax, pos):
    for band in BANDS:
        p = pos[band]
        ax.scatter(*p, s=NODE_SIZE, facecolor="white", edgecolor="0.3",
                   linewidths=1.5, zorder=3)
        ax.text(*p, GREEK[band], ha="center", va="center",
                fontsize=23, zorder=4)


def finalize_panel(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.42, 1.32)
    ax.axis("off")


def panel_label(ax, letter, name, dx=0.0):
    """Upper-left panel label with a bold letter and a non-bold name, e.g.
    "a. Spearman's Correlation" (replaces the per-panel title). `dx` shifts the
    label rightward in data units."""
    parts = [TextArea(f"{letter}.", textprops=dict(fontsize=17, fontweight="bold")),
             TextArea(name, textprops=dict(fontsize=17))]
    box = HPacker(children=parts, align="baseline", pad=0, sep=5)
    # anchor in data coords (left/top of the plotted square) so the label tracks
    # the equal-aspect figure instead of the wider axes rectangle
    x0, _ = ax.get_xlim()
    _, y1 = ax.get_ylim()
    ax.add_artist(AnnotationBbox(box, (x0 + dx, y1 - 0.18), xycoords="data",
                                 box_alignment=(0, 1), frameon=False, pad=0))


def draw_phase_circle(ax, radius=1.0):
    """Faint ring at the phase-clock radius, so the band markers read as phases."""
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, edgecolor="0.7",
                             linewidth=1.2, linestyle=(0, (4, 4)), zorder=0))


def draw_lead_hint(ax):
    """Label indicating that counter-clockwise = phase lead."""
    ax.text(0, -1.05, "counter-clockwise = phase lead",
            ha="center", va="center", fontsize=15, color="0.4")


def draw_sign_hint(ax):
    """Caption noting the diverging-colormap sign convention."""
    ax.text(0, -1.05, "blue is negative, red is positive",
            ha="center", va="center", fontsize=15, color="0.4")


def draw_undirected(ax, edge_values, pos):
    """Straight, un-arrowed edges colored by a 0-centered coolwarm colormap.

    `edge_values` maps (band1, band2) -> signed value (dict or pandas Series).
    """
    edge_values = dict(edge_values.items())
    cmap = plt.get_cmap("coolwarm")
    vals = np.array(list(edge_values.values()), dtype=float)
    vmax = float(np.nanmax(np.abs(vals))) if len(vals) else 1.0
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)

    for (b1, b2), val in edge_values.items():
        if not np.isfinite(val):
            continue
        p1, p2 = pos[b1], pos[b2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=cmap(norm(val)), lw=lw_for(abs(val) / vmax),
                solid_capstyle="round", zorder=1)
    draw_nodes(ax, pos)
    finalize_panel(ax)


def draw_directed(ax, edges, pos, alpha=0.45):
    """One arrow per pair: the stronger significant direction.

    `edges` is a list of (src, tgt, F) tuples (already reduced to one per pair).
    Arrows are always black; F is reflected only in the line width. Each arrow
    is drawn as a curved arc (and shrunk away from the node markers) so the
    directed edges don't overlap each other or pass through other nodes.
    `alpha` sets the arrow transparency.
    """
    vals = np.array([v for _, _, v in edges], dtype=float) if edges else np.array([1.0])
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))

    for src, tgt, val in edges:
        frac = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        rad = 0.35
        arrow = FancyArrowPatch(
            pos[src], pos[tgt], connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=16,
            color="black", alpha=alpha, lw=lw_for(frac),
            shrinkA=SHRINK, shrinkB=SHRINK, zorder=1)
        ax.add_patch(arrow)

    draw_nodes(ax, pos)
    finalize_panel(ax)


def stronger_granger_edges(across):
    """One directed edge per pair: the stronger of the FDR-significant directions.

    `across` is the across-subjects table. For each band pair, keep the
    direction(s) whose FDR-adjusted Granger p-value is < 0.05, then draw only the
    one with the larger across-subject F. Returns (src, tgt, F) tuples; pairs
    with no significant direction are omitted.
    """
    gc = {(r.band1, r.band2): (r.granger_f, r.granger_p_value_fdr)
          for r in across.itertuples()}
    edges = []
    for a, b in combinations(BANDS, 2):
        candidates = []  # (F, src, tgt) for FDR-significant directions
        for src, tgt in ((a, b), (b, a)):
            f, q = gc.get((src, tgt), (np.nan, np.nan))
            if np.isfinite(f) and np.isfinite(q) and q < P_THRESH:
                candidates.append((f, src, tgt))
        if not candidates:
            continue
        f, src, tgt = max(candidates, key=lambda c: c[0])  # stronger by F
        edges.append((src, tgt, f))
    return edges


def pair_similarity(value_by_pair):
    """Convert a (band1, band2)-keyed mapping to a pkey-indexed dict, all band
    pairs present (NaN/missing -> 0 coupling). Accepts a dict or pandas Series."""
    lookup = {pkey(x, y): v for (x, y), v in dict(value_by_pair.items()).items()}
    sim = {}
    for a, b in combinations(BANDS, 2):
        val = lookup.get(pkey(a, b), np.nan)
        sim[pkey(a, b)] = 0.0 if not np.isfinite(val) else float(val)
    return sim


def compute_measures():
    """All per-pair quantities needed to draw the cross-frequency figure.

    Significance is taken from the external validation dataset (SIG_DATASET): an
    edge is drawn for a band pair only where that measure is FDR-significant
    there. The plotted effect sizes come from the pooled across-subjects table
    (PLOT_DATASETS). Granger is intentionally excluded (commented out below)."""
    sig = significant_pairs(SIG_DATASET)
    across = pooled_across(PLOT_DATASETS)

    # undirected edges (color = pooled effect size), shown only for band pairs
    # that are FDR-significant in the external validation dataset
    spearman, plv = {}, {}
    for r in across.itertuples():
        if np.isfinite(r.spearman_corr) and (r.band1, r.band2) in sig["spearman"]:
            spearman[(r.band1, r.band2)] = r.spearman_corr
        if np.isfinite(r.plv) and (r.band1, r.band2) in sig["plv"]:
            plv[(r.band1, r.band2)] = r.plv
    # directed Granger: excluded from the plot
    # granger_edges = stronger_granger_edges(across)

    # embedding similarities (all pairs, no p-filter). Spearman is undirected
    # (only the x->y rows carry it).
    sp_strength = {(r.band1, r.band2): abs(r.spearman_corr)
                   for r in across.itertuples() if np.isfinite(r.spearman_corr)}
    # Granger embedding strength (excluded from the plot):
    # gmean = {(r.band1, r.band2): r.granger_f for r in across.itertuples()}
    # g_strength = {}
    # for a, b in combinations(BANDS, 2):
    #     vals = [v for v in (gmean.get((a, b)), gmean.get((b, a))) if v is not None and np.isfinite(v)]
    #     if vals:
    #         g_strength[(a, b)] = max(vals)

    return dict(spearman=spearman, plv=plv,
                sp_strength=sp_strength,
                Z=phase_coupling_matrix(PLOT_DATASETS))


def plot_measures(measures, out_path):
    pos_sp = embed_positions(pair_similarity(measures["sp_strength"]))
    # nudge gamma rightward so its links don't overlap the neighbouring nodes
    pos_sp["gamma"] = pos_sp["gamma"] + np.array([0.15, 0.0])
    # nudge the isolated delta node leftward
    pos_sp["delta"] = pos_sp["delta"] + np.array([-0.2, 0.0])
    # raise the whole panel-a graph vertically
    for band in pos_sp:
        pos_sp[band] = pos_sp[band] + np.array([0.0, 0.2])
    # Granger layout (excluded from the plot):
    # pos_g = embed_positions(pair_similarity(measures["g_strength"]), dist_floor=0.6)
    # PLV: phase-clock layout (angle = relative ISO phase, CCW = leads)
    pos_plv = phase_clock_positions(phases_from_Z(measures["Z"]), radius=0.85)

    set_prism_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.0))
    draw_undirected(axes[0], measures["spearman"], pos_sp)
    draw_sign_hint(axes[0])
    draw_phase_circle(axes[1], radius=0.85)
    draw_undirected(axes[1], measures["plv"], pos_plv)
    draw_lead_hint(axes[1])
    # Granger panel (excluded from the plot):
    # draw_directed(axes[2], measures["granger_edges"], pos_g)
    # panel labels (bold letter + non-bold name) in place of per-panel titles
    names = ["Spearman's Correlation", "Phase Locking Value"]  # , "Granger Causality"
    for idx, (ax, name) in enumerate(zip(axes, names)):
        panel_label(ax, chr(ord("a") + idx), name, dx=0.18 if idx == 0 else 0.0)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.02, wspace=0.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")#, pad_inches=0.05)
    plt.close(fig)
    print(f"Saved figure to {os.path.abspath(out_path)}")


def main():
    plot_measures(compute_measures(), OUT_PATH)


if __name__ == "__main__":
    main()
