"""stepb: across-subject effects + hierarchical, discovery-gated FDR.

stepa produces the per-recording, per-directed-pair file
`cross_freq_corr_results_{dataset}.csv`. Here we collapse those to across-subject
effect sizes / p-values and apply Benjamini-Hochberg FDR in a hierarchical scheme
that keeps the multiple-comparison burden small:

  1. MSSV is the discovery dataset (largest WT cohort). It is split by `lab`
     (from mastersheet_MSSV.csv). Within each lab we compute the across-subject
     result and BH-FDR over the full set of tests per measure (15 Spearman,
     15 PLV, 30 Granger directions). The discovery set for a measure is the set of
     (band1, band2) hypotheses that are FDR-significant AND have the same effect
     sign in *every* lab -- the reproducible-across-labs core.

  2. OxfordBenchmark (WT) is corrected ONLY over the discovery hypotheses (a much
     smaller number of tests), giving the final WT results.

  3. Oscar (TG) is likewise corrected only over the discovery hypotheses, giving
     the final WT-vs-TG comparison results.

Only the x->y row carries the undirected Spearman/PLV values (15 of 30 rows);
Granger is directional and present in all 30. A hypothesis is a (band1, band2)
key: undirected for Spearman/PLV, directed for Granger. The key labeling is
identical across datasets (set by stepa), so discovery keys match directly.

For each dataset a `*_p_value_fdr` column is written per measure (NaN where the
hypothesis was not tested, i.e. absent measure or, for the gated datasets, not in
the discovery set).

Usage:  python stepb_get_fdr_across_subjects_cross_band_coupling.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

RESULTS = 'cross_freq_corr_results_{dataset}.csv'
OUT = 'cross_freq_corr_results_across_subjects_fdr_{dataset}.csv'
DISCOVERY = 'MSSV'
GATED = ['OxfordBenchmark', 'Oscar']
ALPHA = 0.05

# measure -> (raw p-value column, effect/presence column, FDR column)
MEASURES = {
    'spearman': ('spearman_p_value', 'spearman_corr', 'spearman_p_value_fdr'),
    'plv':      ('plv_p_value', 'plv', 'plv_p_value_fdr'),
    'granger':  ('granger_p_value', 'granger_f', 'granger_p_value_fdr'),
}


def _intercept_pvalue(g, value_col, one_sided):
    """Population mean & its p-value, with subject as the unit of replication.

    Each subject is first collapsed to its mean of `value_col`, then a one-sample
    t-test of those subject means against 0 gives the p-value. Collapsing to one
    value per subject before testing is what keeps recordings/bouts from
    inflating significance: the test's degrees of freedom come from the number of
    subjects, not the number of recordings. (An earlier version fit a
    random-intercept mixed model `value ~ 1 + (1 | subject)` on the recording-level
    rows and reported the Wald intercept p; with few subjects the random-effect
    variance is under-estimated, so that p was driven by the recording count and
    came out wildly anti-conservative -- e.g. ~1e-132 where the subject-level
    t-test gives ~1e-6. The subject-means t-test avoids that and treats every
    dataset the same way.)

    Returns (coef, p); p is halved to a one-sided test when `one_sided` (effect
    expected positive).
    """
    g = g.dropna(subset=[value_col, 'subject'])
    if g['subject'].nunique() < 3:
        return np.nan, np.nan
    sub_means = g.groupby('subject')[value_col].mean().to_numpy()
    coef = float(sub_means.mean())
    p = float(ttest_1samp(sub_means, 0.0).pvalue)
    if one_sided:
        p = p / 2 if coef > 0 else 1 - p / 2
    return coef, p


def across_subjects(df_long, ms):
    """One row per *directed* pair with each measure's across-subject effect size
    and significance, laid out like the per-recording file.

    Each measure is combined with the same subject-level test (`_intercept_pvalue`:
    collapse to one value per subject, then one-sample t-test vs 0), subject
    resolved from the mastersheet `ms` (columns SID, EDFPath):
      - Granger : log(F/F_null) vs 0, one-sided, one test per direction (30)
                  (F_null = d2/(d2-2), the recording's E[F] under H0, so the
                  centre is 0 under no causality rather than assuming E[F]=1)
      - Spearman: Fisher-z(r) vs 0, two-sided (correlation may be +/-)   (15)
      - PLV     : surrogate z vs 0, one-sided                            (15)
    Undirected measures are present only on the x -> y row (NaN on y -> x), so
    Spearman/PLV fill 15 of the 30 rows. Raw (uncorrected) p-values are returned;
    FDR correction happens in `discovery_fdr` / `gated_fdr`.
    """
    smap = {os.path.basename(str(p)): s for s, p in zip(ms['SID'], ms['EDFPath'])}
    df = df_long.copy()
    df['subject'] = df['edf'].map(lambda p: smap.get(os.path.basename(str(p))))
    df = df.dropna(subset=['subject'])
    df['_fisher_z'] = np.arctanh(df['spearman_corr'].clip(-0.999999, 0.999999))
    # centre log(mean F) on log of the recording's null-expected F so the H0
    # centre is 0 (F under H0 has mean d2/(d2-2), not 1)
    df['_log_f'] = np.log(df['granger_f']) - np.log(df['granger_f_null'])

    rows = []
    for (b1, b2), g in df.groupby(['band1', 'band2']):
        row = dict(band1=b1, band2=b2, n_subjects=int(g['subject'].nunique()))
        # Granger: present in every directed row
        _, gc_p = _intercept_pvalue(g, '_log_f', one_sided=True)
        row['granger_f'] = float(g.groupby('subject')['granger_f'].mean().mean())
        row['granger_p_value'] = gc_p
        # Spearman & PLV: only the x -> y row carries the undirected values
        if g['spearman_corr'].notna().any():
            _, sp_p = _intercept_pvalue(g, '_fisher_z', one_sided=False)
            _, plv_p = _intercept_pvalue(g, 'plv_z', one_sided=True)
            row['spearman_corr'] = float(g.groupby('subject')['spearman_corr'].mean().mean())
            row['spearman_p_value'] = sp_p
            row['plv'] = float(g.groupby('subject')['plv'].mean().mean())
            row['plv_z'] = float(g.groupby('subject')['plv_z'].mean().mean())
            row['plv_p_value'] = plv_p
        else:
            row.update(spearman_corr=np.nan, spearman_p_value=np.nan,
                       plv=np.nan, plv_z=np.nan, plv_p_value=np.nan)
        rows.append(row)

    out = pd.DataFrame(rows)
    return out[['band1', 'band2', 'n_subjects',
                'spearman_corr', 'spearman_p_value',
                'plv', 'plv_z', 'plv_p_value',
                'granger_f', 'granger_p_value']]


def _bh_fdr(df, pcol, tested):
    """BH-FDR q-values over the rows where `tested` is True; NaN elsewhere."""
    q = pd.Series(np.nan, index=df.index)
    idx = df.index[tested]
    if len(idx):
        q.loc[idx] = multipletests(df.loc[idx, pcol].to_numpy(), method='fdr_bh')[1]
    return q


def _keys(df):
    """(band1, band2) hypothesis keys (directed for Granger, canonical x->y else)."""
    return pd.Series(list(zip(df['band1'], df['band2'])), index=df.index)


def discovery_fdr():
    """MSSV discovery, split by lab.

    For every lab we build the across-subject result and BH-FDR per measure, then
    take, for each measure, the (band1, band2) hypotheses that are FDR-significant
    with a consistent effect sign in *all* labs as the discovery set.

    Returns sig_keys: measure -> {discovery (band1, band2) key: shared effect sign}.
    """
    ms = pd.read_csv(f'mastersheet_{DISCOVERY}.csv')
    df_res = pd.read_csv(RESULTS.format(dataset=DISCOVERY))
    df_res['EDFPath'] = df_res.edf.str.replace('../', '', regex=False)
    df_res = df_res.merge(ms[['EDFPath', 'lab']], on='EDFPath', how='left')

    # per lab, per measure: {key: sign of effect} for FDR-significant hypotheses
    lab_sig = {}
    for lab, g in df_res.groupby('lab'):
        across = across_subjects(g, ms)
        keys = _keys(across)
        sig = {}
        for m, (pcol, ecol, qcol) in MEASURES.items():
            tested = across[ecol].notna() & across[pcol].notna()
            across[qcol] = _bh_fdr(across, pcol, tested)
            signif = tested & (across[qcol] < ALPHA)
            sig[m] = {keys[i]: np.sign(across.loc[i, ecol]) for i in across.index[signif]}
            print(f'  [{DISCOVERY}/{lab}] {m}: {int(tested.sum())} tests, '
                  f'{len(sig[m])} significant (q<{ALPHA})')
        lab_sig[lab] = sig
        across.to_csv(OUT.format(dataset=f'{DISCOVERY}_{lab}'), index=False)

    labs = list(lab_sig)
    sig_keys = {}
    for m in MEASURES:
        common = set.intersection(*(set(lab_sig[lab][m]) for lab in labs)) if labs else set()
        sig_keys[m] = {}
        for k in common:
            signs = {lab_sig[lab][m][k] for lab in labs}
            # keep only hypotheses whose effect sign agrees across every lab, and
            # record that shared sign so the gated datasets can require it too
            if len(signs) == 1:
                sig_keys[m][k] = signs.pop()
        print(f'  [{DISCOVERY} discovery] {m}: {len(sig_keys[m])} keys '
              f'significant & same-sign across {len(labs)} labs')
    return sig_keys


def gated_fdr(dataset, sig_keys):
    """FDR only over the discovery hypotheses (reduced test set).

    A hypothesis is tested here only if it is a discovery key AND its effect points
    the same way as in discovery (same sign), so a reversed effect can never be
    reported as a confirmation.
    """
    ms = pd.read_csv(f'mastersheet_{dataset}.csv')
    df_res = pd.read_csv(RESULTS.format(dataset=dataset))
    across = across_subjects(df_res, ms)
    keys = _keys(across)
    for m, (pcol, ecol, qcol) in MEASURES.items():
        signs = sig_keys[m]  # discovery key -> shared effect sign
        same_sign = pd.Series(
            [keys[i] in signs and np.sign(across.loc[i, ecol]) == signs[keys[i]]
             for i in across.index], index=across.index)
        tested = across[ecol].notna() & across[pcol].notna() & same_sign
        across[qcol] = _bh_fdr(across, pcol, tested)
        n_sig = int((across[qcol] < ALPHA).sum())
        print(f'  [{dataset}] {m}: {int(tested.sum())} tests (gated by discovery '
              f'key & sign), {n_sig} significant (q<{ALPHA})')
    across.to_csv(OUT.format(dataset=dataset), index=False)


def main():
    print(f'Discovery FDR on {DISCOVERY} (per lab, full test set):')
    sig_keys = discovery_fdr()
    for dataset in GATED:
        print(f'Gated FDR on {dataset}:')
        gated_fdr(dataset, sig_keys)


if __name__ == '__main__':
    main()
