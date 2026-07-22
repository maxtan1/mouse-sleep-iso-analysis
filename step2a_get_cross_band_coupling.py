from collections import defaultdict
from itertools import combinations
import sys, pickle
import numpy as np
import pandas as pd
#import pingouin as pg
from scipy.stats import spearmanr
# from npeet.entropy_estimators import mi
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm


N_SURROGATE = 2000
SEED = 12345


def plv_surrogate_test(phases_a_bouts, phases_b_bouts, rng, n_surrogate=N_SURROGATE):
    """Surrogate significance of the phase-locking value for one recording/pair.

    PLV is the mean resultant length of the phase difference; under the null of
    no phase coupling that difference is circularly uniform. A parametric
    (Rayleigh) test is over-optimistic here because phase samples within a bout
    are strongly autocorrelated (infraslow modulation), so we use a within-bout
    circular-shift surrogate: the second band's phase is circularly rotated
    within every bout by an independent random lag, destroying the cross-band
    relationship while preserving each band's own phase distribution AND its
    within-bout autocorrelation.

    The full circular-shift null for a bout is one FFT: the circular
    cross-correlation ccf[k] = sum_n cx[n] conj(cy[n-k]) is exactly the bout's
    complex PLV sum at shift k, so each surrogate samples one random lag per bout
    and sums the pre-computed complex values across bouts.

    Each argument is a list of one 1-D phase array per bout. Returns:
      plv_p : (1 + #{surrogate PLV >= observed}) / (n_surrogate + 1)
      plv_z : (observed PLV - mean surrogate) / std surrogate
    """
    ccfs, lengths, total, s_obs = [], [], 0, 0.0 + 0.0j
    for pa, pb in zip(phases_a_bouts, phases_b_bouts):
        pa = np.asarray(pa, float)
        pb = np.asarray(pb, float)
        ok = np.isfinite(pa) & np.isfinite(pb)
        if ok.sum() < 2:                      # a length-1 bout has no shift null
            continue
        cx = np.exp(1j * pa[ok])
        cy = np.exp(1j * pb[ok])
        ccf = np.fft.ifft(np.fft.fft(cx) * np.conj(np.fft.fft(cy)))
        ccfs.append(ccf)
        lengths.append(len(ccf))
        total += len(ccf)
        s_obs += ccf[0]                       # k=0 is the true alignment
    if total == 0 or not ccfs:
        return np.nan, np.nan

    plv_obs = np.abs(s_obs) / total
    surr_sum = np.zeros(n_surrogate, dtype=complex)
    for ccf, L in zip(ccfs, lengths):         # random lag per bout, exclude 0
        lags = rng.integers(1, L, size=n_surrogate) if L > 1 else np.zeros(n_surrogate, int)
        surr_sum += ccf[lags]
    plv_surr = np.abs(surr_sum) / total

    p = (1 + np.sum(plv_surr >= plv_obs)) / (n_surrogate + 1)
    sd = plv_surr.std()
    z = (plv_obs - plv_surr.mean()) / sd if sd > 0 else np.nan
    return float(p), float(z)


# undirected per-pair measures (one value per unordered band pair). Granger is
# directional and is handled separately (one value per ordered pair/direction).
UNDIRECTED_COLS = ['spearman_corr', 'spearman_corr_p',  # 'circ_corr', 'circ_corr_p',
                   'plv', 'plv_p', 'plv_z']  # 'circ_mi'


def to_directed_long(df_wide):
    """Reshape the wide per-pair table to one row per *directed* pair (15x2=30).

    Granger is directional, so each unordered pair {x, y} becomes two rows:
      x -> y : granger_f from the band1->band2 columns
      y -> x : granger_f from the band2->band1 columns
    The undirected measures (Spearman, PLV, ...) are kept only on the x -> y row
    and set to NaN on the y -> x row, so exactly one direction per pair carries
    them. Everything downstream can then treat the whole file as (band1=source,
    band2=target).
    """
    keep_ids = [c for c in ['sid', 'edf'] if c in df_wide.columns]
    fwd = df_wide.copy()
    fwd['granger_f'] = df_wide['granger_f_band1_to_band2']
    fwd['granger_f_null'] = df_wide['granger_fnull_band1_to_band2']

    rev = df_wide.copy()
    rev['band1'] = df_wide['band2']
    rev['band2'] = df_wide['band1']
    rev['granger_f'] = df_wide['granger_f_band2_to_band1']
    rev['granger_f_null'] = df_wide['granger_fnull_band2_to_band1']
    rev[UNDIRECTED_COLS] = np.nan

    cols = ['band1', 'band2'] + keep_ids + UNDIRECTED_COLS + ['granger_f', 'granger_f_null']
    return pd.concat([fwd[cols], rev[cols]], ignore_index=True)


def granger_causality(source_bouts, target_bouts, lag=3):
    """Test whether `source` Granger-causes `target`, i.e. whether past values of
    the source band power improve prediction of the target band power.

    Granger causality needs temporally ordered samples, so it is computed within
    each bout separately and then averaged. `grangercausalitytests` treats the
    2nd column as the candidate cause of the 1st column, so we stack [target, source].

    A single fixed `lag` is used (not the best lag across a range) to avoid the
    selection bias that would deflate p-values. With a 4 s sampling step, lag=3
    corresponds to 12 s of history (~1/4 of an infraslow modulation cycle).

    Returns (mean F-stat, mean null-expected F) over the bouts that were long
    enough (NaN, NaN if none are). Under H0 (no Granger causality) the per-bout
    F-statistic is F-distributed with mean d2/(d2-2), not 1, where d2 is the
    denominator df of that bout's test. The second return value is the mean of
    those per-bout null means; the caller centers log(mean F) on log of it so the
    across-subject test has the correct H0 center instead of assuming E[F]=1.

    Significance is *not* computed here: combining per-bout p-values (e.g. Fisher)
    makes the result grow with the number of bouts, reintroducing exactly the
    pseudo-replication that the subject-level test in `_intercept_pvalue` avoids.
    Group significance is instead derived across subjects from log(F/F_null) vs 0
    (see `across_subjects`).
    """
    fvals = []
    fnulls = []
    for src, tgt in zip(source_bouts, target_bouts):
        src = np.asarray(src, dtype=float)
        tgt = np.asarray(tgt, dtype=float)
        ids = (~np.isnan(src)) & (~np.isnan(tgt))
        src = src[ids]
        tgt = tgt[ids]
        # need enough samples to fit the lagged regression
        if len(src) < 3 * lag + 1:
            continue
        data = np.column_stack([tgt, src])
        try:
            res = grangercausalitytests(data, maxlag=lag, verbose=False)
        except Exception:
            continue
        # ssr_ftest = (F, p, df_denom, df_num); df_denom (d2) sets the null mean
        f, _, df_denom, _ = res[lag][0]['ssr_ftest']
        if df_denom <= 2:                 # mean d2/(d2-2) undefined for d2 <= 2
            continue
        fvals.append(f)
        fnulls.append(df_denom / (df_denom - 2.0))
    if len(fvals) == 0:
        return np.nan, np.nan
    return np.nanmean(fvals), np.nanmean(fnulls)


def main():
    suffix = sys.argv[1]
    df = pd.read_csv(f'iso_results_all_subjects_bands_{suffix}.csv')
    df = df.drop_duplicates(subset=['sid', 'edf'], ignore_index=True)
    band_names = ['gamma', 'beta', 'sigma', 'alpha', 'theta', 'delta']

    # load bout_band_powers_all_subjects_bands so that we can use it for cross-band correlation analysis
    with open(f'data_all_subjects_bands_{suffix}.pickle', 'rb') as f:
        res = pickle.load(f)
    bout_phases = res['bout_phases']
    bout_band_powers = res['bout_band_powers_f']

    # circ_rs = defaultdict(list)
    # circ_ps = defaultdict(list)
    sp_rs = defaultdict(list)
    sp_ps = defaultdict(list)
    plvs = defaultdict(list)
    plv_ps = defaultdict(list)
    plv_zs = defaultdict(list)
    # ccmis = defaultdict(list)
    granger_f_xy = defaultdict(list)
    granger_f_yx = defaultdict(list)
    granger_fnull_xy = defaultdict(list)
    granger_fnull_yx = defaultdict(list)
    sidss = defaultdict(list)
    edf_pathss = defaultdict(list)
    rng = np.random.default_rng(SEED)
    for i in tqdm(range(len(df))):
        sid = df.sid.iloc[i]
        edf_path = df.edf.iloc[i]
        n_bout = len(bout_phases[(sid, edf_path, band_names[0])])
        for x,y in combinations(band_names, 2):
            bps_x_bouts = []
            bps_y_bouts = []
            phases_x = []
            phases_y = []
            for bi in range(n_bout):
                bps_x_bouts.append(np.asarray(bout_band_powers[(sid, edf_path, x)][bi], dtype=float))
                bps_y_bouts.append(np.asarray(bout_band_powers[(sid, edf_path, y)][bi], dtype=float))
                phases_x.extend(bout_phases[(sid, edf_path, x)][bi])
                phases_y.extend(bout_phases[(sid, edf_path, y)][bi])
            bps_x = np.concatenate(bps_x_bouts)
            bps_y = np.concatenate(bps_y_bouts)
            ids = (~np.isnan(bps_x))&(~np.isnan(bps_y))
            bps_x = bps_x[ids]
            bps_y = bps_y[ids]
            phases_x = np.array(phases_x)
            phases_y = np.array(phases_y)
            ids = (~np.isnan(phases_x))&(~np.isnan(phases_y))
            phases_x = phases_x[ids]
            phases_y = phases_y[ids]
            # circ_r, circ_p = pg.circ_corrcc(phases_x, phases_y)
            # circ_rs[(x,y)].append(circ_r)
            # circ_ps[(x,y)].append(circ_p)

            phase_diff = phases_x - phases_y
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plvs[(x,y)].append(plv)

            # PLV significance via within-bout circular-shift surrogate (uses the
            # per-bout phase structure, not the pooled phases above)
            plv_p, plv_z = plv_surrogate_test(bout_phases[(sid, edf_path, x)],
                                              bout_phases[(sid, edf_path, y)], rng)
            plv_ps[(x,y)].append(plv_p)
            plv_zs[(x,y)].append(plv_z)

            # mi_value = mi(
            #     np.column_stack([np.cos(phases_x), np.sin(phases_x)]),
            #     np.column_stack([np.cos(phases_y), np.sin(phases_y)]), )
            # ccmis[(x,y)].append(mi_value)

            r, p = spearmanr(bps_x, bps_y)
            sp_rs[(x,y)].append(r)
            sp_ps[(x,y)].append(p)

            # Granger causality on band power, both directions (per-bout, averaged)
            f_xy, fnull_xy = granger_causality(bps_x_bouts, bps_y_bouts)
            f_yx, fnull_yx = granger_causality(bps_y_bouts, bps_x_bouts)
            granger_f_xy[(x,y)].append(f_xy)
            granger_f_yx[(x,y)].append(f_yx)
            granger_fnull_xy[(x,y)].append(fnull_xy)
            granger_fnull_yx[(x,y)].append(fnull_yx)

            sidss[(x,y)].append(sid)
            edf_pathss[(x,y)].append(edf_path)

    rows = []
    for (x, y) in sidss:
        rows.append(pd.DataFrame({
            'band1': x,
            'band2': y,
            'sid': sidss[(x, y)],
            'edf': edf_pathss[(x, y)],
            'spearman_corr': sp_rs[(x, y)],
            'spearman_corr_p': sp_ps[(x, y)],
            # 'circ_corr': circ_rs[(x, y)],
            # 'circ_corr_p': circ_ps[(x, y)],
            'plv':plvs[(x,y)],
            'plv_p':plv_ps[(x,y)],
            'plv_z':plv_zs[(x,y)],
            # 'circ_mi':ccmis[(x,y)],
            'granger_f_band1_to_band2': granger_f_xy[(x, y)],
            'granger_f_band2_to_band1': granger_f_yx[(x, y)],
            'granger_fnull_band1_to_band2': granger_fnull_xy[(x, y)],
            'granger_fnull_band2_to_band1': granger_fnull_yx[(x, y)],
        }))
    df_wide = pd.concat(rows, ignore_index=True)
    # one row per directed pair (Granger is directional; undirected measures kept
    # on the x->y row only, NaN on y->x)
    df_res = to_directed_long(df_wide)
    print(df_res)
    df_res.to_csv(f'cross_freq_corr_results_{suffix}.csv', index=False)

    # across-subject effect sizes, significance, and hierarchical FDR are computed
    # in stepb from this per-recording file.

    df_res = df_res.drop(columns='edf').groupby(['band1','band2','sid']).agg('mean').reset_index()

    """
MSSV:
print(df_res[['band1','band2','spearman_corr']][df_res.spearman_corr_p<0.05].groupby(['band1','band2']).agg('mean').reset_index().sort_values('spearman_corr'))
    band1  band2  spearman_corr
14  theta  delta      -0.775318
0   alpha  delta      -0.642577
3    beta  delta      -0.599793
12  sigma  delta      -0.591931
8   gamma  delta      -0.327787
10  gamma  theta       0.205598
6   gamma  alpha       0.245347
9   gamma  sigma       0.254478
5    beta  theta       0.302552
13  sigma  theta       0.315273
2    beta  alpha       0.450956
7   gamma   beta       0.462187
1   alpha  theta       0.550820
4    beta  sigma       0.563328
11  sigma  alpha       0.623556


df_res[['band1','band2','plv']].groupby(['band1','band2']).agg('mean').reset_index().sort_values('plv')
    band1  band2       plv
10  gamma  theta  0.170270
9   gamma  sigma  0.180682
6   gamma  alpha  0.181413
8   gamma  delta  0.233780
13  sigma  theta  0.276393
5    beta  theta  0.293673
7   gamma   beta  0.377704
2    beta  alpha  0.401272
1   alpha  theta  0.477574
4    beta  sigma  0.484785
3    beta  delta  0.519571
12  sigma  delta  0.522884
11  sigma  alpha  0.558827
0   alpha  delta  0.577858
14  theta  delta  0.704970


OxfordBenchmark:
print(df_res[['band1','band2','spearman_corr']][df_res.spearman_corr_p<0.05].groupby(['band1','band2']).agg('mean').reset_index().sort_values('spearman_corr'))
    band1  band2  spearman_corr
14  theta  delta      -0.802788
0   alpha  delta      -0.664636
12  sigma  delta      -0.560286
3    beta  delta      -0.545575
8   gamma  delta      -0.381806
10  gamma  theta       0.188844
6   gamma  alpha       0.201546
9   gamma  sigma       0.237692
13  sigma  theta       0.254343
5    beta  theta       0.262640
2    beta  alpha       0.384689
4    beta  sigma       0.494401
7   gamma   beta       0.563903
1   alpha  theta       0.573225
11  sigma  alpha       0.608297

df_res[['band1','band2','plv']].groupby(['band1','band2']).agg('mean').reset_index().sort_values('plv')
    band1  band2       plv
6   gamma  alpha  0.106189
9   gamma  sigma  0.109856
10  gamma  theta  0.151534
5    beta  theta  0.190696
13  sigma  theta  0.199829
8   gamma  delta  0.261038
2    beta  alpha  0.326156
7   gamma   beta  0.406461
4    beta  sigma  0.422576
3    beta  delta  0.431953
12  sigma  delta  0.453947
1   alpha  theta  0.472690
11  sigma  alpha  0.552270
0   alpha  delta  0.578043
14  theta  delta  0.727604


ShenLab:
print(df_res[['band1','band2','spearman_corr']][df_res.spearman_corr_p<0.05].groupby(['band1','band2']).agg('mean').reset_index().sort_values('spearman_corr'))
    band1  band2  spearman_corr
14  theta  delta      -0.847484
0   alpha  delta      -0.674809
3    beta  delta      -0.663636
12  sigma  delta      -0.602696
8   gamma  delta      -0.598895
13  sigma  theta       0.388447
9   gamma  sigma       0.439163
5    beta  theta       0.439755
10  gamma  theta       0.457335
6   gamma  alpha       0.466541
2    beta  alpha       0.539018
1   alpha  theta       0.608745
4    beta  sigma       0.634454
11  sigma  alpha       0.654387
7   gamma   beta       0.677167

df_res[['band1','band2','plv']].groupby(['band1','band2']).agg('mean').reset_index().sort_values('plv')
    band1  band2       plv
13  sigma  theta  0.281153
9   gamma  sigma  0.306932
6   gamma  alpha  0.343610
5    beta  theta  0.346654
10  gamma  theta  0.354639
2    beta  alpha  0.436029
8   gamma  delta  0.471943
1   alpha  theta  0.484955
12  sigma  delta  0.487096
11  sigma  alpha  0.515370
4    beta  sigma  0.523795
0   alpha  delta  0.553452
3    beta  delta  0.560350
7   gamma   beta  0.566626
14  theta  delta  0.759930
    """


if __name__=='__main__':
    main()
