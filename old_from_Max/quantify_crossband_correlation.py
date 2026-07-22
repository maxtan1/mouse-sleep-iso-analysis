from collections import defaultdict
from itertools import combinations
import os, datetime
import pickle
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import spearmanr
from npeet.entropy_estimators import mi
from tqdm import tqdm


def main():
    df = pd.read_csv('mastersheet.csv')
    band_names = ['gamma', 'beta', 'sigma', 'alpha', 'theta', 'delta']

    # load bout_band_powers_all_subjects_bands so that we can use it for cross-band correlation analysis
    with open('data_all_subjects_bands.pickle', 'rb') as f:
        res = pickle.load(f)
    bout_phases = res['bout_phases']
    bout_band_powers = res['bout_band_powers']

    circ_rs = defaultdict(list)
    circ_ps = defaultdict(list)
    sp_rs = defaultdict(list)
    sp_ps = defaultdict(list)
    plvs = defaultdict(list)
    ccmis = defaultdict(list)
    sidss = defaultdict(list)
    edf_pathss = defaultdict(list)
    for i in tqdm(range(len(df))):
        sid = df.SID.iloc[i]
        edf_path = df.EDFPath.iloc[i]
        n_bout = len(bout_phases[(sid, edf_path, band_names[0])])
        for x,y in combinations(band_names, 2):
            bps_x = []
            bps_y = []
            phases_x = []
            phases_y = []
            for bi in range(n_bout):
                bps_x.extend(bout_band_powers[(sid, edf_path, x)][bi])
                bps_y.extend(bout_band_powers[(sid, edf_path, y)][bi])
                phases_x.extend(bout_phases[(sid, edf_path, x)][bi])
                phases_y.extend(bout_phases[(sid, edf_path, y)][bi])
            bps_x = np.array(bps_x)
            bps_y = np.array(bps_y)
            phases_x = np.array(phases_x)
            phases_y = np.array(phases_y)
            circ_r, circ_p = pg.circ_corrcc(phases_x, phases_y)
            circ_rs[(x,y)].append(circ_r)
            circ_ps[(x,y)].append(circ_p)

            phase_diff = phases_x - phases_y
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plvs[(x,y)].append(plv)

            mi_value = mi(
                np.column_stack([np.cos(phases_x), np.sin(phases_x)]),
                np.column_stack([np.cos(phases_y), np.sin(phases_y)]), )
            ccmis[(x,y)].append(mi_value)

            r, p = spearmanr(bps_x, bps_y)
            sp_rs[(x,y)].append(r)
            sp_ps[(x,y)].append(p)

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
            'circ_corr': circ_rs[(x, y)],
            'circ_corr_p': circ_ps[(x, y)],
            'plv':plvs[(x,y)],
            'circ_mi':ccmis[(x,y)],
        }))
    df_res = pd.concat(rows, ignore_index=True)
    print(df_res)
    breakpoint()
    df_res.to_csv('cross_freq_corr_results.csv', index=False)

    """
print(df_res[['band1','band2','spearman_corr']][df_res.spearman_corr_p<0.05].groupby(['band1','band2']).agg('mean').reset_index().sort_values('spearman_corr')    band1  band2  spearman_corr
14  theta  delta      -0.751933
0   alpha  delta      -0.625725
3    beta  delta      -0.582548
12  sigma  delta      -0.576788
8   gamma  delta      -0.299735
10  gamma  theta       0.154946
6   gamma  alpha       0.206658
9   gamma  sigma       0.230960
5    beta  theta       0.253227
13  sigma  theta       0.280613
2    beta  alpha       0.422240
7   gamma   beta       0.448661
1   alpha  theta       0.525477
4    beta  sigma       0.539349
11  sigma  alpha       0.608821


print(df_res[['band1','band2','circ_corr']][df_res.circ_corr_p<0.05].groupby(['band1','band2']).agg('mean').reset_index().sort_values('circ_corr'))
    band1  band2  circ_corr
9   gamma  sigma   0.070424
6   gamma  alpha   0.078570
10  gamma  theta   0.112018
8   gamma  delta   0.132183
5    beta  theta   0.217938
13  sigma  theta   0.218716
2    beta  alpha   0.313141
7   gamma   beta   0.321753
3    beta  delta   0.326999
12  sigma  delta   0.347423
4    beta  sigma   0.355108
1   alpha  theta   0.359313
0   alpha  delta   0.415539
11  sigma  alpha   0.484463
14  theta  delta   0.495984


df_res[['band1','band2','plv']].groupby(['band1','band2']).agg('mean').reset_index().sort_values('plv')
    band1  band2       plv
10  gamma  theta  0.155711
6   gamma  alpha  0.169302
9   gamma  sigma  0.169568
8   gamma  delta  0.221621
13  sigma  theta  0.250639
5    beta  theta  0.266921
7   gamma   beta  0.367045
2    beta  alpha  0.380032
1   alpha  theta  0.455829
4    beta  sigma  0.467111
3    beta  delta  0.507394
12  sigma  delta  0.512625
11  sigma  alpha  0.548210
0   alpha  delta  0.565458
14  theta  delta  0.683947


print(df_res[['band1','band2','circ_mi']][df_res.circ_mi>0].groupby(['band1','band2']).agg('mean').reset_index().sort_values('circ_mi'))
    band1  band2   circ_mi
10  gamma  theta  0.082453
9   gamma  sigma  0.083286
6   gamma  alpha  0.099001
8   gamma  delta  0.135658
13  sigma  theta  0.138000
5    beta  theta  0.173570
7   gamma   beta  0.282565
2    beta  alpha  0.307293
1   alpha  theta  0.365891
4    beta  sigma  0.422529
12  sigma  delta  0.474381
3    beta  delta  0.474870
11  sigma  alpha  0.545397
0   alpha  delta  0.586868
14  theta  delta  0.929593
    """


if __name__=='__main__':
    main()