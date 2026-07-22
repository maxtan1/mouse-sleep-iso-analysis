import sys, pickle
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, tukey_hsd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
from dtw import dtw
""" Cite:
T. Giorgino. Computing and Visualizing Dynamic Time Warping Alignments in R: The dtw Package.
J. Stat. Soft., doi:10.18637/jss.v031.i07.
"""


def main():
    # read the result from get_iso.py
    suffix = sys.argv[1]#'_ShenLab'
    df = pd.read_csv(f'iso_results_all_subjects_bands{suffix}.csv')
    band_names = ['gamma', 'beta', 'sigma', 'alpha', 'theta', 'delta']

    # load bout_band_powers_all_subjects_bands so that we can use it for cross-band correlation analysis
    with open(f'data_all_subjects_bands{suffix}.pickle', 'rb') as f:
        result = pickle.load(f)
    iso_freq_all_subjects_bands = result['iso_freq']
    avg_iso_psd_all_subjects_bands = result['avg_iso_psd']

    # get the single peak template based on those band power (BP) among top 10, and peak frequency (PF) between 0.015 to 0.025.
    # sort df by iso_bp_n in descending order
    df2 = df.sort_values('iso_bp_n', ascending=False, ignore_index=True) 
    df_top = df2.iloc[:10]  # take top 10

    # because each mouse can have different length of NREM bout, leading to different frequency resolution in the PSD
    # we need to do linear interpolation to get the same frequency for all mice, and then take average to get the template
    # but what frequencies shall align to?
    # we can take a grid with constant interval 0.001Hz, 0.002Hz, ... 0.05Hz
    iso_freq_grid = np.arange(0.001, 0.05, 0.001)
    iso_dbs = []
    for i in range(len(df_top)):
        sid = df_top.sid.iloc[i]
        edf_path = df_top.edf.iloc[i]
        band_name = df_top.band_name.iloc[i]
        iso_freq = iso_freq_all_subjects_bands[(sid, edf_path, band_name)]
        avg_iso_psd = avg_iso_psd_all_subjects_bands[(sid, edf_path, band_name)]
        iso_db = 10 * np.log10(avg_iso_psd)

        # linear interpolation
        iso_db_interp = np.interp(iso_freq_grid, iso_freq, iso_db, left=np.nan, right=np.nan)
        iso_dbs.append(iso_db_interp) # iso_freq_grid is always the same for all mice

    # take average to get the template
    iso_dbs = np.array(iso_dbs)
    iso_db_template = np.nanmean(iso_dbs, axis=0)

    # remove NaN
    good_mask = ~np.isnan(iso_db_template)
    iso_db_template = iso_db_template[good_mask]
    iso_freq_grid = iso_freq_grid[good_mask]
    iso_dbs = iso_dbs[:, good_mask]

    # plot the template
    plt.close()
    plt.plot(iso_freq_grid, iso_dbs.T, c='k', alpha=0.1)
    plt.plot(iso_freq_grid, iso_db_template, c='r', lw=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB)')
    sns.despine()
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'ISO_PSD_template{suffix}.png', dpi=300, bbox_inches='tight')

    # quantify the "single-peakness" of (iso_freq, iso_db) based on
    # dynamic time warping (DTW) distance to a single-peak template
    # pros: very effective for handling sequences that vary in speed, length, or phase; non-parametric (simple)
    # cons: very slow process and computationally expensive; sensitive to outliers
    # alternatives: scipy find_peaks, or fitting a Gaussian function and looking at the goodness of fit (R^2)
    df['DTWDistToSinglePeak'] = np.nan
    for i in range(len(df)):
        sid = df.sid.iloc[i]
        edf_path = df.edf.iloc[i]
        band_name = df.band_name.iloc[i]
        iso_freq = iso_freq_all_subjects_bands[(sid, edf_path, band_name)]
        avg_iso_psd = avg_iso_psd_all_subjects_bands[(sid, edf_path, band_name)]
        iso_db = 10 * np.log10(avg_iso_psd)
        alignment = dtw(iso_db, iso_db_template, distance_only=True)
        df.loc[i, 'DTWDistToSinglePeak'] = float(alignment.distance)
    df = df.sort_values('DTWDistToSinglePeak', ascending=True, ignore_index=True)
    print(df)
    breakpoint()
    df.to_csv(f'iso_results_all_subjects_bands_with_DTWDist{suffix}.csv', index=False)
    """
MSSV
df[['band_name','DTWDistToSinglePeak']].groupby('band_name').agg('mean').reset_index().sort_values('DTWDistToSinglePeak')
  band_name  DTWDistToSinglePeak
0     alpha            91.807757
5     theta           121.487359
2     delta           231.273627
4     sigma           273.066545

1      beta           405.566233

3     gamma          3201.717126

f_oneway(*[df.DTWDistToSinglePeak[df.band_name==x].values for x in band_names])
F_onewayResult(statistic=259.94930167495704, pvalue=1.115323958861537e-170)

band_names
['gamma', 'beta', 'sigma', 'alpha', 'theta', 'delta']
print(tukey_hsd(*[df.DTWDistToSinglePeak[df.band_name==x].values for x in band_names]))
Tukey's HSD Pairwise Group Comparisons (95.0% Confidence Interval)
Comparison  Statistic  p-value  Lower CI  Upper CI
 (0 - 1)   2796.151     0.000  2490.378  3101.923
 (0 - 2)   2928.651     0.000  2622.878  3234.423
 (0 - 3)   3109.909     0.000  2804.137  3415.682
 (0 - 4)   3080.230     0.000  2774.457  3386.002
 (0 - 5)   2970.443     0.000  2664.671  3276.216
 (1 - 2)    132.500     0.818  -173.273   438.272
 (1 - 3)    313.758     0.040     7.986   619.531
 (1 - 4)    284.079     0.086   -21.694   589.851
 (1 - 5)    174.293     0.580  -131.480   480.065
 (2 - 3)    181.259     0.537  -124.514   487.031
 (2 - 4)    151.579     0.717  -154.193   457.352
 (2 - 5)     41.793     0.999  -263.980   347.565
 (3 - 4)    -29.680     1.000  -335.452   276.093
 (3 - 5)   -139.466     0.784  -445.238   166.307
 (4 - 5)   -109.786     0.909  -415.559   195.986


OxfordBenchmark:
df[['band_name','DTWDistToSinglePeak']].groupby('band_name').agg('mean').reset_index().sort_values('DTWDistToSinglePeak')
  band_name  DTWDistToSinglePeak
0     alpha            38.064372
5     theta            75.787038
2     delta           173.443541

4     sigma           456.882555
1      beta           778.046296
3     gamma          1714.158626

f_oneway(*[df.DTWDistToSinglePeak[df.band_name==x].values for x in band_names])
F_onewayResult(statistic=256.60856567293524, pvalue=4.311597592898174e-27)

band_names
['gamma', 'beta', 'sigma', 'alpha', 'theta', 'delta']
print(tukey_hsd(*[df.DTWDistToSinglePeak[df.band_name==x].values for x in band_names]))
Tukey's HSD Pairwise Group Comparisons (95.0% Confidence Interval)
Comparison  Statistic  p-value  Lower CI  Upper CI
 (0 - 1)    936.112     0.000   766.289  1105.936
 (0 - 2)   1257.276     0.000  1087.453  1427.099
 (0 - 3)   1676.094     0.000  1506.271  1845.918
 (0 - 4)   1638.372     0.000  1468.548  1808.195
 (0 - 5)   1540.715     0.000  1370.892  1710.538
 (1 - 2)    321.164     0.000   151.340   490.987
 (1 - 3)    739.982     0.000   570.159   909.805
 (1 - 4)    702.259     0.000   532.436   872.083
 (1 - 5)    604.603     0.000   434.779   774.426
 (2 - 3)    418.818     0.000   248.995   588.642
 (2 - 4)    381.096     0.000   211.272   550.919
 (2 - 5)    283.439     0.000   113.616   453.262
 (3 - 4)    -37.723     0.984  -207.546   132.101
 (3 - 5)   -135.379     0.184  -305.203    34.444
 (4 - 5)    -97.657     0.522  -267.480    72.167

ShenLab: template is not single peak
    """



if __name__ == "__main__":
    main() 
