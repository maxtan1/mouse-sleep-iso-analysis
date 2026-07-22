from itertools import groupby
import os, datetime
import pickle
import pyedflib
import numpy as np
import pandas as pd
import mne
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

from dtw import dtw


def main():
    # read the result from get_iso.py
    df = pd.read_csv('iso_results_all_subjects_bands.csv')

    band_names = ['gamma', 'beta', 'sigma', 'alpha', 'theta', 'delta']

    # load bout_band_powers_all_subjects_bands so that we can use it for cross-band correlation analysis
    with open('data_all_subjects_bands.pickle', 'rb') as f:
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
        band_name = df_top.band_name.iloc[i]

        iso_freq = iso_freq_all_subjects_bands[(sid, band_name)]
        avg_iso_psd = avg_iso_psd_all_subjects_bands[(sid, band_name)]
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
    plt.savefig('ISO_PSD_template.png', dpi=300, bbox_inches='tight')

    # quantify the "single-peakness" of (iso_freq, iso_db) based on
    # dynamic time warping (DTW) distance to a single-peak template
    # pros: very effective for handling sequences that vary in speed, length, or phase; non-parametric (simple)
    # cons: very slow process and computationally expensive; sensitive to outliers
    # alternatives: scipy find_peaks, or fitting a Gaussian function and looking at the goodness of fit (R^2)
    df['DTWDistToSinglePeak'] = np.nan
    for i in range(len(df)):
        sid = df.sid.iloc[i]
        band_name = df.band_name.iloc[i]

        iso_freq = iso_freq_all_subjects_bands[(sid, band_name)]
        avg_iso_psd = avg_iso_psd_all_subjects_bands[(sid, band_name)]
        iso_db = 10 * np.log10(avg_iso_psd)

        alignment = dtw(iso_db, iso_db_template, distance_only=True)
        df.loc[i, 'DTWDistToSinglePeak'] = float(alignment.distance)

    df = df=df.sort_values('DTWDistToSinglePeak', ascending=True, ignore_index=True)
    print(df)
    df.to_csv('iso_results_all_subjects_bands_with_DTWDist.csv', index=False)


if __name__ == "__main__":
    main() 