from itertools import groupby
import os, datetime, glob
import pickle
import pyedflib
import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
from scipy.integrate import trapezoid
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import seaborn as sns


def get_iso(edf_path, annot_path, band_lb, band_ub, band_name, to_plot=False):
    # load signal
    signals, signal_hdrs, hdr = pyedflib.highlevel.read_edf(edf_path)
    #print(len(signals))
    #print(signal_hdrs)
    #print(hdr)

    # find the index of the EEG signal, and get its sampling frequency
    ch_names = [h['label'] for h in signal_hdrs]
    #print(ch_names)
    EEG_index = [i for i, name in enumerate(ch_names) if 'EEG' in str(name).upper()]
    EEG_index = EEG_index[0]  # this is the index of the EEG signal, it is an integer
    fs = signal_hdrs[EEG_index]['sample_frequency']  # this is the sampling frequency, it is a float number
    eeg = signals[EEG_index]  # unit = uV

    epoch_duration = 4 # seconds
    epoch_size = int(round(epoch_duration*fs))

    # type(eeg) = numpy.ndarray which means it is a n-dimensional array
    # eeg.shape = (#sample points,)
    # #sample points = sampling_frequency x duration

    # load sleep stages
    # we want to create a variable `sleep_stages`
    # it should be numpy.ndarray data type, it should have the same length as eeg
    # so that we can easily segment eeg based on sleep stages
    # W = 1, NREM = 2, REM = 3

    df_annot = pd.read_csv(annot_path, sep='\t')
    #print(df_annot)
    sleep_stages_ = df_annot.stage.values
    # make brief not-NREM to NREM
    sleep_stages = np.array(sleep_stages_)
    for i in range(len(sleep_stages_)-4):
        if (sleep_stages_[i+2]!=2) and (sleep_stages_[i]==sleep_stages_[i+1]==sleep_stages_[i+3]==sleep_stages_[i+4]==2):
            sleep_stages[i+2] = 2
    for i in range(len(sleep_stages_)-5):
        if (sleep_stages_[i+2]!=2 and sleep_stages_[i+3]!=1) and (sleep_stages_[i]==sleep_stages_[i+1]==sleep_stages_[i+4]==sleep_stages_[i+5]==2):
            sleep_stages[i+2] = 2
            sleep_stages[i+3] = 2
    L = len(eeg)//epoch_size
    sleep_stages = sleep_stages[:L]

    total_band_lb = 0.3
    total_band_ub = 45  # because they are recorded in Europe with AC freq=50Hz, so the upper bound of frequency is 50Hz, but we want to exclude the AC noise, therefore set it to 45Hz

    #3.1 For this analysis, select only non-REM sleep bouts lasting ≥ 96 s (i.e., at least 24 epochs of 4 s); see Figure 2. NOTE: Customized routines are available upon request1.
    minimum_duration = 96 # seconds
    # in `sleep_stages`, find consecutive sequence of 2, check if the duration is >=`minimum_size`
    # if yes, save the start and end sampling index

    # supporse sleep_stages=[1,1,1,2,2,2], then groupby will give us (1, [0,1,2]) and (2, [3,4,5])
    start_end_epochs_NREM = []
    counter = 0
    for k,l in groupby(sleep_stages):
        l2 = list(l)
        if k == 2:
            start_index = counter
            end_index = counter+len(l2)
            length = end_index - start_index
            if length*epoch_duration >= minimum_duration:
                start_end_epochs_NREM.append((start_index, end_index))
        counter +=len(l2)

    #3.2 Extract the power values for the sigma frequency band (10-15 Hz) spectral power in 4-s bins ( Figure 2A and B) using calculations of fast Fourier transforms (FFT)1.
    # get the timeseries of band power
    # method #1: apply band pass filter (e.g., sigma 11-15Hz, do a 11-15Hz band pass filter), and then do Hilbert-Huang transform to get its envelope
    # ---> method #2: first segment the signal into overlapping windows, and for each window, do FFT, and get the band power

    #3.3 Calculate the baseline spectral power for non-REM sleep by averaging the values in each frequency bin for all non-REM sleep epochs (artifacts and epochs of transition between vigilance states are excluded from this averaging). Normalize the sigma power values of each epoch to the mean power of the sigma band during non-REM sleep over the time period of interest. Plot against time (Figure 2C).
    all_bout_band_powers = [] 
    all_bout_phases = [] 

    # segemnt eeg_bout to 4-second epochs, where each epoch has size `epoch_size`
    epochs = np.array([
        eeg[e * epoch_size : (e + 1) * epoch_size]
        for e in range(len(sleep_stages))
    ])  # epochs.shape = (n_epochs, epoch_size)
    # do FFT using mne.time_frquency.multitaper_array (something like this, not exactly, please search for it)
    psd, freqs = mne.time_frequency.psd_array_multitaper(
        epochs, sfreq = fs,
        fmin = total_band_lb,
        fmax = total_band_ub,
        bandwidth = 2.0,
        low_bias = True,
        normalization = 'full',
        verbose = False
    )  # psd.shape = (n_epochs, n_freqs), freqs.shape = (n_freqs,)

    # the unit of `psd` is uV^2/Hz
    # the unit of `freq` is Hz
    # get the band power based on area under curve using the trapezoid area equation
    # search function to do it
    total_band_powers = trapezoid(psd, freqs)  # band_powers.shape = (n_epochs,)

    # get band of interest power
    freq_mask = (freqs >= band_lb) & (freqs <= band_ub)
    band_powers = trapezoid(psd[:, freq_mask], freqs[freq_mask])  # band_powers.shape = (n_epochs,)

    # normalize the band power by the total band power
    band_powers = band_powers / total_band_powers  # band_powers.shape

    band_powers_f = mne.filter.filter_data(band_powers, 1/epoch_duration, 0.01,0.03, verbose=False)
    is_nan_mask = np.isnan(band_powers_f)
    if is_nan_mask.sum()>0:
        assert (np.diff(np.where(is_nan_mask)[0])==1).all()
        iphase = np.zeros(len(band_powers_f))+np.nan
        iphase[~is_nan_mask] = np.angle(hilbert(band_powers_f[~is_nan_mask]))
    else:
        iphase = np.angle(hilbert(band_powers_f))
    for start_index, end_index in start_end_epochs_NREM:
        # for each NREM bout
        all_bout_band_powers.append(band_powers[start_index:end_index])
        all_bout_phases.append(iphase[start_index:end_index])

    # all_bout_band_powers is a list of numpy arrays, where each array is the band power time course of one NREM bout, and the length of each array is the number of epochs in that bout

    #3.4 Calculate the FFT of the sigma power time course with Hamming windowing (welch method) to reveal the oscillatory frequency components of the power dynamics (Figure 2D)1.
    # search for the pros and cons for Welch method vs. the multi-taper method, and decide which one to use for this step
    # Welch is simpler, faster, and sets n_fft to the full signal length and maximizes frequency resolution
    # Multitaper uses multiple orthogonal tapers to reduce variance without sacrificing frequency resolution, but it would not be as good here because the sigma power is so short
    # Should use Welch (Hamming window)
    bout_lens = []
    bout_psds = []
    bout_freqs = []
    for band_powers in all_bout_band_powers:
        psd, freq = mne.time_frequency.psd_array_welch(
            band_powers,
            sfreq = 1/epoch_duration, # because the sampling frequency of the band power time course is 1/epoch_duration
            fmin = 0.005, # because we want to analyze the oscillation of the band power, so the lower bound of frequency should be very low, for example 0.01Hz
            fmax = 0.1, # because we want to analyze the oscillation of the band power, so the upper bound of frequency should be very low, for example 0.5Hz
            n_fft = len(band_powers), # because we want to get the frequency resolution as high as possible, so we can set n_fft to be the length of the band power time course
            window = 'hamming',
            verbose = False
        )  # freq.shape = (n_freqs,), psd.shape = (n_freqs,)
        bout_lens.append(len(band_powers))
        bout_psds.append(psd)
        bout_freqs.append(freq)

    #3.5 Note that since the non-REM sleep bouts have different durations, the resulting FFTs have different frequency resolutions. Interpolate to adjust the resolution to the highest one obtained from the longest non-REM sleep bout and average the FFTs of all bouts.
    # find the longest NREM bout, get its band power time course, do FFT to get the frequency resolution, and then interpolate the FFTs of all bouts to this frequency resolution, and then average the FFTs of all bouts 
    longest_id = np.argmax(bout_lens)
    iso_freq = bout_freqs[longest_id]
    iso_psds = []
    for i in range(len(bout_psds)):
        if i != longest_id:
            # interpolate bout_psds[i] to the frequency resolution of bout_psds[longest_id]
            # search for function to do it
            iso_psds.append( np.interp(iso_freq, bout_freqs[i], bout_psds[i], left=np.nan, right=np.nan) )
        else:
            iso_psds.append(bout_psds[i])

    # now we have iso_psds, which is a list of numpy arrays, where each array is the interpolated PSD of one NREM bout, and all arrays have the same length and frequency resolution
    # now we can average the FFTs of all bouts
    avg_iso_psd = np.nanmean(iso_psds, axis=0)  # avg_psd.shape = (n_freqs,)

    # get peak frequency, peak psd, ISO band power (0.01-0.03Hz) from iso_freq and avg_iso_psd
    # save them so that we can have these values for all EDF files
    iso_band_mask =  (iso_freq >= .01) & (iso_freq <= 0.03)
    peak_freq = iso_freq[np.argmax(avg_iso_psd)]  
    peak_psd = avg_iso_psd[np.argmax(avg_iso_psd)]
    iso_bp = trapezoid(avg_iso_psd[iso_band_mask], iso_freq[iso_band_mask])
    iso_bp_n = iso_bp / trapezoid(avg_iso_psd, iso_freq)  # normalized ISO band power

    #print(sid)
    #print(f"Peak frequency : {peak_freq} Hz")
    #print(f"Peak PSD : {peak_psd} dB")
    #print(f"ISO band power : {iso_bp} dB")
    #print(f"Normalized ISO band power : {iso_bp_n}")

    if to_plot:
        # plot frequency vs. decible
        sns.set_style('ticks')
        plt.close()
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1,1,1)
        ax.plot(iso_freq, 10 * np.log10(avg_iso_psd))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(f'{band_name} ISO PSD (dB)')
        sns.despine()

        plt.tight_layout()
        #plt.show()
        plt.savefig(f'figures_iso/iso_psd_{os.path.basename(edf_path)[:-8]}_{band_name}.png', dpi=300, bbox_inches='tight')

    return peak_freq, peak_psd, iso_bp, iso_bp_n, all_bout_band_powers, all_bout_phases, iso_freq, avg_iso_psd


def main():
    os.makedirs('figures_iso_ShenLab', exist_ok=True)

    df = pd.read_excel('../analysis_code/mastersheet_with_video_annot.xlsx')
    df = df[np.in1d(df.SID, [2,3])&(df.Condition=='BL')].reset_index(drop=True)

    band_names = ['gamma', 'beta', 'sigma', 'alpha', 'theta', 'delta']
    band_name_to_freq = {
        'gamma': (30, 45),
        'beta': (15, 30),
        'sigma': (11, 15),
        'alpha': (8, 12),
        'theta': (5, 10),  # for mice
        'delta': (1, 4)
    }

    df_res = []
    bout_band_powers_all_subjects_bands = {}
    bout_phases_all_subjects_bands = {}
    iso_freq_all_subjects_bands = {}
    avg_iso_psd_all_subjects_bands = {}
    for i in tqdm(range(len(df))):
        sid = df.SID.iloc[i]
        sig_id = df.SignalID.iloc[i]
        day_night = df.Day_Night.iloc[i]
        edf_path = f'../analysis_code/clean_edf/SID{sid}_SigID{sig_id}_{day_night}_BL.edf'
        annot_path = f'../analysis_code/sleep_stages_pred_usleep/SID{sid}_SigID{sig_id}_events-pred.tsv'
        for band_name in band_names:
            #print(f'{datetime.datetime.now()}: Processing {os.path.basename(edf_path)} - {band_name} band')
            band_lb, band_ub = band_name_to_freq[band_name]
            peak_freq, peak_psd, iso_bp, iso_bp_n, all_bout_band_powers, all_bout_phases, iso_freq, avg_iso_psd = \
                get_iso(edf_path, annot_path, band_lb, band_ub, band_name, to_plot=False)#True)

            bout_band_powers_all_subjects_bands[(sid, edf_path, band_name)] = all_bout_band_powers
            bout_phases_all_subjects_bands[(sid, edf_path, band_name)] = all_bout_phases
            iso_freq_all_subjects_bands[(sid, edf_path, band_name)] = iso_freq
            avg_iso_psd_all_subjects_bands[(sid, edf_path, band_name)] = avg_iso_psd

            df_res.append(pd.DataFrame(data={
                'sid': [sid],
                'edf': [edf_path],
                'band_name': [band_name],
                'peak_freq': [peak_freq],
                'peak_psd': [peak_psd],
                'iso_bp': [iso_bp],
                'iso_bp_n': [iso_bp_n],
            }))

    df_res = pd.concat(df_res, ignore_index=True)
    print(df_res)

    df_res.to_csv('iso_results_all_subjects_bands_ShenLab.csv', index=False)

    # save bout_band_powers_all_subjects_bands so that we can use it for cross-band correlation analysis
    with open('data_all_subjects_bands_ShenLab.pickle', 'wb') as f:
        pickle.dump({
            'bout_band_powers': bout_band_powers_all_subjects_bands,
            'bout_phases': bout_phases_all_subjects_bands,
            'iso_freq': iso_freq_all_subjects_bands,
            'avg_iso_psd': avg_iso_psd_all_subjects_bands
        }, f)


if __name__ == "__main__":
    main()
