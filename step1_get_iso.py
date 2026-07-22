from itertools import groupby
import argparse
import os, datetime, glob
import pickle
import pyedflib
import mat73
import h5py
import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
from scipy.integrate import trapezoid
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import seaborn as sns


DATASETS = ['MSSV', 'OxfordBenchmark', 'Oscar']#, 'ShenLab'
DATASET_BASE_DIR = '..'


def find_channel(keys, token, required=True):
    """Return the first key containing `token` (case-insensitive)."""
    t = token.lower()
    for k in keys:
        if t in k.lower():
            return k
    if required:
        raise KeyError(f'no channel key containing {token!r}')
    return None


def parse_bl_hour(bl_hour):
    """Parse the Oscar `BL_hour` cell into a list of continuous (start_h, end_h)
    windows, one per '|'-separated segment (which marks a discontinuity).

    - 'none'  -> [] (row should be dropped)
    - 'all'   -> [(0.0, None)] (whole recording)
    - 'a-b'   -> the inclusive window [a, b] hours (start hour a, end hour b).

    A '|' (e.g. '0-23|72-143') means the baseline is *not* continuous, so each
    side becomes its own window / row.
    """
    s = str(bl_hour).strip().lower()
    if s in ('none', 'nan', ''):
        return []
    windows = []
    for seg in s.split('|'):
        seg = seg.strip()
        if seg == 'all':
            windows.append((0.0, None))
        else:
            a, b = seg.split('-')
            windows.append((float(a), float(b)))
    return windows


def load_annot(path, dataset):
    if dataset in ('MSSV', 'ShenLab'):
        return pd.read_csv(path, sep='\t')
    elif dataset in ('Oscar'):
        return pd.read_csv(path)
    elif dataset == 'OxfordBenchmark':
        #with open(path) as f:
        #    txt = f.readline()
        #total_duration = float(txt.split()[1])
        df = pd.read_csv(path, comment='*',sep='\t',header=None)
        df.columns = ['stage', 'start']
        df['duration'] = np.diff(np.r_[0, df.start.values])
        #df['start'] = np.r_[0, df.start.values][:-1]
        #df = df[df.duration>0].reset_index(drop=True)
        df.loc[df.stage.astype(str).str.strip().str.lower()=='awake', 'stage'] = 1
        df.loc[df.stage.astype(str).str.strip().str.lower()=='non-rem', 'stage'] = 2
        df.loc[df.stage.astype(str).str.strip().str.lower()=='rem', 'stage'] = 3
        df.loc[df.stage.astype(str).str.strip().str.lower()=='undefined', 'stage'] = 4
        df['stage'] = df.stage.astype(int)
        assert (df.duration>0).all()
        df2 = []
        for i in range(len(df)):
            df2.extend([df.stage.iloc[i]]*int(df.duration.iloc[i]//4))
        df2 = pd.DataFrame(data={'stage':df2})
        df2['start'] = np.arange(len(df2))*4
        return df2

    else:
        raise ValueError(f'Unknown dataset: {dataset}')


def load_signal(path, dataset, eeg_ch):
    """Return (eeg_uV, fs_Hz)."""
    if dataset == 'Oscar':
        # Spike2 .mat (v7.3) export: each channel is a struct with `values`,
        # `interval` (sampling period in s) and `start`. Channel keys carry a
        # per-recording prefix and trailing index (e.g. 'ORP576_Inh_EEG1_2'),
        # so the requested channel ('EEG1'/'EEG2') is matched by substring.
        with h5py.File(path, 'r') as f:
            chs = list(f.keys())
        eeg_key = find_channel(chs, eeg_ch)
        m = mat73.loadmat(path, only_include=[eeg_key])
        ch = m[eeg_key]
        eeg = np.asarray(ch['values']).ravel()
        fs = round(1.0 / float(ch['interval']))
    else:
        # EDF-based datasets (MSSV, OxfordBenchmark, ShenLab)
        signals, signal_hdrs, hdr = pyedflib.highlevel.read_edf(path)
        # find the index of the EEG signal, and get its sampling frequency
        ch_names = [h['label'] for h in signal_hdrs]
        EEG_index = [i for i, name in enumerate(ch_names) if eeg_ch.upper() in str(name).upper()]
        EEG_index = EEG_index[0]  # this is the index of the EEG signal, it is an integer
        fs = signal_hdrs[EEG_index]['sample_frequency']  # this is the sampling frequency, it is a float number
        eeg = signals[EEG_index]

    if eeg.std()<0.01:
        eeg *= 1e6  # unit = uV
    return eeg, fs


def get_file_list(dataset):
    """Return (sids, edf_paths, annot_paths, eeg_chs, bl_windows) for the given
    dataset. The subjects always come from mastersheet_<dataset>.csv.

    `bl_windows` holds one entry per row: None means "use the whole recording",
    otherwise a (start_h, end_h) baseline window in hours (Oscar only)."""
    df = pd.read_csv(f'mastersheet_{dataset}.csv')
    if dataset in ('MSSV', 'OxfordBenchmark', 'ShenLab'):
        # these sheets have SID, EDFPath, AnnotPath columns (ShenLab's sheet
        # does not exist yet; it will be read the same way once created).
        sids = df.SID.tolist()
        edf_paths = [os.path.join(DATASET_BASE_DIR, p) for p in df.EDFPath]
        annot_paths = [os.path.join(DATASET_BASE_DIR, p) for p in df.AnnotPath]
        if dataset == 'OxfordBenchmark':
            eeg_chs = []
            for edf_path in edf_paths:
                if '/test/' in edf_path:
                    eeg_chs.append('Signal 3')
                elif '/pilot/' in edf_path:
                    eeg_chs.append('Signal 1')
                else:
                    raise ValueError(f'Cannot determine EEG channel for {edf_path}')
        else:  # MSSV, ShenLab
            eeg_chs = ['EEG'] * len(sids)
        bl_windows = [None] * len(sids)

    elif dataset == 'Oscar':
        annot_dir = 'sleep_stages_pred_Oscar'
        sids = []
        edf_paths = []
        annot_paths = []
        eeg_chs = []
        bl_windows = []
        for _, row in df.iterrows():
            # drop rows with no baseline; split non-continuous baselines ('|')
            # into one row per continuous window
            windows = parse_bl_hour(row.BL_hour)
            if not windows:
                continue
            edf_path = os.path.join(DATASET_BASE_DIR, row.EDFPath)
            stem = os.path.splitext(os.path.basename(str(row.EDFPath)))[0]
            # the staged sleep-stage file lives under sleep_stages_pred_Oscar and
            # encodes the EEG channel used for staging (see sleep_stage_Oscar.py)
            annot_path = os.path.join(annot_dir, f'{stem}_EEG1_sleep_stages_pred.csv')
            for start_h, end_h in windows:
                win_tag = 'all' if end_h is None else f'h{start_h:g}-{end_h:g}'
                sids.append(f'{stem}_{win_tag}')  # unique per (recording, window)
                edf_paths.append(edf_path)
                annot_paths.append(annot_path)
                eeg_chs.append('EEG1')
                bl_windows.append((start_h, end_h))

    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    return sids, edf_paths, annot_paths, eeg_chs, bl_windows


def get_iso(edf_path, annot_path, band_lb, band_ub, band_name, dataset, to_plot=False, eeg_ch='EEG',
            bl_window=None):
    eeg, fs = load_signal(edf_path, dataset, eeg_ch)

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

    df_annot = load_annot(annot_path, dataset)
    #print(df_annot)
    sleep_stages_ = df_annot.stage.values
    # make brief not-NREM to NREM
    sleep_stages = np.array(sleep_stages_)
    for i in range(len(sleep_stages_)-4):
        if (sleep_stages_[i+2]!=2) and (sleep_stages_[i]==sleep_stages_[i+1]==sleep_stages_[i+3]==sleep_stages_[i+4]==2):
            sleep_stages[i+2] = 2
    #for i in range(len(sleep_stages_)-5):
    #    if (sleep_stages_[i+2]!=2 and sleep_stages_[i+3]!=1) and (sleep_stages_[i]==sleep_stages_[i+1]==sleep_stages_[i+4]==sleep_stages_[i+5]==2):
    #        sleep_stages[i+2] = 2
    #        sleep_stages[i+3] = 2
    L = len(eeg)//epoch_size
    #sleep_stages = sleep_stages[:L]
    assert len(sleep_stages)==L

    # take only the baseline (BL) duration for both the signal and the sleep
    # stages: crop to the inclusive [start_h, end_h] hour window (Oscar). end_h
    # None means "to the end of the recording".
    if bl_window is not None:
        start_h, end_h = bl_window
        start_ep = int(round(start_h * 3600 / epoch_duration))
        end_ep = len(sleep_stages) if end_h is None else int(round(end_h * 3600 / epoch_duration))
        start_ep = max(0, min(start_ep, len(sleep_stages)))
        end_ep = max(start_ep, min(end_ep, len(sleep_stages)))
        sleep_stages = sleep_stages[start_ep:end_ep]
        sleep_stages_ = sleep_stages_[start_ep:end_ep]
        eeg = eeg[start_ep * epoch_size:end_ep * epoch_size]

    total_band_lb = 0.3
    total_band_ub = 45  # because they are recorded in Europe with AC freq=50Hz, so the upper bound of frequency is 50Hz, but we want to exclude the AC noise, therefore set it to 45Hz

    #3.1 For this analysis, select only non-REM sleep bouts lasting ≥ 96 s (i.e., at least 24 epochs of 4 s); see Figure 2
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

    # get band of interest power
    freq_mask = (freqs >= band_lb) & (freqs <= band_ub)
    band_powers = trapezoid(psd[:, freq_mask], freqs[freq_mask])  # band_powers.shape = (n_epochs,)

    #3.3 Calculate the baseline spectral power for non-REM sleep by averaging the values in each frequency bin for all non-REM sleep epochs
    # (artifacts and epochs of transition between vigilance states are excluded from this averaging).
    # Normalize the sigma power values of each epoch to the mean power of the sigma band during non-REM sleep over the time period of interest. Plot against time (Figure 2C).
    """
    # normalize the band power by the total band power (optional, see --normalize)
    if normalize:
        total_band_powers = trapezoid(psd, freqs)  # band_powers.shape = (n_epochs,)
        band_powers /= total_band_powers  # band_powers.shape
    """
    nrem_ids = np.array([ii for ii in range(1, len(sleep_stages_)-1) if (sleep_stages_[ii-1:ii+2]==2).all() ])
    bl_power = np.nanmean(psd[nrem_ids], axis=0)
    bl_power = trapezoid(bl_power[freq_mask], freqs[freq_mask])
    band_powers /= bl_power

    band_powers_f = mne.filter.filter_data(band_powers, 1/epoch_duration, 0.01,0.03, verbose=False)
    is_nan_mask = np.isnan(band_powers_f)
    if is_nan_mask.sum()>0:
        assert (np.diff(np.where(is_nan_mask)[0])==1).all()
        iphase = np.zeros(len(band_powers_f))+np.nan
        iphase[~is_nan_mask] = np.angle(hilbert(band_powers_f[~is_nan_mask]))
    else:
        iphase = np.angle(hilbert(band_powers_f))
    all_bout_band_powers = []
    all_bout_band_powers_f = []
    all_bout_phases = []
    for start_index, end_index in start_end_epochs_NREM:
        # for each NREM bout
        all_bout_band_powers.append(band_powers[start_index:end_index])
        all_bout_band_powers_f.append(band_powers_f[start_index:end_index])
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
        # `to_plot` is the output figure path
        plt.savefig(to_plot, dpi=300, bbox_inches='tight')

    return peak_freq, peak_psd, iso_bp, iso_bp_n, all_bout_band_powers, all_bout_band_powers_f, all_bout_phases, iso_freq, avg_iso_psd


def main(dataset):
    figure_dir = f'figures_iso_{dataset}'
    os.makedirs(figure_dir, exist_ok=True)

    # get the list of EDF files, their corresponding annotation files, subject IDs, and EEG channel names
    sids, edf_paths, annot_paths, eeg_chs, bl_windows = get_file_list(dataset)
    assert all([os.path.exists(x) for x in edf_paths])
    assert all([os.path.exists(x) for x in annot_paths])

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
    bout_band_powers_f_all_subjects_bands = {}
    bout_phases_all_subjects_bands = {}
    iso_freq_all_subjects_bands = {}
    avg_iso_psd_all_subjects_bands = {}
    for sid, edf_path, annot_path, eeg_ch, bl_window in tqdm(
            zip(sids, edf_paths, annot_paths, eeg_chs, bl_windows), total=len(sids)):
        for band_name in band_names:
            #print(f'{datetime.datetime.now()}: Processing {os.path.basename(edf_path)} - {band_name} band')
            band_lb, band_ub = band_name_to_freq[band_name]
            plot_path = os.path.join(figure_dir, f'iso_psd_{sid}_{band_name}.png')
            peak_freq, peak_psd, iso_bp, iso_bp_n, all_bout_band_powers, all_bout_band_powers_f, all_bout_phases, iso_freq, avg_iso_psd = \
                get_iso(edf_path, annot_path, band_lb, band_ub, band_name, dataset, eeg_ch=eeg_ch, to_plot=plot_path,
                        bl_window=bl_window)

            bout_band_powers_all_subjects_bands[(sid, edf_path, band_name)] = all_bout_band_powers
            bout_band_powers_f_all_subjects_bands[(sid, edf_path, band_name)] = all_bout_band_powers_f
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

    df_res.to_csv(f'iso_results_all_subjects_bands_{dataset}.csv', index=False)

    # save bout_band_powers_all_subjects_bands so that we can use it for cross-band correlation analysis
    with open(f'data_all_subjects_bands_{dataset}.pickle', 'wb') as f:
        pickle.dump({
            'bout_band_powers': bout_band_powers_all_subjects_bands,
            'bout_band_powers_f': bout_band_powers_f_all_subjects_bands,
            'bout_phases': bout_phases_all_subjects_bands,
            'iso_freq': iso_freq_all_subjects_bands,
            'avg_iso_psd': avg_iso_psd_all_subjects_bands
        }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute ISO metrics for a dataset.')
    parser.add_argument('--dataset', default='MSSV', choices=DATASETS,
                        help='Which dataset to process (default: MSSV)')
    #parser.add_argument('--normalize', dest='normalize', action='store_true',
    #                    help='Normalize band power by the total band power (off by default)')
    args = parser.parse_args()

    main(args.dataset)#, normalize=args.normalize)
