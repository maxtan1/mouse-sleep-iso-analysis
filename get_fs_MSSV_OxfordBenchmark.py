"""Read the MSSV and OxfordBenchmark mastersheets and, for each EDF file listed
in them, record the sampling rate of each channel. (Oscar is handled separately
by get_fs_Oscar.py.)

For each mastersheet_<name>.csv a file fs_<name>.csv is written with the same
rows plus a `ChannelFS` column formatted as:
    CHANNEL_NAME1=FS1|CHANNEL_NAME2=FS2|...
and a `DurationHours` column with the recording duration in hours.
"""
import os
import numpy as np
import pandas as pd
import pyedflib

try:
    import h5py
except ImportError:
    h5py = None


def format_channel_fs(pairs):
    """pairs: list of (channel_name, fs) -> 'NAME1=FS1|NAME2=FS2|...'."""
    parts = []
    for name, fs in pairs:
        # drop a trailing .0 so 500.0 -> 500 for readability, keep decimals otherwise
        fs = float(fs)
        fs_str = str(int(fs)) if fs.is_integer() else repr(fs)
        parts.append(f'{name}={fs_str}')
    return '|'.join(parts)


def get_channel_fs(path):
    """Return (pairs, duration_seconds) for one recording, where pairs is a list
    of (channel_name, sampling_frequency)."""
    ext = os.path.splitext(path)[1].lower()

    if ext == '.edf':
        f = pyedflib.EdfReader(path)
        try:
            labels = f.getSignalLabels()
            fss = f.getSampleFrequencies()  # one entry per channel
            pairs = [(labels[i], fss[i]) for i in range(len(labels))]
            return pairs, float(f.getFileDuration())
        finally:
            f.close()

    # Spike2 .mat (v7.3 = HDF5) export (e.g. Oscar dataset): each channel is a
    # struct with an `interval` field (sampling period in s); fs = 1/interval.
    # Read only the `interval` scalars via h5py so the large `values` arrays are
    # never loaded into memory. Duration = interval * n_samples (the `values`
    # dataset shape is read from metadata without loading the array).
    if ext == '.mat':
        if h5py is None:
            raise ImportError('h5py is required to read .mat recordings')
        pairs = []
        duration = 0.0
        with h5py.File(path, 'r') as f:
            for key, item in f.items():
                if isinstance(item, h5py.Group) and 'interval' in item:
                    interval = float(np.asarray(item['interval']).ravel()[0])
                    if interval > 0:
                        pairs.append((key, 1.0 / interval))
                        if 'values' in item:
                            n_samples = int(np.prod(item['values'].shape))
                            duration = max(duration, interval * n_samples)
        return pairs, duration

    raise ValueError(f'Unsupported recording format: {path}')


def resolve_path(path, bases):
    """Return the first existing resolution of `path` against the candidate
    base dirs, or the first candidate if none exist (so the error is clear)."""
    if os.path.isabs(path):
        return path
    candidates = [os.path.normpath(os.path.join(b, path)) for b in bases]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)  # relative paths in some sheets are rooted here
    bases = [here, project_root]
    datasets = ['MSSV', 'OxfordBenchmark']
    mastersheets = [os.path.join(here, f'mastersheet_{d}.csv') for d in datasets]
    mastersheets = [p for p in mastersheets if os.path.exists(p)]
    if not mastersheets:
        print('No MSSV/OxfordBenchmark mastersheet_*.csv files found.')
        return

    for ms_path in mastersheets:
        name = os.path.basename(ms_path)[len('mastersheet_'):]  # e.g. 'MSSV.csv'
        out_path = os.path.join(here, 'fs_' + name)
        df = pd.read_csv(ms_path)
        print(f'\n{os.path.basename(ms_path)}: {len(df)} recording(s)')

        channel_fs = []
        duration_hours = []
        for _, row in df.iterrows():
            edf_path = str(row['EDFPath'])
            resolved = resolve_path(edf_path, bases)
            try:
                pairs, duration_s = get_channel_fs(resolved)
                channel_fs.append(format_channel_fs(pairs))
                duration_hours.append(duration_s / 3600.0)
            except Exception as e:
                print(f'  [WARN] {row.get("SID", edf_path)}: {e}')
                channel_fs.append('')
                duration_hours.append(np.nan)

        df['ChannelFS'] = channel_fs
        df['DurationHours'] = duration_hours
        df.to_csv(out_path, index=False)
        print(f'  -> wrote {os.path.basename(out_path)}')


if __name__ == '__main__':
    main()
