import time
t0 = time.time()
import cfg, os, glob, shutil, numpy as np
this_file = os.path.basename(__file__)
this_folder = os.path.basename(os.path.dirname(__file__))
print(f'Executing {this_folder}/{this_file}')
print(f'Import basic modules: {time.time() - t0:.3f} seconds')
t1 = time.time()
import pandas as pd
print(f'Import pandas modules: {time.time() - t1:.3f} seconds')
t1 = time.time()
import common
print(f'Import common modules: {time.time() - t1:.3f} seconds')
common.start_time(t0)
print('-' * 80)
def np_shift(arr, n, fill_value=0):
    result = np.full_like(arr, fill_value)
    if n > 0:
        result[n:] = arr[:-n]
    elif n < 0:
        result[:n] = arr[-n:]
    else:
        result[:] = arr
    return result
def truncate_and_pad(df_in, target_length, peak_position):
    grouped = df_in.groupby(cfg.sample)
    n_samples = len(grouped)
    sample_vals = np.empty((n_samples, target_length))
    time_vals = np.zeros((n_samples, target_length))
    time_vals[:] = np.linspace(0, (target_length - 1) / cfg.sampling_rate_hz, target_length)
    other_cols = [c for c in df_in.columns if c not in (cfg.time_s, cfg.sample)]
    other_arrays = {c: np.zeros((n_samples, target_length), dtype=float) for c in other_cols}
    for idx, (i, group) in enumerate(grouped):
        torque_vals = group[cfg.torque_nm].values
        pad_size = target_length - len(torque_vals)
        if pad_size > 0:
            torque_vals = np.pad(torque_vals, (0, pad_size), mode='constant', constant_values=0)
        peak_idx = np.argmax(torque_vals)
        start_idx = peak_position - peak_idx
        orig_len = len(group)
        end_idx = start_idx + orig_len
        torque_vals_shifted = np_shift(torque_vals, start_idx)
        if pad_size < 0:
            torque_vals = torque_vals[:target_length]
        sample_vals[idx] = np.full(target_length, group[cfg.sample].values[0])
        for c in other_cols:
            arr = np.zeros(target_length, dtype=float)
            arr[max(0, start_idx):min(target_length, end_idx)] = group[c].values[max(0, -start_idx):min(orig_len, target_length - start_idx)]
            other_arrays[c][idx] = arr
    df_dict = {cfg.time_s: time_vals.flatten(), cfg.sample: sample_vals.flatten()}
    for c in other_cols:
        df_dict[c] = other_arrays[c].flatten()
    df_out = pd.DataFrame(df_dict)
    df_out = df_out[df_in.columns]
    return df_out
def main():
    print('Start loading.')
    input_folder_path = common.find_input_folder(this_file)
    df_label, df_series_in = common.load_data_csv(input_folder_path)
    print('Loading complete.')
    print('-' * 80)
    print('Start processing.')
    df_series_out = truncate_and_pad(df_series_in, cfg.target_sample_length, cfg.peak_position)
    print('Processing complete.')
    print('-' * 80)
    print('Start saving.')
    common.save_data_csv(this_file, df_label, df_series_out)
    print('Saving complete.')
    print('-' * 80)
    common.end_time(t0)
if __name__ == '__main__':
    main()
