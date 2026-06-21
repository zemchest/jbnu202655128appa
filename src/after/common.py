import cfg, glob, io, matplotlib, numpy as np, os, pandas as pd, re, shutil, time
from datetime import datetime
from scipy.stats import gmean, hmean
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def start_time(start_time):
    start_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f'Started on: {start_time}')
def end_time(start_time):
    end_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print(f'Ended on: {end_time}')
    print(f'Total running time: {time.time() - start_time:.3f} s')
def find_input_folder(file_name):
    n_in = int(file_name[:2])
    target_prefix = f"{n_in} "
    folder_paths = [f for f in os.listdir('.') if f.startswith(target_prefix) and os.path.isdir(f)]
    folder_path = folder_paths[0]
    if len(folder_paths) > 1:
        print(f'{len(folder_paths)} input folders found. Use {folder_paths[0]} folder only.')
        print('-' * 80)
    return folder_path
def save_data_label_csv(df, output_folder):
    t1 = time.time()
    os.makedirs(f'{output_folder}', exist_ok=True)
    output_path = os.path.join(output_folder, cfg.data_label_csv)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'{len(df):,} samples saved to "{output_path}". ({time.time() - t1:.3f} s)')
def save_data_series_csv(df, output_folder):
    t1 = time.time()
    os.makedirs(f'{output_folder}', exist_ok=True)
    output_path = os.path.join(output_folder, cfg.data_series_csv)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'{len(df):,} points saved to "{output_path}". ({time.time() - t1:.3f} s)')
def save_data_case_info_txt(df_label, output_folder):
    messages = []
    case_count_list = [(case, len(group)) for case, group in df_label.groupby(cfg.case)]
    messages.append(f'{cfg.case}: {len(case_count_list):,}')
    messages.append('-' * 20)
    for case, case_count in case_count_list:
        messages.append(f'Case {case}\t:\t{case_count:,}')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, cfg.data_case_info_txt)
    with open(output_path, 'w') as f:
        f.write('\n'.join(messages))
    print(f'"{output_path}" saved.')
def save_data_length_info_txt(df_series, output_folder):
    messages = []
    segment_lengths = [len(segment) for _, segment in df_series.groupby(cfg.sample)]
    messages.append(f'{cfg.sample}: {len(segment_lengths):,}')
    messages.append('-' * 20)
    messages.append(f'{cfg.min}: {min(segment_lengths):,}')
    messages.append(f'{cfg.max}: {max(segment_lengths):,}')
    messages.append(f'{cfg.median}: {np.median(segment_lengths):,.1f}')
    messages.append('-' * 20)
    messages.append(f'{cfg.a_mean}: {np.mean(segment_lengths):,.1f}')
    messages.append(f'{cfg.g_mean}: {gmean(segment_lengths):,.1f}')
    messages.append(f'{cfg.h_mean}: {hmean(segment_lengths):,.1f}')
    messages.append('-' * 20)
    messages.append(f'{cfg.std}: {pd.Series(segment_lengths).std():,.1f}')
    messages.append(f'{cfg.skew}: {pd.Series(segment_lengths).skew():.3f}')
    messages.append(f'{cfg.kurt}: {pd.Series(segment_lengths).kurt():.3f}')
    os.makedirs(output_folder, exist_ok=True)
    txt_path = os.path.join(output_folder, cfg.data_length_info_txt)
    with open(txt_path, 'w') as f:
        f.write('\n'.join(messages))
    print(f'"{cfg.data_length_info_txt}" saved.')
def find_output_folder(file_name):
    name, _ = os.path.splitext(file_name)
    m = re.match(r"(\d+)", name)
    n_in = int(m.group(1))
    n_out = n_in + 1
    output_folder = re.sub(r"^\d+", str(n_out), name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder
def save_data_csv(file_name, df_label, df_series):
    output_folder = find_output_folder(file_name)
    save_data_label_csv(df_label, output_folder)
    save_data_series_csv(df_series, output_folder)
    save_data_case_info_txt(df_label, output_folder)
    save_data_length_info_txt(df_series, output_folder)
    return output_folder
def load_sample_length_info(file_name):
    t1 = time.time()
    input_folder_path = find_input_folder(file_name)
    file_path = os.path.join(input_folder_path, cfg.data_length_info_txt)
    stats = {}
    start_collecting = False
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if cfg.sample in line:
                start_collecting = True
            if start_collecting and ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().replace(",", "")
                try:
                    val_num = float(val) if "." in val else int(val)
                    stats[key] = val_num
                except ValueError:
                    stats[key] = val
    print(f'Sample length informations loaded from "{cfg.data_length_info_txt}". ({time.time() - t1:.3f} s), Max length: {stats[cfg.max]}.')
    return stats
def load_data_csv(file_name):
    t1 = time.time()
    input_folder_path = find_input_folder(file_name)
    file_path = os.path.join(input_folder_path, cfg.data_label_csv)
    df_label = pd.read_csv(file_path)
    print(f'{len(df_label):,} samples loaded from "{file_path}". ({time.time() - t1:.3f} s)')
    t1 = time.time()
    file_path = os.path.join(input_folder_path, cfg.data_series_csv)
    df_series = pd.read_csv(file_path)
    input_file = os.path.basename(file_path)
    print(f'{len(df_series):,} points loaded from "{file_path}". ({time.time() - t1:.3f} s)')
    return df_label, df_series
def reset_time(df):
    current_time = 0
    for sample_id, seg in df.groupby(cfg.sample):
        seg_times = seg[cfg.time_s].values
        seg_duration = seg_times[-1] - seg_times[0]
        normalized_times = seg_times - seg_times[0]
        new_times = current_time + normalized_times
        df.loc[seg.index, cfg.time_s] = new_times
        current_time = new_times.max()
    return df
def save_label_list_txt(label, output_folder):
    t1 = time.time()
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, cfg.label_list_txt)
    with open(file_path, "w", encoding="utf-8") as f:
        for col in label:
            f.write(col + "\n")
    print(f'"{file_path}" saved. ({time.time() - t1:.3f} s)')
def save_data_label_info_txt(label, df, output_folder):
    n_sample = len(df)
    messages = []
    messages.append(f'{cfg.sample}: {n_sample:,}')
    messages.append('-' * 80)
    df_spec_bool = df[label].replace({'TRUE': True, 'FALSE': False}).astype(bool)
    spec_counts = df_spec_bool.sum()
    for col in label:
        messages.append(f'{col}\t:\t{spec_counts[col]}')
    file_path = os.path.join(output_folder, cfg.data_label_info_txt)
    with open(file_path, 'w') as f:
        f.write('\n'.join(messages))
    print(f'"{file_path}" saved.')
def save_label_txt(file_name, label, df):
    output_folder = find_output_folder(file_name)
    save_label_list_txt(label, output_folder)
    save_data_label_info_txt(label, df, output_folder)
    return
def load_label_list_txt(file_name):
    input_folder_path = find_input_folder(file_name)
    file_path = os.path.join(input_folder_path, cfg.label_list_txt)
    with open(file_path, "r", encoding="utf-8") as f:
        label = [line.strip() for line in f]
    print(f'{len(label)} labels loaded.')
    return label
def save_data_split_info_txt(file_name, df):
    output_folder = find_output_folder(file_name)
    n_sample = df[cfg.sample].nunique()
    counts = df.groupby(cfg.data_type)[cfg.sample].nunique()
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, cfg.data_split_info_txt)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{cfg.sample}: {n_sample}\n")
        f.write('-' * 20)
        for dtype in [cfg.train, cfg.valid, cfg.test]:
            n = counts.get(dtype, 0)
            f.write(f"\n{dtype}: {n}")
    print(f'Data split info saved to "{output_path}".')
def save_model_list_csv(file_name, df):
    output_folder = find_output_folder(file_name)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, cfg.model_list_csv)
    df.to_csv(output_path, index=False)
    print(f'Model list have been saved to "{output_path}".')
def load_model_list_csv(file_name):
    input_folder_path = find_input_folder(file_name)
    file_path = os.path.join(input_folder_path, cfg.model_list_csv)
    df = pd.read_csv(file_path)
    print(f'List of {len(df):,} models loaded from "{file_path}".')
    return df
