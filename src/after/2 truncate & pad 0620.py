import time
t0 = time.time()
t1 = time.perf_counter()
import cfg, numpy as np, os
from collections import defaultdict
from dataclasses import dataclass
this_file = os.path.basename(__file__)
this_folder = os.path.basename(os.path.dirname(__file__))
print(f'Executing {this_folder}/{this_file}')
print(f'Import basic modules: {time.perf_counter() - t1:.3f} seconds')
t1 = time.perf_counter()
import pandas as pd
print(f'Import pandas modules: {time.perf_counter() - t1:.3f} seconds')
t1 = time.perf_counter()
import common
print(f'Import common modules: {time.perf_counter() - t1:.3f} seconds')
common.start_time(t0)
print('-' * 80)

@dataclass(frozen=True)
class PreprocessConfig:
    input_path: str
    output_path: str
    chunk_size: int
    chunk_size_mb: int
    sample_col: str
    torque_col: str
    time_col: str
    target_length: int
    peak_position: int
    sampling_rate_hz: float
    pad_value: float

class SeriesProcessor:
    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        self.common_time_axis = np.linspace(0, (self.cfg.target_length - 1) / self.cfg.sampling_rate_hz, self.cfg.target_length)

    def allocate_buffers(self, n_samples: int, other_cols: list) -> tuple:
        out_samples = np.empty((n_samples, self.cfg.target_length), dtype=object)
        out_times = np.tile(self.common_time_axis, (n_samples, 1))
        out_buffers = {
            c: np.full((n_samples, self.cfg.target_length), self.cfg.pad_value, dtype=float) 
            for c in other_cols
        }
        return out_samples, out_times, out_buffers

    def build_output_dataframe(self, out_times: np.ndarray, out_samples: np.ndarray, out_buffers: dict, other_cols: list, original_columns: list) -> pd.DataFrame:
        df_dict = {
            self.cfg.time_col: out_times.flatten(),
            self.cfg.sample_col: out_samples.flatten()
        }
        for c in other_cols:
            df_dict[c] = out_buffers[c].flatten()
        return pd.DataFrame(df_dict)[original_columns]

    def process_chunk_vectorized(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        group_indices = df_chunk.groupby(self.cfg.sample_col).indices
        n_samples = len(group_indices)
        if n_samples == 0:
            return pd.DataFrame()
        other_cols = [c for c in df_chunk.columns if c not in (self.cfg.time_col, self.cfg.sample_col)]
        raw_arrays = {c: df_chunk[c].values for c in df_chunk.columns}
        out_samples, out_times, out_buffers = self.allocate_buffers(n_samples, other_cols)
        for idx, (sample_id, locs) in enumerate(group_indices.items()):
            orig_len = len(locs)
            torque_vals = raw_arrays[self.cfg.torque_col][locs]
            dest_start, dest_end, src_start, src_end = self.calculate_shift_indices(torque_vals, orig_len)
            out_samples[idx] = sample_id
            for c in other_cols:
                if (dest_end > dest_start) and (src_end > src_start):
                    out_buffers[c][idx, dest_start:dest_end] = raw_arrays[c][locs][src_start:src_end]
        return self.build_output_dataframe(out_times, out_samples, out_buffers, other_cols, df_chunk.columns)

    def calculate_shift_indices(self, torque_vals: np.ndarray, orig_len: int) -> tuple:
        peak_idx = np.argmax(torque_vals)
        start_idx = self.cfg.peak_position - peak_idx
        end_idx = start_idx + orig_len
        return (
            max(0, start_idx),
            min(self.cfg.target_length, end_idx),
            max(0, -start_idx),
            min(orig_len, self.cfg.target_length - start_idx)
        )
    
class StreamPipeline:
    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        self.processor = SeriesProcessor(config)

    def calculate_optimal_chunk_size(self, target_mb: float) -> int:
        if not os.path.exists(self.cfg.input_path):
            return self.cfg.chunk_size
        file_size_bytes = os.path.getsize(self.cfg.input_path)
        with open(self.cfg.input_path, 'rb') as f:
            row_count = sum(1 for _ in f)
        if row_count == 0:
            return self.cfg.chunk_size
        avg_row_bytes = file_size_bytes / row_count
        target_bytes = target_mb * 1024 * 1024
        optimal_chunk_size = int(target_bytes / avg_row_bytes)
        return optimal_chunk_size

    def append_to_csv(self, df: pd.DataFrame, is_first: bool):
        if df.empty:
            return
        if is_first:
            df.to_csv(self.cfg.output_path, mode='w', index=False)
        else:
            df.to_csv(self.cfg.output_path, mode='a', index=False, header=False)

    def run(self):
        first_chunk = True
        remainder_buffer = pd.DataFrame()
        dynamic_chunk_size = self.calculate_optimal_chunk_size(self.cfg.chunk_size_mb)
        for chunk in pd.read_csv(self.cfg.input_path, chunksize=dynamic_chunk_size):
            if not remainder_buffer.empty:
                chunk = pd.concat([remainder_buffer, chunk], ignore_index=True)
            last_sample_id = chunk[self.cfg.sample_col].iloc[-1]
            is_last_sample = chunk[self.cfg.sample_col] == last_sample_id
            current_processing_data = chunk[~is_last_sample]
            remainder_buffer = chunk[is_last_sample]
            if current_processing_data.empty:
                continue
            df_out = self.processor.process_chunk_vectorized(current_processing_data)
            self.append_to_csv(df_out, is_first=first_chunk)
            first_chunk = False
        if not remainder_buffer.empty:
            df_out = self.processor.process_chunk_vectorized(remainder_buffer)
            self.append_to_csv(df_out, is_first=first_chunk)
            
def main():
    input_folder_path = common.find_input_folder(this_file)
    output_folder_path = common.find_output_folder(this_file)
    config_obj = PreprocessConfig(
        input_path = os.path.join(input_folder_path, cfg.data_series_csv),
        output_path = os.path.join(output_folder_path, cfg.data_series_csv),
        chunk_size = cfg.chunk_size,
        chunk_size_mb = cfg.chunk_size_mb,
        sample_col = cfg.sample,
        torque_col = cfg.torque_nm,
        time_col = cfg.time_s,
        sampling_rate_hz = cfg.sampling_rate_hz,
        target_length = cfg.target_sample_length,
        peak_position = cfg.peak_position,
        pad_value = cfg.pad_value,
    )
    pipeline = StreamPipeline(config_obj)
    pipeline.run()
    common.end_time(t0)
if __name__ == '__main__':
    main()
