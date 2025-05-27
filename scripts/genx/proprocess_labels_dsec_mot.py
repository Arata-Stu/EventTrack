import os
import numpy as np
from pathlib import Path
from multiprocessing import Pool, set_start_method, cpu_count
import argparse
from glob import glob

def process_one_file(label_path):
    suffix = "_with_motion"
    base = os.path.basename(label_path)
    name, ext = os.path.splitext(base)

    if name.endswith(suffix):
        return f"[!] Skipped (already processed): {label_path}"

    save_path = os.path.join(os.path.dirname(label_path), f"{name}{suffix}{ext}")
    if os.path.exists(save_path):
        return f"[!] Skipped (output exists): {save_path}"

    try:
        labels = np.load(label_path)
        N = len(labels)

        cx = labels['x'] + labels['w'] / 2
        cy = labels['y'] + labels['h'] / 2

        motion_fields = [
            ('dx_prev', 'f8'), ('dy_prev', 'f8'),
            ('dx_next', 'f8'), ('dy_next', 'f8')
        ]
        extended_dtype = np.dtype(labels.dtype.descr + motion_fields)

        output = np.zeros(N, dtype=extended_dtype)
        for name in labels.dtype.names:
            output[name] = labels[name]

        track_ids = np.unique(labels['track_id'])

        for tid in track_ids:
            idxs = np.where(labels['track_id'] == tid)[0]
            if len(idxs) < 2:
                continue

            sorted_idxs = idxs[np.argsort(labels['t'][idxs])]
            for i, curr_idx in enumerate(sorted_idxs):
                if i > 0:
                    prev_idx = sorted_idxs[i - 1]
                    output['dx_prev'][curr_idx] = cx[curr_idx] - cx[prev_idx]
                    output['dy_prev'][curr_idx] = cy[curr_idx] - cy[prev_idx]
                if i < len(sorted_idxs) - 1:
                    next_idx = sorted_idxs[i + 1]
                    output['dx_next'][curr_idx] = cx[next_idx] - cx[curr_idx]
                    output['dy_next'][curr_idx] = cy[next_idx] - cy[curr_idx]

        np.save(save_path, output)
        return f"[✓] Processed: {label_path} → {save_path}"

    except Exception as e:
        return f"[✗] Failed: {label_path} — {e}"

def collect_dsec_mot_npy_files(input_dir):
    """対象: input_dir/sequence_name/tracking/tracks.npy"""
    return sorted(Path(input_dir).glob("*/tracking/tracks.npy"))

def batch_process_dsec_mot(input_dir, num_workers=None):
    npy_files = collect_dsec_mot_npy_files(input_dir)
    print(f"[*] DSEC-MOT — Found {len(npy_files)} .npy files in {input_dir}")

    if num_workers is None:
        num_workers = min(cpu_count(), len(npy_files))

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_one_file, npy_files)

    for msg in results:
        print(msg)

if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Batch motion preprocessor for DSEC-MOT .npy tracking labels")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing sequence folders")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    batch_process_dsec_mot(args.input_dir, args.num_workers)
