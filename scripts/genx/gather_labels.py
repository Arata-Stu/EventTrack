import os
import numpy as np
from glob import glob
import argparse

def gather_raw_data(npy_files, use_npz=False):
    """
    ファイルから生のモーションデータ配列のリストを収集する。
    ここではまだ結合しない。
    """
    all_dx_prev, all_dy_prev = [], []
    all_dx_next, all_dy_next = [], []

    for path in npy_files:
        try:
            if use_npz:
                with np.load(path, allow_pickle=True) as data:
                    labels = data["labels"]
                    all_dx_prev.append(labels['dx_prev'])
                    all_dy_prev.append(labels['dy_prev'])
                    all_dx_next.append(labels['dx_next'])
                    all_dy_next.append(labels['dy_next'])
            else: # .npy
                data = np.load(path, allow_pickle=True).item()
                all_dx_prev.append(data['dx_prev'])
                all_dy_prev.append(data['dy_prev'])
                all_dx_next.append(data['dx_next'])
                all_dy_next.append(data['dy_next'])
        except Exception as e:
            print(f"Warning: Could not process file {path}. Error: {e}")

    return all_dx_prev, all_dy_prev, all_dx_next, all_dy_next


def main(args):
    print(f"Searching for files in: {args.input_dir}")
    if args.use_npz:
        # .npz ファイルを再帰的に検索
        all_files = sorted(glob(os.path.join(args.input_dir, "**/labels.npz"), recursive=True))
    else:
        # .npy ファイルを検索 (データセット構造に合わせて調整が必要な場合があります)
        all_files = sorted(glob(os.path.join(args.input_dir, "**/*_with_motion.npy"), recursive=True))

    if not all_files:
        print("Error: No data files found. Please check your --input_dir and path patterns.")
        return

    print(f"Found {len(all_files)} files to process.")

    # 全てのファイルからデータを収集
    dx_prev_list, dy_prev_list, dx_next_list, dy_next_list = gather_raw_data(all_files, args.use_npz)

    # 収集したデータをカテゴリごとに1つの大きな配列に結合
    print("Concatenating all data...")
    dx_prev = np.concatenate(dx_prev_list)
    dy_prev = np.concatenate(dy_prev_list)
    dx_next = np.concatenate(dx_next_list)
    dy_next = np.concatenate(dy_next_list)
    print("Concatenation complete.")

    # 1つの .npz ファイルに圧縮して保存
    print(f"Saving combined data to: {args.output_file}")
    np.savez_compressed(
        args.output_file,
        dx_prev=dx_prev,
        dy_prev=dy_prev,
        dx_next=dx_next,
        dy_next=dy_next,
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and combine all motion data into a single file.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory for the dataset.")
    parser.add_argument("--output_file", type=str, default="combined_motion_data.npz", help="Path to save the combined .npz file.")
    parser.add_argument("--use_npz", action="store_true", help="Look for .npz files instead of .npy files.")
    
    args = parser.parse_args()
    main(args)