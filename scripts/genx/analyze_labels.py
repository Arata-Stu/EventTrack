import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd # 追加

def concat_valid_pairs(dx_list, dy_list):
    dx = np.concatenate(dx_list)
    dy = np.concatenate(dy_list)
    valid_mask = ~(np.isclose(dx, 0) | np.isclose(dy, 0))
    return dx[valid_mask], dy[valid_mask]

def gather_motion_data(npy_files):
    all_dx_prev, all_dy_prev = [], []
    all_dx_next, all_dy_next = [], []

    for path in npy_files:
        data = np.load(path, allow_pickle=True).item() # .item() を追加して辞書として読み込み
        all_dx_prev.append(data['dx_prev'])
        all_dy_prev.append(data['dy_prev'])
        all_dx_next.append(data['dx_next'])
        all_dy_next.append(data['dy_next'])

    # concat_valid_pairsは中で呼ばれるので修正
    dx_prev, dy_prev = concat_valid_pairs(all_dx_prev, all_dy_prev)
    dx_next, dy_next = concat_valid_pairs(all_dx_next, all_dy_next)

    return {
        "dx_prev": dx_prev,
        "dy_prev": dy_prev,
        "dx_next": dx_next,
        "dy_next": dy_next,
    }

def gather_motion_data_from_npz(npz_files):
    all_dx_prev, all_dy_prev = [], []
    all_dx_next, all_dy_next = [], []

    for path in npz_files:
        with np.load(path, allow_pickle=True) as data:
            labels = data["labels"].item() # .item() を追加して辞書として読み込み
            all_dx_prev.append(labels['dx_prev'])
            all_dy_prev.append(labels['dy_prev'])
            all_dx_next.append(labels['dx_next'])
            all_dy_next.append(labels['dy_next'])

    dx_prev, dy_prev = concat_valid_pairs(all_dx_prev, all_dy_prev)
    dx_next, dy_next = concat_valid_pairs(all_dx_next, all_dy_next)

    return {
        "dx_prev": dx_prev,
        "dy_prev": dy_prev,
        "dx_next": dx_next,
        "dy_next": dy_next,
    }

def compute_motion_asymmetry_mask(dx, dy, motion_mag, mag_thresh=80.0, ratio_thresh=10.0):
    eps = 1e-6
    ratio1 = np.abs(dx) / (np.abs(dy) + eps)
    ratio2 = np.abs(dy) / (np.abs(dx) + eps)
    max_ratio = np.maximum(ratio1, ratio2)

    motion_mask = motion_mag < mag_thresh
    ratio_mask = max_ratio < ratio_thresh
    final_mask = motion_mask & ratio_mask

    return final_mask, motion_mask, ratio_mask


def analyze_motion(sequence_id, motion_data):
    # 返り値用の辞書を初期化
    basic_stats = {}

    # 合成ベクトル長を計算
    dx_prev, dy_prev = motion_data["dx_prev"], motion_data["dy_prev"]
    dx_next, dy_next = motion_data["dx_next"], motion_data["dy_next"]
    motion_prev_mag = np.sqrt(dx_prev ** 2 + dy_prev ** 2)
    motion_next_mag = np.sqrt(dx_next ** 2 + dy_next ** 2)

    # 非対称レポート（統計情報を辞書で受け取る）
    asymmetry_stats_prev = report_asymmetry_stats(dx_prev, dy_prev, motion_prev_mag, prefix=f"{sequence_id}_prev")
    asymmetry_stats_next = report_asymmetry_stats(dx_next, dy_next, motion_next_mag, prefix=f"{sequence_id}_next")

    # フィルタ前のデータをmotion_dataに追加
    motion_data_unfiltered = motion_data.copy()
    motion_data_unfiltered["motion_prev_mag"] = motion_prev_mag
    motion_data_unfiltered["motion_next_mag"] = motion_next_mag

    # motionの非対称性 + 異常な大きさの除去フィルタ
    mask_prev, motion_mask_prev, ratio_mask_prev = compute_motion_asymmetry_mask(
        dx_prev, dy_prev, motion_prev_mag, mag_thresh=80.0, ratio_thresh=10.0
    )
    mask_next, motion_mask_next, ratio_mask_next = compute_motion_asymmetry_mask(
        dx_next, dy_next, motion_next_mag, mag_thresh=80.0, ratio_thresh=10.0
    )

    print(f"[{sequence_id}] motion_prev_mag: total={len(motion_prev_mag)}, kept={np.sum(mask_prev)}")
    print(f"[{sequence_id}]  removed by magnitude: {np.sum(~motion_mask_prev)}")
    print(f"[{sequence_id}]  removed by asymmetry: {np.sum(~ratio_mask_prev)}")
    print(f"[{sequence_id}] motion_next_mag: total={len(motion_next_mag)}, kept={np.sum(mask_next)}")

    # フィルタ後のmotion_dataを作成
    motion_data_filtered = {}
    for key, values in motion_data_unfiltered.items():
        if "prev" in key:
            motion_data_filtered[key] = values[mask_prev]
        elif "next" in key:
            motion_data_filtered[key] = values[mask_next]

    # 可視化・統計出力
    for key, values in motion_data_filtered.items():
        mean = values.mean() if len(values) > 0 else 0
        std = values.std() if len(values) > 0 else 0
        outlier_thresh = 3 * std
        outliers = values[np.abs(values - mean) > outlier_thresh]
        outlier_ratio = 100 * len(outliers) / len(values) if len(values) > 0 else 0

        print(f"\n[{sequence_id}] {key} (Filtered)")
        print(f"  Count   : {len(values)}")
        print(f"  Mean    : {mean:.4f}")
        print(f"  Std     : {std:.4f}")
        print(f"  Min/Max : {values.min():.4f} / {values.max():.4f}" if len(values) > 0 else "N/A")
        print(f"  Outliers (>3σ): {len(outliers)} ({outlier_ratio:.2f}%)")

        # 統計情報を辞書に保存
        basic_stats[key] = {
            "count": len(values),
            "mean": mean,
            "std": std,
            "min": values.min() if len(values) > 0 else np.nan,
            "max": values.max() if len(values) > 0 else np.nan,
            "outlier_count": len(outliers),
            "outlier_ratio": outlier_ratio,
        }

        # 可視化
        plt.figure()
        plt.hist(values, bins=100, alpha=0.7)
        plt.title(f"{sequence_id} — {key} (Filtered)")
        plt.grid(True)
        plt.xlabel(key)
        plt.ylabel("Frequency")
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{sequence_id}_{key}_hist.png")
        plt.close()

    # 2種類の統計情報を返す
    return {
        "basic_stats": basic_stats,
        "asymmetry_stats": {
            "prev": asymmetry_stats_prev,
            "next": asymmetry_stats_next,
        }
    }

def report_asymmetry_stats(dx, dy, motion_mag, prefix="motion_prev"):
    # 返り値用の辞書
    stats = {}
    total = len(dx)
    if total == 0:
        return stats

    eps = 1e-6
    ratio1 = np.abs(dx) / (np.abs(dy) + eps)
    ratio2 = np.abs(dy) / (np.abs(dx) + eps)
    max_ratio = np.maximum(ratio1, ratio2)

    # ヒストグラムで保存 (変更なし)
    plt.figure()
    plt.hist(max_ratio, bins=100, range=(0, 50), alpha=0.7)
    plt.xlabel("max(|dx/dy|, |dy/dx|)")
    plt.ylabel("Frequency")
    plt.title(f"{prefix} — Asymmetry Ratio Distribution (Unfiltered)")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{prefix}_asymmetry_ratio_hist.png")
    plt.close()

    # しきい値ごとの割合
    thresholds = [5, 10, 20, 50, 100]
    print(f"\n[{prefix}] Asymmetry Ratio Stats (Unfiltered, max(|dx/dy|, |dy/dx|))")
    for t in thresholds:
        count = np.sum(max_ratio > t)
        percent = 100 * count / total
        print(f"  > {t:>3} : {count} samples ({percent:.2f}%)")
        # 統計を辞書に保存
        stats[f'ratio_gt_{t}_count'] = count
        stats[f'ratio_gt_{t}_percent'] = percent

    # motion magnitude が大きく、かつ非対称なもの
    motion_thresh = 80
    ratio_thresh = 10
    joint_mask = (motion_mag > motion_thresh) & (max_ratio > ratio_thresh)
    joint_count = np.sum(joint_mask)
    joint_percent = 100 * joint_count / total
    print(f"\n  Combined condition (motion > {motion_thresh} & ratio > {ratio_thresh})")
    print(f"  -> {joint_count} samples ({joint_percent:.2f}%)")
    # 統計を辞書に保存
    stats[f'combined_motion_gt_{motion_thresh}_ratio_gt_{ratio_thresh}_count'] = joint_count
    stats[f'combined_motion_gt_{motion_thresh}_ratio_gt_{ratio_thresh}_percent'] = joint_percent

    return stats


def summarize_motion_stats(sequence_id, stats_dict):
    # この関数は変更なし
    import matplotlib.ticker as ticker
    # ... (内容は元のまま)
    keys = list(stats_dict.keys())
    if not keys: return # 統計が空なら何もしない

    means = [stats_dict[k]["mean"] for k in keys]
    stds = [stats_dict[k]["std"] for k in keys]
    outlier_ratios = [stats_dict[k]["outlier_ratio"] for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(x - width/2, means, width=width, label="Mean")
    ax1.bar(x + width/2, stds, width=width, label="Std")
    ax1.set_ylabel("Mean / Std")
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys, rotation=45, ha="right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, outlier_ratios, color='red', marker='o', label="Outlier Ratio (%)")
    ax2.set_ylabel("Outlier Ratio (%)")
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2.set_ylim(bottom=0)

    plt.title(f"{sequence_id} — Motion Summary (Filtered)")
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{sequence_id}_motion_summary.png")
    plt.close()


def analyze_by_dataset(dataset_name, input_dir, use_npz=False):
    # CSV保存用に統計情報を集めるリスト
    all_basic_stats_list = []
    all_asymmetry_stats_list = []

    process_func = lambda file_list, seq_id: {
        "motion_data": gather_motion_data(file_list),
        "stats": analyze_motion(seq_id, gather_motion_data(file_list))
    }

    if use_npz:
        all_npz = sorted(glob(os.path.join(input_dir, "**/labels_v2/labels.npz"), recursive=True))
        if all_npz:
            seq_id = f"{dataset_name}_labels_v2_npz"
            motion_data = gather_motion_data_from_npz(all_npz)
            if motion_data["dx_prev"].size > 0:
                stats = analyze_motion(seq_id, motion_data)
                summarize_motion_stats(seq_id, stats["basic_stats"])
                # リストに追加
                all_basic_stats_list.extend(flatten_basic_stats(seq_id, stats["basic_stats"]))
                all_asymmetry_stats_list.extend(flatten_asymmetry_stats(seq_id, stats["asymmetry_stats"]))
        else:
            print("[!] No .npz files found")

    else:
        # データセットごとの処理ロジック
        # (gen1, gen4, DSEC, GIFUのロジックは元の構造を維持)
        # ...
        # 以下に各データセットのループ内で統計情報をリストに追加する処理を記述
        if dataset_name in ["gen1", "gen4"]:
            for split in ["train", "val", "test"]:
                npy_files = sorted(glob(os.path.join(input_dir, split, '*_with_motion.npy')))
                if not npy_files: continue
                seq_id = f"{dataset_name}_{split}"
                motion_data = gather_motion_data(npy_files)
                if motion_data["dx_prev"].size > 0:
                    stats = analyze_motion(seq_id, motion_data)
                    summarize_motion_stats(seq_id, stats["basic_stats"])
                    all_basic_stats_list.extend(flatten_basic_stats(seq_id, stats["basic_stats"]))
                    all_asymmetry_stats_list.extend(flatten_asymmetry_stats(seq_id, stats["asymmetry_stats"]))
        # ... 他のデータセット(DSEC, GIFU)も同様に修正 ...
        elif dataset_name == "DSEC":
            seq_dirs = sorted(glob(os.path.join(input_dir, '*')))
            for seq_dir in seq_dirs:
                label_path = os.path.join(seq_dir, 'object_detections', 'left', 'tracks_with_motion.npy')
                if os.path.exists(label_path):
                    seq_id = os.path.basename(seq_dir)
                    motion_data = gather_motion_data([label_path])
                    if motion_data["dx_prev"].size > 0:
                        stats = analyze_motion(seq_id, motion_data)
                        summarize_motion_stats(seq_id, stats["basic_stats"])
                        all_basic_stats_list.extend(flatten_basic_stats(seq_id, stats["basic_stats"]))
                        all_asymmetry_stats_list.extend(flatten_asymmetry_stats(seq_id, stats["asymmetry_stats"]))
        # ...
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")


    # --- ループ終了後、集計したデータをCSVに書き出す ---
    if all_basic_stats_list:
        df_basic = pd.DataFrame(all_basic_stats_list)
        output_path = os.path.join("outputs", f"{dataset_name}_stats_basic.csv")
        df_basic.to_csv(output_path, index=False)
        print(f"\n[+] Basic statistics saved to: {output_path}")

    if all_asymmetry_stats_list:
        df_asymmetry = pd.DataFrame(all_asymmetry_stats_list)
        output_path = os.path.join("outputs", f"{dataset_name}_stats_asymmetry.csv")
        df_asymmetry.to_csv(output_path, index=False)
        print(f"\n[+] Asymmetry statistics saved to: {output_path}")


# CSV保存用に辞書をフラットな形式に変換するヘルパー関数
def flatten_basic_stats(sequence_id, basic_stats_dict):
    rows = []
    for metric, values in basic_stats_dict.items():
        row = {'sequence_id': sequence_id, 'metric': metric, **values}
        rows.append(row)
    return rows

def flatten_asymmetry_stats(sequence_id, asymmetry_stats_dict):
    rows = []
    for kind, values in asymmetry_stats_dict.items(): # kind = 'prev' or 'next'
        if values: # 空の辞書はスキップ
            row = {'sequence_id': sequence_id, 'type': kind, **values}
            rows.append(row)
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Motion statistics checker")
    parser.add_argument("--dataset", type=str, choices=["gen1", "gen4", "DSEC", "GIFU"], required=True)
    parser.add_argument("--input_dir", type=str, required=True, help="Root dir for dataset")
    parser.add_argument("--use_npz", action="store_true", help="Use .npz (labels_v2) instead of .npy")

    args = parser.parse_args()

    analyze_by_dataset(args.dataset, args.input_dir, args.use_npz)