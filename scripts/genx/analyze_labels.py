import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def concat_valid_pairs(dx_list, dy_list):
    dx = np.concatenate(dx_list)
    dy = np.concatenate(dy_list)
    valid_mask = ~(np.isclose(dx, 0) | np.isclose(dy, 0))  # どちらかが0に近ければ除外
    return dx[valid_mask], dy[valid_mask]

def gather_motion_data(npy_files):
    all_dx_prev, all_dy_prev = [], []
    all_dx_next, all_dy_next = [], []

    for path in npy_files:
        data = np.load(path)
        all_dx_prev.append(data['dx_prev'])
        all_dy_prev.append(data['dy_prev'])
        all_dx_next.append(data['dx_next'])
        all_dy_next.append(data['dy_next'])

    return {
        "dx_prev": concat_valid_pairs(all_dx_prev),
        "dy_prev": concat_valid_pairs(all_dy_prev),
        "dx_next": concat_valid_pairs(all_dx_next),
        "dy_next": concat_valid_pairs(all_dy_next),
    }

def gather_motion_data_from_npz(npz_files):
    all_dx_prev, all_dy_prev = [], []
    all_dx_next, all_dy_next = [], []

    for path in npz_files:
        data = np.load(path)
        labels = data["labels"]
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
    stats = {}

    # 合成ベクトル長を計算
    dx_prev, dy_prev = motion_data["dx_prev"], motion_data["dy_prev"]
    dx_next, dy_next = motion_data["dx_next"], motion_data["dy_next"]
    motion_prev_mag = np.sqrt(dx_prev ** 2 + dy_prev ** 2)
    motion_next_mag = np.sqrt(dx_next ** 2 + dy_next ** 2)

    motion_data["motion_prev_mag"] = motion_prev_mag
    motion_data["motion_next_mag"] = motion_next_mag

        # 合成ベクトル長を計算
    dx_prev, dy_prev = motion_data["dx_prev"], motion_data["dy_prev"]
    dx_next, dy_next = motion_data["dx_next"], motion_data["dy_next"]
    motion_prev_mag = np.sqrt(dx_prev ** 2 + dy_prev ** 2)
    motion_next_mag = np.sqrt(dx_next ** 2 + dy_next ** 2)

    # 非対称レポート
    report_asymmetry_stats(dx_prev, dy_prev, motion_prev_mag, prefix=f"{sequence_id}_prev")
    report_asymmetry_stats(dx_next, dy_next, motion_next_mag, prefix=f"{sequence_id}_next")


    # motionの非対称性 + 異常な大きさの除去フィルタ（prev）
    mask_prev, motion_mask_prev, ratio_mask_prev = compute_motion_asymmetry_mask(
        dx_prev, dy_prev, motion_prev_mag, mag_thresh=80.0, ratio_thresh=10.0
    )
    # nextバージョンも同様
    mask_next, motion_mask_next, ratio_mask_next = compute_motion_asymmetry_mask(
        dx_next, dy_next, motion_next_mag, mag_thresh=80.0, ratio_thresh=10.0
    )

    print(f"[{sequence_id}] motion_prev_mag: total={len(motion_prev_mag)}, kept={np.sum(mask_prev)}")
    print(f"[{sequence_id}]  removed by magnitude: {np.sum(~motion_mask_prev)}")
    print(f"[{sequence_id}]  removed by asymmetry: {np.sum(~ratio_mask_prev)}")
    print(f"[{sequence_id}] motion_next_mag: total={len(motion_next_mag)}, kept={np.sum(mask_next)}")

    # 各項目に対して prev, next でマスク適用
    for key in motion_data:
        if "prev" in key:
            motion_data[key] = motion_data[key][mask_prev]
        elif "next" in key:
            motion_data[key] = motion_data[key][mask_next]

    # 可視化・統計出力
    for key, values in motion_data.items():
        mean = values.mean()
        std = values.std()
        outlier_thresh = 3 * std
        outliers = values[np.abs(values - mean) > outlier_thresh]
        outlier_ratio = 100 * len(outliers) / len(values)

        print(f"\n[{sequence_id}] {key}")
        print(f"  Count   : {len(values)}")
        print(f"  Mean    : {mean:.4f}")
        print(f"  Std     : {std:.4f}")
        print(f"  Min/Max : {values.min():.4f} / {values.max():.4f}")
        print(f"  Outliers (>3σ): {len(outliers)} ({outlier_ratio:.2f}%)")

        plt.figure()
        plt.hist(values, bins=100, alpha=0.7)
        plt.title(f"{sequence_id} — {key}")
        plt.grid(True)
        plt.xlabel(key)
        plt.ylabel("Frequency")
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{sequence_id}_{key}_hist.png")
        plt.close()

        stats[key] = {
            "mean": mean,
            "std": std,
            "outlier_ratio": outlier_ratio,
        }

    return stats

def report_asymmetry_stats(dx, dy, motion_mag, prefix="motion_prev"):
    eps = 1e-6
    ratio1 = np.abs(dx) / (np.abs(dy) + eps)
    ratio2 = np.abs(dy) / (np.abs(dx) + eps)
    max_ratio = np.maximum(ratio1, ratio2)

    # ヒストグラムで保存
    plt.figure()
    plt.hist(max_ratio, bins=100, range=(0, 50), alpha=0.7)
    plt.xlabel("max(|dx/dy|, |dy/dx|)")
    plt.ylabel("Frequency")
    plt.title(f"{prefix} — Asymmetry Ratio Distribution")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{prefix}_asymmetry_ratio_hist.png")
    plt.close()

    # しきい値ごとの割合
    thresholds = [5, 10, 20, 50, 100]
    total = len(max_ratio)
    print(f"\n[{prefix}] Asymmetry Ratio Stats (max(|dx/dy|, |dy/dx|))")
    for t in thresholds:
        count = np.sum(max_ratio > t)
        print(f"  > {t:>3} : {count} samples ({100 * count / total:.2f}%)")

    # motion magnitude が大きく、かつ非対称なもの
    motion_thresh = 80
    ratio_thresh = 10
    joint_mask = (motion_mag > motion_thresh) & (max_ratio > ratio_thresh)
    joint_count = np.sum(joint_mask)
    print(f"\n  Combined condition (motion > {motion_thresh} & ratio > {ratio_thresh})")
    print(f"  -> {joint_count} samples ({100 * joint_count / total:.2f}%)")


def summarize_motion_stats(sequence_id, stats_dict):
    import matplotlib.ticker as ticker

    keys = list(stats_dict.keys())
    means = [stats_dict[k]["mean"] for k in keys]
    stds = [stats_dict[k]["std"] for k in keys]
    outlier_ratios = [stats_dict[k]["outlier_ratio"] for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左軸：平均・標準偏差
    ax1.bar(x - width/2, means, width=width, label="Mean")
    ax1.bar(x + width/2, stds, width=width, label="Std")
    ax1.set_ylabel("Mean / Std")
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 右軸：外れ値割合（別スケール）
    ax2 = ax1.twinx()
    ax2.plot(x, outlier_ratios, color='red', marker='o', label="Outlier Ratio (%)")
    ax2.set_ylabel("Outlier Ratio (%)")
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())

    # タイトルと凡例
    plt.title(f"{sequence_id} — Motion Summary")
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # 保存
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{sequence_id}_motion_summary.png")
    plt.close()



def analyze_by_dataset(dataset_name, input_dir, use_npz=False):
    if use_npz:
        all_npz = []
        for split in ["train", "val", "test"]:
            pattern = os.path.join(input_dir, split, "*", "labels_v2", "labels.npz")
            all_npz += sorted(glob(pattern))
        if all_npz:
            motion_data = gather_motion_data_from_npz(all_npz)
            stats = analyze_motion(f"{dataset_name}_labels_v2_npz", motion_data)
            summarize_motion_stats(f"{dataset_name}_labels_v2_npz", stats)
        else:
            print("[!] No .npz files found")
        return

    if dataset_name in ["gen1", "gen4"]:
        for split in ["train", "val", "test"]:
            npy_files = sorted(glob(os.path.join(input_dir, split, '*_with_motion.npy')))
            if not npy_files:
                continue
            motion_data = gather_motion_data(npy_files)
            stats = analyze_motion(f"{dataset_name}_{split}", motion_data)
            summarize_motion_stats(f"{dataset_name}_{split}", stats)

    elif dataset_name == "DSEC":
        seq_dirs = sorted(glob(os.path.join(input_dir, '*')))
        for seq_dir in seq_dirs:
            label_path = os.path.join(seq_dir, 'object_detections', 'left', 'tracks_with_motion.npy')
            if os.path.exists(label_path):
                motion_data = gather_motion_data([label_path])
                seq_name = os.path.basename(seq_dir)
                stats = analyze_motion(seq_name, motion_data)
                summarize_motion_stats(seq_name, stats)

    elif dataset_name == "GIFU":
        for split in ["train", "val", "test"]:
            seq_dirs = sorted(glob(os.path.join(input_dir, split, '*')))
            for seq_dir in seq_dirs:
                label_path = os.path.join(seq_dir, 'labels', 'labels_events_with_motion.npy')
                if os.path.exists(label_path):
                    motion_data = gather_motion_data([label_path])
                    seq_name = os.path.basename(seq_dir)
                    stats = analyze_motion(seq_name, motion_data)
                    summarize_motion_stats(seq_name, stats)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Motion statistics checker")
    parser.add_argument("--dataset", type=str, choices=["gen1", "gen4", "DSEC", "GIFU"], required=True)
    parser.add_argument("--input_dir", type=str, required=True, help="Root dir for dataset")
    parser.add_argument("--use_npz", action="store_true", help="Use .npz (labels_v2) instead of .npy")

    args = parser.parse_args()

    analyze_by_dataset(args.dataset, args.input_dir, args.use_npz)
