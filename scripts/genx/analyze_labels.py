import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib.ticker as ticker

# --- ヘルパー関数群 ---

def concat_valid_pairs(dx, dy):
    """
    dx, dy のペアのうち、どちらも0に近くない有効なペアのみを返す
    """
    if dx.size == 0 or dy.size == 0:
        return np.array([]), np.array([])
    valid_mask = ~(np.isclose(dx, 0) | np.isclose(dy, 0))
    return dx[valid_mask], dy[valid_mask]

def compute_motion_asymmetry_mask(dx, dy, motion_mag, mag_thresh=80.0, ratio_thresh=10.0):
    """
    モーションの大きさと非対称性に基づいてフィルタリングマスクを計算する
    """
    eps = 1e-6
    ratio1 = np.abs(dx) / (np.abs(dy) + eps)
    ratio2 = np.abs(dy) / (np.abs(dx) + eps)
    max_ratio = np.maximum(ratio1, ratio2)

    motion_mask = motion_mag < mag_thresh
    ratio_mask = max_ratio < ratio_thresh
    final_mask = motion_mask & ratio_mask

    return final_mask, motion_mask, ratio_mask

def report_asymmetry_stats(dx, dy, motion_mag, prefix, verbose=False):
    """
    非対称性の統計を計算し、verboseモードであればコンソールに表示する
    """
    stats = {}
    total = len(dx)
    if total == 0:
        return stats

    eps = 1e-6
    max_ratio = np.maximum(np.abs(dx) / (np.abs(dy) + eps), np.abs(dy) / (np.abs(dx) + eps))

    plt.figure()
    plt.hist(max_ratio, bins=100, range=(0, 50), alpha=0.7)
    plt.xlabel("max(|dx/dy|, |dy/dx|)")
    plt.ylabel("Frequency")
    plt.title(f"{prefix} — Asymmetry Ratio Distribution (Unfiltered)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/{prefix}_asymmetry_ratio_hist.png")
    plt.close()

    thresholds = [5, 10, 20, 50, 100]
    if verbose:
        print(f"\n[{prefix}] Asymmetry Ratio Stats (Unfiltered, max(|dx/dy|, |dy/dx|))")
    
    for t in thresholds:
        count = np.sum(max_ratio > t)
        percent = 100 * count / total
        if verbose:
            print(f"  > {t:>3} : {count} samples ({percent:.2f}%)")
        stats[f'ratio_gt_{t}_count'] = count
        stats[f'ratio_gt_{t}_percent'] = percent

    motion_thresh, ratio_thresh = 80, 10
    joint_mask = (motion_mag > motion_thresh) & (max_ratio > ratio_thresh)
    joint_count = np.sum(joint_mask)
    joint_percent = 100 * joint_count / total
    
    if verbose:
        print(f"\n  Combined condition (motion > {motion_thresh} & ratio > {ratio_thresh})")
        print(f"  -> {joint_count} samples ({joint_percent:.2f}%)")
        
    stats[f'combined_motion_gt_{motion_thresh}_ratio_gt_{ratio_thresh}_count'] = joint_count
    stats[f'combined_motion_gt_{motion_thresh}_ratio_gt_{ratio_thresh}_percent'] = joint_percent

    return stats

def summarize_motion_stats(sequence_id, stats_dict):
    """
    統計情報のサマリーグラフを生成・保存する
    """
    keys = list(stats_dict.keys())
    if not keys: return

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

    plt.savefig(f"outputs/{sequence_id}_motion_summary.png")
    plt.close()

def flatten_basic_stats(sequence_id, basic_stats_dict):
    """CSV保存用に基本統計の辞書をフラットなリストに変換する"""
    rows = []
    for metric, values in basic_stats_dict.items():
        row = {'sequence_id': sequence_id, 'metric': metric, **values}
        rows.append(row)
    return rows

def flatten_asymmetry_stats(sequence_id, asymmetry_stats_dict):
    """CSV保存用に非対称性統計の辞書をフラットなリストに変換する"""
    rows = []
    for kind, values in asymmetry_stats_dict.items():
        if values:
            row = {'sequence_id': sequence_id, 'type': kind, **values}
            rows.append(row)
    return rows

def generate_threshold_justification_report(dx, dy, data_type="prev"):
    """
    しきい値決定の根拠となるレポート（パーセンタイル値とヒストグラム）を生成する。
    """
    total_count = len(dx)
    if total_count == 0: return

    print("\n" + "#"*60)
    print(f"## Threshold Justification Report ({data_type})")
    print("#"*60)

    metrics = {
        'Magnitude': np.sqrt(dx**2 + dy**2),
        'Asymmetry_Ratio': np.maximum(np.abs(dx) / (np.abs(dy) + 1e-6), np.abs(dy) / (np.abs(dx) + 1e-6)),
        'abs_dx': np.abs(dx),
        'abs_dy': np.abs(dy)
    }

    percentiles_to_calc = [90, 95, 98, 99, 99.5, 99.9]
    print("\n[Percentile Values (Value below which X% of observations fall)]")
    header = f"{'Metric':<18} |" + " |".join([f"{p:^7.1f}%" for p in percentiles_to_calc])
    print(header)
    print("-" * len(header))
    
    for name, data in metrics.items():
        percentile_values = np.percentile(data, percentiles_to_calc)
        row_str = f"{name:<18} |" + " |".join([f"{v:7.2f}" for v in percentile_values])
        print(row_str)

    print("\n[Distribution Histograms]")
    print("Generating histograms for each metric (saved to 'outputs' directory)...")
    
    for name, data in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=200, alpha=0.75, range=(0, np.percentile(data, 99.9)))
        plt.yscale('log')
        plt.title(f"Distribution of {name} ({data_type}, y-axis is log-scaled)")
        plt.xlabel(f"Value of {name}")
        plt.ylabel("Frequency (log scale)")
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        
        filename = f"outputs/justification_hist_{data_type}_{name.replace('|','').replace(' ','_')}.png"
        plt.savefig(filename)
        plt.close()
        print(f" -> Saved {filename}")

    print("\nReport generation complete.")
    print("#"*60 + "\n")


# --- メイン分析関数 ---

def analyze_motion(sequence_id, motion_data, verbose=False):
    """
    モーションデータを分析し、統計情報とグラフを生成する
    """
    basic_stats = {}
    asymmetry_stats = {}

    if 'dx_prev' in motion_data:
        dx_prev, dy_prev = motion_data["dx_prev"], motion_data["dy_prev"]
        motion_prev_mag = np.sqrt(dx_prev ** 2 + dy_prev ** 2)
        asymmetry_stats["prev"] = report_asymmetry_stats(dx_prev, dy_prev, motion_prev_mag, prefix=f"{sequence_id}_prev", verbose=verbose)
        mask_prev, motion_mask_prev, ratio_mask_prev = compute_motion_asymmetry_mask(dx_prev, dy_prev, motion_prev_mag)
        if verbose:
            print(f"\n[{sequence_id}] motion_prev_mag: total={len(motion_prev_mag)}, kept={np.sum(mask_prev)}")
            print(f"[{sequence_id}]  removed by magnitude: {np.sum(~motion_mask_prev)}")
            print(f"[{sequence_id}]  removed by asymmetry: {np.sum(~ratio_mask_prev)}")
    
    if 'dx_next' in motion_data:
        dx_next, dy_next = motion_data["dx_next"], motion_data["dy_next"]
        motion_next_mag = np.sqrt(dx_next ** 2 + dy_next ** 2)
        asymmetry_stats["next"] = report_asymmetry_stats(dx_next, dy_next, motion_next_mag, prefix=f"{sequence_id}_next", verbose=verbose)
        mask_next, motion_mask_next, ratio_mask_next = compute_motion_asymmetry_mask(dx_next, dy_next, motion_next_mag)
        if verbose:
            print(f"\n[{sequence_id}] motion_next_mag: total={len(motion_next_mag)}, kept={np.sum(mask_next)}")
            print(f"[{sequence_id}]  removed by magnitude: {np.sum(~motion_mask_next)}")
            print(f"[{sequence_id}]  removed by asymmetry: {np.sum(~ratio_mask_next)}")

    motion_data_unfiltered = motion_data.copy()
    if 'dx_prev' in motion_data_unfiltered: motion_data_unfiltered["motion_prev_mag"] = motion_prev_mag
    if 'dx_next' in motion_data_unfiltered: motion_data_unfiltered["motion_next_mag"] = motion_next_mag

    motion_data_filtered = {}
    for key in motion_data_unfiltered:
        if "prev" in key and 'dx_prev' in motion_data:
            motion_data_filtered[key] = motion_data_unfiltered[key][mask_prev]
        elif "next" in key and 'dx_next' in motion_data:
            motion_data_filtered[key] = motion_data_unfiltered[key][mask_next]

    for key, values in motion_data_filtered.items():
        if len(values) == 0: continue
        mean, std = values.mean(), values.std()
        outliers = values[np.abs(values - mean) > (3 * std)]
        outlier_ratio = 100 * len(outliers) / len(values)

        print(f"\n[{sequence_id}] {key} (Filtered)")
        print(f"  Count   : {len(values)}")
        print(f"  Mean    : {mean:.4f}")
        print(f"  Std     : {std:.4f}")
        print(f"  Min/Max : {values.min():.4f} / {values.max():.4f}")
        print(f"  Outliers (>3σ): {len(outliers)} ({outlier_ratio:.2f}%)")
        
        basic_stats[key] = {
            "count": len(values), "mean": mean, "std": std,
            "min": values.min(), "max": values.max(),
            "outlier_count": len(outliers), "outlier_ratio": outlier_ratio,
        }

        plt.figure()
        plt.hist(values, bins=100, alpha=0.7)
        plt.title(f"{sequence_id} — {key} (Filtered)")
        plt.grid(True)
        plt.xlabel(key)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"outputs/{sequence_id}_{key}_hist.png")
        plt.close()

    return {"basic_stats": basic_stats, "asymmetry_stats": asymmetry_stats}


# --- スクリプト実行のメインロジック ---

def main(args):
    print(f"Loading combined data from: {args.input_file}")
    data = np.load(args.input_file)

    if args.justify_thresholds:
        if args.eval_type in ['all', 'prev']:
            generate_threshold_justification_report(data['dx_prev'], data['dy_prev'], data_type="prev")
        if args.eval_type in ['all', 'next']:
            generate_threshold_justification_report(data['dx_next'], data['dy_next'], data_type="next")
        
        print("Justification report generated. Continuing with main analysis...")

    motion_data = {}
    if args.eval_type in ['all', 'prev']:
        dx_prev, dy_prev = concat_valid_pairs(data['dx_prev'], data['dy_prev'])
        motion_data["dx_prev"] = dx_prev
        motion_data["dy_prev"] = dy_prev

    if args.eval_type in ['all', 'next']:
        dx_next, dy_next = concat_valid_pairs(data['dx_next'], data['dy_next'])
        motion_data["dx_next"] = dx_next
        motion_data["dy_next"] = dy_next
        
    if not motion_data:
        print("No data selected for evaluation. Exiting.")
        return

    print("\nData loaded. Starting main analysis...")
    
    sequence_id = f"{os.path.splitext(os.path.basename(args.input_file))[0]}_{args.eval_type}"
    stats = analyze_motion(sequence_id, motion_data, verbose=args.verbose)
    
    summarize_motion_stats(sequence_id, stats["basic_stats"])

    all_basic_stats_list = flatten_basic_stats(sequence_id, stats["basic_stats"])
    all_asymmetry_stats_list = flatten_asymmetry_stats(sequence_id, stats["asymmetry_stats"])

    if all_basic_stats_list:
        df_basic = pd.DataFrame(all_basic_stats_list)
        output_path = os.path.join("outputs", f"{sequence_id}_stats_basic.csv")
        df_basic.to_csv(output_path, index=False)
        print(f"\n[+] Basic statistics saved to: {output_path}")

    if all_asymmetry_stats_list:
        df_asymmetry = pd.DataFrame(all_asymmetry_stats_list)
        output_path = os.path.join("outputs", f"{sequence_id}_stats_asymmetry.csv")
        df_asymmetry.to_csv(output_path, index=False)
        print(f"\n[+] Asymmetry statistics saved to: {output_path}")

    print("\nAnalysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze combined motion data from a single .npz file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the combined .npz file.")
    parser.add_argument(
        "--eval_type", 
        type=str, 
        default="all", 
        choices=['all', 'prev', 'next'], 
        help="Type of motion data to evaluate."
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed filtering logs.")
    parser.add_argument(
        "--justify_thresholds", 
        action="store_true", 
        help="Generate a special report to help justify filter threshold values."
    )
    
    args = parser.parse_args()
    os.makedirs("outputs", exist_ok=True)
    main(args)