import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

def load_tracks_txt_as_dsec_det_format(txt_path: Path):
    # 正しい dtype に従い構造化配列に変換
    dtype = np.dtype([
        ('t', '<u8'),
        ('x', '<f4'),
        ('y', '<f4'),
        ('w', '<f4'),
        ('h', '<f4'),
        ('class_id', 'u1'),
        ('class_confidence', '<f4'),
        ('track_id', '<u4')
    ])

    # 生データ読み込み（float64やintなどで推定）
    raw = np.genfromtxt(txt_path, delimiter=',', dtype=float)

    # 1行だけだった場合に備えて形状を強制変換
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)

    # 7列なことを確認
    if raw.shape[1] != 7:
        raise ValueError(f"{txt_path} の列数が7ではありません（{raw.shape[1]}列）")

    # 項目整列して構造化配列に変換
    structured = np.array([
        (
            int(row[0]),        # t (timestamp)
            float(row[2]),      # x
            float(row[3]),      # y
            float(row[4]),      # w
            float(row[5]),      # h
            int(row[6]),        # class_id
            1.0,                # class_confidence（固定）
            int(row[1])         # track_id
        )
        for row in raw
    ], dtype=dtype)

    return structured


def convert_txt_to_npy_as_dsec_det(input_dir: Path, split_yaml: Path):
    split_cfg = OmegaConf.load(split_yaml)
    split_names = set()
    for split_key in ["train", "val", "test"]:
        if split_key in split_cfg:
            split_names.update(split_cfg[split_key])

    for name in split_names:
        txt_path = input_dir / name / "tracking" / f"{name}.txt"
        npy_path = input_dir / name / "tracking" / "tracks.npy"

        if not txt_path.exists():
            print(f"⚠️ {txt_path} が存在しません。スキップします。")
            continue

        print(f"✅ {name}: txt → npy (DSEC-DET形式) に変換中")
        data = load_tracks_txt_as_dsec_det_format(txt_path)
        np.save(npy_path, data)

# 実行部
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("input_dir", help="各 sequence が含まれる親ディレクトリ")
    parser.add_argument("split_yaml", help="split.yaml のパス")
    args = parser.parse_args()

    convert_txt_to_npy_as_dsec_det(Path(args.input_dir), Path(args.split_yaml))
