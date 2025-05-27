import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

def load_tracks_txt_as_structured_array(txt_path: Path):
    dtype = np.dtype([
        ('t', '<u8'),
        ('class_id', 'u1'),
        ('x', '<f4'),
        ('y', '<f4'),
        ('w', '<f4'),
        ('h', '<f4'),
        ('track_id', '<u4')
    ])
    raw = np.genfromtxt(txt_path, delimiter=',', dtype=None, encoding=None)
    structured = np.array([tuple(row) for row in raw], dtype=dtype)
    return structured

def convert_txt_to_npy_only_for_split(input_dir: Path, split_yaml: Path):
    # YAMLファイルの読み込み
    split_cfg = OmegaConf.load(split_yaml)
    # train, val, test に含まれるすべてのシーケンスを取得（valがない場合もある）
    split_names = set()
    for split_key in ["train", "val", "test"]:
        if split_key in split_cfg:
            split_names.update(split_cfg[split_key])

    for name in split_names:
        txt_path = input_dir / name / "tracking" / f"{name}.txt"
        npy_path = input_dir / name / "tracking" / f"{name}.npy"

        if not txt_path.exists():
            print(f"⚠️ {txt_path} が存在しません。スキップします。")
            continue

        print(f"✅ {name}: txt → npy 変換中")
        data = load_tracks_txt_as_structured_array(txt_path)
        np.save(npy_path, data)

# 使用例
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("input_dir", help="input_dir (各 sequence が含まれる親ディレクトリ)")
    parser.add_argument("split_yaml", help="split.yaml のパス")
    args = parser.parse_args()

    convert_txt_to_npy_only_for_split(Path(args.input_dir), Path(args.split_yaml))
