import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from multimodal_sft.image_utils import blend_polygon, blend_sign, detect_shape


def blend_and_save(
    img: np.ndarray,
    sign_name: str,
    shape: str,
    annotation: str,
    area: np.array,
    sign_path: Path,
) -> np.ndarray:
    img_sign = cv2.imread(str(sign_path / f"{sign_name}.png"), -1)
    img_sign_bin = (img_sign[:, :, 3] > 0).astype(np.uint8)
    _, area_sign = detect_shape(img_sign_bin)

    if shape == "circle":
        img = blend_sign(img, img_sign, annotation)
    else:
        img = blend_polygon(img, img_sign, area, area_sign, annotation)
    return img


def main():
    parser = argparse.ArgumentParser(description="合成画像を生成するスクリプト")
    parser.add_argument("--data", type=str, default="data", help="データを保存しているディレクトリ")
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_path = data_dir / "train"
    sign_path = Path("../data/road_signs/")
    output_path = Path("../data/synth_data")
    df = pd.read_csv("../data/shape.csv", header=None, names=["name", "shape", "color"])

    # 1枚の画像に対して3通りの合成画像を生成する
    synth_per_image = 3
    result_dict = dict(image_name=[], sign_name=[], shape=[], color=[], position=[])
    for j, path in enumerate((train_path / "images").glob("*.jpg")):
        img = cv2.imread(str(path))
        height, width, _ = img.shape
        with open(train_path / "labels" / (path.stem + ".txt")) as f:
            for i, line in enumerate(f):
                annotation = list(map(float, line.split()))
                _, x, y, w, h = annotation
                cx, cy = int(x * width), int(y * height)
                # 画像中の標識の位置を判定. 画像の左1/3, 中央1/3, 右1/3に分ける
                if cx < width // 3:
                    position = "left"
                elif cx > width * 2 // 3:
                    position = "right"
                else:
                    position = "center"

                # 画像中の標識の領域を切り出す
                x0, y0 = int((x - w / 2) * width), int((y - h / 2) * height)
                segment = np.load(
                    train_path / "segmentations" / (path.stem + f"_{i}.npy")
                )
                h, w = segment.shape
                # 標識の平均色を計算
                color = np.sum(
                    img[y0 : y0 + h, x0 : x0 + w] * segment[..., np.newaxis],
                    axis=(0, 1),
                ) / np.sum(segment)
                dominant_color = "bgr"[color.argmax()]

                # 標識の形を判定
                shape, area = detect_shape(segment.astype(np.uint8))
                if shape == "others":
                    continue

                # 色と形が一致するサインを探す
                sign_candidates = df[
                    (df["shape"] == shape) & (df["color"] == dominant_color)
                ]["name"].values
                if len(sign_candidates) == 0:
                    continue

                # 一致するサインの中からランダムに選び合成する
                n = len(sign_candidates)
                sign_names = np.random.choice(sign_candidates, min(synth_per_image, n))
                for sign_name in sign_names:
                    img_synth = blend_and_save(
                        img, sign_name, shape, annotation, area, sign_path
                    )
                    cv2.imwrite(
                        str(output_path / f"{path.stem}_{sign_name}.png"), img_synth
                    )
                    result_dict["image_name"].append(path.name)
                    result_dict["sign_name"].append(sign_name)
                    result_dict["shape"].append(shape)
                    result_dict["color"].append(dominant_color)
                    result_dict["position"].append(position)

    pd.DataFrame(result_dict).to_csv(data_dir / "synthetic_info.csv", index=False)
