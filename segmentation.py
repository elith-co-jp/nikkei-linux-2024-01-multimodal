import argparse
from pathlib import Path

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator


def main():
    parser = argparse.ArgumentParser(description="標識のセグメンテーションを行うスクリプト")
    parser.add_argument("--data", type=str, default="data", help="データを保存しているディレクトリ")
    args = parser.parse_args()

    data_dir = Path(args.data)

    sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth").to("cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)

    for j, path in enumerate((data_dir / "train" / "images").glob("*.jpg")):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        with open(data_dir / "train" / "labels" / f"{path.stem}.txt") as f:
            for i, line in enumerate(f):
                c, x, y, w, h = map(float, line.split())
                x0 = int((x - w / 2) * width)
                y0 = int((y - h / 2) * height)
                x1 = int((x + w / 2) * width)
                y1 = int((y + h / 2) * height)
                img = image[y0:y1, x0:x1]
                try:
                    masks = mask_generator.generate(img)
                except:
                    continue
                masks = sorted(masks, key=lambda x: -x["area"])
                np.save(
                    data_dir / "train" / "segmentations" / f"{path.stem}_{i}.npy",
                    masks[0]["segmentation"],
                )
