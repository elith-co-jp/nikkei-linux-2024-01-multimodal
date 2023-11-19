import argparse
from pathlib import Path

import pandas as pd

from multimodal_sft.instruction_utils import fill_template


def main():
    parser = argparse.ArgumentParser(description="合成画像に関する質問と回答を生成するスクリプト")
    parser.add_argument("--data", type=str, default="data", help="データを保存しているディレクトリ")
    args = parser.parse_args()

    data_dir = Path(args.data)

    # 合成画像データのディレクトリ
    image_dir = data_dir / "synth_data"
    # 合成画像の情報
    df_info = pd.read_csv(data_dir / "synthetic_info.csv")
    # 標識に関する説明
    df_sign = pd.read_csv(data_dir / "roadsign_detail.csv")
    # 質問と回答のテンプレート
    df_qa = pd.read_csv(data_dir / "qa.csv")

    # 1枚の画像に対して3通りの質問・回答を生成する
    question_per_image = 3
    result_dict = {
        "image_name": [],
        "question": [],
        "answer": [],
    }

    # 形に関する日本語表現
    shape2jp = {
        "circle": "丸い",
        "triangle": "三角の",
        "rectangle": "四角い",
        "inverted_triangle": "逆三角の",
    }
    # 色に関する日本語表現
    color2jp = {
        "r": "赤",
        "b": "青",
    }
    # 位置に関する日本語表現
    position2jp = {"left": "左", "right": "右", "center": "中央"}
    for i, row in df_info.iterrows():
        image_name = row["image_name"]
        sign_name = row["sign_name"]
        image_path = image_dir / (Path(image_name).stem + "_" + sign_name + ".png")
        if not image_path.exists():
            continue

        # df_sign のうち、 name が sign_name と一致する行を取得
        sign = df_sign[df_sign["number"] == sign_name].iloc[0]
        # info_dict に情報を追加
        info_dict = {
            "name": sign["name"],
            "feature": sign["feature"],
            "detail": sign["detail"],
            "shape": shape2jp[row["shape"]],
            "color": color2jp[row["color"]],
            "position": position2jp[row["position"]],
        }
        # df_qa からランダムに3行取得
        qa = df_qa.sample(question_per_image)
        for j, qa_row in qa.iterrows():
            # テンプレートを埋める
            question_text = fill_template(qa_row["question"], info_dict)
            answer_text = fill_template(qa_row["answer"], info_dict)

            # 結果を保存
            result_dict["image_name"].append(image_path.name)
            result_dict["question"].append(question_text)
            result_dict["answer"].append(answer_text)

    df_result = pd.DataFrame(result_dict)
    df_result.to_csv(data_dir / "instructions.csv", index=False)


if __name__ == "__main__":
    main()
    main()
