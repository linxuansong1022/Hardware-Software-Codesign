from pathlib import Path
import csv
import numpy as np
from PIL import Image
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_CSV = PROJECT_ROOT / "data" / "metadata" / "image_manifest_split.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "gray48_cnn_center_w001" / "gray48_cnn_center.keras"
OUTPUT_DIR = PROJECT_ROOT / "models" / "gray48_cnn_center_w001_int8"

IMAGE_SIZE = (48, 48)
NUM_CHANNELS = 1
REPRESENTATIVE_SAMPLES = 200


#读取 manifest
def read_manifest(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["order_idx"] = int(row["order_idx"])
            row["is_usable"] = int(row["is_usable"])
            rows.append(row)
    return rows


def filter_train_rows(rows):
    filtered = []
    for row in rows:
        if row["split"] != "train":
            continue
        if row["is_usable"] != 1:
            continue
        if row["subject_id"] not in {"person_a", "person_b", "person_c"}:
            continue
        filtered.append(row)
    return filtered


# 路径解析
def resolve_image_path(filepath_in_manifest):
    return PROJECT_ROOT / filepath_in_manifest



#图片预处理（必须和训练保持一致）

def load_one_image_gray(img_path, image_size):
    img = Image.open(img_path).convert("L")
    img = img.resize(image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)   # (48,48) -> (48,48,1)
    return arr



# representative dataset
def representative_dataset_gen(train_rows):
    selected_rows = train_rows[:REPRESENTATIVE_SAMPLES]

    for row in selected_rows:
        img_path = resolve_image_path(row["filepath"])
        x = load_one_image_gray(img_path, IMAGE_SIZE)
        x = np.expand_dims(x, axis=0).astype(np.float32)   # (1,48,48,1)
        yield [x]



#主程序
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("读取 manifest ...")
    rows = read_manifest(MANIFEST_CSV)
    train_rows = filter_train_rows(rows)

    print(f"train 样本数: {len(train_rows)}")
    print(f"representative dataset 使用前 {min(REPRESENTATIVE_SAMPLES, len(train_rows))} 张")

    print("\n加载 Keras 模型 ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"模型路径: {MODEL_PATH}")

    print("\n开始 TFLite 转换（full INT8）...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(train_rows)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # 输入/输出也强制成 int8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = OUTPUT_DIR / "gray48_cnn_center_int8.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"\n已保存 INT8 TFLite 模型: {tflite_path}")
    print(f"模型大小: {len(tflite_model) / 1024:.2f} KB")


if __name__ == "__main__":
    main()