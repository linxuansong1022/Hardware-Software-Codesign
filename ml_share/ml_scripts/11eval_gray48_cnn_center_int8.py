from pathlib import Path
import csv
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


# =========================
# 1. 路径与配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_CSV = PROJECT_ROOT / "data" / "metadata" / "image_manifest_split.csv"
TFLITE_PATH = PROJECT_ROOT / "models" / "gray48_cnn_center_w001_int8" / "gray48_cnn_center_int8.tflite"

IMAGE_SIZE = (48, 48)
CLASS_NAMES = ["person_a", "person_b", "person_c"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# =========================
# 2. 读取 manifest
# =========================
def read_manifest(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["order_idx"] = int(row["order_idx"])
            row["is_usable"] = int(row["is_usable"])
            rows.append(row)
    return rows


def filter_rows(rows, split_name):
    filtered = []
    for row in rows:
        if row["split"] != split_name:
            continue
        if row["is_usable"] != 1:
            continue
        if row["subject_id"] not in CLASS_TO_INDEX:
            continue
        filtered.append(row)
    return filtered


def print_split_distribution(rows, split_name):
    counter = Counter(row["subject_id"] for row in rows)
    print(f"\n[{split_name}] 样本数量: {len(rows)}")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {counter[class_name]}")


# =========================
# 3. 图片预处理（和训练保持一致）
# =========================
def resolve_image_path(filepath_in_manifest):
    return PROJECT_ROOT / filepath_in_manifest


def load_one_image_gray(img_path, image_size):
    img = Image.open(img_path).convert("L")
    img = img.resize(image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)   # (48,48) -> (48,48,1)
    return arr


def build_numpy_dataset(rows, image_size):
    x_list = []
    y_list = []

    for row in rows:
        img_path = resolve_image_path(row["filepath"])
        img_array = load_one_image_gray(img_path, image_size)

        label = CLASS_TO_INDEX[row["subject_id"]]
        x_list.append(img_array)
        y_list.append(label)

    x = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return x, y


# =========================
# 4. 跑 TFLite INT8 推理
# =========================
def run_tflite_inference(x_data, tflite_path):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    print("\nTFLite 输入信息:")
    print("  shape:", input_details["shape"])
    print("  dtype:", input_details["dtype"])
    print("  quantization:", input_details["quantization"])

    print("\nTFLite 输出信息:")
    print("  shape:", output_details["shape"])
    print("  dtype:", output_details["dtype"])
    print("  quantization:", output_details["quantization"])

    y_pred = []
    y_prob_dequant = []

    for i in range(len(x_data)):
        x = x_data[i:i+1]   # shape: (1,48,48,1)

        # float32 -> int8
        x_q = np.round(x / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details["index"], x_q)
        interpreter.invoke()

        output_q = interpreter.get_tensor(output_details["index"])   # int8
        output_f = (output_q.astype(np.float32) - output_zero_point) * output_scale

        pred = int(np.argmax(output_f[0]))
        y_pred.append(pred)
        y_prob_dequant.append(output_f[0])

    y_pred = np.array(y_pred, dtype=np.int64)
    y_prob_dequant = np.array(y_prob_dequant, dtype=np.float32)

    return y_pred, y_prob_dequant


# =========================
# 5. 主程序
# =========================
def main():
    print("读取 manifest ...")
    rows = read_manifest(MANIFEST_CSV)

    test_rows = filter_rows(rows, "test")
    print_split_distribution(test_rows, "test")

    print("\n加载测试集图片 ...")
    x_test, y_test = build_numpy_dataset(test_rows, IMAGE_SIZE)

    print("\n数据形状:")
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    print(f"\n加载 TFLite 模型: {TFLITE_PATH}")
    y_pred, y_prob = run_tflite_inference(x_test, TFLITE_PATH)

    test_acc = float(np.mean(y_pred == y_test))
    print(f"\nINT8 TFLite test_acc = {test_acc:.4f}")

    print("\n测试集分类报告:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    ))

    print("\n测试集混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()