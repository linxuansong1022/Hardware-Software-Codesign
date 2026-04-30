from pathlib import Path
import csv
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_CSV = PROJECT_ROOT / "data" / "metadata" / "image_manifest_split.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "gray48_cnn_center_w001" / "gray48_cnn_center.keras"
OUTPUT_DIR = PROJECT_ROOT / "models" / "gray48_cnn_center_w001_int8"

_ML_DATA_DIR = PROJECT_ROOT / "data" / "data"
_FACE_DATA_DIR = PROJECT_ROOT.parent / "face" / "data"

if _ML_DATA_DIR.exists():
    DATA_DIR = _ML_DATA_DIR
elif _FACE_DATA_DIR.exists():
    DATA_DIR = _FACE_DATA_DIR
else:
    raise FileNotFoundError(
        f"找不到图片数据目录，检查过:\n  {_ML_DATA_DIR}\n  {_FACE_DATA_DIR}"
    )

IMAGE_SIZE = (48, 48)
NUM_CHANNELS = 1
REPRESENTATIVE_SAMPLES = 200
KNOWN_SUBJECTS = ("person_a", "person_b", "person_c")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
EMBEDDING_DIM = 32


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
        if row["subject_id"] not in KNOWN_SUBJECTS:
            continue
        filtered.append(row)
    return filtered


# 路径解析
def resolve_image_path(filepath_in_manifest):
    parts = Path(filepath_in_manifest).parts
    for i, part in enumerate(parts):
        if part in {"person_a", "person_b", "person_c", "unknown"}:
            return DATA_DIR / Path(*parts[i:])
    return PROJECT_ROOT / filepath_in_manifest



#图片预处理（必须和训练保持一致）

def load_one_image_gray(img_path, image_size):
    img = Image.open(img_path).convert("L")
    img = img.resize(image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)   # (48,48) -> (48,48,1)
    return arr


def collect_unknown_image_paths():
    unknown_dir = DATA_DIR / "unknown"
    if not unknown_dir.exists():
        return []
    return sorted(
        p for p in unknown_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def build_representative_items(train_rows):
    """Build a deterministic, class-balanced calibration set.

    The manifest is sorted by subject, so using train_rows[:N] calibrates the
    INT8 model almost entirely on person_a. Quantization calibration does not
    need labels; include known classes evenly and some unknown images so the
    activation ranges match deployment inputs better.
    """
    grouped = {subject: [] for subject in KNOWN_SUBJECTS}
    for row in train_rows:
        grouped[row["subject_id"]].append(row)

    unknown_paths = collect_unknown_image_paths()
    buckets = [[("row", row) for row in grouped[subject]] for subject in KNOWN_SUBJECTS]
    if unknown_paths:
        buckets.append([("path", path) for path in unknown_paths])

    selected = []
    cursor = 0
    while len(selected) < REPRESENTATIVE_SAMPLES:
        added = False
        for bucket in buckets:
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                added = True
                if len(selected) >= REPRESENTATIVE_SAMPLES:
                    break
        if not added:
            break
        cursor += 1

    return selected



# representative dataset
def representative_dataset_gen(train_rows):
    selected_items = build_representative_items(train_rows)

    for kind, item in selected_items:
        if kind == "row":
            img_path = resolve_image_path(item["filepath"])
        else:
            img_path = item
        x = load_one_image_gray(img_path, IMAGE_SIZE)
        x = np.expand_dims(x, axis=0).astype(np.float32)   # (1,48,48,1)
        yield [x]


def build_dual_output_model(model):
    """Expose both softmax and Dense(32) embedding for open-set rejection."""
    inp = keras.Input(shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], NUM_CHANNELS))
    x = inp
    embedding_output = None

    for layer in model.layers:
        x = layer(x)
        if isinstance(layer, layers.Dense) and layer.units == EMBEDDING_DIM:
            embedding_output = x

    if embedding_output is None:
        raise ValueError(f"找不到 Dense({EMBEDDING_DIM}) embedding 层")

    softmax_output = x
    return keras.Model(
        inputs=inp,
        outputs=[softmax_output, embedding_output],
        name="face_softmax_embedding",
    )



#主程序
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("读取 manifest ...")
    rows = read_manifest(MANIFEST_CSV)
    train_rows = filter_train_rows(rows)
    representative_items = build_representative_items(train_rows)

    print(f"train 样本数: {len(train_rows)}")
    print(f"数据目录: {DATA_DIR}")
    print(f"representative dataset: {len(representative_items)} 张，按 person_a/b/c/unknown 均衡抽样")

    print("\n加载 Keras 模型 ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"模型路径: {MODEL_PATH}")
    deploy_model = build_dual_output_model(model)
    print("部署模型输出: softmax(3) + embedding(32)")

    print("\n开始 TFLite 转换（full INT8）...")
    converter = tf.lite.TFLiteConverter.from_keras_model(deploy_model)

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
