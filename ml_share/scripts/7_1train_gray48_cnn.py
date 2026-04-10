from pathlib import Path
import csv
from collections import Counter
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 路径和基本配置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = PROJECT_ROOT / "data" / "metadata" / "image_manifest_split.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "gray48_cnn"

IMAGE_SIZE = (48, 48)
NUM_CHANNELS = 1 #灰度图
BATCH_SIZE = 32
EPOCHS = 100

CLASS_NAMES = ["person_a", "person_b", "person_c"]

# 类别名到数字标签的映射
CLASS_TO_INDEX = {
    "person_a": 0,
    "person_b": 1,
    "person_c": 2,
}


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
    # 根据 split 过滤数据
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
    # 打印每个 split 里的类别分布
    counter = Counter()

    for row in rows:
        subject_id = row["subject_id"]
        counter[subject_id] += 1

    print(f"\n[{split_name}] 样本数量: {len(rows)}")

    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {counter[class_name]}")


def load_one_image_gray(img_path, image_size):
    # 读取一张图片
    img = Image.open(img_path)

    # 转成灰度图
    img = img.convert("L")

    # resize到指定大小
    img = img.resize(image_size)

    # 转成numpy，并归一化到[0,1]
    arr = np.asarray(img, dtype=np.float32)
    arr = arr / 255.0

    # 增加一个通道维度，变成(H, W, 1)
    arr = np.expand_dims(arr, axis=-1)

    return arr


def build_numpy_dataset(rows, project_root, image_size):
    # 把一个 split 的图片全部加载进内存
    x_list = []
    y_list = []

    for row in rows:
        img_path = project_root / row["filepath"]
        img_array = load_one_image_gray(img_path, image_size)

        class_name = row["subject_id"]
        label = CLASS_TO_INDEX[class_name]

        x_list.append(img_array)
        y_list.append(label)

    x = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    return x, y


def build_model(input_shape, num_classes):
    # 构建一个比较小的CNN
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(8, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.1),

        layers.Conv2D(16, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.1),

        layers.Conv2D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.1),

        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    # 创建输出文件夹
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("读取 manifest ...")
    rows = read_manifest(MANIFEST_CSV)

    train_rows = filter_rows(rows, "train")
    val_rows = filter_rows(rows, "val")
    test_rows = filter_rows(rows, "test")

    print_split_distribution(train_rows, "train")
    print_split_distribution(val_rows, "val")
    print_split_distribution(test_rows, "test")

    print("\n加载图片到内存 ...")
    x_train, y_train = build_numpy_dataset(train_rows, PROJECT_ROOT, IMAGE_SIZE)
    x_val, y_val = build_numpy_dataset(val_rows, PROJECT_ROOT, IMAGE_SIZE)
    x_test, y_test = build_numpy_dataset(test_rows, PROJECT_ROOT, IMAGE_SIZE)

    print("\n数据形状:")
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_val:  ", x_val.shape)
    print("y_val:  ", y_val.shape)
    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)

    print("\n构建模型 ...")
    input_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0], NUM_CHANNELS)
    num_classes = len(CLASS_NAMES)

    model = build_model(input_shape, num_classes)

    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    callbacks = [early_stopping]

    print("\n开始训练 ...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks,
    )

    print("\n在验证集上评估 ...")
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"val_loss = {val_loss:.4f}")
    print(f"val_acc  = {val_acc:.4f}")

    print("\n在测试集上评估 ...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"test_loss = {test_loss:.4f}")
    print(f"test_acc  = {test_acc:.4f}")

    print("\n生成测试集预测 ...")
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print("\n测试集分类报告:")
    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    print(report)

    print("\n测试集混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    keras_model_path = OUTPUT_DIR / "gray48_cnn.keras"
    model.save(keras_model_path)
    print(f"\n已保存模型: {keras_model_path}")

    label_map_path = OUTPUT_DIR / "label_map.csv"

    with open(label_map_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "label"])

        for class_name in CLASS_NAMES:
            label = CLASS_TO_INDEX[class_name]
            writer.writerow([class_name, label])

    print(f"已保存标签映射: {label_map_path}")


if __name__ == "__main__":
    main()