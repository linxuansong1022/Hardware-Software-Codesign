from pathlib import Path
import argparse
import csv
from collections import Counter
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── 路径和常量 ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = PROJECT_ROOT / "data" / "metadata" / "image_manifest_split.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "gray48_cnn_center"

# 图片数据目录（兼容两种路径）
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

UNKNOWN_DIR = DATA_DIR / "unknown"

IMAGE_SIZE = (48, 48)
NUM_CHANNELS = 1
BATCH_SIZE = 32
EPOCHS = 100

CLASS_NAMES = ["person_a", "person_b", "person_c"]
NUM_CLASSES = len(CLASS_NAMES)
EMBEDDING_DIM = 32

CLASS_TO_INDEX = {"person_a": 0, "person_b": 1, "person_c": 2}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Center Loss 超参数
CENTER_LOSS_WEIGHT = 0.1   # λ: center loss 权重
CENTER_LR = 0.5            # 质心更新速率


# ── 数据加载工具函数（复用自 7_2） ────────────────────────────

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
    counter = Counter()
    for row in rows:
        counter[row["subject_id"]] += 1
    print(f"\n[{split_name}] 样本数量: {len(rows)}")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {counter[class_name]}")


def load_one_image_gray(img_path, image_size):
    img = Image.open(img_path)
    img = img.convert("L")
    img = img.resize(image_size)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return arr


def resolve_image_path(filepath_in_manifest):
    parts = Path(filepath_in_manifest).parts
    for i, part in enumerate(parts):
        if part in {"person_a", "person_b", "person_c", "unknown"}:
            relative = Path(*parts[i:])
            return DATA_DIR / relative
    return PROJECT_ROOT / filepath_in_manifest


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


def load_unknown_images(unknown_dir, image_size):
    if not unknown_dir.exists():
        raise FileNotFoundError(f"Unknown 目录不存在: {unknown_dir}")
    image_files = sorted(
        p for p in unknown_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(image_files) == 0:
        raise ValueError(f"Unknown 目录为空: {unknown_dir}")
    x_list = []
    filenames = []
    for p in image_files:
        arr = load_one_image_gray(p, image_size)
        x_list.append(arr)
        filenames.append(p.name)
    x = np.array(x_list, dtype=np.float32)
    return x, filenames


def save_history_csv(history, output_csv_path):
    history_dict = history.history
    fieldnames = list(history_dict.keys())
    num_epochs = len(history_dict[fieldnames[0]])
    with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + fieldnames)
        for i in range(num_epochs):
            row = [i + 1]
            for key in fieldnames:
                row.append(history_dict[key][i])
            writer.writerow(row)


# ── 模型构建 ────────────────────────────────────────────────

def build_base_model(input_shape, num_classes):
    """构建和 7_2 完全一样的 CNN（不 compile）"""
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ], name="data_augmentation")

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,

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
        layers.Dense(EMBEDDING_DIM, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])

    return model


def build_dual_output_model(base_model, input_shape):
    """构建双输出模型：同时输出 softmax 和 embedding

    共享 base_model 的权重，一次前向传播拿到两个输出。
    """
    inp = keras.Input(shape=input_shape)
    x = inp
    embedding_output = None

    for layer in base_model.layers:
        x = layer(x)
        # 找到 Dense(32) 层，记录它的输出
        if isinstance(layer, layers.Dense) and layer.units == EMBEDDING_DIM:
            embedding_output = x

    softmax_output = x

    dual_model = keras.Model(
        inputs=inp,
        outputs=[softmax_output, embedding_output]
    )

    return dual_model


# ── Center Loss Model ───────────────────────────────────────

class CenterLossModel(keras.Model):
    """带 center loss 的训练包装器

    total_loss = crossentropy + λ * center_loss
    center_loss = 每个样本到其类质心的平均平方距离
    """

    def __init__(self, base_model, input_shape, num_classes, embedding_dim,
                 center_loss_weight, center_lr, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.dual_model = build_dual_output_model(base_model, input_shape)
        self.center_loss_weight = center_loss_weight
        self.center_lr = center_lr
        self.num_classes = num_classes

        # 质心：不参与梯度优化，手动更新
        self.centroids = tf.Variable(
            tf.zeros([num_classes, embedding_dim]),
            trainable=False,
            name="centroids"
        )

        # 追踪指标
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.ce_loss_tracker = keras.metrics.Mean(name="ce_loss")
        self.center_loss_tracker = keras.metrics.Mean(name="center_loss")
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.loss_tracker, self.ce_loss_tracker,
                self.center_loss_tracker, self.acc_tracker]

    def call(self, inputs, training=False):
        """标准前向传播（用于 predict/evaluate 兼容性）"""
        return self.base_model(inputs, training=training)

    def _compute_losses(self, x, y, training):
        """计算 crossentropy + center loss"""
        y_pred, embeddings = self.dual_model(x, training=training)

        # Crossentropy
        ce_loss = keras.losses.sparse_categorical_crossentropy(y, y_pred)
        ce_loss = tf.reduce_mean(ce_loss)

        # Center Loss: 每个样本到其类质心的平方距离
        # batch_centroids: 取出每个样本对应类别的质心
        y_int = tf.cast(y, tf.int32)
        batch_centroids = tf.gather(self.centroids, y_int)  # (batch, 32)
        center_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(embeddings - batch_centroids), axis=1)
        )

        total_loss = ce_loss + self.center_loss_weight * center_loss

        return total_loss, ce_loss, center_loss, y_pred, embeddings

    def _update_centroids(self, embeddings, y):
        """手动更新质心（移动平均）"""
        y_int = tf.cast(y, tf.int32)

        # 计算 batch 内每类的平均 embedding
        batch_means = tf.math.unsorted_segment_mean(
            embeddings, y_int, num_segments=self.num_classes
        )

        # 检查哪些类在这个 batch 里有样本
        class_counts = tf.math.unsorted_segment_sum(
            tf.ones_like(y, dtype=tf.float32), y_int,
            num_segments=self.num_classes
        )
        has_samples = class_counts > 0  # (num_classes,)

        # 只更新有样本的类的质心
        updated = tf.where(
            has_samples[:, tf.newaxis],
            self.centroids + self.center_lr * (batch_means - self.centroids),
            self.centroids
        )
        self.centroids.assign(updated)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            total_loss, ce_loss, center_loss, y_pred, embeddings = \
                self._compute_losses(x, y, training=True)

        # 只更新 base_model 的参数（不更新质心）
        grads = tape.gradient(total_loss, self.dual_model.trainable_variables)
        self.optimizer.apply(grads, self.dual_model.trainable_variables)

        # 手动更新质心
        self._update_centroids(embeddings, y)

        # 更新指标
        self.loss_tracker.update_state(total_loss)
        self.ce_loss_tracker.update_state(ce_loss)
        self.center_loss_tracker.update_state(center_loss)
        self.acc_tracker.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        total_loss, ce_loss, center_loss, y_pred, embeddings = \
            self._compute_losses(x, y, training=False)

        self.loss_tracker.update_state(total_loss)
        self.ce_loss_tracker.update_state(ce_loss)
        self.center_loss_tracker.update_state(center_loss)
        self.acc_tracker.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


# ── 评估工具 ────────────────────────────────────────────────

def build_embedding_extractor(base_model, input_shape):
    """从 base_model 构建 embedding 提取器"""
    inp = keras.Input(shape=input_shape)
    x = inp
    for layer in base_model.layers:
        x = layer(x)
        if isinstance(layer, layers.Dense) and layer.units == EMBEDDING_DIM:
            break
    return keras.Model(inputs=inp, outputs=x)


def compute_centroids_from_embeddings(embeddings, labels, num_classes):
    """从 embedding 计算质心"""
    centroids = {}
    for class_name in CLASS_NAMES:
        class_idx = CLASS_TO_INDEX[class_name]
        mask = labels == class_idx
        centroids[class_name] = np.mean(embeddings[mask], axis=0)
    return centroids


def eval_embedding_quality(embedding_model, x_train, y_train, x_test, y_test,
                           x_unknown):
    """内联评估：embedding 紧凑度和 unknown rejection"""
    print("\n========== Embedding 质量评估 ==========")

    emb_train = embedding_model.predict(x_train, verbose=0)
    emb_test = embedding_model.predict(x_test, verbose=0)
    emb_unknown = embedding_model.predict(x_unknown, verbose=0)

    # 质心
    centroids = compute_centroids_from_embeddings(emb_train, y_train, NUM_CLASSES)

    # 已知人脸距离
    centroid_matrix = np.array([centroids[cn] for cn in CLASS_NAMES])
    diff_test = emb_test[:, np.newaxis, :] - centroid_matrix[np.newaxis, :, :]
    test_dists = np.linalg.norm(diff_test, axis=2)
    test_min_dist = np.min(test_dists, axis=1)

    # 陌生人距离
    diff_unk = emb_unknown[:, np.newaxis, :] - centroid_matrix[np.newaxis, :, :]
    unk_dists = np.linalg.norm(diff_unk, axis=2)
    unk_min_dist = np.min(unk_dists, axis=1)

    print(f"\n距离分布:")
    print(f"  {'':12} {'已知(test)':>12} {'陌生人(unknown)':>16}")
    print(f"  {'均值':<12} {np.mean(test_min_dist):>12.4f} {np.mean(unk_min_dist):>16.4f}")
    print(f"  {'中位数':<12} {np.median(test_min_dist):>12.4f} {np.median(unk_min_dist):>16.4f}")
    print(f"  {'最小值':<12} {np.min(test_min_dist):>12.4f} {np.min(unk_min_dist):>16.4f}")
    print(f"  {'最大值':<12} {np.max(test_min_dist):>12.4f} {np.max(unk_min_dist):>16.4f}")

    # Nearest-centroid 准确率
    test_nearest = np.argmin(test_dists, axis=1)
    nearest_acc = np.mean(test_nearest == y_test)
    print(f"\n  Nearest-centroid 准确率: {nearest_acc:.4f}")

    # 两阶段评估
    # softmax
    softmax_test = embedding_model.predict(x_test, verbose=0)  # 这里用的是 embedding model，不是 full model
    # 需要 full model 的 softmax 输出，用 base_model
    # 先跳过 softmax 部分，只评估 embedding 距离

    # 简单阈值扫描 (只做 embedding 距离)
    all_dists = np.concatenate([test_min_dist, unk_min_dist])
    true_labels = np.concatenate([np.ones(len(test_min_dist)), np.zeros(len(unk_min_dist))])

    best_f1 = 0
    best_thresh = 0
    best_result = {}

    for thresh in np.linspace(np.min(all_dists), np.max(all_dists), 200):
        preds = (all_dists <= thresh).astype(int)
        tp = int(np.sum((preds == 1) & (true_labels == 1)))
        fp = int(np.sum((preds == 1) & (true_labels == 0)))
        tn = int(np.sum((preds == 0) & (true_labels == 0)))
        fn = int(np.sum((preds == 0) & (true_labels == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_result = {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                           "precision": precision, "recall": recall, "f1": f1}

    print(f"\n  最佳 Embedding 距离阈值: {best_thresh:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    print(f"  F1:        {best_result['f1']:.4f}")
    print(f"  已知正确接受:  {best_result['tp']}/{best_result['tp']+best_result['fn']}")
    print(f"  陌生人正确拒绝: {best_result['tn']}/{best_result['tn']+best_result['fp']}")

    return {
        "test_mean_dist": float(np.mean(test_min_dist)),
        "unknown_mean_dist": float(np.mean(unk_min_dist)),
        "nearest_acc": float(nearest_acc),
        "best_f1": float(best_f1),
        "best_threshold": float(best_thresh),
    }


# ── 主函数 ──────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Center Loss 超参数: λ={CENTER_LOSS_WEIGHT}, center_lr={CENTER_LR}")
    print(f"数据目录: {DATA_DIR}")

    # ── 1. 加载数据 ──
    print("\n读取 manifest ...")
    rows = read_manifest(MANIFEST_CSV)
    train_rows = filter_rows(rows, "train")
    val_rows = filter_rows(rows, "val")
    test_rows = filter_rows(rows, "test")

    print_split_distribution(train_rows, "train")
    print_split_distribution(val_rows, "val")
    print_split_distribution(test_rows, "test")

    print("\n加载图片到内存 ...")
    x_train, y_train = build_numpy_dataset(train_rows, IMAGE_SIZE)
    x_val, y_val = build_numpy_dataset(val_rows, IMAGE_SIZE)
    x_test, y_test = build_numpy_dataset(test_rows, IMAGE_SIZE)

    print(f"x_train: {x_train.shape}, x_val: {x_val.shape}, x_test: {x_test.shape}")

    # 加载 unknown（用于训练后评估）
    print("加载 unknown 图片 ...")
    x_unknown, _ = load_unknown_images(UNKNOWN_DIR, IMAGE_SIZE)
    print(f"x_unknown: {x_unknown.shape}")

    # ── 2. 构建模型 ──
    print("\n构建模型 ...")
    input_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0], NUM_CHANNELS)

    base_model = build_base_model(input_shape, NUM_CLASSES)
    base_model.summary()

    model = CenterLossModel(
        base_model=base_model,
        input_shape=input_shape,
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        center_loss_weight=CENTER_LOSS_WEIGHT,
        center_lr=CENTER_LR,
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

    # ── 3. 初始化质心 ──
    print("\n初始化质心 ...")
    embedding_extractor = build_embedding_extractor(base_model, input_shape)
    init_emb = embedding_extractor.predict(x_train, verbose=0)

    init_centroids = np.zeros((NUM_CLASSES, EMBEDDING_DIM), dtype=np.float32)
    for class_name in CLASS_NAMES:
        idx = CLASS_TO_INDEX[class_name]
        mask = y_train == idx
        init_centroids[idx] = np.mean(init_emb[mask], axis=0)

    model.centroids.assign(init_centroids)
    print("质心已初始化")

    # ── 4. 训练 ──
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )

    print("\n开始训练 ...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[early_stopping],
    )

    # ── 5. 评估分类准确率 ──
    print("\n在测试集上评估 ...")
    test_results = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    print(f"test_loss:        {test_results['loss']:.4f}")
    print(f"test_ce_loss:     {test_results['ce_loss']:.4f}")
    print(f"test_center_loss: {test_results['center_loss']:.4f}")
    print(f"test_accuracy:    {test_results['accuracy']:.4f}")

    # 分类报告
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print("\n测试集分类报告:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    print("测试集混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    # ── 6. 保存模型 ──
    print("\n========== 保存模型 ==========")

    # 保存 base_model（纯 Sequential，eval 脚本可直接加载）
    model_path = OUTPUT_DIR / "gray48_cnn_center.keras"
    base_model.save(model_path)
    print(f"已保存模型: {model_path}")

    # label_map
    label_map_path = OUTPUT_DIR / "label_map.csv"
    with open(label_map_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "label"])
        for class_name, label in CLASS_TO_INDEX.items():
            writer.writerow([class_name, label])
    print(f"已保存标签映射: {label_map_path}")

    # 训练历史
    history_path = OUTPUT_DIR / "history.csv"
    save_history_csv(history, history_path)
    print(f"已保存训练历史: {history_path}")

    # 训练得到的质心
    centroids_path = OUTPUT_DIR / "centroids_trained.csv"
    trained_centroids = model.centroids.numpy()
    with open(centroids_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        header = ["class_name"] + [f"dim_{d}" for d in range(EMBEDDING_DIM)]
        writer.writerow(header)
        for class_name in CLASS_NAMES:
            idx = CLASS_TO_INDEX[class_name]
            row = [class_name] + [float(v) for v in trained_centroids[idx]]
            writer.writerow(row)
    print(f"已保存训练质心: {centroids_path}")

    # ── 7. Embedding 质量评估 ──
    print("\n重新构建 embedding 提取器（从保存的模型加载）...")
    saved_model = keras.models.load_model(model_path)
    embedding_model = build_embedding_extractor(saved_model, input_shape)

    eval_results = eval_embedding_quality(
        embedding_model, x_train, y_train, x_test, y_test, x_unknown
    )

    # ── 8. 与之前的结果对比 ──
    print("\n========== 与原模型对比 ==========")
    print(f"{'指标':<30} {'原模型(7_2)':>14} {'Center Loss':>14}")
    print(f"{'已知平均距离':<28} {'6.1049':>14} {eval_results['test_mean_dist']:>14.4f}")
    print(f"{'陌生人平均距离':<28} {'8.9186':>14} {eval_results['unknown_mean_dist']:>14.4f}")
    print(f"{'Nearest-centroid准确率':<28} {'0.6364':>14} {eval_results['nearest_acc']:>14.4f}")
    print(f"{'Embedding距离 F1':<28} {'0.5373':>14} {eval_results['best_f1']:>14.4f}")

    print(f"\n完整评估请运行:")
    print(f"  修改 9_eval_embedding_distance.py 中 MODEL_PATH 指向:")
    print(f"  {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Center Loss 训练")
    parser.add_argument("--center-weight", type=float, default=CENTER_LOSS_WEIGHT,
                        help=f"Center loss 权重 λ (默认: {CENTER_LOSS_WEIGHT})")
    parser.add_argument("--center-lr", type=float, default=CENTER_LR,
                        help=f"质心更新速率 (默认: {CENTER_LR})")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="输出目录后缀，如 '_lam05' → models/gray48_cnn_center_lam05")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 覆盖超参数
    CENTER_LOSS_WEIGHT = args.center_weight
    CENTER_LR = args.center_lr
    if args.output_suffix:
        OUTPUT_DIR = PROJECT_ROOT / "models" / f"gray48_cnn_center{args.output_suffix}"
    main()
