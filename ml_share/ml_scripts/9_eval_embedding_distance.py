from pathlib import Path
import csv
from collections import Counter
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── 路径和常量 ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = PROJECT_ROOT / "data" / "metadata" / "image_manifest_split.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gray48_cnn_aug" / "gray48_cnn_aug.keras"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "gray48_cnn_aug" / "embedding_eval"

# 支持命令行参数覆盖
import argparse
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model", type=str, default=None)
_parser.add_argument("--output-dir", type=str, default=None)
_args, _ = _parser.parse_known_args()

MODEL_PATH = Path(_args.model) if _args.model else DEFAULT_MODEL_PATH
OUTPUT_DIR = Path(_args.output_dir) if _args.output_dir else DEFAULT_OUTPUT_DIR

# 图片数据可能在两个位置：ml_share/data/data/ 或 face/data/
# 优先使用 ml_share/data/data/，不存在则回退到 face/data/
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
CLASS_NAMES = ["person_a", "person_b", "person_c"]
CLASS_TO_INDEX = {"person_a": 0, "person_b": 1, "person_c": 2}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


# ── 数据加载工具函数 ────────────────────────────────────────

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


def load_one_image_gray(img_path, image_size):
    img = Image.open(img_path)
    img = img.convert("L")
    img = img.resize(image_size)

    arr = np.asarray(img, dtype=np.float32)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=-1)

    return arr


def resolve_image_path(filepath_in_manifest):
    """将 manifest 里的相对路径映射到实际的图片路径"""
    # manifest 中格式: data/data/person_a/xxx.png
    # 提取 person_a/xxx.png 部分
    parts = Path(filepath_in_manifest).parts
    # 找到 person_a/person_b/person_c 所在位置
    for i, part in enumerate(parts):
        if part in {"person_a", "person_b", "person_c", "unknown"}:
            relative = Path(*parts[i:])
            return DATA_DIR / relative

    # 回退：直接拼
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


# ── Embedding 提取 ──────────────────────────────────────────

def find_dense32_layer(model):
    """在模型中找到 Dense(32) 层（倒数第二层）"""
    for i, layer in enumerate(model.layers):
        if isinstance(layer, layers.Dense) and layer.units == 32:
            return i, layer

    raise ValueError("找不到 Dense(32) 层，请检查模型结构")


def build_embedding_model(model, input_shape):
    layer_idx, target_layer = find_dense32_layer(model)
    print(f"找到 Dense(32) 层: index={layer_idx}, name='{target_layer.name}'")

    # Keras 3 嵌套 Sequential 无法直接用 model.input
    # 改用 Functional API 手动构建子模型
    inp = keras.Input(shape=input_shape)
    x = inp
    for layer in model.layers:
        x = layer(x)
        if layer.name == target_layer.name:
            break

    embedding_model = keras.Model(inputs=inp, outputs=x)

    return embedding_model


# ── 质心计算 ────────────────────────────────────────────────

def compute_centroids(embeddings, labels):
    """计算每个类别的质心和统计信息"""
    centroids = {}
    stats = {}

    for class_name in CLASS_NAMES:
        class_idx = CLASS_TO_INDEX[class_name]
        mask = labels == class_idx
        class_emb = embeddings[mask]

        centroid = np.mean(class_emb, axis=0)
        centroids[class_name] = centroid

        # 每个样本到质心的距离
        distances = np.linalg.norm(class_emb - centroid, axis=1)

        stats[class_name] = {
            "count": int(np.sum(mask)),
            "mean_dist": float(np.mean(distances)),
            "std_dist": float(np.std(distances)),
            "max_dist": float(np.max(distances)),
            "median_dist": float(np.median(distances)),
        }

    return centroids, stats


def print_centroid_stats(stats):
    print("\n========== 质心诊断 ==========")
    print(f"{'类别':<12} {'样本数':>6} {'平均距离':>10} {'标准差':>10} {'最大距离':>10} {'中位数':>10}")

    for class_name in CLASS_NAMES:
        s = stats[class_name]
        print(
            f"{class_name:<12} {s['count']:>6} "
            f"{s['mean_dist']:>10.4f} {s['std_dist']:>10.4f} "
            f"{s['max_dist']:>10.4f} {s['median_dist']:>10.4f}"
        )


# ── 距离计算 ────────────────────────────────────────────────

def compute_distances_to_centroids(embeddings, centroids):
    """计算每个 embedding 到所有质心的欧氏距离

    返回:
        distances: (N, 3) 数组，每行是到 3 个质心的距离
        min_distances: (N,) 最小距离
        nearest_classes: (N,) 最近质心对应的类别索引
    """
    centroid_matrix = np.array([centroids[cn] for cn in CLASS_NAMES])  # (3, 32)

    # (N, 1, 32) - (1, 3, 32) → (N, 3, 32) → norm → (N, 3)
    diff = embeddings[:, np.newaxis, :] - centroid_matrix[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)

    min_distances = np.min(distances, axis=1)
    nearest_classes = np.argmin(distances, axis=1)

    return distances, min_distances, nearest_classes


# ── 阈值扫描和评估 ──────────────────────────────────────────

def threshold_sweep(known_scores, unknown_scores, num_steps=200, higher_is_known=True):
    """扫描阈值，计算每个阈值下的 precision/recall/F1

    known_scores: 已知人脸的分数（应该被接受）
    unknown_scores: 陌生人的分数（应该被拒绝）
    higher_is_known: True 表示分数越高越可能是已知（softmax），
                     False 表示分数越低越可能是已知（距离）
    """
    all_scores = np.concatenate([known_scores, unknown_scores])
    # 真实标签：1=已知, 0=未知
    true_labels = np.concatenate([
        np.ones(len(known_scores), dtype=int),
        np.zeros(len(unknown_scores), dtype=int),
    ])

    lo = float(np.min(all_scores))
    hi = float(np.max(all_scores))
    thresholds = np.linspace(lo, hi, num_steps)

    results = []

    for thresh in thresholds:
        if higher_is_known:
            # softmax: score >= threshold → 接受
            predictions = (all_scores >= thresh).astype(int)
        else:
            # 距离: score <= threshold → 接受
            predictions = (all_scores <= thresh).astype(int)

        tp = int(np.sum((predictions == 1) & (true_labels == 1)))
        fp = int(np.sum((predictions == 1) & (true_labels == 0)))
        tn = int(np.sum((predictions == 0) & (true_labels == 0)))
        fn = int(np.sum((predictions == 0) & (true_labels == 1)))

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "threshold": float(thresh),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        })

    return results


def find_best_threshold(sweep_results):
    """找到 F1 最高的阈值"""
    best = max(sweep_results, key=lambda r: r["f1"])
    return best


def two_stage_sweep(known_softmax_probs, known_min_dists,
                    unknown_softmax_probs, unknown_min_dists,
                    softmax_steps=30, dist_steps=30):
    """二维阈值扫描：softmax 阈值 × 距离阈值

    放行条件：softmax_max_prob >= softmax_thresh AND min_dist <= dist_thresh
    """
    softmax_lo = float(min(np.min(known_softmax_probs), np.min(unknown_softmax_probs)))
    softmax_hi = float(max(np.max(known_softmax_probs), np.max(unknown_softmax_probs)))
    dist_lo = float(min(np.min(known_min_dists), np.min(unknown_min_dists)))
    dist_hi = float(max(np.max(known_min_dists), np.max(unknown_min_dists)))

    softmax_thresholds = np.linspace(softmax_lo, softmax_hi, softmax_steps)
    dist_thresholds = np.linspace(dist_lo, dist_hi, dist_steps)

    n_known = len(known_softmax_probs)
    n_unknown = len(unknown_softmax_probs)

    best = {"f1": -1}
    results = []

    for s_thresh in softmax_thresholds:
        for d_thresh in dist_thresholds:
            # 已知人脸：两个条件都满足 → 接受
            known_accepted = (
                (known_softmax_probs >= s_thresh) &
                (known_min_dists <= d_thresh)
            )
            # 陌生人：两个条件都满足 → 错误接受
            unknown_accepted = (
                (unknown_softmax_probs >= s_thresh) &
                (unknown_min_dists <= d_thresh)
            )

            tp = int(np.sum(known_accepted))
            fn = n_known - tp
            fp = int(np.sum(unknown_accepted))
            tn = n_unknown - fp

            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            row = {
                "softmax_threshold": float(s_thresh),
                "dist_threshold": float(d_thresh),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }
            results.append(row)

            if f1 > best["f1"]:
                best = row

    return results, best


# ── 输出函数 ────────────────────────────────────────────────

def save_predictions_csv(output_path, test_rows, unknown_filenames,
                         test_distances, test_min_dist, test_nearest,
                         unknown_distances, unknown_min_dist, unknown_nearest,
                         test_softmax_probs, unknown_softmax_probs,
                         dist_threshold, softmax_threshold):
    """保存每张图的详细预测结果"""
    rows_out = []

    # 已知人脸（test set）
    for i, row in enumerate(test_rows):
        rows_out.append({
            "filename": row["filename"],
            "ground_truth": row["subject_id"],
            "softmax_pred": CLASS_NAMES[int(np.argmax(test_softmax_probs[i]))],
            "softmax_max_prob": float(np.max(test_softmax_probs[i])),
            "nearest_centroid": CLASS_NAMES[test_nearest[i]],
            "min_distance": float(test_min_dist[i]),
            "dist_to_a": float(test_distances[i, 0]),
            "dist_to_b": float(test_distances[i, 1]),
            "dist_to_c": float(test_distances[i, 2]),
            "embedding_accepted": test_min_dist[i] <= dist_threshold,
            "softmax_accepted": float(np.max(test_softmax_probs[i])) >= softmax_threshold,
        })

    # 陌生人
    for i, fname in enumerate(unknown_filenames):
        rows_out.append({
            "filename": fname,
            "ground_truth": "unknown",
            "softmax_pred": CLASS_NAMES[int(np.argmax(unknown_softmax_probs[i]))],
            "softmax_max_prob": float(np.max(unknown_softmax_probs[i])),
            "nearest_centroid": CLASS_NAMES[unknown_nearest[i]],
            "min_distance": float(unknown_min_dist[i]),
            "dist_to_a": float(unknown_distances[i, 0]),
            "dist_to_b": float(unknown_distances[i, 1]),
            "dist_to_c": float(unknown_distances[i, 2]),
            "embedding_accepted": unknown_min_dist[i] <= dist_threshold,
            "softmax_accepted": float(np.max(unknown_softmax_probs[i])) >= softmax_threshold,
        })

    fieldnames = [
        "filename", "ground_truth",
        "softmax_pred", "softmax_max_prob",
        "nearest_centroid", "min_distance",
        "dist_to_a", "dist_to_b", "dist_to_c",
        "embedding_accepted", "softmax_accepted",
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"已保存预测结果: {output_path}")


def save_threshold_sweep_csv(output_path, embedding_sweep, softmax_sweep):
    """保存阈值扫描数据"""
    rows_out = []

    for r in embedding_sweep:
        row = dict(r)
        row["method"] = "embedding_distance"
        rows_out.append(row)

    for r in softmax_sweep:
        row = dict(r)
        row["method"] = "softmax_confidence"
        rows_out.append(row)

    fieldnames = ["method", "threshold", "precision", "recall", "f1",
                  "accuracy", "tp", "fp", "tn", "fn"]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"已保存阈值扫描: {output_path}")


def save_centroids_csv(output_path, centroids, stats):
    """保存质心向量"""
    rows_out = []

    for class_name in CLASS_NAMES:
        row = {"class_name": class_name}

        for d in range(len(centroids[class_name])):
            row[f"dim_{d}"] = float(centroids[class_name][d])

        row["mean_radius"] = stats[class_name]["mean_dist"]
        row["max_radius"] = stats[class_name]["max_dist"]
        rows_out.append(row)

    fieldnames = ["class_name"] + [f"dim_{d}" for d in range(32)] + ["mean_radius", "max_radius"]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"已保存质心: {output_path}")


def export_c_header(output_path, centroids, threshold):
    """生成 ESP32 可用的 C header"""
    lines = []
    lines.append("/* 自动生成 — 请勿手动修改 */")
    lines.append("/* 由 9_eval_embedding_distance.py 生成 */")
    lines.append("")
    lines.append("#ifndef CENTROIDS_CONFIG_H")
    lines.append("#define CENTROIDS_CONFIG_H")
    lines.append("")
    lines.append(f"#define EMBEDDING_DIM 32")
    lines.append(f"#define NUM_CLASSES {len(CLASS_NAMES)}")
    lines.append(f"#define DISTANCE_THRESHOLD {threshold:.6f}f")
    lines.append("")
    lines.append("/*")
    lines.append(" * 拒识逻辑：")
    lines.append(" *   1. 从 Dense(32) 层提取 32 维 embedding")
    lines.append(" *   2. 计算到每个质心的欧氏距离")
    lines.append(" *   3. 若最小距离 > DISTANCE_THRESHOLD → 拒绝（unknown）")
    lines.append(" */")
    lines.append("")
    lines.append("static const float centroids[NUM_CLASSES][EMBEDDING_DIM] = {")

    for i, class_name in enumerate(CLASS_NAMES):
        vec = centroids[class_name]
        vals = ", ".join(f"{v:.6f}f" for v in vec)
        comma = "," if i < len(CLASS_NAMES) - 1 else ""
        lines.append(f"    /* {class_name} */ {{{vals}}}{comma}")

    lines.append("};")
    lines.append("")
    lines.append(f'static const char* class_names[NUM_CLASSES] = {{')

    for i, class_name in enumerate(CLASS_NAMES):
        comma = "," if i < len(CLASS_NAMES) - 1 else ""
        lines.append(f'    "{class_name}"{comma}')

    lines.append("};")
    lines.append("")
    lines.append("#endif /* CENTROIDS_CONFIG_H */")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"已生成 C header: {output_path}")


# ── 主函数 ──────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"数据目录: {DATA_DIR}")
    print(f"Unknown 目录: {UNKNOWN_DIR}")

    # ── 1. 加载数据 ──
    print("\n读取 manifest ...")
    rows = read_manifest(MANIFEST_CSV)
    train_rows = filter_rows(rows, "train")
    test_rows = filter_rows(rows, "test")

    print(f"train: {len(train_rows)} 张, test: {len(test_rows)} 张")

    print("加载训练集图片 ...")
    x_train, y_train = build_numpy_dataset(train_rows, IMAGE_SIZE)

    print("加载测试集图片 ...")
    x_test, y_test = build_numpy_dataset(test_rows, IMAGE_SIZE)

    print("加载 unknown 图片 ...")
    x_unknown, unknown_filenames = load_unknown_images(UNKNOWN_DIR, IMAGE_SIZE)
    print(f"unknown: {len(unknown_filenames)} 张")

    # ── 2. 加载模型 + 构建 embedding 子模型 ──
    print("\n加载模型 ...")
    model = keras.models.load_model(MODEL_PATH)
    model.summary()

    input_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0], NUM_CHANNELS)
    embedding_model = build_embedding_model(model, input_shape)

    # ── 3. 提取 embedding ──
    print("\n提取 embedding ...")
    emb_train = embedding_model.predict(x_train, verbose=0)
    emb_test = embedding_model.predict(x_test, verbose=0)
    emb_unknown = embedding_model.predict(x_unknown, verbose=0)

    print(f"embedding shape: train={emb_train.shape}, test={emb_test.shape}, unknown={emb_unknown.shape}")

    # 同时获取 softmax 输出用于对比
    softmax_test = model.predict(x_test, verbose=0)
    softmax_unknown = model.predict(x_unknown, verbose=0)

    # ── 4. 计算质心 ──
    centroids, centroid_stats = compute_centroids(emb_train, y_train)
    print_centroid_stats(centroid_stats)

    # ── 5. 计算距离 ──
    print("\n计算测试集距离 ...")
    test_distances, test_min_dist, test_nearest = compute_distances_to_centroids(emb_test, centroids)

    print("计算 unknown 距离 ...")
    unknown_distances, unknown_min_dist, unknown_nearest = compute_distances_to_centroids(emb_unknown, centroids)

    # ── 6. 距离分布对比 ──
    print("\n========== 距离分布对比 ==========")
    print(f"{'指标':<12} {'已知(test)':>12} {'陌生人(unknown)':>16}")
    print(f"{'均值':<12} {np.mean(test_min_dist):>12.4f} {np.mean(unknown_min_dist):>16.4f}")
    print(f"{'中位数':<12} {np.median(test_min_dist):>12.4f} {np.median(unknown_min_dist):>16.4f}")
    print(f"{'最小值':<12} {np.min(test_min_dist):>12.4f} {np.min(unknown_min_dist):>16.4f}")
    print(f"{'最大值':<12} {np.max(test_min_dist):>12.4f} {np.max(unknown_min_dist):>16.4f}")
    print(f"{'标准差':<12} {np.std(test_min_dist):>12.4f} {np.std(unknown_min_dist):>16.4f}")

    # ── 7. Nearest-centroid 分类准确率（sanity check） ──
    nearest_correct = np.sum(test_nearest == y_test)
    nearest_acc = nearest_correct / len(y_test)
    print(f"\nNearest-centroid 分类准确率 (test set): {nearest_acc:.4f} ({nearest_correct}/{len(y_test)})")

    # ── 8. 阈值扫描 ──
    print("\n========== 阈值扫描 ==========")

    # Embedding 距离：分数越低 = 越可能是已知
    emb_sweep = threshold_sweep(
        test_min_dist, unknown_min_dist,
        num_steps=200, higher_is_known=False
    )
    best_emb = find_best_threshold(emb_sweep)

    # Softmax 置信度：分数越高 = 越可能是已知
    test_max_prob = np.max(softmax_test, axis=1)
    unknown_max_prob = np.max(softmax_unknown, axis=1)
    softmax_sweep = threshold_sweep(
        test_max_prob, unknown_max_prob,
        num_steps=200, higher_is_known=True
    )
    best_softmax = find_best_threshold(softmax_sweep)

    # ── 9. 对比结果 ──
    print("\n========== 最佳阈值对比 ==========")
    print(f"{'方法':<22} {'阈值':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print(
        f"{'Embedding距离':<20} {best_emb['threshold']:>10.4f} "
        f"{best_emb['precision']:>10.4f} {best_emb['recall']:>10.4f} "
        f"{best_emb['f1']:>10.4f} {best_emb['accuracy']:>10.4f}"
    )
    print(
        f"{'Softmax置信度':<20} {best_softmax['threshold']:>10.4f} "
        f"{best_softmax['precision']:>10.4f} {best_softmax['recall']:>10.4f} "
        f"{best_softmax['f1']:>10.4f} {best_softmax['accuracy']:>10.4f}"
    )

    print(f"\n  Embedding: 已知正确接受 {best_emb['tp']}/{best_emb['tp']+best_emb['fn']}，"
          f"陌生人正确拒绝 {best_emb['tn']}/{best_emb['tn']+best_emb['fp']}")
    print(f"  Softmax:   已知正确接受 {best_softmax['tp']}/{best_softmax['tp']+best_softmax['fn']}，"
          f"陌生人正确拒绝 {best_softmax['tn']}/{best_softmax['tn']+best_softmax['fp']}")

    # ── 10. 两阶段方案评估 ──
    print("\n========== 两阶段方案（Softmax + Embedding 距离） ==========")
    print("放行条件: softmax_max_prob >= S阈值 AND min_distance <= D阈值")
    print("扫描中 ...")

    two_stage_results, best_two_stage = two_stage_sweep(
        test_max_prob, test_min_dist,
        unknown_max_prob, unknown_min_dist,
        softmax_steps=50, dist_steps=50,
    )

    print(f"\n最佳两阶段阈值:")
    print(f"  Softmax 阈值: {best_two_stage['softmax_threshold']:.4f}")
    print(f"  距离阈值:     {best_two_stage['dist_threshold']:.4f}")
    print(f"  Precision:    {best_two_stage['precision']:.4f}")
    print(f"  Recall:       {best_two_stage['recall']:.4f}")
    print(f"  F1:           {best_two_stage['f1']:.4f}")
    print(f"  Accuracy:     {best_two_stage['accuracy']:.4f}")
    print(f"  已知正确接受: {best_two_stage['tp']}/{best_two_stage['tp']+best_two_stage['fn']}")
    print(f"  陌生人正确拒绝: {best_two_stage['tn']}/{best_two_stage['tn']+best_two_stage['fp']}")

    # ── 11. 三种方案总对比 ──
    print("\n========== 三种方案总对比 ==========")
    print(f"{'方法':<24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10} {'已知接受':>10} {'陌生拒绝':>10}")
    print(
        f"{'Softmax单独':<22} "
        f"{best_softmax['precision']:>10.4f} {best_softmax['recall']:>10.4f} "
        f"{best_softmax['f1']:>10.4f} {best_softmax['accuracy']:>10.4f} "
        f"{best_softmax['tp']:>5}/{best_softmax['tp']+best_softmax['fn']:<4} "
        f"{best_softmax['tn']:>5}/{best_softmax['tn']+best_softmax['fp']:<4}"
    )
    print(
        f"{'Embedding单独':<22} "
        f"{best_emb['precision']:>10.4f} {best_emb['recall']:>10.4f} "
        f"{best_emb['f1']:>10.4f} {best_emb['accuracy']:>10.4f} "
        f"{best_emb['tp']:>5}/{best_emb['tp']+best_emb['fn']:<4} "
        f"{best_emb['tn']:>5}/{best_emb['tn']+best_emb['fp']:<4}"
    )
    print(
        f"{'两阶段(推荐)':<22} "
        f"{best_two_stage['precision']:>10.4f} {best_two_stage['recall']:>10.4f} "
        f"{best_two_stage['f1']:>10.4f} {best_two_stage['accuracy']:>10.4f} "
        f"{best_two_stage['tp']:>5}/{best_two_stage['tp']+best_two_stage['fn']:<4} "
        f"{best_two_stage['tn']:>5}/{best_two_stage['tn']+best_two_stage['fp']:<4}"
    )

    # ── 12. 保存输出 ──
    print("\n========== 保存结果 ==========")

    save_predictions_csv(
        OUTPUT_DIR / "embedding_predictions.csv",
        test_rows, unknown_filenames,
        test_distances, test_min_dist, test_nearest,
        unknown_distances, unknown_min_dist, unknown_nearest,
        softmax_test, softmax_unknown,
        best_two_stage["dist_threshold"], best_two_stage["softmax_threshold"],
    )

    save_threshold_sweep_csv(
        OUTPUT_DIR / "threshold_sweep.csv",
        emb_sweep, softmax_sweep,
    )

    # 保存两阶段扫描结果
    two_stage_csv = OUTPUT_DIR / "two_stage_sweep.csv"
    fieldnames_2s = ["softmax_threshold", "dist_threshold", "precision", "recall",
                     "f1", "accuracy", "tp", "fp", "tn", "fn"]
    with open(two_stage_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_2s)
        writer.writeheader()
        writer.writerows(two_stage_results)
    print(f"已保存两阶段扫描: {two_stage_csv}")

    save_centroids_csv(
        OUTPUT_DIR / "centroids.csv",
        centroids, centroid_stats,
    )

    export_c_header(
        OUTPUT_DIR / "centroids_config.h",
        centroids, best_two_stage["dist_threshold"],
    )

    # ── 13. ESP32 部署说明 ──
    print("\n========== ESP32 部署说明（两阶段方案） ==========")
    print(f"  推理流程:")
    print(f"    1. 跑完整模型，获取 softmax 概率和 Dense(32) embedding")
    print(f"    2. 第一关: softmax_max_prob >= {best_two_stage['softmax_threshold']:.4f} ?")
    print(f"    3. 第二关: min_distance <= {best_two_stage['dist_threshold']:.4f} ?")
    print(f"    4. 两关都通过 → LED 亮（放行），否则 → LED 灭（拒绝）")
    print(f"")
    print(f"  内存开销:")
    print(f"    质心常量: 3 × 32 × 4 = 384 bytes")
    print(f"    两个阈值: 8 bytes")
    print(f"  计算开销: 3 × 32 次减法 + 3 × 32 次乘法 + 3 次 sqrt")


if __name__ == "__main__":
    main()
