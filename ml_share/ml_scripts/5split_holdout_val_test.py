from pathlib import Path
import csv
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "metadata" / "image_manifest.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "metadata" / "image_manifest_split.csv"
KNOWN_SUBJECTS = {"person_a", "person_b", "person_c"}

def read_manifest():
    rows = []

    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row["order_idx"] = int(row["order_idx"])
            row["is_usable"] = int(row["is_usable"])
            rows.append(row)

    return rows


def split_holdout_rows(rows):
    # 对每个subject单独处理
    # train保持不变
    # holdout按order_idx排序
    # 前一半改成val,后一半改成test
    grouped = defaultdict(list)

    for row in rows:
        subject_id = row["subject_id"]
        grouped[subject_id].append(row)

    output_rows = []

    for subject_id, subject_rows in grouped.items():
        # 先按顺序编号排序，确保是时间顺序
        subject_rows = sorted(subject_rows, key=lambda r: r["order_idx"])

        train_rows = []
        holdout_rows = []

        for r in subject_rows:
            if r["split"] == "train":
                train_rows.append(r)
            elif r["split"] == "holdout":
                holdout_rows.append(r)

        # 对holdout再切分
        n_holdout = len(holdout_rows)
        n_val = n_holdout // 2   #前一半给val
        n_test = n_holdout - n_val

        for i, row in enumerate(holdout_rows):
            if i < n_val:
                row["split"] = "val"
            else:
                row["split"] = "test"

        for row in train_rows:
            output_rows.append(row)

        for row in holdout_rows:
            output_rows.append(row)

        print(f"{subject_id}: train={len(train_rows)}, val={n_val}, test={n_test}")

    #最后再整体排一下，输出更整齐
    output_rows = sorted(output_rows, key=lambda r: (r["subject_id"], r["order_idx"]))

    return output_rows


def write_manifest(rows):
    if not rows:
        raise ValueError("没有可写入的数据")

    fieldnames = list(rows[0].keys())

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = read_manifest()

    filtered_rows = []
    for r in rows:
        if r["subject_id"] in KNOWN_SUBJECTS:
            filtered_rows.append(r)

    split_rows = split_holdout_rows(filtered_rows)
    write_manifest(split_rows)

    total_train = 0
    total_val = 0
    total_test = 0

    for r in split_rows:
        if r["split"] == "train":
            total_train += 1
        elif r["split"] == "val":
            total_val += 1
        elif r["split"] == "test":
            total_test += 1

    print("\n========== 总结 ==========")
    print(f"总 train: {total_train}")
    print(f"总 val:   {total_val}")
    print(f"总 test:  {total_test}")
    print(f"已生成: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()