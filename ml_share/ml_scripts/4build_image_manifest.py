from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_INDEX_CSV = PROJECT_ROOT / "data" / "metadata" / "image_index.csv"
RANGE_MANIFEST_CSV = PROJECT_ROOT / "data" / "metadata" / "range_manifest.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "metadata" / "image_manifest.csv"


def read_image_index():
    # 返回一个列表，每个元素代表一张图片
    rows = []

    with open(IMAGE_INDEX_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row["order_idx"] = int(row["order_idx"])
            rows.append(row)

    return rows


def read_range_manifest():
    # 返回一个列表，每个元素代表一个范围段
    rows = []

    with open(RANGE_MANIFEST_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row["start_ord"] = int(row["start_ord"])
            row["end_ord"] = int(row["end_ord"])
            row["is_usable"] = int(row["is_usable"])
            rows.append(row)

    return rows


def find_matching_range(subject_id, order_idx, range_rows):
    # 给定一张图片的 subject_id和 order_idx,去 range_manifest里找到它属于哪一个范围段,找到后就返回那一行范围信息
    for r in range_rows:
        if r["subject_id"] != subject_id:
            continue

        if r["start_ord"] <= order_idx <= r["end_ord"]:
            return r

    return None


def convert_time_group_to_split(time_group):
    # 先做一个最简单的 split 映射
    # before_1729 -> train
    # after_1729 -> holdout
    # 这里先不急着把 holdout 再分成 val/test
    if time_group == "before_1729":
        return "train"
    elif time_group == "after_1729":
        return "holdout"
    else:
        return "unknown"


def main():
    image_rows = read_image_index()
    range_rows = read_range_manifest()

    output_rows = []

    for img in image_rows:
        subject_id = img["subject_id"]

        # 目前先只处理 person_a/ person_b/ person_c
        if subject_id not in {"person_a", "person_b", "person_c"}:
            continue

        order_idx = img["order_idx"]

        matched_range = find_matching_range(subject_id, order_idx, range_rows)

        if matched_range is None:
            raise ValueError(
                f"找不到匹配范围：subject_id={subject_id}, order_idx={order_idx}"
            )

        time_group = matched_range["time_group"]
        split = convert_time_group_to_split(time_group)

        row = {}
        row["subject_id"] = subject_id
        row["order_idx"] = order_idx
        row["filename"] = img["filename"]
        row["filepath"] = img["filepath"]
        row["timestamp_from_name"] = img["timestamp_from_name"]
        row["time_group"] = time_group
        row["split"] = split
        row["task_group"] = matched_range["task_group"]
        row["background"] = matched_range["background"]
        row["lighting"] = matched_range["lighting"]
        row["accessories"] = matched_range["accessories"]
        row["pose"] = matched_range["pose"]
        row["expression"] = matched_range["expression"]
        row["distance"] = matched_range["distance"]
        row["is_usable"] = matched_range["is_usable"]
        row["notes"] = matched_range["notes"]

        output_rows.append(row)

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "subject_id",
            "order_idx",
            "filename",
            "filepath",
            "timestamp_from_name",
            "time_group",
            "split",
            "task_group",
            "background",
            "lighting",
            "accessories",
            "pose",
            "expression",
            "distance",
            "is_usable",
            "notes",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"已生成: {OUTPUT_CSV}")
    print(f"共写入 {len(output_rows)} 行")


if __name__ == "__main__":
    main()