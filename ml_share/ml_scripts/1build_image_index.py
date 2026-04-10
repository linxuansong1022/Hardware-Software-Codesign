from pathlib import Path
import csv
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "data"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_CSV = METADATA_DIR / "image_index.csv"
SUBJECT_FOLDERS = ["person_a", "person_b", "person_c", "unknown"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_timestamp_from_filename(filename):
    # 去掉后缀，只保留文件名主体
    stem = Path(filename).stem
    parts = stem.split("_") # 按_分开

    # 格式：image_YYYYMMDD_HHMMSS_microseconds
    if len(parts) != 4:
        return ""

    prefix = parts[0]
    date_part = parts[1]
    time_part = parts[2]
    micro_part = parts[3]

    if prefix != "image":
        return ""

    try:
        dt = datetime.strptime(
            date_part + time_part + micro_part,
            "%Y%m%d%H%M%S%f"
        )
        result = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        return result
    except ValueError:
        return ""


def main():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject_id in SUBJECT_FOLDERS:
        folder_path = RAW_DATA_DIR / subject_id

        if not folder_path.exists():
            print(f"跳过：文件夹不存在 -> {folder_path}")
            continue

        image_files = []

        #只保留图片文件
        for p in folder_path.iterdir():
            if p.is_file():
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    image_files.append(p)

        #按文件名排序
        image_files = sorted(image_files, key=lambda p: p.name)

        #给这个文件夹里的图片按顺序编号
        for idx, image_path in enumerate(image_files, start=1):
            row = {}

            row["subject_id"] = subject_id
            row["order_idx"] = idx
            row["filename"] = image_path.name
            row["filepath"] = str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            row["timestamp_from_name"] = parse_timestamp_from_filename(image_path.name)

            rows.append(row)

    # 写出 csv
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "subject_id",
            "order_idx",
            "filename",
            "filepath",
            "timestamp_from_name",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"已生成: {OUTPUT_CSV}")
    print(f"共写入 {len(rows)} 行")


if __name__ == "__main__":
    main()