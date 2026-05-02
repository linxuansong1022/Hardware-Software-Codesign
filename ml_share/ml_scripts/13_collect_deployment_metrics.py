from pathlib import Path
import argparse
import json
import math
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_TFLITE = PROJECT_ROOT / "models" / "gray48_cnn_center_w001_int8" / "gray48_cnn_center_int8.tflite"
DEFAULT_ESP32_BUILD = REPO_ROOT / "face" / "esp32" / "build"
DEFAULT_OUTPUT_MD = REPO_ROOT / "face" / "deployment_metrics.md"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "face" / "deployment_metrics.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect deployment-side size and build metrics for the ESP32 face model."
    )
    parser.add_argument("--tflite", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--esp32-build-dir", type=Path, default=DEFAULT_ESP32_BUILD)
    parser.add_argument(
        "--app-partition-size",
        type=lambda x: int(x, 0),
        default=0x100000,
        help="Smallest app partition size in bytes. Default matches the current build output: 0x100000.",
    )
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def file_size(path):
    if not path.exists():
        return None
    return path.stat().st_size


def human_size(num_bytes):
    if num_bytes is None:
        return "missing"
    if num_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    idx = min(int(math.log(num_bytes, 1024)), len(units) - 1)
    value = num_bytes / (1024 ** idx)
    return f"{value:.1f} {units[idx]}"


def load_project_description(build_dir):
    path = build_dir / "project_description.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_metrics(args):
    build_dir = args.esp32_build_dir
    project_description = load_project_description(build_dir)

    app_bin_name = project_description.get("app_bin", "face-recognition.bin")
    app_bin = build_dir / app_bin_name
    elf_name = project_description.get("app_elf", "face-recognition.elf")
    app_elf = build_dir / elf_name
    bootloader_bin = build_dir / "bootloader" / "bootloader.bin"
    partition_bin = build_dir / "partition_table" / "partition-table.bin"

    tflite_size = file_size(args.tflite)
    app_bin_size = file_size(app_bin)
    bootloader_size = file_size(bootloader_bin)
    partition_table_size = file_size(partition_bin)
    app_elf_size = file_size(app_elf)

    free_app_partition = (
        args.app_partition_size - app_bin_size
        if app_bin_size is not None else None
    )

    return {
        "paths": {
            "tflite": str(args.tflite),
            "esp32_build_dir": str(build_dir),
            "app_bin": str(app_bin),
            "app_elf": str(app_elf),
            "bootloader_bin": str(bootloader_bin),
            "partition_table_bin": str(partition_bin),
        },
        "project": {
            "name": project_description.get("project_name"),
            "target": project_description.get("target"),
            "idf_path": project_description.get("idf_path"),
            "git_revision": project_description.get("git_revision"),
        },
        "metrics": {
            "input_shape": "48x48x1 grayscale",
            "model_format": "INT8 TFLite",
            "tflite_model_bytes": tflite_size,
            "firmware_app_bin_bytes": app_bin_size,
            "firmware_elf_bytes": app_elf_size,
            "bootloader_bytes": bootloader_size,
            "partition_table_bytes": partition_table_size,
            "app_partition_bytes": args.app_partition_size,
            "free_app_partition_bytes": free_app_partition,
            "app_partition_used_percent": (
                100.0 * app_bin_size / args.app_partition_size
                if app_bin_size is not None and args.app_partition_size else None
            ),
        },
        "notes": [
            "Latency and FPS should be filled from ESP32 serial logs after flashing timing instrumentation.",
            "Tensor arena size is configured in face/esp32/main/inference.cpp.",
        ],
    }


def write_outputs(args, metrics):
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    m = metrics["metrics"]
    p = metrics["project"]
    lines = [
        "# Deployment Metrics",
        "",
        "## Build Context",
        "",
        f"- Project: {p.get('name') or 'unknown'}",
        f"- Target: {p.get('target') or 'unknown'}",
        f"- Model format: {m['model_format']}",
        f"- Input shape: {m['input_shape']}",
        "",
        "## Size Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| TFLite model size | {human_size(m['tflite_model_bytes'])} |",
        f"| Firmware app binary size | {human_size(m['firmware_app_bin_bytes'])} |",
        f"| Firmware ELF size | {human_size(m['firmware_elf_bytes'])} |",
        f"| Bootloader size | {human_size(m['bootloader_bytes'])} |",
        f"| Partition table size | {human_size(m['partition_table_bytes'])} |",
        f"| App partition size | {human_size(m['app_partition_bytes'])} |",
        f"| Free app partition | {human_size(m['free_app_partition_bytes'])} |",
    ]
    if m["app_partition_used_percent"] is not None:
        lines.append(f"| App partition used | {m['app_partition_used_percent']:.1f}% |")

    lines.extend([
        "",
        "## Runtime Metrics To Fill From Serial Logs",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Capture latency | TBD ms |",
        "| Preprocess latency | TBD ms |",
        "| Inference latency | TBD ms |",
        "| Total frame latency | TBD ms |",
        "| Approximate FPS | TBD |",
        "",
        "## Source Paths",
        "",
        f"- TFLite: `{metrics['paths']['tflite']}`",
        f"- App binary: `{metrics['paths']['app_bin']}`",
        f"- Build directory: `{metrics['paths']['esp32_build_dir']}`",
        "",
    ])

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    metrics = collect_metrics(args)
    write_outputs(args, metrics)
    print(f"Wrote JSON metrics: {args.output_json}")
    print(f"Wrote Markdown metrics: {args.output_md}")


if __name__ == "__main__":
    main()
