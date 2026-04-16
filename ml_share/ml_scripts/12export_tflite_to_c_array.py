from pathlib import Path


# =========================
# 1. 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TFLITE_PATH = PROJECT_ROOT / "models" / "gray48_cnn_center_w001_int8" / "gray48_cnn_center_int8.tflite"
OUTPUT_DIR = PROJECT_ROOT / "models" / "gray48_cnn_center_w001_int8" / "c_export"

ARRAY_NAME = "g_model_data"


# =========================
# 2. 工具函数
# =========================
def format_bytes_as_c_array(data: bytes, bytes_per_line: int = 12) -> str:
    lines = []
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i + bytes_per_line]
        line = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append("  " + line)
    return ",\n".join(lines)


# =========================
# 3. 主程序
# =========================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TFLITE_PATH.exists():
        raise FileNotFoundError(f"找不到 TFLite 文件: {TFLITE_PATH}")

    data = TFLITE_PATH.read_bytes()
    model_len = len(data)

    c_array_text = format_bytes_as_c_array(data)

    header_path = OUTPUT_DIR / "model_data.h"
    source_path = OUTPUT_DIR / "model_data.c"

    header_text = f"""#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#ifdef __cplusplus
extern "C" {{
#endif

extern const unsigned char {ARRAY_NAME}[];
extern const unsigned int {ARRAY_NAME}_len;

#ifdef __cplusplus
}}
#endif

#endif  // MODEL_DATA_H_
"""

    source_text = f"""#include "model_data.h"

const unsigned char {ARRAY_NAME}[] = {{
{c_array_text}
}};

const unsigned int {ARRAY_NAME}_len = {model_len};
"""

    header_path.write_text(header_text, encoding="utf-8")
    source_path.write_text(source_text, encoding="utf-8")

    print(f"已生成: {header_path}")
    print(f"已生成: {source_path}")
    print(f"模型字节长度: {model_len}")
    print(f"模型大小: {model_len / 1024:.2f} KB")


if __name__ == "__main__":
    main()