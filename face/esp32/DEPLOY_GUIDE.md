# Face Recognition ESP32 部署指南

## 概述

将训练好的人脸识别 CNN 模型部署到 XIAO ESP32S3 Sense 上。

整体流程：摄像头拍照 → 预处理（裁剪+缩放+灰度） → TFLite Micro 推理 → LED 反馈 + 串口输出

---

## 1. 项目结构

需要创建的文件（参考 `keywords/esp32/main/` 的结构）：

```
face/esp32/main/
├── CMakeLists.txt        # 构建配置
├── main.cpp              # 主循环：camera → preprocess → inference → LED → serial
├── camera.h / camera.cpp # 直接复制 camera/esp32/main/（不改）
├── preprocess.h / .cpp   # 新写：RGB565 → 48×48 灰度 → 归一化
├── inference.h / .cpp    # 改编自 keywords：输入改为图片，输出 3 类
├── model_data.h / .c     # Kirsi 生成好的（从 ml_share 复制）
├── test.h / test.cpp     # 改编自 keywords：用图片测试向量
└── idf_component.yml     # TFLite Micro 依赖
```

---

## 2. 逐步实现

### Step 1: 搭建项目框架

从 `keywords/esp32/` 复制整个目录结构，删掉 audio 相关文件。

需要复制的文件：
- `camera/esp32/main/camera.h` + `camera.cpp` → 直接用，不改
- `ml_share/models/gray48_cnn_center_w001_int8/c_export/model_data.c` + `model_data.h` → 模型文件
- `keywords/esp32/main/idf_component.yml` → TFLite Micro 依赖
- `keywords/esp32/main/CMakeLists.txt` → 改一下源文件列表

CMakeLists.txt 需要改成：
```cmake
idf_component_register(
    SRCS "main.cpp" "camera.cpp" "preprocess.cpp" "inference.cpp" "model_data.c" "test.cpp"
    INCLUDE_DIRS "."
)
```

### Step 2: preprocess.cpp — 图片预处理

这是**最关键的新代码**。需要把摄像头的 320×240 RGB565 帧转成模型需要的 48×48 灰度图。

#### 输入
- 320×240 RGB565 原始帧（`uint8_t image_buffer[320*240*2]`）

#### 处理步骤

```
320×240 RGB565
    ↓
1. 中心裁剪：取中间 240×240 的正方形区域
   （左右各裁掉 40 像素：x_offset = (320-240)/2 = 40）
    ↓
2. 缩放到 48×48：每个 5×5 的像素块取平均（240/48 = 5）
    ↓
3. RGB565 → 灰度：Y = (5*R + 9*G + 2*B) >> 4
    ↓
4. 归一化：pixel / 255.0
    ↓
输出：float face_features[48*48]
```

#### RGB565 格式说明

每个像素 2 字节（big-endian），排列：`RRRRR GGGGGG BBBBB`

```c
// 从 RGB565 提取 R/G/B（0-255 范围）
uint16_t pixel = (buf[i] << 8) | buf[i+1];  // big-endian
uint8_t r = (pixel >> 11) & 0x1F;  // 5-bit → 左移3位扩展到8位
uint8_t g = (pixel >> 5) & 0x3F;   // 6-bit → 左移2位扩展到8位
uint8_t b = pixel & 0x1F;          // 5-bit → 左移3位扩展到8位
r = (r << 3) | (r >> 2);
g = (g << 2) | (g >> 4);
b = (b << 3) | (b >> 2);
```

#### 区域平均下采样

对每个 5×5 块，累加所有像素的灰度值，再除以 25：

```c
// 伪代码
for out_y in 0..47:
    for out_x in 0..47:
        sum = 0
        for dy in 0..4:
            for dx in 0..4:
                src_x = crop_x_offset + out_x * 5 + dx
                src_y = crop_y_offset + out_y * 5 + dy
                pixel = read_rgb565(src_x, src_y)
                gray = rgb_to_gray(pixel)
                sum += gray
        face_features[out_y * 48 + out_x] = (sum / 25.0f) / 255.0f
```

#### 接口设计

```c
// preprocess.h
#pragma once
#include <stdint.h>

#define FACE_W 48
#define FACE_H 48

// 从 320×240 RGB565 帧提取 48×48 灰度特征
void preprocess_frame(const uint8_t *rgb565_frame, float *face_features);
```

#### 对齐验证（重要！）

Python 端和 ESP32 端的预处理**必须产生相同的结果**。验证方法：
1. 用 Python 读一张图，走 PIL resize + grayscale 流程，保存特征向量
2. 同一张图的 RGB565 数据送进 ESP32 的 `preprocess_frame()`
3. 对比两边的输出，误差应该 < 0.01

注意：PIL 的 resize 用的是双线性插值，ESP32 用的是区域平均。两者不会完全一致，
但在低分辨率下差异很小。如果误差太大，可以在 Python 端也改用区域平均来训练。

### Step 3: inference.cpp — 推理引擎

改编自 `keywords/esp32/main/inference.cpp`，主要改动：

#### 3.1 模型头文件

```c
// 改 #include "model.h" 为：
#include "model_data.h"

// 加载模型时：
model = tflite::GetModel(g_model_data);  // 对应 model_data.h 里的变量名
```

#### 3.2 常量定义

在 `inference.h` 或单独的头文件里定义：

```c
#define FACE_INPUT_SIZE  (48 * 48 * 1)  // 输入大小
#define NUM_CLASSES      3               // person_a, person_b, person_c
#define CONFIDENCE_THRESHOLD 0.90f       // 可调阈值
```

#### 3.3 Tensor Arena 大小

keywords 项目用了 30KB，我们的模型更大一些（30KB 的 tflite），arena 建议设 **50-80KB**：

```c
#define TENSOR_ARENA_SIZE (60 * 1024)  // 60KB，如果不够再加
```

ESP32S3 有 512KB SRAM，60KB 完全没问题。

#### 3.4 Op Resolver

我们的模型用到的算子（和 keywords 项目基本一样）：

```c
static tflite::MicroMutableOpResolver<5> micro_op_resolver;
micro_op_resolver.AddConv2D();
micro_op_resolver.AddMaxPool2D();
micro_op_resolver.AddFullyConnected();
micro_op_resolver.AddSoftmax();
micro_op_resolver.AddReshape();
```

注意：数据增强层（RandomFlip 等）在推理时**不生效**，TFLite 导出时已经被移除了。

#### 3.5 量化/反量化

和 keywords 完全一样的逻辑，不需要改：

```c
// 输入：float → int8（量化）
int8_t* inference_put_features(const float *features) {
    for (int i = 0; i < FACE_INPUT_SIZE; i++) {
        float val = roundf(features[i] / input->params.scale) + input->params.zero_point;
        val = fmaxf(-128.0f, fminf(127.0f, val));
        input->data.int8[i] = (int8_t)val;
    }
    return input->data.int8;
}

// 输出：int8 → float（反量化）
bool inference_predict(float *prediction) {
    if (interpreter->Invoke() != kTfLiteOk) return false;
    for (int i = 0; i < NUM_CLASSES; i++) {
        prediction[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    }
    return true;
}
```

### Step 4: main.cpp — 主循环

整合所有组件：

```
setup():
    1. camera_init()         — 初始化摄像头
    2. inference_init()      — 加载 TFLite 模型
    3. test_pipeline()       — 运行对齐测试
    4. USB serial init       — 初始化串口
    5. LED init              — 初始化 GPIO

loop():
    1. camera_capture_frame(image_buffer)    — 拍一帧
    2. preprocess_frame(image_buffer, features) — 预处理
    3. inference_put_features(features)       — 量化+送入模型
    4. inference_predict(prediction)          — 推理
    5. 判断结果 + LED 控制 + 串口输出
```

#### LED 逻辑

```c
// 找到最大概率和对应类别
int best_class = 0;
float max_score = prediction[0];
for (int i = 1; i < NUM_CLASSES; i++) {
    if (prediction[i] > max_score) {
        max_score = prediction[i];
        best_class = i;
    }
}

// 判断：置信度够高 → 开门
if (max_score >= CONFIDENCE_THRESHOLD) {
    gpio_set_level(LED_PIN, 0);  // LED 亮（active-low）= 放行
} else {
    gpio_set_level(LED_PIN, 1);  // LED 灭 = 拒绝
}
```

#### 串口输出（给 Python display.py 用）

```c
// 先发帧数据（给 display 显示画面）
usb_serial_jtag_write_bytes("===FRAME===\n", ...);
// 分块发送 image_buffer（和 camera 项目一样）

// 再发推理结果
usb_serial_jtag_write_bytes("===RESULT===\n", ...);
// 发送 1 byte class_index + 12 bytes scores (3×float32)
```

### Step 5: test.cpp — 对齐测试

开机时自动运行，验证 ESP32 端的预处理和推理与 Python 端一致。

需要 Kirsi 用 `generate_test_case.py`（还没写）生成 `test_case.h`，包含：
- 一张测试图片的 RGB565 原始数据
- Python 端预处理后的特征向量（ground truth）
- Python 端推理后的预测结果（ground truth）

ESP32 端对同一张图跑预处理和推理，对比结果。

**这一步可以先跳过，等模型跑通了再加。**

---

## 3. 多帧投票（可选增强）

在 main.cpp 的 loop 里加一个简单的滑动窗口：

```c
#define VOTE_WINDOW 5
#define VOTE_THRESHOLD 3

static int vote_buffer[VOTE_WINDOW];
static int vote_idx = 0;

// 每帧更新
if (max_score >= CONFIDENCE_THRESHOLD) {
    vote_buffer[vote_idx] = best_class + 1;  // 1,2,3 = 已知
} else {
    vote_buffer[vote_idx] = 0;               // 0 = 拒绝
}
vote_idx = (vote_idx + 1) % VOTE_WINDOW;

// 统计最近 5 帧
int accept_count = 0;
for (int i = 0; i < VOTE_WINDOW; i++) {
    if (vote_buffer[i] > 0) accept_count++;
}

// 3/5 帧通过 → 开门
if (accept_count >= VOTE_THRESHOLD) {
    gpio_set_level(LED_PIN, 0);  // 开门
} else {
    gpio_set_level(LED_PIN, 1);  // 拒绝
}
```

---

## 4. 内存预算

| 缓冲区 | 大小 | 位置 |
|--------|-----:|------|
| `image_buffer[320×240×2]` | 153,600 B | PSRAM |
| `face_features[48×48]` | 9,216 B | SRAM |
| `tensor_arena` | ~60,000 B | SRAM |
| `prediction[3]` | 12 B | SRAM |
| `vote_buffer[5]` | 20 B | SRAM |
| **SRAM 总计** | **~69 KB** | 在 512 KB 预算内 |

---

## 5. 实施顺序建议

| 优先级 | 任务 | 依赖 |
|:---:|------|------|
| 1 | 复制项目框架 + model_data.c/.h | 无 |
| 2 | 写 preprocess.cpp | camera.h |
| 3 | 改 inference.cpp | model_data.h |
| 4 | 写 main.cpp（不含串口输出） | 以上全部 |
| 5 | 编译 + 烧录 + 调试 LED | 以上全部 |
| 6 | 加串口输出 + display.py | Step 5 通过 |
| 7 | 加多帧投票 | Step 5 通过 |
| 8 | 加 test.cpp 对齐测试 | 需要 test_case.h |

---

## 6. 关键注意事项

1. **预处理对齐** — ESP32 和 Python 的预处理结果必须一致，否则模型输出垃圾。最容易出错的地方是 RGB565 字节序和灰度转换公式。

2. **Tensor Arena 大小** — 如果 `AllocateTensors()` 失败，说明 arena 不够大，往上加。

3. **Op Resolver 算子** — 如果模型加载报错说缺算子，根据报错信息在 resolver 里加对应的 `Add*()` 调用。

4. **PSRAM** — ESP32S3 Sense 有 8MB PSRAM。image_buffer 建议放 PSRAM（用 `heap_caps_malloc(size, MALLOC_CAP_SPIRAM)`），SRAM 留给 tensor arena。

5. **阈值调整** — `CONFIDENCE_THRESHOLD` 从 0.90 开始，实际测试时根据效果调整。
