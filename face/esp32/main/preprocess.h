// ============================================================================
// preprocess.h — 图片预处理接口
//
// 只暴露一个函数给外部使用：
//   preprocess_frame() — 把摄像头的原始帧转成模型需要的 48×48 灰度 float
//
// 常量定义也放在这里，让 inference.cpp 和 main.cpp 都能用到。
// ============================================================================

#pragma once
// #pragma once 的作用：防止这个头文件被重复包含。
// 如果 main.cpp 和 inference.cpp 都 #include "preprocess.h"，
// 没有这行的话，里面的定义会出现两次，编译器会报错。

#include <stdint.h>  // 提供 uint8_t, uint16_t 等类型定义

// ── 摄像头参数（来自 camera.h）────────────────────────────
// 摄像头输出 320×240 像素，每个像素 2 字节（RGB565 格式）
#define CAMERA_W  320
#define CAMERA_H  240

// ── 模型输入参数 ─────────────────────────────────────────
// 我们的 CNN 模型需要 48×48 的灰度图
#define FACE_W    48
#define FACE_H    48

// 模型输入的总像素数（用于 inference.cpp 里遍历）
#define FACE_INPUT_SIZE  (FACE_W * FACE_H)

// ── 函数声明 ─────────────────────────────────────────────

// 把 320×240 RGB565 帧转成 48×48 灰度 float 数组
// 参数：
//   rgb565_frame — 摄像头原始数据，320×240×2 = 153,600 字节
//   face_features — 输出数组，48×48 = 2,304 个 float，值域 [0.0, 1.0]
void preprocess_frame(const uint8_t *rgb565_frame, float *face_features);
