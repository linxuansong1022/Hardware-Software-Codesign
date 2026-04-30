// ============================================================================
// inference.h — 推理引擎接口
//
// 和 keywords 版本完全一样的三个函数，只是处理的数据从音频变成了图片。
// ============================================================================

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifndef FACE_NUM_CLASSES
#define FACE_NUM_CLASSES 3
#endif

#ifndef FACE_EMBEDDING_DIM
#define FACE_EMBEDDING_DIM 32
#endif

// 初始化推理引擎（加载模型、分配内存）
bool inference_init(void);

// 把 float 特征量化成 int8 写入模型输入
int8_t *inference_put_features(const float *features);

// 执行推理，结果写入 prediction[3] 和 embedding[32]
bool inference_predict(float *prediction, float *embedding);
