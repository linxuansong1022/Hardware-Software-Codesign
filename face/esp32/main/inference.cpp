// ============================================================================
// inference.cpp — TFLite Micro 推理引擎（人脸项目版）
//
// 基于 keywords/esp32/main/inference.cpp 改编。
// 改动点用 [改动] 标注，方便对比。
// ============================================================================

#include <cmath>   // roundf
#include <cstddef> // size_t
#include <cstdint> // int8_t, uint8_t

#include "esp_log.h"

// [改动 1] 头文件：model.h → model_data.h
// keywords 的模型文件叫 model.h（里面有 model_binary 数组）
// 我们的模型文件叫 model_data.h（里面有 g_model_data 数组，由 Kirsi
// 的脚本生成）
#include "model_data.h"

// [改动 2] 引入 preprocess.h 获取常量
// FACE_INPUT_SIZE = 48×48 = 2304（替代 keywords 的 SPECTRUM_WIDTH *
// SPECTRUM_HEIGHT） 这样如果以后改分辨率，只需要改 preprocess.h 里的定义
#include "preprocess.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// [改动 3] Arena 大小：30KB → 60KB
// 我们的模型（30KB tflite）比 keywords 的大，中间层也更大（图片 vs 音频）
// 如果 AllocateTensors() 失败，就加大这个值
#define TENSOR_ARENA_SIZE (60 * 1024)

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static TfLiteTensor *input = nullptr;
static TfLiteTensor *softmax_output = nullptr;
static TfLiteTensor *embedding_output = nullptr;
static const char *TAG_INF = "Inference";

bool inference_init() {
  // [改动 5] 变量名：model_binary → g_model_data
  // g_model_data 定义在 model_data.c 里，是 Kirsi 用 export 脚本生成的
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG_INF, "Model schema mismatch!");
    return false;
  }

  static tflite::MicroMutableOpResolver<9> micro_op_resolver;
  micro_op_resolver.AddConv2D();         // 卷积层
  micro_op_resolver.AddMaxPool2D();      // 池化层
  micro_op_resolver.AddFullyConnected(); // 全连接层 (Dense)
  micro_op_resolver.AddSoftmax();        // 输出激活
  micro_op_resolver.AddReshape();        // Flatten 需要
  micro_op_resolver.AddShape();          // Flatten 需要
  micro_op_resolver.AddExpandDims();     // Flatten 需要
  micro_op_resolver.AddStridedSlice();   // Flatten 需要
  micro_op_resolver.AddPack();           // Flatten 需要

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG_INF, "Failed to allocate tensors!");
    return false;
  }

  input = interpreter->input(0);
  softmax_output = nullptr;
  embedding_output = nullptr;

  //   Input:  int8, shape: 1, 48, 48, 1
  //   Output 0/1: int8, shapes: 1×3 softmax and 1×32 embedding
  ESP_LOGI(TAG_INF, "Input tensor type: %s, dims: %d",
           TfLiteTypeGetName(input->type), input->dims->size);
  for (int i = 0; i < input->dims->size; i++) {
    ESP_LOGI(TAG_INF, "  input dim[%d] = %d", i, input->dims->data[i]);
  }

  for (size_t out_idx = 0; out_idx < interpreter->outputs_size(); out_idx++) {
    TfLiteTensor *candidate = interpreter->output(out_idx);
    ESP_LOGI(TAG_INF, "Output[%d] tensor type: %s, dims: %d",
             static_cast<int>(out_idx), TfLiteTypeGetName(candidate->type),
             candidate->dims->size);
    for (int i = 0; i < candidate->dims->size; i++) {
      ESP_LOGI(TAG_INF, "  output[%d] dim[%d] = %d", static_cast<int>(out_idx),
               i, candidate->dims->data[i]);
    }

    int last_dim = candidate->dims->data[candidate->dims->size - 1];
    if (last_dim == FACE_NUM_CLASSES) {
      softmax_output = candidate;
    } else if (last_dim == FACE_EMBEDDING_DIM) {
      embedding_output = candidate;
    }
  }

  if (softmax_output == nullptr) {
    ESP_LOGE(TAG_INF, "Could not find softmax output with %d classes!",
             FACE_NUM_CLASSES);
    return false;
  }
  if (embedding_output == nullptr) {
    ESP_LOGE(TAG_INF, "Could not find embedding output with %d dimensions!",
             FACE_EMBEDDING_DIM);
    return false;
  }

  return true;
}

int8_t *inference_put_features(const float *features) {
  // 循环次数：SPECTRUM_WIDTH * SPECTRUM_HEIGHT → FACE_INPUT_SIZE
  // FACE_INPUT_SIZE = 48 × 48 = 2304（定义在 preprocess.h）
  for (size_t i = 0; i < FACE_INPUT_SIZE; ++i) {
    // 量化逻辑完全不变 — 公式是通用的
    float val_quant_float =
        roundf(features[i] / input->params.scale) + input->params.zero_point;
    if (val_quant_float > 127.0f) {
      val_quant_float = 127.0f;
    } else if (val_quant_float < -128.0f) {
      val_quant_float = -128.0f;
    }
    input->data.int8[i] = static_cast<int8_t>(val_quant_float);
  }

  return input->data.int8;
}

static void dequantize_tensor(const TfLiteTensor *tensor, float *dst,
                              size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dst[i] =
        (static_cast<float>(tensor->data.int8[i]) - tensor->params.zero_point) *
        tensor->params.scale;
  }
}

bool inference_predict(float *prediction, float *embedding) {
  // 推理
  if (interpreter->Invoke() != kTfLiteOk) {
    return false;
  }

  // 反量化
  dequantize_tensor(softmax_output, prediction, FACE_NUM_CLASSES);
  dequantize_tensor(embedding_output, embedding, FACE_EMBEDDING_DIM);

  return true;
}
