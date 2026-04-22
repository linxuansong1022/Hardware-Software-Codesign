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

// [改动 4] 输出类别数
// keywords: 3 类（Other, Yes, No）
// 人脸: 3 类（person_a, person_b, person_c）
// 数量恰好一样，但含义不同
#define NUM_CLASSES 3

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;
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
  output = interpreter->output(0);

  //   Input:  int8, shape: 1, 48, 48, 1
  //   Output: int8, shape: 1, 3
  ESP_LOGI(TAG_INF, "Input tensor type: %s, dims: %d",
           TfLiteTypeGetName(input->type), input->dims->size);
  for (int i = 0; i < input->dims->size; i++) {
    ESP_LOGI(TAG_INF, "  input dim[%d] = %d", i, input->dims->data[i]);
  }
  ESP_LOGI(TAG_INF, "Output tensor type: %s, dims: %d",
           TfLiteTypeGetName(output->type), output->dims->size);
  for (int i = 0; i < output->dims->size; i++) {
    ESP_LOGI(TAG_INF, "  output dim[%d] = %d", i, output->dims->data[i]);
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

bool inference_predict(float *prediction) {
  // 推理
  if (interpreter->Invoke() != kTfLiteOk) {
    return false;
  }

  // 反量化
  for (size_t i = 0; i < NUM_CLASSES; ++i) {
    prediction[i] =
        (static_cast<float>(output->data.int8[i]) - output->params.zero_point) *
        output->params.scale;
  }

  return true;
}
