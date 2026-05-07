#include <cmath>
#include <cstddef>
#include <cstdint>

#include "esp_log.h"

#include "inference.h"
#include "model_data.h"
#include "preprocess.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#define TENSOR_ARENA_SIZE (60 * 1024)

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static TfLiteTensor *input = nullptr;
static TfLiteTensor *softmax_output = nullptr;
static TfLiteTensor *embedding_output = nullptr;
static const char *TAG_INF = "Inference";

bool inference_init() {
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG_INF, "Model schema mismatch!");
    return false;
  }

  static tflite::MicroMutableOpResolver<9> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddShape();
  micro_op_resolver.AddExpandDims();
  micro_op_resolver.AddStridedSlice();
  micro_op_resolver.AddPack();

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

  // The model has two outputs, but their order is not assumed here.
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
  for (size_t i = 0; i < FACE_INPUT_SIZE; ++i) {
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
  if (interpreter->Invoke() != kTfLiteOk) {
    return false;
  }

  // Keep the rejection code in float even though the model itself is int8.
  dequantize_tensor(softmax_output, prediction, FACE_NUM_CLASSES);
  dequantize_tensor(embedding_output, embedding, FACE_EMBEDDING_DIM);

  return true;
}
