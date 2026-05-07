#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifndef FACE_NUM_CLASSES
#define FACE_NUM_CLASSES 3
#endif

#ifndef FACE_EMBEDDING_DIM
#define FACE_EMBEDDING_DIM 32
#endif

bool inference_init(void);

int8_t *inference_put_features(const float *features);

bool inference_predict(float *prediction, float *embedding);
