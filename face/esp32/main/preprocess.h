#pragma once

#include <stdint.h>

#define CAMERA_W  320
#define CAMERA_H  240

#define FACE_W    48
#define FACE_H    48

#define FACE_INPUT_SIZE  (FACE_W * FACE_H)

void preprocess_frame(const uint8_t *rgb565_frame, float *face_features);
