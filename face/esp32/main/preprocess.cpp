#include "preprocess.h"

static const int CROP_SIZE = 240;
static const int CROP_X_OFFSET = (CAMERA_W - CROP_SIZE) / 2;
static const int CROP_Y_OFFSET = (CAMERA_H - CROP_SIZE) / 2;
static const int BLOCK_SIZE = CROP_SIZE / FACE_W;

static uint8_t rgb565_to_gray(uint8_t byte0, uint8_t byte1)
{
    uint16_t pixel = ((uint16_t)byte0 << 8) | byte1;

    uint8_t r5 = (pixel >> 11) & 0x1F;
    uint8_t g6 = (pixel >> 5)  & 0x3F;
    uint8_t b5 = pixel         & 0x1F;

    uint8_t r = (r5 << 3) | (r5 >> 2);
    uint8_t g = (g6 << 2) | (g6 >> 4);
    uint8_t b = (b5 << 3) | (b5 >> 2);

    // integer version of the usual RGB-to-luma weighting
    return (5 * r + 9 * g + 2 * b) >> 4;
}

void preprocess_frame(const uint8_t *rgb565_frame, float *face_features)
{
    // crop and downsample in one pass; no extra 240x240 buffer
    for (int out_y = 0; out_y < FACE_H; out_y++)
    {
        for (int out_x = 0; out_x < FACE_W; out_x++)
        {
            int gray_sum = 0;

            for (int dy = 0; dy < BLOCK_SIZE; dy++)
            {
                for (int dx = 0; dx < BLOCK_SIZE; dx++)
                {
                    int src_x = CROP_X_OFFSET + out_x * BLOCK_SIZE + dx;
                    int src_y = CROP_Y_OFFSET + out_y * BLOCK_SIZE + dy;

                    int byte_offset = (src_y * CAMERA_W + src_x) * 2;
                    uint8_t byte0 = rgb565_frame[byte_offset];
                    uint8_t byte1 = rgb565_frame[byte_offset + 1];
                    gray_sum += rgb565_to_gray(byte0, byte1);
                }
            }

            // average the 5x5 block and normalize to the model input range
            face_features[out_y * FACE_W + out_x] = (float)gray_sum / (BLOCK_SIZE * BLOCK_SIZE * 255.0f);
        }
    }
}
