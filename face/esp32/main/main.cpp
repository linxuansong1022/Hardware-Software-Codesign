#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cinttypes>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/usb_serial_jtag.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"

#include "camera.h"
#include "centroids_config.h"
#include "preprocess.h"
#include "inference.h"

// Onboard LED is active-low on the XIAO ESP32-S3 Sense.
#define LED_PIN GPIO_NUM_21

#define VOTE_WINDOW    5
#define VOTE_THRESHOLD 4

static const char *CLASS_NAMES[] = {"person_a", "person_b", "person_c"};
#define NUM_CLASSES 3

static const char *TAG = "FaceRecog";

// The browser side looks for these markers in the serial stream.
static constexpr size_t SERIAL_CHUNK_SIZE = 512;
static constexpr size_t FRAME_SIZE_BYTES = FRAME_W * FRAME_H * FRAME_C;
static const char *READY_PREAMBLE = "\n===READY===\n";
static const char *METRICS_PREAMBLE = "\n===METRICS===\n";
static const char *FRAME_PREAMBLE = "\n===FRAME===\n";
static const char *START_STREAM_COMMAND = "START_STREAM\n";

static uint8_t image_buffer[FRAME_W * FRAME_H * FRAME_C];
static float face_features[FACE_INPUT_SIZE];
static float prediction[NUM_CLASSES];
static float embedding[FACE_EMBEDDING_DIM];

// ring buffer for the last few frame decisions
// -1 = unknown, 0..2 = known people
static int vote_buffer[VOTE_WINDOW];
static int vote_idx = 0;
static bool stream_enabled = false;

// same CRC32 polynomial as the frontend checker
static uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t length)
{
    crc = ~crc;
    for (size_t i = 0; i < length; ++i)
    {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
        {
            const uint32_t mask = static_cast<uint32_t>(-(static_cast<int32_t>(crc & 1U)));
            crc = (crc >> 1) ^ (0xEDB88320U & mask);
        }
    }
    return ~crc;
}

static uint32_t crc32_bytes(const uint8_t *data, size_t length)
{
    return crc32_update(0U, data, length);
}

static void serial_write_all(const uint8_t *data, size_t length)
{
    for (size_t offset = 0; offset < length;)
    {
        const size_t chunk = (length - offset > SERIAL_CHUNK_SIZE) ? SERIAL_CHUNK_SIZE : (length - offset);
        const int written = usb_serial_jtag_write_bytes(data + offset, chunk, pdMS_TO_TICKS(1000));
        if (written > 0)
        {
            offset += static_cast<size_t>(written);
        }
        else
        {
            vTaskDelay(1);
        }
    }
}

static void serial_write_text(const char *text)
{
    serial_write_all(reinterpret_cast<const uint8_t *>(text), strlen(text));
}

static void poll_stream_command(void)
{
    static char command_buffer[32];
    static size_t command_len = 0;

    char incoming[32];
    const int bytes_read = usb_serial_jtag_read_bytes(incoming, sizeof(incoming), 0);
    if (bytes_read <= 0)
    {
        return;
    }

    for (int i = 0; i < bytes_read; ++i)
    {
        const char c = incoming[i];
        if (command_len < sizeof(command_buffer) - 1)
        {
            command_buffer[command_len++] = c;
            command_buffer[command_len] = '\0';
        }
        else
        {
            command_len = 0;
            command_buffer[0] = '\0';
        }

        if (c == '\n')
        {
            if (strcmp(command_buffer, START_STREAM_COMMAND) == 0)
            {
                stream_enabled = true;
                serial_write_text("\n===STREAMING===\n");
                serial_write_text("Web stream started.\n");
            }
            command_len = 0;
            command_buffer[0] = '\0';
        }
    }
}

static int find_nearest_centroid(const float *emb, float *min_dist_sq)
{
    int nearest_class = 0;
    float best_dist_sq = 0.0f;

    for (int c = 0; c < NUM_CLASSES; c++)
    {
        float dist_sq = 0.0f;
        for (int d = 0; d < FACE_EMBEDDING_DIM; d++)
        {
            float diff = emb[d] - FACE_CENTROIDS[c][d];
            dist_sq += diff * diff;
        }

        if (c == 0 || dist_sq < best_dist_sq)
        {
            best_dist_sq = dist_sq;
            nearest_class = c;
        }
    }

    *min_dist_sq = best_dist_sq;
    return nearest_class;
}

void setup(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    usb_serial_jtag_driver_config_t usb_serial_jtag_config = {
        .tx_buffer_size = 2048,
        .rx_buffer_size = 256,
    };
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_serial_jtag_config));

    // If either model or camera init fails, there is no useful fallback.
    if (!inference_init())
    {
        ESP_LOGE(TAG, "Failed to initialize inference!");
        abort();
    }

    if (!camera_init())
    {
        ESP_LOGE(TAG, "Failed to initialize camera!");
        abort();
    }

    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(LED_PIN, 1);

    for (int i = 0; i < VOTE_WINDOW; i++)
    {
        vote_buffer[i] = -1;
    }

    ESP_LOGI(TAG, "Initialization complete. Waiting for START_STREAM command.");
    serial_write_text(READY_PREAMBLE);
    serial_write_text("Send START_STREAM to begin hardware video streaming.\n");
}

void loop(void)
{
    int64_t t_loop_start_us = esp_timer_get_time();
    poll_stream_command();

    if (!camera_capture_frame(image_buffer))
    {
        ESP_LOGW(TAG, "Frame capture failed, skipping...");
        return;
    }
    int64_t t_capture_done_us = esp_timer_get_time();

    preprocess_frame(image_buffer, face_features);
    int64_t t_preprocess_done_us = esp_timer_get_time();

    inference_put_features(face_features);

    if (!inference_predict(prediction, embedding))
    {
        ESP_LOGE(TAG, "Inference failed!");
        return;
    }
    int64_t t_inference_done_us = esp_timer_get_time();

    int best_class = 0;
    float max_score = prediction[0];

    for (int i = 1; i < NUM_CLASSES; i++)
    {
        if (prediction[i] > max_score)
        {
            max_score = prediction[i];
            best_class = i;
        }
    }

    float min_dist_sq = 0.0f;
    int nearest_centroid = find_nearest_centroid(embedding, &min_dist_sq);

    int frame_result;
    bool softmax_ok = max_score >= SOFTMAX_THRESHOLD;
    bool distance_ok = min_dist_sq <= DISTANCE_THRESHOLD_SQ;
    bool class_agree = nearest_centroid == best_class;

    // all three checks must pass before we count this frame as a known user
    if (softmax_ok && distance_ok && class_agree)
    {
        frame_result = best_class;
    }
    else
    {
        frame_result = -1;
    }

    vote_buffer[vote_idx] = frame_result;
    vote_idx = (vote_idx + 1) % VOTE_WINDOW;

    // voting is deliberately conservative for the door-access demo
    int counts[NUM_CLASSES] = {0};
    for (int i = 0; i < VOTE_WINDOW; i++)
    {
        if (vote_buffer[i] >= 0 && vote_buffer[i] < NUM_CLASSES)
        {
            counts[vote_buffer[i]]++;
        }
    }

    int vote_winner = -1;
    int vote_max_count = 0;
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        if (counts[i] > vote_max_count)
        {
            vote_max_count = counts[i];
            vote_winner = i;
        }
    }

    bool access_granted = (vote_max_count >= VOTE_THRESHOLD);

    if (access_granted)
    {
        gpio_set_level(LED_PIN, 0);
    }
    else
    {
        gpio_set_level(LED_PIN, 1);
    }

    const char *frame_name = (frame_result >= 0) ? CLASS_NAMES[frame_result] : "UNKNOWN";
    const char *vote_name = access_granted ? CLASS_NAMES[vote_winner] : "UNKNOWN";

    int64_t capture_ms = (t_capture_done_us - t_loop_start_us) / 1000;
    int64_t preprocess_ms = (t_preprocess_done_us - t_capture_done_us) / 1000;
    int64_t inference_ms = (t_inference_done_us - t_preprocess_done_us) / 1000;
    int64_t total_ms = (esp_timer_get_time() - t_loop_start_us) / 1000;
    float fps = total_ms > 0 ? 1000.0f / (float)total_ms : 0.0f;

    // plain monitor mode: readable logs only, no raw frame spam
    if (!stream_enabled)
    {
        ESP_LOGI(TAG,
                 "Frame: %s (%.2f, dist_sq=%.3f, nearest=%s) | Vote: %s (%d/%d) | "
                 "[A=%.3f B=%.3f C=%.3f] | gates[S=%d D=%d C=%d]",
                 frame_name, max_score,
                 min_dist_sq, CLASS_NAMES[nearest_centroid],
                 vote_name, vote_max_count, VOTE_WINDOW,
                 prediction[0], prediction[1], prediction[2],
                 softmax_ok, distance_ok, class_agree);

        ESP_LOGI(TAG,
                 "timing_ms[capture=%lld preprocess=%lld inference=%lld total=%lld fps=%.2f]",
                 (long long)capture_ms,
                 (long long)preprocess_ms,
                 (long long)inference_ms,
                 (long long)total_ms,
                 fps);
    }

    if (!stream_enabled)
    {
        vTaskDelay(pdMS_TO_TICKS(30));
        return;
    }

    // stream mode: JSON metrics first, then one RGB565 frame
    char metrics_json[512];
    const int metrics_len = snprintf(
        metrics_json,
        sizeof(metrics_json),
        "{\"frame\":\"%s\",\"frameConfidence\":%.3f,\"vote\":\"%s\",\"voteCount\":%d,\"voteWindow\":%d,"
        "\"scores\":{\"A\":%.3f,\"B\":%.3f,\"C\":%.3f},"
        "\"nearest\":\"%s\",\"distSq\":%.3f,"
        "\"gates\":{\"softmax\":%d,\"distance\":%d,\"classAgreement\":%d},"
        "\"timingMs\":{\"capture\":%lld,\"preprocess\":%lld,\"inference\":%lld,\"total\":%lld,\"fps\":%.2f}}\n",
        frame_name,
        max_score,
        vote_name,
        vote_max_count,
        VOTE_WINDOW,
        prediction[0],
        prediction[1],
        prediction[2],
        CLASS_NAMES[nearest_centroid],
        min_dist_sq,
        softmax_ok,
        distance_ok,
        class_agree,
        (long long)capture_ms,
        (long long)preprocess_ms,
        (long long)inference_ms,
        (long long)total_ms,
        fps);

    if (metrics_len > 0)
    {
        serial_write_text(METRICS_PREAMBLE);
        serial_write_all(reinterpret_cast<const uint8_t *>(metrics_json), static_cast<size_t>(metrics_len));
    }

    const uint32_t frame_crc32 = crc32_bytes(image_buffer, FRAME_SIZE_BYTES);
    char frame_header[96];
    const int frame_header_len = snprintf(
        frame_header,
        sizeof(frame_header),
        "{\"length\":%u,\"crc32\":\"%08" PRIx32 "\"}\n",
        static_cast<unsigned>(FRAME_SIZE_BYTES),
        frame_crc32);

    serial_write_text(FRAME_PREAMBLE);
    if (frame_header_len > 0)
    {
        serial_write_all(reinterpret_cast<const uint8_t *>(frame_header), static_cast<size_t>(frame_header_len));
    }
    serial_write_all(image_buffer, FRAME_SIZE_BYTES);
}

extern "C" void app_main(void)
{
    setup();

    while (true)
    {
        loop();
    }
}
