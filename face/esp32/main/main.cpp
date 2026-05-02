// ============================================================================
// main.cpp — 人脸识别门禁系统主程序
//
// 这是整个 ESP32 固件的入口。程序结构很简单：
//   setup() — 开机初始化（只跑一次）
//   loop()  — 主循环（无限重复）：拍照 → 预处理 → 推理 → LED 反馈
//
// 数据流：
//   OV2640 摄像头 (320×240 RGB565)
//       ↓ camera_capture_frame()
//   image_buffer (153,600 字节)
//       ↓ preprocess_frame()
//   face_features (48×48 = 2,304 个 float)
//       ↓ inference_put_features() — 量化成 int8
//       ↓ inference_predict()     — 跑 CNN
//   prediction (3 个 float: person_a, person_b, person_c 的概率)
//       ↓ 判断逻辑
//   LED 亮/灭 + 串口输出
// ============================================================================

#include <cstdio>
#include <cstring>

// ESP-IDF 系统头文件
#include "freertos/FreeRTOS.h"  // 实时操作系统（ESP32 的底层调度系统）
#include "freertos/task.h"      // 任务延时函数 vTaskDelay()
#include "driver/gpio.h"        // GPIO 控制（用于 LED）
#include "driver/usb_serial_jtag.h"
#include "esp_log.h"            // 日志打印（ESP_LOGI, ESP_LOGE 等）
#include "esp_timer.h"          // 高精度计时，用于部署端 latency 测量
#include "nvs_flash.h"

// 项目模块
#include "camera.h"      // 摄像头：初始化 + 拍照
#include "centroids_config.h"  // 二阶段拒识：embedding 质心和距离阈值
#include "preprocess.h"  // 预处理：RGB565 → 48×48 灰度 float
#include "inference.h"   // 推理：加载模型 + 量化 + Invoke + 反量化

// ============================================================================
// 常量定义
// ============================================================================

// LED 引脚 — XIAO ESP32S3 Sense 的板载 LED
// active-low：GPIO 输出 0 = LED 亮，GPIO 输出 1 = LED 灭
#define LED_PIN GPIO_NUM_21

// ── 多帧投票参数 ─────────────────────────────────────────
// 原理：最近 VOTE_WINDOW 帧里，如果同一个人出现 >= VOTE_THRESHOLD 次，
//       才认为真的是这个人。否则判为 UNKNOWN。
// 为什么需要：
//   - 手掌/桌面等非人脸输入，模型预测不稳定（一会 A 一会 B）→ 投票通不过
//   - 真人脸，模型预测稳定（连续都是 A）→ 投票通过
//   - 这样不改模型就能提升 unknown rejection
#define VOTE_WINDOW    5   // 滑动窗口大小（看最近 5 帧）
#define VOTE_THRESHOLD 4   // 至少 4 帧一致才放行

// 类别名称 — 用于串口打印，方便调试
static const char *CLASS_NAMES[] = {"person_a", "person_b", "person_c"};

// 输出类别数
#define NUM_CLASSES 3

// 日志标签
static const char *TAG = "FaceRecog";

// Web/串口前端协议常量
static constexpr size_t SERIAL_CHUNK_SIZE = 512;
static constexpr size_t FRAME_SIZE_BYTES = FRAME_W * FRAME_H * FRAME_C;
static const char *READY_PREAMBLE = "\n===READY===\n";
static const char *METRICS_PREAMBLE = "\n===METRICS===\n";
static const char *FRAME_PREAMBLE = "\n===FRAME===\n";
static const char *START_STREAM_COMMAND = "START_STREAM\n";

// ============================================================================
// 缓冲区 — 存储摄像头帧和处理结果
// ============================================================================

// 摄像头原始帧：320×240 像素 × 2 字节/像素 = 153,600 字节
// 这是整个程序里最大的一块内存
// 注意：如果内存不够，可以用 heap_caps_malloc(size, MALLOC_CAP_SPIRAM)
// 把它放到 PSRAM（8MB 外部内存）而不是 SRAM（512KB 内部内存）
static uint8_t image_buffer[FRAME_W * FRAME_H * FRAME_C];

// 预处理后的特征：48×48 = 2,304 个 float
static float face_features[FACE_INPUT_SIZE];

// 推理结果：3 个 float（person_a, person_b, person_c 的概率）
static float prediction[NUM_CLASSES];

// Dense(32) embedding，用于计算到三个人质心的距离
static float embedding[FACE_EMBEDDING_DIM];

// ── 多帧投票缓冲区 ──────────────────────────────────────
// vote_buffer[i] 存第 i 帧的判断结果：
//   -1 = UNKNOWN（置信度不够）
//   0/1/2 = person_a / person_b / person_c
// vote_idx 是循环写入的位置（环形缓冲区）
static int vote_buffer[VOTE_WINDOW];
static int vote_idx = 0;
static bool stream_enabled = false;

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

// ============================================================================
// setup() — 开机初始化
//
// 初始化顺序有讲究：
//   1. 先初始化推理引擎（如果模型加载失败，后面都不用做了）
//   2. 再初始化摄像头
//   3. 最后初始化 LED
// ============================================================================
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

    // 初始化推理引擎 — 加载模型、注册算子、分配 tensor arena
    if (!inference_init())
    {
        ESP_LOGE(TAG, "Failed to initialize inference!");
        abort();  // 模型加载失败，直接停止（没法继续）
    }

    // 初始化摄像头 — 配置 OV2640，320×240 RGB565
    if (!camera_init())
    {
        ESP_LOGE(TAG, "Failed to initialize camera!");
        abort();  // 摄像头初始化失败，也没法继续
    }

    // 初始化 LED
    // gpio_reset_pin: 把引脚恢复到默认状态（清除之前的配置）
    // gpio_set_direction: 设为输出模式
    // gpio_set_level(1): 初始状态 LED 灭（active-low，1 = 灭）
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(LED_PIN, 1);  // 开机时 LED 灭（门关着）

    // 初始化投票缓冲区：全部设为 -1（UNKNOWN）
    for (int i = 0; i < VOTE_WINDOW; i++)
    {
        vote_buffer[i] = -1;
    }

    ESP_LOGI(TAG, "Initialization complete. Waiting for START_STREAM command.");
    serial_write_text(READY_PREAMBLE);
    serial_write_text("Send START_STREAM to begin hardware video streaming.\n");
}

// ============================================================================
// loop() — 主循环（每次执行 = 处理一帧）
//
// 流程：
//   1. 拍一帧照片
//   2. 预处理（RGB565 → 48×48 灰度 float）
//   3. 量化 + 推理
//   4. 找最大概率的类别
//   5. 判断是否开门 + LED 控制
//   6. 串口输出结果（调试用）
// ============================================================================
void loop(void)
{
    int64_t t_loop_start_us = esp_timer_get_time();
    poll_stream_command();

    // ── 1. 拍照 ──────────────────────────────────────────
    // camera_capture_frame() 从 OV2640 拍一帧，写入 image_buffer
    // 返回 false 表示拍照失败（摄像头可能卡了），跳过这一帧
    if (!camera_capture_frame(image_buffer))
    {
        ESP_LOGW(TAG, "Frame capture failed, skipping...");
        return;
    }
    int64_t t_capture_done_us = esp_timer_get_time();

    // ── 2. 预处理 ────────────────────────────────────────
    // 320×240 RGB565 → 48×48 灰度 float [0.0, 1.0]
    // 这一步做了：中心裁剪、5×5 区域平均缩放、灰度转换、归一化
    preprocess_frame(image_buffer, face_features);
    int64_t t_preprocess_done_us = esp_timer_get_time();

    // ── 3. 量化 + 推理 ──────────────────────────────────
    // put_features: float [0.0, 1.0] → int8 [-128, 127]（量化）
    // predict: 跑 CNN（Invoke），输出 int8 → float 概率（反量化）
    inference_put_features(face_features);

    if (!inference_predict(prediction, embedding))
    {
        ESP_LOGE(TAG, "Inference failed!");
        return;
    }
    int64_t t_inference_done_us = esp_timer_get_time();

    // ── 4. 找最大概率的类别 ─────────────────────────────
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

    // ── 5. 多帧投票 ──────────────────────────────────────
    // 先判断这一帧的单帧结果
    int frame_result;
    bool softmax_ok = max_score >= SOFTMAX_THRESHOLD;
    bool distance_ok = min_dist_sq <= DISTANCE_THRESHOLD_SQ;
    bool class_agree = nearest_centroid == best_class;

    if (softmax_ok && distance_ok && class_agree)
    {
        frame_result = best_class;  // 0/1/2 = person_a/b/c
    }
    else
    {
        frame_result = -1;  // UNKNOWN
    }

    // 写入环形缓冲区
    vote_buffer[vote_idx] = frame_result;
    vote_idx = (vote_idx + 1) % VOTE_WINDOW;

    // 统计最近 VOTE_WINDOW 帧里每个类别出现了多少次
    // counts[0] = person_a 次数, counts[1] = person_b 次数, counts[2] = person_c 次数
    int counts[NUM_CLASSES] = {0};
    for (int i = 0; i < VOTE_WINDOW; i++)
    {
        if (vote_buffer[i] >= 0 && vote_buffer[i] < NUM_CLASSES)
        {
            counts[vote_buffer[i]]++;
        }
    }

    // 找票数最多的类别
    int vote_winner = -1;    // -1 = 没人赢 = UNKNOWN
    int vote_max_count = 0;
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        if (counts[i] > vote_max_count)
        {
            vote_max_count = counts[i];
            vote_winner = i;
        }
    }

    // 只有票数 >= VOTE_THRESHOLD 才认为有效
    bool access_granted = (vote_max_count >= VOTE_THRESHOLD);

    // ── 6. LED 控制 ─────────────────────────────────────
    if (access_granted)
    {
        gpio_set_level(LED_PIN, 0);  // LED 亮 = 放行
    }
    else
    {
        gpio_set_level(LED_PIN, 1);  // LED 灭 = 拒绝
    }

    const char *frame_name = (frame_result >= 0) ? CLASS_NAMES[frame_result] : "UNKNOWN";
    const char *vote_name = access_granted ? CLASS_NAMES[vote_winner] : "UNKNOWN";

    ESP_LOGI(TAG,
             "Frame: %s (%.2f, dist_sq=%.3f, nearest=%s) | Vote: %s (%d/%d) | "
             "[A=%.3f B=%.3f C=%.3f] | gates[S=%d D=%d C=%d]",
             frame_name, max_score,
             min_dist_sq, CLASS_NAMES[nearest_centroid],
             vote_name, vote_max_count, VOTE_WINDOW,
             prediction[0], prediction[1], prediction[2],
             softmax_ok, distance_ok, class_agree);

    int64_t capture_ms = (t_capture_done_us - t_loop_start_us) / 1000;
    int64_t preprocess_ms = (t_preprocess_done_us - t_capture_done_us) / 1000;
    int64_t inference_ms = (t_inference_done_us - t_preprocess_done_us) / 1000;
    int64_t total_ms = (esp_timer_get_time() - t_loop_start_us) / 1000;
    float fps = total_ms > 0 ? 1000.0f / (float)total_ms : 0.0f;

    ESP_LOGI(TAG,
             "timing_ms[capture=%lld preprocess=%lld inference=%lld total=%lld fps=%.2f]",
             (long long)capture_ms,
             (long long)preprocess_ms,
             (long long)inference_ms,
             (long long)total_ms,
             fps);

    if (!stream_enabled)
    {
        vTaskDelay(pdMS_TO_TICKS(30));
        return;
    }

    // ── 7. 串口输出 ─────────────────────────────────────
    // 同时发元数据和原始 RGB565 帧，供 Web Serial 前端直接展示。
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

    serial_write_text(FRAME_PREAMBLE);
    serial_write_all(image_buffer, FRAME_SIZE_BYTES);
}

// ============================================================================
// app_main() — ESP-IDF 的程序入口（相当于 Arduino 的 main）
//
// extern "C" 是因为 ESP-IDF 的启动代码是 C 语言写的，
// 但我们的代码是 C++。extern "C" 告诉编译器：
// "这个函数用 C 的命名规则，这样 C 代码才能找到它"
// ============================================================================
extern "C" void app_main(void)
{
    setup();

    // 无限循环 — 每次循环处理一帧
    while (true)
    {
        loop();
    }
}
