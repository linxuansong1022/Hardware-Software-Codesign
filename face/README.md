# Face Recognition Door Access System
# 人脸识别门禁系统 — 技术设计文档

**Course / 课程**: DTU 02214 — Embedded AI
**Team / 团队**: 3 members
**Hardware / 硬件**: XIAO ESP32S3 Sense (OV2640 camera, 8MB PSRAM, 8MB Flash, onboard LED, PDM microphone)
**Date / 日期**: 2026-02

---

## 1. Project Overview / 项目概述

### English
A real-time face recognition system running fully on the XIAO ESP32S3 Sense microcontroller. The system captures images from the onboard OV2640 camera, runs TensorFlow Lite Micro inference to recognize the 3 team members, and rejects all other faces. Results are shown via the onboard LED and a live Python display with recognition overlay.

### 中文
一个完全运行在 XIAO ESP32S3 Sense 微控制器上的实时人脸识别系统。系统通过 OV2640 摄像头采集图像，在设备端运行 TensorFlow Lite Micro 推理，识别三位团队成员并拒绝其他人脸。识别结果通过板载 LED 和 Python 端实时预览窗口（含识别叠加层）展示。

---

## 2. System Architecture / 系统架构

```
┌─────────────────────────────────────────────────────┐
│                  XIAO ESP32S3 Sense                 │
│                                                     │
│  OV2640 Camera                                      │
│       │ 320×240 RGB565                              │
│       ▼                                             │
│  Center-crop + Downsample (5×5 area avg)            │
│       │ 48×48 grayscale float                       │
│       ▼                                             │
│  TFLite Micro (INT8 CNN, ~40KB model)               │
│       │ 4-class softmax                             │
│       ▼                                             │
│  Person A / B / C / Unknown                         │
│       │                                             │
│  LED (green=member, off=unknown)                    │
│  USB Serial ──────────────────────────────────────► │
└─────────────────────────────────────────────────────┘
                                                      │
              ┌───────────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────────┐
│                Python Display App                   │
│                                                     │
│  ===FRAME===  → RGB565 decode → pygame display      │
│  ===RESULT=== → cls + 4 scores → overlay render     │
│                                                     │
│  [Live camera feed with name + confidence bar]      │
│  [Green/red border around frame]                    │
│  [Access log panel (Bonus)]                         │
└─────────────────────────────────────────────────────┘
```

---

## 3. Data Collection Plan / 数据收集计划

*(This section also serves as the Lab 2 submission / 本节同时作为 Lab 2 作业提交)*

### 3.1 Application / 应用描述

**English**: A face-based access control application that grants access to registered team members and denies all others. The system operates in indoor environments, under varied lighting, at distances of 30–100cm from the camera. It responds by lighting a green LED and displaying the member's name when a team member is recognized with confidence ≥ 0.80, or lighting no LED and showing "Unknown" otherwise.

**中文**: 一个基于人脸的门禁控制应用，为已注册的团队成员开放访问权限，拒绝所有其他人。系统在室内环境下运行，支持不同光照条件，摄像头距人脸 30–100cm 范围内有效。当以 ≥ 0.80 的置信度识别到团队成员时，点亮绿色 LED 并显示姓名；否则不亮灯，显示"Unknown"。

---

### 3.2 Classes and Labeling Guidelines / 类别与标注指南

| Class Index | Class Name | Description |
|:-----------:|:----------:|:----------- |
| 0 | Person A | Team member 1 — full frontal face in frame |
| 1 | Person B | Team member 2 — full frontal face in frame |
| 2 | Person C | Team member 3 — full frontal face in frame |
| 3 | Unknown  | Any face not belonging to team members, or no face |

**Labeling rules / 标注规则**:
- Label as **Person X** (0/1/2): the face of that team member occupies ≥ 30% of the image area and is clearly recognizable. Includes images with glasses, slight head tilt (< 45°), mild occlusion (chin/forehead cut off).
- Label as **Unknown** (3): any other person's face, empty backgrounds, objects, animals, printed photos of faces, faces with > 45° tilt, or faces where identity is ambiguous.
- **Discard** (do not save): motion-blurred images where facial features are indistinguishable, or images where the face is < 15% of the image area.
- **Edge case**: if two people are in frame simultaneously, label as "Unknown" unless one team member clearly dominates the center and the other is peripheral.

---

### 3.3 Conditional Factors / 条件因素

| Factor / 因素 | Variations / 变化范围 |
|:-------------|:---------------------|
| Distance / 距离 | 30cm, 60cm, 100cm |
| Angle / 角度 | Frontal (0°), left 30°, right 30°, slight up/down tilt |
| Lighting / 光照 | Window daylight, overhead fluorescent, low-light (lamp only), backlit |
| Expression / 表情 | Neutral, slight smile, talking |
| Accessories / 配饰 | No glasses, glasses, sunglasses (some shots) |
| Background / 背景 | Lab desk, wall, outdoors |
| **Unknown variations** | Empty desk, empty wall, outdoors scene, strangers (consented), partially visible faces, posters/printed photos |

---

### 3.4 Collection Strategy / 收集策略

**Target counts / 目标数量**: 150 images × 4 classes = **600 images total**

**Per-person session (each member self-collects)**:
- Session A: frontal, neutral, no accessories — 40 images × 3 distances
- Session B: angled (±30°), varied expressions — 30 images
- Session C: different lighting conditions — 30 images
- Session D: with glasses/accessories — 20 images
- Session E: varied backgrounds — 30 images

**Unknown class**:
- 60 images: empty backgrounds (desk, wall, outdoors)
- 40 images: other people (with explicit verbal consent)
- 30 images: partially visible / edge-of-frame faces
- 20 images: printed photos of faces held up to camera (for anti-spoofing robustness)

**Balance check**: Verify counts after collection with `ls face/data/person_X/ | wc -l` before training.

---

### 3.5 Sources / 数据来源

- **Primary camera**: OV2640 on XIAO ESP32S3 Sense — 320×240 RGB565, saved as PNG via `collect.py`
- **No public datasets used** — all images captured in-session for privacy compliance
- **Camera settings**: fixed at 320×240, 1 fps capture rate, no manual focus adjustment

---

### 3.6 Documentation / 文档记录

Each image filename encodes: `image_YYYYMMDD_HHMMSS.png` (generated automatically by `collect.py`).

Additional session log (`data/session_log.csv`) recorded manually:

| Field | Description |
|:------|:----------- |
| `timestamp` | Session start datetime |
| `class` | person_a / person_b / person_c / unknown |
| `collector` | Name of person who collected this batch |
| `lighting` | daylight / overhead / low / backlit |
| `accessories` | none / glasses / sunglasses |
| `background` | lab / wall / outdoors |
| `notes` | Any deviations or special conditions |

---

### 3.7 Privacy and Legal Considerations / 隐私与法律合规

- **Team members**: all 3 team members give explicit informed consent for their images to be collected, used for training, and stored locally on team members' computers only.
- **Other people in "Unknown" class**: verbal consent obtained before capturing. Images of consenting individuals are used only for training and immediately deleted after the course project is complete.
- **Printed photos**: only public figures' photos from freely available sources (e.g., Wikipedia) are used.
- **Data storage**: all images stored locally; not uploaded to any cloud service or shared with third parties.
- **GDPR compliance**: data is used solely for educational purposes within the course. Upon project completion, all images of third parties are deleted.

---

## 4. Machine Learning Design / 机器学习设计

### 4.1 Preprocessing Pipeline / 预处理流程

**Python (training)**:
```
PNG image (320×240 or any size)
    → convert to grayscale (PIL .convert('L'))
    → resize to 48×48 (bilinear interpolation)
    → normalize: pixel / 255.0
    → z-score: (x - IMAGE_MEAN) / IMAGE_STD
    → output shape: (48, 48, 1) float32
```

**ESP32 (inference)** — from 320×240 RGB565 raw frame:
```
320×240 RGB565
    → center-crop 240×240 (skip 40px left/right)
    → 5×5 area-average downsample → 48×48
    → RGB565→grayscale: Y = (5R + 9G + 2B) >> 4
    → normalize: (Y/255.0 - IMAGE_MEAN) / IMAGE_STD
    → output: float[48×48]
```

**Why 48×48 grayscale / 为什么选 48×48 灰度**:
- 48×48×1 INT8 model ≈ 30–50KB → fits in 8MB Flash
- Tensor arena ≈ 80KB → fits in 512KB SRAM
- Inference time ≈ 50–100ms on ESP32S3 @ 240MHz → well within 1fps camera rate
- Color is largely irrelevant for identity recognition; grayscale reduces first-layer compute 3×

---

### 4.2 Model Architecture / 模型架构

**Framework**: TensorFlow 2.x / Keras → TFLite INT8 → TFLite Micro

```
Input (48, 48, 1)
    │
Conv2D(8, 3×3, relu)    → (46, 46, 8)
MaxPooling2D(2×2)        → (23, 23, 8)
Dropout(0.1)
    │
Conv2D(16, 3×3, relu)   → (21, 21, 16)
MaxPooling2D(2×2)        → (10, 10, 16)
Dropout(0.1)
    │
Conv2D(32, 3×3, relu)   → (8, 8, 32)
MaxPooling2D(2×2)        → (4, 4, 32)
Dropout(0.1)
    │
Flatten                  → (512)
Dense(32, relu)          → (32)
Dense(4, softmax)        → (4)  ← Person A / B / C / Unknown

Total params: ~22,400
TFLite INT8 model size: ~40KB
```

**Key**: uses only `Conv2D`, `MaxPool2D`, `Flatten`, `FullyConnected`, `Softmax` — the **same op resolver** as the existing keywords ESP32 code. Zero new TFLite ops needed.

**Training config / 训练配置**:
- Optimizer: Adam (lr=0.001)
- Loss: sparse_categorical_crossentropy
- Epochs: 100 with EarlyStopping (patience=20, monitor=val_loss)
- Data augmentation: horizontal flip, ±15° rotation, brightness [0.7, 1.3], zoom 0.1

**Quantization / 量化**:
- Full INT8 quantization with representative dataset (200 training samples)
- Input/output tensors both INT8
- Export: `model.c` + `model.h` (same `write_model_c_file()` utility as keywords project)

---

## 5. ESP32 Firmware Design / ESP32固件设计

### 5.1 File Structure / 文件结构

```
face/esp32/main/
├── main.cpp          # application loop
├── camera.h/.cpp     # COPY from camera/esp32/main/ (verbatim)
├── preprocess.h/.cpp # new: image resize + normalize
├── inference.h/.cpp  # ADAPT from keywords/esp32/main/inference.cpp
├── model.h/.c        # GENERATED by python/main.py
├── test_case.h       # GENERATED by python/generate_test_case.py
└── test.h/.cpp       # ADAPT from keywords/esp32/main/test.cpp
```

### 5.2 Serial Protocol / 串口协议

Each inference cycle sends:
```
[Frame packet]
  "===FRAME===\n"               ← same preamble as camera project
  <320×240×2 bytes RGB565>

[Result packet]
  "===RESULT===\n"              ← new result preamble
  <1 byte: uint8 class_index>
  <16 bytes: 4×float32 scores> ← little-endian
```
Python parses: `cls, *scores = struct.unpack('<Bffff', data[1:18])`

### 5.3 Memory Layout / 内存布局

| Buffer | Size | Placement |
|:-------|-----:|:----------|
| `image_buffer[320×240×2]` | 153,600 B | PSRAM (SPIRAM) |
| `face_features[48×48]` | 9,216 B | SRAM |
| `tensor_arena[80×1024]` | 81,920 B | SRAM (increase if AllocateTensors fails) |
| `prediction[4]` | 16 B | SRAM |

### 5.4 LED Logic / LED逻辑

```c
// Onboard LED, active-low
if (predicted_class < 3)       // Person A, B, or C
    gpio_set_level(LED_PIN, 0); // ON = green = access granted
else                            // Unknown
    gpio_set_level(LED_PIN, 1); // OFF = access denied
```

---

## 6. Python Application Design / Python应用设计

### 6.1 Scripts / 脚本说明

| Script | Purpose |
|:-------|:--------|
| `collect.py` | Data collection: receives camera feed, press 1/2/3/0 to label and save |
| `main.py` | Training pipeline: preprocess → train → quantize → export to C |
| `preprocess.py` | Image loading, resizing, normalization, dataset split |
| `generate_test_case.py` | Generates `test_case.h` for ESP32 validation |
| `display.py` | Live inference display: frame + recognition overlay |

### 6.2 Display Overlay / 显示叠加

```
┌─────────────────────────────┐
│ Person A    87%  ███████░░  │  ← name + confidence
│                             │
│   [live 320×240 camera]     │
│                             │
│ ░ A ███████░░               │  ← 4-class confidence bars
│ ░ B ██░░░░░░░               │
│ ░ C █░░░░░░░░               │
│ ░ ? ░░░░░░░░░               │
└─────────────────────────────┘
  ↑ green border = member recognized
    red border   = unknown
```

---

## 7. Bonus Features / 进阶功能

### Bonus 1: Mask/Glasses Attribute Detection / 口罩/眼镜检测
- Second tiny CNN (same 48×48 grayscale input): 3-class output (none / mask / glasses)
- Model size: ~5KB; separate 32KB tensor arena (`model2.c`, `inference2.cpp`)
- Both models share `face_features[]` — run sequentially in main loop
- Python overlay adds attribute label badge

### Bonus 2: Door Access Scenario / 门禁场景化
- Python-only changes (no firmware update needed)
- Debounce: access event fires when 3/5 consecutive frames agree with confidence > 0.80
- Animated "ACCESS GRANTED — Welcome, Alice!" (green) or "ACCESS DENIED" (red) slide-in panel
- Scrolling access log with timestamps; written to `access_log.csv`
- Optional: `pygame.mixer` plays audio on PC (no ESP32 speaker needed)

### Bonus 3: Liveness Detection / 活体检测
- Python-only; no new model needed
- Rolling buffer of last 8 frames as numpy arrays
- `is_live = np.mean(np.var(frame_stack, axis=0)[center_crop]) > THRESHOLD`
- Calibrate: printed photo ≈ 5–10 variance; live face ≈ 50–200 variance
- Gate "ACCESS GRANTED" on both `confidence > 0.80` AND `is_live == True`

---

## 8. Implementation Steps / 实施步骤

| # | Step | Output |
|:-:|:-----|:-------|
| 1 | Flash `camera/esp32` unchanged | Board streams frames |
| 2 | Create `face/python/collect.py` | Data collection tool ready |
| 3 | Collect 600 images (all 4 classes) | `face/data/` populated |
| 4 | Create `face/python/preprocess.py` | Image preprocessing verified |
| 5 | Create `face/python/main.py` | Training pipeline runs |
| 6 | Run `python main.py` | `gen/model.c`, `gen/model.h` generated; val acc ≥ 85% |
| 7 | Create `face/python/generate_test_case.py` | `gen/test_case.h` generated |
| 8 | Set up `face/esp32/` project (copy camera.*, idf_component.yml) | Project compiles |
| 9 | Write `preprocess.cpp` | RGB565 → 48×48 grayscale pipeline |
| 10 | Adapt `inference.cpp` from keywords version | TFLite Micro runs |
| 11 | Write `main.cpp` integrating all components | Full loop runs |
| 12 | Adapt `test.cpp` with image test vectors | Preprocessing + inference validated on-device |
| 13 | `idf.py build && idf.py flash monitor` | LED responds correctly |
| 14 | Create `face/python/display.py` | Live overlay working end-to-end |
| 15 | **[Bonus]** Train attribute model, integrate `model2` | Mask/glasses detected |
| 16 | **[Bonus]** Add access log panel to `display.py` | Door access demo ready |
| 17 | **[Bonus]** Add liveness buffer to `display.py` | Anti-spoofing active |

---

## 9. Key File References / 关键文件参考

| New file | Source | Change |
|:---------|:-------|:-------|
| `face/python/collect.py` | `camera/python/main.py` | Key→folder mapping, HUD label |
| `face/python/main.py` | `keywords/python/main.py` | Model arch (Conv2D 2D), 4 classes, image input |
| `face/python/preprocess.py` | `keywords/python/preprocess.py` | PIL image load instead of audio FFT |
| `face/python/generate_test_case.py` | `keywords/python/generate_test_case.py` | Image input instead of audio |
| `face/python/display.py` | `camera/python/main.py` | + result packet parsing + pygame overlay |
| `face/python/utils/*.py` | `keywords/python/utils/*.py` | **Copy verbatim — no changes** |
| `face/esp32/main/camera.*` | `camera/esp32/main/camera.*` | **Copy verbatim — no changes** |
| `face/esp32/main/inference.*` | `keywords/esp32/main/inference.*` | Arena size 80KB, input (1,48,48,1), output (1,4) |
| `face/esp32/main/preprocess.*` | *(new)* | RGB565 center-crop→area-avg→grayscale→normalize |
| `face/esp32/main/main.cpp` | *(new)* | Integrate camera + preprocess + inference + LED + serial |

---

## 10. Verification Checklist / 验证清单

- [ ] `python main.py` reports test accuracy ≥ 85%; confusion matrix diagonal dominant
- [ ] TFLite evaluation matches float model accuracy within 2%
- [ ] `idf.py build` succeeds with no errors
- [ ] `test.cpp` passes: preprocessing and inference match `test_case.h` within tolerance
- [ ] `python display.py --port <port>`: team member face → green border + name; stranger/photo → red border + "Unknown"
- [ ] LED lights for members, off for unknown
- [ ] **[Bonus]** Printed photo → "DENIED" even if face looks similar (liveness gate)
- [ ] **[Bonus]** Access log CSV written correctly
