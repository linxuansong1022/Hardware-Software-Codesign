# Face Recognition Door Access System — Developer Specification

**Course**: DTU 02214 — Embedded AI (Spring 2026)
**Team**: 3 members
**Hardware**: XIAO ESP32S3 Sense (OV2640 camera, 8MB PSRAM, 8MB Flash, onboard LED)
**Date**: 2026-02

---

## 1. Application Description

### 1.1 Purpose

A real-time face recognition system running entirely on the XIAO ESP32S3 Sense microcontroller, designed as a door access control prototype. The system grants access to registered team members and denies all others.

### 1.2 Scope

- **Environment**: Indoor only (lab, office, hallway)
- **Users**: 3 registered team members + all others classified as "Unknown"
- **Distance**: 30–100 cm from camera
- **Output**: Onboard LED feedback + live Python display with recognition overlay via USB serial

### 1.3 Functional Requirements

| ID | Requirement |
|:--:|:------------|
| FR-1 | The system captures frames from the OV2640 camera at ≥ 1 FPS |
| FR-2 | The system classifies each frame into one of 4 classes: Person A, Person B, Person C, or Unknown |
| FR-3 | When a team member is recognized with confidence ≥ 0.80, the onboard LED turns on (green) |
| FR-4 | When the face is Unknown or confidence < 0.80, the LED remains off |
| FR-5 | Each frame and inference result is sent to the host PC via USB serial |
| FR-6 | The Python display app shows the live camera feed with the recognized name and confidence bars |

### 1.4 Non-Functional Requirements

| ID | Metric | Target | Rationale |
|:--:|:-------|:------:|:----------|
| NFR-1 | Test accuracy | ≥ 90% | 4-class task with self-collected data; distributions are close, 90% is a reasonable baseline |
| NFR-2 | Per-class recall | ≥ 85% | Prevents systematic misclassification of any one person |
| NFR-3 | Inference latency | ≤ 200 ms | ESP32S3 @ 240 MHz with INT8 CNN; expected ~50–100 ms |
| NFR-4 | Model binary size | ≤ 50 KB | Fits comfortably in 8 MB Flash |
| NFR-5 | Tensor arena size | ≤ 100 KB | Must fit in ~512 KB SRAM alongside other buffers |
| NFR-6 | Frame rate (end-to-end) | ≥ 1 FPS | Adequate for door access; not a real-time video scenario |
| NFR-7 | INT8 quantization loss | ≤ 2% accuracy drop | Quantized model must remain close to float baseline |

### 1.5 How It Works

```
┌─────────────────────────────────────────────────────┐
│                  XIAO ESP32S3 Sense                 │
│                                                     │
│  OV2640 Camera                                      │
│       │ 320×240 RGB565                              │
│       ▼                                             │
│  Center-crop + Downsample (area avg)                │
│       │ 64×64 grayscale float                       │
│       ▼                                             │
│  TFLite Micro (INT8 CNN, ~40 KB model)              │
│       │ 4-class softmax                             │
│       ▼                                             │
│  Person A / B / C / Unknown                         │
│       │                                             │
│  LED (green = member, off = unknown)                │
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
└─────────────────────────────────────────────────────┘
```

---

## 2. Dataset Specification

### 2.1 Classes and Labeling Guidelines

| Class Index | Class Name | Description |
|:-----------:|:----------:|:------------|
| 0 | Person A | Team member 1 — full frontal face in frame |
| 1 | Person B | Team member 2 — full frontal face in frame |
| 2 | Person C | Team member 3 — full frontal face in frame |
| 3 | Unknown  | Any face not belonging to team members, or no face |

**Labeling rules**:
- **Person X** (0/1/2): the face occupies ≥ 30% of the image area and is clearly recognizable. Includes glasses, slight head tilt (< 45°), mild occlusion.
- **Unknown** (3): any other person's face, empty backgrounds, objects, printed photos, faces with > 45° tilt, or ambiguous identity.
- **Discard**: motion-blurred images or faces < 15% of image area.
- **Edge case**: two people in frame simultaneously → label as Unknown unless one team member clearly dominates center.

### 2.2 Conditional Factors

| Factor | Variations |
|:-------|:-----------|
| Distance | 30 cm, 60 cm, 100 cm |
| Angle | Frontal (0°), left 30°, right 30°, slight up/down tilt |
| Lighting | Window daylight, overhead fluorescent, low-light (lamp only), backlit |
| Expression | Neutral, slight smile, talking |
| Accessories | No glasses, glasses, sunglasses (some shots) |
| Background | Lab desk, wall, outdoors |
| Unknown variations | Empty desk, empty wall, outdoors scene, strangers (consented), partially visible faces, printed photos |

### 2.3 Collection Strategy

**Target**: 200–250 images × 4 classes = **800–1000 images total**

**Per-person session** (each member self-collects, target ~220 images):
- Session A: frontal, neutral, no accessories — 50 images × 3 distances (30/60/100 cm)
- Session B: angled (±30°), slight up/down tilt, varied expressions — 40 images
- Session C: different lighting conditions (daylight, fluorescent, low-light, backlit) — 40 images
- Session D: with glasses/accessories — 30 images
- Session E: varied backgrounds (lab, wall, outdoors) — 30 images
- Session F: talking/dynamic expressions — 30 images

**Unknown class** (target ~220 images):
- 90 images (~40%): strangers' frontal faces (in-person, with verbal consent)
- 35 images (~15%): face photos displayed on tablet/phone screen (for anti-spoofing training)
- 35 images (~15%): partially occluded / edge-of-frame faces
- 35 images (~15%): empty backgrounds (desk, wall, outdoors)
- 25 images (~15%): non-face objects (hands, books, cups, etc. — to prevent false triggers)

**Data quality requirements**:
- Class counts must not differ by more than 20% (minimum 200 images per class)
- Each member's data must cover all conditional factor combinations; avoid "all frontal, daylight" bias
- Frames from the same continuous capture session must not appear in both train and test sets (split by session, not random shuffle)

**Balance check**: verify counts with `ls face/data/<class>/ | wc -l` before training.

### 2.4 Data Source and Format

- **Camera**: OV2640 on XIAO ESP32S3 Sense — 320×240 RGB565, saved as PNG via `collect.py`
- **No public datasets used** — all images captured in-session for privacy compliance
- **Camera settings**: fixed at 320×240, ~1 FPS capture rate, no manual focus adjustment
- **Filename format**: `image_YYYYMMDD_HHMMSS.png` (auto-generated)

### 2.5 Session Log

Each collection session is recorded in `data/session_log.csv`:

| Field | Description |
|:------|:------------|
| `timestamp` | Session start datetime |
| `class` | person_a / person_b / person_c / unknown |
| `collector` | Name of person who collected this batch |
| `lighting` | daylight / overhead / low / backlit |
| `accessories` | none / glasses / sunglasses |
| `background` | lab / wall / outdoors |
| `notes` | Any deviations or special conditions |

### 2.6 Privacy and Legal Considerations

- **Team members**: all 3 give explicit informed consent. Images stored locally only.
- **Strangers in Unknown class**: verbal consent obtained before capturing. Images deleted after project completion.
- **Screen-displayed photos**: face images from the internet shown on a tablet/phone screen, used only for anti-spoofing training.
- **GDPR compliance**: data used solely for educational purposes. All third-party images deleted upon completion.

---

## 3. Design Space and Justification

This section presents the main design dimensions, the candidate options considered, and the rationale for the current recommendation. The objective is not only to state a final design, but also to show why that design is a reasonable trade-off for the ESP32-S3 resource budget and the face-recognition task.

本节展示系统的主要设计维度、考虑过的候选方案以及当前推荐决策的依据。重点不只是给出最终选择，还要说明为什么该选择适合 ESP32-S3 的资源约束，以及为什么它适合“识别组员并拒绝其他人”的任务。

### 3.1 Input Resolution

| Option | Model Input Size | Tensor Arena (est.) | Pros | Cons |
|:-------|:----------------:|:-------------------:|:-----|:-----|
| 32×32 | 1,024 B | ~40 KB | Smallest model, fastest inference | Loses fine facial detail; likely underfits |
| 48×48 | 2,304 B | ~80 KB | Low cost; easier to fit | Some identity cues may be too compressed |
| **64×64** | 4,096 B | ~100-140 KB | Better eye, nose, and face-shape detail | Higher SRAM and latency cost |
| 96×96 | 9,216 B | ~200 KB+ | Strong detail preservation; common in some TFLM examples | Too expensive for this project budget |

**Recommended choice: 64×64 grayscale**.

**English**: 32×32 is likely too aggressive for identity recognition, because small geometric differences between team members are easily lost. 96×96 is feasible for simple microcontroller vision tasks, but it is unnecessarily expensive for this closed-set 4-class problem. 64×64 is therefore the recommended starting point: it preserves substantially more facial structure than 48×48 while remaining realistic for ESP32-S3 deployment. If board-side measurements later show unacceptable memory or latency cost, 48×48 remains the fallback option.

**中文**：32×32 对身份识别来说过于激进，因为组员之间的细微几何差异很容易被压缩掉。96×96 虽然在一些微控制器视觉任务中可以运行，但对本项目这种封闭集 4 分类任务来说开销偏大。因此当前推荐从 64×64 灰度输入开始：它比 48×48 保留更多眼睛、鼻梁、脸型等结构信息，同时资源消耗仍然有希望控制在 ESP32-S3 可接受范围内。如果后续实测发现内存或时延压力过大，再回退到 48×48 作为备选。

### 3.2 Color Channels

| Option | Input Shape | First-Layer Compute | Pros | Cons |
|:-------|:----------:|:-------------------:|:-----|:-----|
| **Grayscale** | (64, 64, 1) | 1× | Smaller model, simpler preprocessing, lower memory | Loses color information |
| RGB | (64, 64, 3) | 3× | Richer features | 3× first-layer compute; more sensitive to lighting |

**Recommended choice: Grayscale**.

**English**: For this project, identity is expected to depend mainly on facial geometry rather than color. RGB would triple the first-layer input cost and increase both memory traffic and preprocessing complexity on the ESP32-S3. Since indoor lighting variation makes color unstable anyway, grayscale is the better engineering trade-off.

**中文**：对本项目而言，身份信息主要来自人脸几何结构，而不是颜色。RGB 会使第一层输入开销增加到 3 倍，同时提升内存访问和预处理复杂度；而室内光照变化又会让颜色特征变得不稳定。因此，灰度输入是更合理的工程取舍。

### 3.3 Model Architecture Depth

| Option | Params (est.) | Model Size (INT8) | Arena (est.) | Notes |
|:-------|:-------------:|:-----------------:|:------------:|:------|
| 2-layer Conv | ~8,000 | ~15 KB | ~50-70 KB | Fast, but likely too weak for identity discrimination |
| **3-layer Conv** | ~25,000-35,000 | ~35-55 KB | ~100-140 KB | Best baseline for 64×64 input |
| 4-layer Conv | ~50,000+ | ~70 KB+ | ~150 KB+ | More capacity, but larger activation cost |
| MobileNet-style | ~30,000 | ~45 KB | ~90-130 KB | Efficient in principle, but more complex to tune and justify |

**Recommended choice: 3-layer Conv2D + Dense**.

**English**: A 2-layer CNN is attractive from a latency perspective, but it is likely too small for distinguishing three visually similar identities plus an Unknown class. A 4-layer network or a larger mobile-style model would provide more capacity, but the extra activation memory and implementation complexity are difficult to justify unless the smaller model clearly fails. A 3-layer CNN is therefore the recommended baseline: it is expressive enough for a small face-classification task while remaining straightforward to deploy with TFLite Micro.

**中文**：2 层卷积网络虽然时延更低，但对于“三个相似身份 + Unknown”这个任务，表达能力可能不足。4 层卷积网络或更大的 MobileNet 风格模型虽然容量更强，但 activation 内存和实现复杂度也明显增加，除非小模型实测效果明显不够，否则很难 justify。综合来看，3 层卷积网络是当前最推荐的基线：表达能力足够，同时部署复杂度仍然可控。

### 3.4 Quantization Strategy

| Option | Model Size | Inference Speed | Accuracy Impact |
|:-------|:----------:|:---------------:|:----------------|
| Float32 (no quant) | ~90 KB | Slower (float ops on integer core) | Baseline |
| **Full INT8** | ~40 KB | Fastest (native int8 on ESP32S3) | Typically ≤ 2% drop |
| Mixed (INT8 weights, float activations) | ~50 KB | Medium | ~1% drop |

**Recommended choice: Full INT8 quantization**.

**English**: The ESP32-S3 is far better suited to integer inference than full float32 inference. INT8 quantization reduces flash usage, shrinks tensor sizes, and improves speed. Any accuracy loss must be measured experimentally, but for this application the expected trade-off is acceptable.

**中文**：ESP32-S3 更适合整数推理而不是完整的 float32 推理。INT8 量化可以降低 Flash 占用、缩小张量尺寸并提升速度。量化带来的精度损失需要后续实测，但对本项目来说，这样的代价通常是可以接受的。

### 3.5 Downsampling Method (ESP32 side)

| Option | Implementation Complexity | Quality |
|:-------|:-------------------------:|:--------|
| Nearest-neighbor | Very simple | Aliasing artifacts, noisy output |
| Bilinear interpolation | Moderate (float math per pixel) | Smooth but computationally heavier |
| **Center-crop + area average** | Simple (integer accumulation) | Good noise suppression; deterministic implementation |

**Recommended choice: center-crop followed by integer area averaging**.

**English**: On-device preprocessing should stay simple and reproducible. Nearest-neighbor is cheap but produces unstable inputs, while bilinear interpolation is smoother but more expensive. For a 64×64 input, the practical solution is to center-crop the frame to a square region and then downsample using integer block averaging or a close equivalent. This keeps the embedded implementation deterministic and also suppresses high-frequency noise.

**中文**：设备端预处理应尽量简单、稳定、可复现。最近邻虽然便宜，但输入不稳定；双线性插值更平滑，但实现和计算都更重。对于 64×64 输入，更合适的做法是先中心裁剪出正方形区域，再用整数块平均或等价的低成本方法下采样。这样既便于嵌入式实现，也能抑制高频噪声。

### 3.6 Output Formulation

| Option | Output | Pros | Cons |
|:-------|:-------|:-----|:-----|
| **4-class softmax** | A / B / C / Unknown | Simplest training and firmware logic | Unknown class is heterogeneous |
| 3-class + threshold reject | A / B / C, reject if max score too low | More natural reject mechanism | Threshold tuning becomes critical |
| Embedding + distance threshold | Feature vector + template matching | Closer to real face recognition | Harder training and evaluation pipeline |

**Recommended choice: 4-class softmax as the project baseline**.

**English**: A threshold-based reject mechanism is theoretically more flexible, but a 4-class softmax formulation is the most practical choice for the first working system. It keeps the training objective simple, the firmware logic minimal, and the end-to-end implementation achievable within the project scope. A threshold-based reject mechanism can still be evaluated later as an advanced comparison.

**中文**：从理论上讲，基于阈值的拒识机制更灵活，但作为第一版可交付系统，4 类 softmax 更实际。它训练目标直接，固件逻辑简单，整体实现难度也更适合课程项目周期。后续如果时间允许，再把“3 类身份 + 阈值拒识”作为进阶比较方案会更稳妥。

### 3.7 Summary Table

| Design Dimension | Chosen | Alternatives Rejected | Key Rationale |
|:-----------------|:-------|:----------------------|:--------------|
| Input resolution | 64×64 | 32×32, 48×48, 96×96 | Better identity detail without 96×96 cost |
| Color channels | Grayscale | RGB | Lower compute and more stable under lighting variation |
| Model depth | 3-layer Conv | 2-layer, 4-layer, MobileNet | Best baseline trade-off for capacity vs. deployability |
| Output formulation | 4-class softmax | 3-class + threshold, embedding | Simplest end-to-end baseline |
| Quantization | Full INT8 | Float32, mixed | Smaller and faster on ESP32-S3 |
| Downsampling | Center-crop + area avg | Nearest, bilinear | Low-cost, deterministic, and robust |

---

## 4. Machine Learning Pipeline

### 4.1 Preprocessing Pipeline

**Python (training)**:
```
PNG image (320×240 or any size)
    → center-crop to square region
    → convert to grayscale (PIL .convert('L'))
    → resize to 64×64
    → normalize: pixel / 255.0
    → z-score: (x - IMAGE_MEAN) / IMAGE_STD
    → output shape: (64, 64, 1) float32
```

**ESP32 (inference)**:
```
320×240 RGB565 raw frame
    → center-crop square region
    → integer area-average downsample → 64×64
    → RGB565 → grayscale: Y = (5R + 9G + 2B) >> 4
    → normalize: (Y / 255.0 - IMAGE_MEAN) / IMAGE_STD
    → output: float[64×64]
```

**Alignment**: both pipelines must produce numerically equivalent output for the same input image. This is verified by `test_case.h` (see Section 7).

### 4.2 Model Architecture

**Framework**: TensorFlow 2.x / Keras → TFLite INT8 → TFLite Micro

```
Input (64, 64, 1)
    │
Conv2D(8, 3×3, relu)    → (62, 62, 8)
MaxPooling2D(2×2)        → (31, 31, 8)
Dropout(0.1)
    │
Conv2D(16, 3×3, relu)   → (29, 29, 16)
MaxPooling2D(2×2)        → (14, 14, 16)
Dropout(0.1)
    │
Conv2D(32, 3×3, relu)   → (12, 12, 32)
MaxPooling2D(2×2)        → (6, 6, 32)
Dropout(0.1)
    │
Flatten                  → (1152)
Dense(32, relu)          → (32)
Dense(4, softmax)        → (4)  ← Person A / B / C / Unknown

Total params: ~42,000
TFLite INT8 model size: ~45-60 KB
```

**Op resolver**: `Conv2D`, `MaxPool2D`, `Flatten`, `FullyConnected`, `Softmax` — the same set as the keywords project. Zero new TFLite ops required.

### 4.3 Training Configuration

| Parameter | Value |
|:----------|:------|
| Optimizer | Adam (lr = 0.001) |
| Loss | sparse_categorical_crossentropy |
| Epochs | 100 with EarlyStopping (patience = 20, monitor = val_loss) |
| Batch size | 32 |
| Data split | 70% train / 15% val / 15% test |
| Data augmentation | Horizontal flip, ±15° rotation, brightness [0.7, 1.3], zoom 0.1 |

### 4.4 Quantization and Export

- Full INT8 quantization with representative dataset (200 training samples)
- Input/output tensors both INT8
- Export: `model.c` + `model.h` via `write_model_c_file()` utility (reused from keywords project)
- Validation: evaluate TFLite INT8 model on test set; compare accuracy with float baseline

---

## 5. ESP32 Firmware Design

### 5.1 File Structure

```
face/esp32/main/
├── main.cpp          # application loop: camera → preprocess → inference → LED → serial
├── camera.h/.cpp     # COPY from camera/esp32/main/ (verbatim)
├── preprocess.h/.cpp # NEW: RGB565 → 64×64 grayscale → normalize
├── inference.h/.cpp  # ADAPTED from keywords/esp32/main/ (2D input, 4 classes, ~128 KB arena target)
├── model.h/.c        # GENERATED by python/main.py
├── test_case.h       # GENERATED by python/generate_test_case.py
└── test.h/.cpp       # ADAPTED from keywords/esp32/main/ (image test vectors)
```

### 5.2 Serial Protocol

Each inference cycle sends two packets:

```
[Frame packet]
  "===FRAME===\n"               ← same preamble as camera project
  <320×240×2 bytes RGB565>      ← 153,600 bytes

[Result packet]
  "===RESULT===\n"              ← new result preamble
  <1 byte: uint8 class_index>
  <16 bytes: 4×float32 scores>  ← little-endian
```

Python parses: `cls, *scores = struct.unpack('<Bffff', data[1:18])`

### 5.3 Memory Layout

| Buffer | Size | Placement |
|:-------|-----:|:----------|
| `image_buffer[320×240×2]` | 153,600 B | PSRAM (SPIRAM) |
| `face_features[64×64]` | 16,384 B | SRAM |
| `tensor_arena[128×1024]` | 131,072 B | SRAM |
| `prediction[4]` | 16 B | SRAM |
| **Total SRAM** | **~147 KB** | Within 512 KB budget |

### 5.4 LED Logic

```c
if (max_score >= 0.80f && predicted_class < 3)
    gpio_set_level(LED_PIN, 0);   // ON (active-low) = access granted
else
    gpio_set_level(LED_PIN, 1);   // OFF = access denied
```

---

## 6. Python Application Design

### 6.1 Scripts

| Script | Purpose |
|:-------|:--------|
| `collect.py` | Data collection: receives camera feed, press 0/1/2/3 to label and save |
| `preprocess.py` | Image loading, resizing, normalization, dataset split |
| `main.py` | Training pipeline: preprocess → train → quantize → export to C |
| `generate_test_case.py` | Generates `test_case.h` for ESP32 validation |
| `display.py` | Live inference display: frame + recognition overlay |

### 6.2 Display Overlay

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
  green border = member recognized
  red border   = unknown
```

### 6.3 Key File References

| New File | Source | Change |
|:---------|:-------|:-------|
| `face/python/collect.py` | `camera/python/main.py` | Key→folder mapping, HUD label |
| `face/python/main.py` | `keywords/python/main.py` | Conv2D model, 4 classes, image input |
| `face/python/preprocess.py` | `keywords/python/preprocess.py` | PIL image load instead of audio FFT |
| `face/python/generate_test_case.py` | `keywords/python/generate_test_case.py` | Image input instead of audio |
| `face/python/display.py` | `camera/python/main.py` | + result packet parsing + overlay |
| `face/python/utils/*.py` | `keywords/python/utils/*.py` | Copy verbatim |
| `face/esp32/main/camera.*` | `camera/esp32/main/camera.*` | Copy verbatim |
| `face/esp32/main/inference.*` | `keywords/esp32/main/inference.*` | Arena target ~128 KB, input (1,64,64,1), output (1,4) |
| `face/esp32/main/preprocess.*` | *(new)* | RGB565 → 64×64 grayscale → normalize |
| `face/esp32/main/main.cpp` | *(new)* | Camera + preprocess + inference + LED + serial |

---

## 7. Verification and Test Plan

### 7.1 Level 1 — Offline Model Evaluation (Python)

| Test | Pass Criteria |
|:-----|:-------------|
| Overall test accuracy | ≥ 85% |
| Per-class recall (Person A, B, C) | Each ≥ 80% |
| Confusion matrix | Diagonal-dominant; no systematic off-diagonal cluster |
| TFLite INT8 accuracy vs. float baseline | Drop ≤ 2% |

**Metrics reported**: accuracy, per-class precision, recall, F1, confusion matrix.

### 7.2 Level 2 — On-Device Alignment (test_case.h)

| Test | Pass Criteria |
|:-----|:-------------|
| Preprocessing alignment | Python features vs. ESP32 features: element-wise error < 1e-4 |
| Quantization alignment | Python int8 vs. ESP32 int8: element-wise difference ≤ 1 |
| Inference alignment | Python prediction vs. ESP32 prediction: error < 0.05 per class |

These tests run automatically on boot via `test_pipeline()` before entering the main loop.

### 7.3 Level 3 — Real-World Functional Test

| Test Scenario | Method | Pass Criteria |
|:-------------|:-------|:-------------|
| Team member recognition | Each member stands at 30/60/100 cm, 5 trials each | LED on ≥ 80% of trials; correct name displayed |
| Stranger rejection | 3 non-members stand at 60 cm, 5 trials each | LED off 100%; "Unknown" displayed |
| Printed photo attack | Hold printed photo at 60 cm | LED off; "Unknown" displayed |
| Different lighting | Repeat member test under low-light and backlit | LED on ≥ 70% of trials |
| End-to-end latency | Measure time from frame capture to LED response | ≤ 200 ms |

Results will be recorded in a table with trial-by-trial outcomes.

---

## 8. Bonus Features

### Bonus 1: Mask/Glasses Attribute Detection
- Second tiny CNN (same 64×64 grayscale input): 3-class output (none / mask / glasses)
- Model size: ~5 KB; separate 32 KB tensor arena
- Both models share `face_features[]` — run sequentially
- Python overlay adds attribute label badge

### Bonus 2: Door Access Scenario
- Python-only changes (no firmware update)
- Debounce: access event fires when 3/5 consecutive frames agree with confidence > 0.80
- Animated "ACCESS GRANTED / DENIED" panel
- Scrolling access log written to `access_log.csv`

### Bonus 3: Liveness Detection
- Python-only; no new model needed
- Rolling buffer of last 8 frames as numpy arrays
- `is_live = np.mean(np.var(frame_stack, axis=0)[center_crop]) > THRESHOLD`
- Gate access on both `confidence > 0.80` AND `is_live == True`

---

## 9. Implementation Steps

| # | Step | Output |
|:-:|:-----|:-------|
| 1 | Flash `camera/esp32` unchanged | Board streams frames |
| 2 | Create `face/python/collect.py` | Data collection tool ready |
| 3 | Collect 600 images (all 4 classes) | `face/data/` populated |
| 4 | Create `face/python/preprocess.py` | Preprocessing verified |
| 5 | Create `face/python/main.py` | Training pipeline runs |
| 6 | Run `python main.py` | `model.c`, `model.h` generated; val acc ≥ 85% |
| 7 | Create `face/python/generate_test_case.py` | `test_case.h` generated |
| 8 | Set up `face/esp32/` project | Project compiles |
| 9 | Write `preprocess.cpp` | RGB565 → 64×64 grayscale pipeline |
| 10 | Adapt `inference.cpp` | TFLite Micro runs |
| 11 | Write `main.cpp` | Full loop runs |
| 12 | Adapt `test.cpp` | Preprocessing + inference validated on-device |
| 13 | Build, flash, monitor | LED responds correctly |
| 14 | Create `face/python/display.py` | Live overlay working end-to-end |
| 15 | [Bonus] Attribute model | Mask/glasses detected |
| 16 | [Bonus] Access log panel | Door access demo ready |
| 17 | [Bonus] Liveness buffer | Anti-spoofing active |

---

## 10. Team Member Contributions

| Member | Responsibilities |
|:-------|:----------------|
| TODO | TODO |
| TODO | TODO |
| TODO | TODO |

*(To be filled in before report submission.)*
