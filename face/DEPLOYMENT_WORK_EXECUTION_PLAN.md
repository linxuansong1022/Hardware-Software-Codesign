# Deployment-Side Work Execution Plan

This plan lists only the deployment-side work that is useful and realistic for our current project deadline. It intentionally does not include pruning, custom SIMD kernels, or NPU acceleration, because these are low-value for our current model and time budget.

## Goal

Strengthen the embedded deployment part of the project by evaluating and improving the model after conversion to ESP32-S3 format.

The goal is not to completely solve unknown recognition through hardware optimization. The goal is to show that:

1. The model can be deployed on ESP32-S3.
2. INT8 quantization keeps the model small and runnable.
3. Quantization does not significantly damage closed-set accuracy.
4. Remaining errors are mainly caused by open-set rejection and camera/preprocessing mismatch.
5. The final system is evaluated as an embedded ML system, not only as an offline classifier.

## Work Package 1: Deployment Metrics

### Purpose

Measure the embedded cost of the deployed system. This connects directly to the course material on neural network optimization and hardware/software co-design.

### What to Measure

- TFLite model size.
- ESP32 firmware binary size.
- Flash usage.
- Free app partition space.
- Inference latency per frame.
- Approximate FPS.
- RAM/tensor arena usage if available from logs.

### How to Do It

1. Check the deployed TFLite model size:

```bash
ls -lh ml_share/models/gray48_cnn_center_w001_int8/gray48_cnn_center_int8.tflite
```

2. Build the ESP32 firmware:

```bash
cd face/esp32
source /Users/songlinxuan/esp/esp-idf/export.sh
idf.py build
```

3. Record the build output:

The build output prints the firmware binary size and free partition space, for example:

```text
face-recognition.bin binary size ...
Smallest app partition is ...
... bytes free
```

4. Add or use timing logs around inference on ESP32:

```cpp
int64_t t0 = esp_timer_get_time();
bool ok = inference_predict(input, prediction, embedding);
int64_t t1 = esp_timer_get_time();
ESP_LOGI(TAG, "inference_time_ms=%lld", (t1 - t0) / 1000);
```

5. Run serial monitor and record several inference times:

```bash
idf.py -p /dev/cu.usbmodemXXXX monitor
```

### Expected Output

A table for the report:

| Metric | Value |
|---|---:|
| Input size | 48x48 grayscale |
| TFLite model format | INT8 |
| TFLite model size | TBD |
| Firmware binary size | TBD |
| Free app partition | TBD |
| Inference latency | TBD ms |
| Approximate FPS | TBD |

## Work Package 2: Float Model vs INT8 Model Evaluation

### Purpose

Show whether deployment conversion changes model accuracy.

This is the most important deployment evaluation:

- Before deployment: float/Keras model.
- After deployment conversion: INT8 TFLite model.

### What to Compare

- Closed-set accuracy on person A/B/C test images.
- Confusion matrix.
- Unknown rejection with softmax threshold.
- Unknown rejection with the final two-stage rejection logic.

### How to Do It

1. Run the float model evaluation script if available.

Expected command pattern:

```bash
keywords/python/.venv/bin/python ml_share/ml_scripts/9_eval_embedding_distance.py
```

2. Run the INT8 deployment evaluation:

```bash
keywords/python/.venv/bin/python ml_share/ml_scripts/11eval_gray48_cnn_center_int8.py
```

3. If evaluating external LFW unknown holdout:

```bash
keywords/python/.venv/bin/python ml_share/ml_scripts/11eval_gray48_cnn_center_int8.py --unknown-dir face/data/unknown_holdout_lfw
```

### Expected Output

A report table:

| Evaluation stage | Model | Known accuracy / accept | Unknown false accept | Comment |
|---|---|---:|---:|---|
| Before deployment | Float model | TBD | TBD | Baseline model |
| After deployment | INT8 TFLite | TBD | TBD | Quantized model |
| Final rejection logic | INT8 + gates + voting | TBD | TBD | Deployed decision rule |

### How to Interpret

If INT8 closed-set accuracy is close to the float model, quantization is not the main source of error.

If unknown false accepts remain high, the issue is open-set recognition, not deployment conversion alone.

## Work Package 3: Deployment-Aware Preprocessing Alignment

### Purpose

Reduce mismatch between training images and ESP32 camera input.

This is likely more important for real-world behavior than additional model compression.

### Problem

The model is trained on preprocessed images, but deployment uses live ESP32 camera frames. Accuracy can drop if the following differ:

- Face position.
- Crop area.
- Brightness.
- Contrast.
- Resize method.
- Grayscale conversion.
- Distance from camera.

### What to Do

1. Document the Python preprocessing pipeline:

- Input image source.
- Face crop or center crop.
- Resize to 48x48.
- Grayscale conversion.
- Normalization / quantization.

2. Document the ESP32 preprocessing pipeline:

- Camera frame format.
- Crop policy.
- Resize method.
- Grayscale conversion.
- Input tensor quantization.

3. Make both pipelines as close as possible.

4. For remote teammate data:

- Ask for original phone photos, not manually cropped 48x48 images.
- Use the same preprocessing script to crop/resize them.
- Add the processed images to the corresponding known-person class.

### Expected Output

A report paragraph:

```text
To reduce deployment mismatch, all training and remote-collected images were processed through the same 48x48 grayscale pipeline used by the embedded model. This improves consistency between the offline dataset and the ESP32 camera input.
```

Optional table:

| Step | Python training pipeline | ESP32 deployment pipeline | Matched? |
|---|---|---|---|
| Grayscale | TBD | TBD | TBD |
| Crop | TBD | TBD | TBD |
| Resize | 48x48 | 48x48 | Yes |
| Quantization | INT8 calibration | INT8 input tensor | Yes |

## Work Package 4: On-Device Behavior Test

### Purpose

Check whether offline deployment evaluation matches real ESP32 behavior.

Offline INT8 evaluation is useful, but the final system uses a live camera. We need a small manual test to understand real behavior.

### Test Setup

Use the ESP32 serial monitor:

```bash
cd face/esp32
source /Users/songlinxuan/esp/esp-idf/export.sh
idf.py -p /dev/cu.usbmodemXXXX monitor
```

### Test Cases

Test under stable lighting:

- Person A in front of camera.
- Person B in front of camera.
- Person C in front of camera.
- Unknown person.
- Empty/background if relevant.

For each case, record around 20-30 log lines.

### What to Record

From serial output:

```text
Frame: UNKNOWN/person_x
Vote: UNKNOWN/person_x
softmax probabilities [A=... B=... C=...]
dist_sq=...
nearest=...
gates[S=... D=... C=...]
```

### How to Interpret

- If `S=0`, softmax confidence is too low.
- If `D=0`, embedding distance is too far from known centroids.
- If `C=0`, softmax class and nearest centroid disagree.
- If single frames are correct but `Vote` stays unknown, voting may be too strict.
- If unknown often passes all gates, thresholds are too loose.

### Expected Output

A small table:

| Test subject | Expected | Observed vote | Main failure gate | Comment |
|---|---|---|---|---|
| Person A | person_a | TBD | TBD | TBD |
| Person B | person_b | TBD | TBD | TBD |
| Person C | person_c | TBD | TBD | TBD |
| Unknown | UNKNOWN | TBD | TBD | TBD |

## Work Package 5: Optional QAT vs PTQ Experiment

### Purpose

Try one additional course-related optimization technique if time remains.

QAT means quantization-aware training. It simulates quantization during training, so the final INT8 model may become more robust.

### When to Do It

Only do this after Work Packages 1-4 are done.

### What to Compare

- PTQ INT8 model.
- QAT INT8 model.

Metrics:

- Closed-set accuracy.
- Known accept rate.
- Unknown false accept rate.
- Model size.

### Expected Result

QAT may improve INT8 stability slightly, but it may not solve unknown rejection. If QAT does not improve unknown performance, that is still a valid result because unknown rejection is mainly an open-set problem.

## Things We Should Not Do Now

### Do Not Prioritize Pruning

The model is already small. Pruning adds training complexity and may not make inference faster unless the runtime supports sparse kernels.

### Do Not Write Custom SIMD Kernels

ESP32-S3 supports vector-style acceleration, but writing custom kernels is too risky for the deadline. It is enough to explain that INT8 operators are hardware-friendly.

### Do Not Claim NPU Acceleration

ESP32-S3 does not provide an NPU for this project. We should mention NPU only as a course concept, not as something we used.

### Do Not Train Unknown as a Fourth Class as the Main Fix

Unknown is too broad to represent as one class. A fourth-class model can still fail on unseen unknown people. Our current open-set rejection method is more appropriate.

## Final Recommended Order

1. Measure model size and firmware size.
2. Run INT8 deployment evaluation and save the numbers.
3. Compare float vs INT8 if float evaluation is available.
4. Add inference timing logs on ESP32.
5. Run a small on-device serial test for known and unknown faces.
6. Document preprocessing alignment.
7. Optionally run QAT if there is still time.

## Report Sections This Supports

This work can be used in:

- Method: model compression and open-set rejection.
- Training and Quantization: PTQ, representative calibration, INT8 conversion.
- Embedded Deployment: ESP32-S3, TFLite Micro, model size, latency, memory.
- Experimental Results: float vs INT8, unknown rejection, on-device test.
- Discussion: deployment mismatch, limitations, and why unknown remains difficult.

