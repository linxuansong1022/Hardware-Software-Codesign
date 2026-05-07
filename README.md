# Open-Set Face Recognition on ESP32-S3

This repository contains the final DTU 02214 Hardware/Software Codesign project.
The application is an embedded face-access prototype running on the XIAO
ESP32-S3 Sense. It recognizes three enrolled users and rejects unknown inputs
using a compact int8 CNN, embedding-distance rejection, class agreement, and
multi-frame voting.

## Repository Structure

```text
.
├── face/
│   ├── esp32/        ESP-IDF firmware for the XIAO ESP32-S3 Sense
│   ├── frontend/     React/Vite browser dashboard using Web Serial
│   ├── python/       ESP32 camera data collection utility
│   ├── data/         Dataset collection note and logs; raw images are not included
│   ├── deployment_metrics.json
│   └── int8_deployment_eval.json
├── ml_share/
│   ├── ml_scripts/   Training, evaluation, quantization, and export scripts
│   └── data/         Metadata manifests and data split files
└── README.md
```

The final project code is split into two main parts:

- `face/` contains the embedded application, browser interface, and face-project
  utilities.
- `ml_share/` contains the machine-learning pipeline used to build and evaluate
  the deployed model.

Raw face images are intentionally not included in the final repository.
The `ml_share/models/` directory is generated locally by the training and
export scripts and is also not submitted.

## System Overview

The deployed ESP32-S3 pipeline is:

```text
OV2640 camera frame
-> center crop and grayscale resize to 48x48x1
-> int8 TensorFlow Lite Micro CNN inference
-> softmax confidence check
-> 32-D embedding distance check against enrolled-user centroids
-> class agreement check
-> 5-frame voting
-> LED, serial log, and browser dashboard output
```

The model is trained as a three-class classifier for:

```text
person_a
person_b
person_c
```

Unknown inputs are not trained as a normal fourth class in the final model.
Instead, unknown rejection is handled by the deployment-time rejection logic.

## Main Results

The final int8 TensorFlow Lite model keeps strong closed-set accuracy while
adding conservative rejection behavior for unknown inputs.

| Metric | Value |
|---|---:|
| INT8 closed-set test accuracy | 98.70% |
| Test images | 77 |
| Local unknown evaluation images | 259 |
| Softmax-only false accepts at threshold 0.99 | 45/259 |
| Two-stage rejection false accepts | 11/259 |
| TFLite model size | 29.7 KB |
| Firmware app binary size | 395.2 KB |
| App partition used | 38.6% |

Typical live ESP32-S3 timing from serial logs:

| Runtime metric | Typical value |
|---|---:|
| Capture latency | about 137-142 ms |
| Preprocessing latency | about 20 ms |
| Inference latency | about 29 ms |
| Total loop latency | about 189-195 ms |
| Effective frame rate | about 5.1-5.3 FPS |

## Machine-Learning Pipeline

The main scripts are in `ml_share/ml_scripts/`.

```text
1build_image_index.py
4build_image_manifest.py
5split_holdout_val_test.py
7_1train_gray48_cnn.py
7_2train_gray48_cnn_aug.py
7_3train_center_loss.py
9_eval_embedding_distance.py
10quantize_gray48_cnn_center_int8.py
11eval_gray48_cnn_center_int8.py
12export_tflite_to_c_array.py
13_collect_deployment_metrics.py
```

The intended order is:

```text
build image index
-> build manifest
-> split train/validation/test
-> train compact grayscale CNN
-> train augmented CNN
-> train center-loss CNN
-> evaluate embedding-distance rejection
-> quantize to int8 TFLite
-> evaluate int8 deployment model
-> export model_data.c/model_data.h for ESP32 firmware
```

The deployed model expects a `48x48x1` grayscale input and produces two useful
outputs:

- a three-class softmax output;
- a 32-dimensional embedding used for centroid-distance rejection.

## Rebuild and Evaluate the Model

Run from the repository root:

```bash
python3 ml_share/ml_scripts/10quantize_gray48_cnn_center_int8.py
python3 ml_share/ml_scripts/11eval_gray48_cnn_center_int8.py
python3 ml_share/ml_scripts/12export_tflite_to_c_array.py
```

Use a Python environment with TensorFlow and the required ML packages installed.

The export script generates firmware model files under the ignored local
`ml_share/models/` directory:

```text
ml_share/models/gray48_cnn_center_w001_int8/c_export/model_data.c
ml_share/models/gray48_cnn_center_w001_int8/c_export/model_data.h
```

For the submitted firmware, the deployed copy is already placed in:

```text
face/esp32/main/model_data.c
face/esp32/main/model_data.h
```

## Build and Flash the ESP32 Firmware

Run from the repository root:

```bash
cd face/esp32
source /path/to/esp-idf/export.sh
idf.py build
```

Flash and monitor:

```bash
idf.py -p /dev/cu.usbmodemXXXX flash monitor
```

Replace `/path/to/esp-idf/export.sh` with the local ESP-IDF installation path.
On macOS, check the available serial ports with:

```bash
ls /dev/cu.*
```

Exit the ESP-IDF monitor with:

```text
Ctrl+]
```

Close `idf.py monitor` before opening the browser frontend, because only one
program can use the USB serial port at a time.

## Run the Frontend

The frontend is in `face/frontend/`.

```bash
cd face/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 4174
```

Open the Vite URL in Chrome or Edge. The browser connects to the ESP32 through
the Web Serial API. After connecting, the page sends `START_STREAM` and displays
the camera frame, class probabilities, frame result, vote result, rejection
status, and timing logs.

## ESP32 Firmware Structure

```text
face/esp32/main/
├── main.cpp              Runtime loop, rejection gates, voting, logs, LED output
├── camera.cpp/.h         OV2640 camera setup and frame capture
├── preprocess.cpp/.h     RGB565 center crop, grayscale conversion, 48x48 resize
├── inference.cpp/.h      TFLite Micro setup, int8 input, inference, dequantization
├── centroids_config.h    Class names, 32-D centroids, thresholds
├── model_data.c/.h       Quantized TFLite model compiled as a C array
├── test.cpp/.h           Small test helpers
└── CMakeLists.txt
```

## Frontend Structure

```text
face/frontend/
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── App.jsx
    ├── main.jsx
    └── index.css
```

## Notes for Final Submission

- Raw face images are excluded.
- The repository is intended to contain source code, configuration, scripts,
  and small metadata/evaluation files.
- Generated build directories should not be submitted.
- The written project report is submitted separately as a PDF.
