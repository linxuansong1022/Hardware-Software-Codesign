# Face Open-Set Rejection Note

## 1. Problem

Our face model recognizes `person_a`, `person_b`, and `person_c` well on the closed-set test split, but it still misclassifies many `unknown` faces as one of the three known people.

The main issue is not ordinary classification accuracy. It is open-set rejection.

## 2. Current Solution

We now use a two-stage rejection pipeline:

1. Softmax confidence check
2. Embedding distance check against class centroids
3. Class agreement check between softmax prediction and nearest centroid
4. 5-frame voting on the ESP32 side

The model itself is still a 3-class model. We do not train `unknown` as a normal output class.

## 3. What Changed

### Python side

- Added embedding-based evaluation in `ml_share/ml_scripts/9_eval_embedding_distance.py`
- Exported class centroids and deployment thresholds
- Updated INT8 quantization so calibration is class-balanced and not dominated by `person_a`
- Extended INT8 evaluation to support an external unknown folder

### ESP32 side

- Switched the deployed model to a dual-output INT8 TFLite model:
  - softmax output: 3 classes
  - embedding output: 32-dim vector
- Added centroid-distance rejection in `face/esp32/main/main.cpp`
- Kept the existing 5-frame voting logic

## 4. Data Usage

### Known people

These are used for training and closed-set evaluation:

- `face/data/person_a`
- `face/data/person_b`
- `face/data/person_c`

### Unknown data

These are used only for rejection evaluation:

- `face/data/unknown`
- `face/data/unknown_holdout_lfw`

`unknown_holdout_lfw` is the external test set from LFW and should not be used for training.

## 5. How To Run

### Rebuild the INT8 model

```bash
keywords/python/.venv/bin/python ml_share/ml_scripts/10quantize_gray48_cnn_center_int8.py
keywords/python/.venv/bin/python ml_share/ml_scripts/12export_tflite_to_c_array.py
```

### Evaluate closed-set and unknown rejection

```bash
keywords/python/.venv/bin/python ml_share/ml_scripts/11eval_gray48_cnn_center_int8.py \
  --unknown-dir face/data/unknown_holdout_lfw
```

### Generate centroid config

```bash
keywords/python/.venv/bin/python ml_share/ml_scripts/9_eval_embedding_distance.py \
  --model ml_share/models/gray48_cnn_center_w001/gray48_cnn_center.keras \
  --output-dir ml_share/models/gray48_cnn_center_w001/embedding_eval \
  --unknown-dir face/data/unknown_holdout_lfw
```

## 6. Current Result

On the current LFW holdout:

- Closed-set accuracy stays at about `98.7%`
- Softmax-only still accepts too many unknown faces
- Two-stage rejection is stricter and reduces false accepts

This is better than softmax-only rejection, but it still shows a clear precision/recall tradeoff. The system is now more suitable for a door-access style deployment, where false accept is more dangerous than false reject.

## 7. Notes

- The deployed threshold values are exported into `face/esp32/main/centroids_config.h`
- The deployed INT8 model is exported into `face/esp32/main/model_data.c` and `model_data.h`
- ESP32 build verification was limited by the local environment, not by the Python pipeline
