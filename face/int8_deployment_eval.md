# INT8 Deployment Evaluation Summary

- TFLite model: `/Users/songlinxuan/Desktop/DTU-02214/ml_share/models/gray48_cnn_center_w001_int8/gray48_cnn_center_int8.tflite`
- Data directory: `/Users/songlinxuan/Desktop/DTU-02214/face/data`
- Unknown directory: `/Users/songlinxuan/Desktop/DTU-02214/face/data/unknown`
- Test samples: 77
- Unknown samples: 259
- INT8 closed-set accuracy: 0.9870

## Softmax Threshold Sweep

| Threshold | Known accept | Unknown reject | Unknown false accept |
|---:|---:|---:|---:|
| 0.950 | 56/77 | 126/259 | 133/259 |
| 0.980 | 44/77 | 177/259 | 82/259 |
| 0.990 | 36/77 | 214/259 | 45/259 |
| 0.995 | 27/77 | 219/259 | 40/259 |
| 0.999 | 0/77 | 259/259 | 0/259 |

## Two-Stage Rejection

| Metric | Value |
|---|---:|
| Softmax threshold | 0.990000 |
| Distance threshold | 1.875593 |
| Known accept | 35/77 |
| Unknown reject | 248/259 |
| Unknown false accept | 11/259 |
| Known mean min distance | 1.7599 |
| Unknown mean min distance | 2.0923 |
