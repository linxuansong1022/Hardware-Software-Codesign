# Deployment Metrics

## Build Context

- Project: face-recognition
- Target: esp32s3
- Model format: INT8 TFLite
- Input shape: 48x48x1 grayscale

## Size Metrics

| Metric | Value |
|---|---:|
| TFLite model size | 29.7 KB |
| Firmware app binary size | 395.2 KB |
| Firmware ELF size | 10.7 MB |
| Bootloader size | 20.6 KB |
| Partition table size | 3.0 KB |
| App partition size | 1.0 MB |
| Free app partition | 628.8 KB |
| App partition used | 38.6% |

## Runtime Metrics To Fill From Serial Logs

| Metric | Value |
|---|---:|
| Capture latency | TBD ms |
| Preprocess latency | TBD ms |
| Inference latency | TBD ms |
| Total frame latency | TBD ms |
| Approximate FPS | TBD |

## Source Paths

- TFLite: `/Users/songlinxuan/Desktop/DTU-02214/ml_share/models/gray48_cnn_center_w001_int8/gray48_cnn_center_int8.tflite`
- App binary: `/Users/songlinxuan/Desktop/DTU-02214/face/esp32/build/face-recognition.bin`
- Build directory: `/Users/songlinxuan/Desktop/DTU-02214/face/esp32/build`
