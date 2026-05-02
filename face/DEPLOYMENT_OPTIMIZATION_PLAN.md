# Deployment-Side Optimization Plan

This note summarizes how the course material on neural network optimization and hardware acceleration can be connected to our ESP32-S3 face recognition project.

## 1. Why Deployment Optimization Matters

Our project is not only a face recognition model. It is an embedded machine learning system that must run on an ESP32-S3 microcontroller. This means the model must fit within limited flash, RAM, and compute resources while still producing stable recognition results from a real camera.

The unknown-face problem is mainly an open-set recognition problem, not a pure hardware acceleration problem. However, deployment can still affect recognition quality through:

- INT8 quantization error.
- Differences between Python preprocessing and ESP32 camera preprocessing.
- Lighting, camera noise, face position, and image crop differences.
- Limited frame rate and unstable single-frame predictions.

Therefore, the deployment work should be presented as an important part of the system design, while being clear that hardware optimization itself does not directly solve unknown rejection.

## 2. Techniques From the Course Material

The neural network optimization and hardware acceleration material discusses several deployment-related techniques:

| Technique | Purpose | Used in our project? | Comment |
|---|---|---|---|
| Small model design | Reduce compute, memory, and latency | Yes | We use a compact CNN and 48x48 grayscale input. |
| Post-training quantization (PTQ) | Convert a trained float model to INT8 | Yes | This is the main deployment optimization currently used. |
| Representative dataset calibration | Estimate activation ranges for INT8 quantization | Yes | We improved this by using class-balanced calibration data. |
| Full integer quantization | Quantize both weights and activations | Yes | The deployed TFLite model uses INT8 input/output tensors. |
| Quantization-aware training (QAT) | Train while simulating quantization noise | Not yet | Possible extra experiment if time allows. |
| Pruning / sparsity | Remove less important weights | Not used | Low priority because the model is already small and sparse kernels may not help on ESP32. |
| Optimized kernels | Use hardware-specific Conv/Dense implementations | Indirectly | TFLite Micro / ESP-NN may use optimized kernels when available. |
| SIMD/vector instructions | Execute multiple INT8 operations in parallel | Indirectly | ESP32-S3 supports vector-style acceleration, but we did not write custom SIMD kernels. |
| NPU acceleration | Offload neural network operators to a neural processor | Not used | ESP32-S3 does not provide an NPU for this project. |
| Hardware/software co-design | Adapt model and pipeline to hardware constraints | Yes | We use small input, small CNN, INT8 quantization, and simple temporal voting. |

## 3. What We Already Did

### INT8 Quantization

We converted the trained Keras/TensorFlow model into an INT8 TFLite model for ESP32 deployment. This reduces model size and makes inference more suitable for microcontroller execution.

The deployment model uses:

- 48x48 grayscale input.
- INT8 input tensor.
- INT8 output tensors.
- Full integer quantization.
- Representative dataset calibration.

### Balanced Representative Dataset

The representative dataset is important because it determines the activation ranges used during quantization. If calibration data is biased toward one person, the INT8 model can become biased or unstable.

We therefore changed the calibration process to use a more balanced set of known-person images. Unknown images are not used as a fourth class; they are used only for rejection evaluation or calibration checks.

### Open-Set Rejection Logic

The deployed system does not simply choose the largest softmax class. It uses:

- Softmax confidence threshold.
- Embedding distance threshold.
- Agreement between softmax class and nearest embedding centroid.
- Multi-frame voting.

This is important because a 3-class softmax model will always try to classify every input as person A, B, or C unless we add rejection logic.

## 4. What We Can Still Add on the Deployment Side

### A. Measure Deployment Cost

This is the most useful extra deployment work because it connects directly to the hardware acceleration material.

We can report:

- TFLite model size.
- Firmware binary size.
- Flash usage.
- Free partition space.
- Inference latency per frame.
- Approximate FPS.
- RAM/tensor arena usage if available.

This would show that we evaluated not only accuracy, but also embedded deployment cost.

### B. Compare Float Model and INT8 Model

We should explicitly compare:

- Float model closed-set accuracy.
- INT8 model closed-set accuracy.
- INT8 unknown false accept rate.
- INT8 known accept rate under rejection thresholds.

This supports the argument that quantization did not destroy closed-set accuracy, while unknown rejection remains the main challenge.

### C. Improve Preprocessing Alignment

This is probably the most important deployment-side improvement for real camera behavior.

We should make training and ESP32 input processing as similar as possible:

- Same grayscale conversion.
- Same face crop or center crop policy.
- Same resize target: 48x48.
- Similar brightness/contrast distribution.
- Similar face distance and position.

For the teammate who is not in Denmark, phone photos can still be useful. The teammate should send original photos, and we should process them using the same preprocessing pipeline instead of manually cropping them inconsistently.

### D. Optional: QAT Experiment

If time allows, we can try quantization-aware training.

The goal would be to check whether QAT improves the INT8 deployment model compared with PTQ.

Expected comparison:

| Model | Quantization | Closed-set accuracy | Unknown rejection | Size |
|---|---|---:|---:|---:|
| Baseline CNN | Float32 | TBD | TBD | Larger |
| CNN + PTQ | INT8 | Current result | Current result | Smaller |
| CNN + QAT | INT8 | TBD | TBD | Similar |

QAT may not solve unknown rejection, because unknown rejection is mostly an open-set problem. However, it is a relevant deployment-aware experiment from the course material.

### E. Do Not Prioritize Pruning

Pruning is not recommended as the next step because:

- The current model is already small.
- Pruning requires extra training and validation.
- Sparse models are not automatically faster on microcontrollers.
- Runtime support for sparse acceleration is required to get real speedups.

For a one-week deadline, latency/memory measurement and preprocessing alignment are more valuable.

## 5. Suggested Report Framing

The report should frame the project as an embedded open-set face recognition system:

> This project focuses on open-set face recognition under embedded deployment constraints. Besides training a compact CNN, we investigate deployment-aware optimization techniques including INT8 quantization, representative dataset calibration, memory/latency evaluation, and hardware-aware model design for ESP32-S3.

Important wording:

- Do not claim that hardware acceleration directly solves unknown recognition.
- Say that deployment optimization makes the model feasible on ESP32-S3.
- Say that unknown rejection is handled by confidence, embedding distance, class agreement, and voting.
- Say that deployment can still affect real-world accuracy through quantization and preprocessing mismatch.

## 6. Recommended Next Steps

1. Measure model size, firmware size, and flash usage.
2. Add inference timing logs on ESP32 if not already available.
3. Compare float model vs INT8 model accuracy.
4. Test the ESP32 serial output for each known person and several unknown faces.
5. Align preprocessing between Python and ESP32.
6. If time remains, run a small QAT-vs-PTQ experiment.

The best short-term focus is not custom hardware acceleration. The best short-term focus is deployment-aware evaluation: prove that the model is small, fast enough, quantized correctly, and that remaining errors come mainly from open-set recognition and camera/preprocessing mismatch.
