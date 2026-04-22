## 量化与验证
将 gray48_cnn_center.keras 转换为 full INT8 TFLite 模型
输出文件：gray48_cnn_center_int8.tflite
## INT8 模型评估
在与 float 模型相同的 test set 上重新评估
INT8 TFLite test accuracy = 98.7%
person_a / person_b / person_c 三类的 precision、recall、F1 基本保持不变
混淆矩阵与 float 模型一致，仅有 1 张 person_c 被误分为 person_a
## 结论
量化后模型几乎没有精度损失
当前 gray48_cnn_center_int8.tflite 已可作为 ESP32 部署候选模型

已导出成 `model_data.c / model_data.h`
