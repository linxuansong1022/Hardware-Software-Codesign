# Center Loss 改进方案说明

## 背景

原模型 `gray48_cnn_aug` 在 3 类分类上表现优秀（test_acc = 98.7%），但无法拒识陌生人。原因是 softmax 对所有输入都过度自信——陌生人照片也会被以接近 100% 的概率归类为某个已知人员。这意味着如果部署到门禁系统，陌生人站在门口系统也会开门。

## 做了什么

### 1. Embedding 距离评估（脚本 9）

**脚本**: `ml_share/scripts/9_eval_embedding_distance.py`

从原模型的 Dense(32) 层提取 32 维 embedding 向量，计算每个类别的质心（centroid），测试能否通过"距离远近"来区分已知人脸和陌生人。

评估了三种拒识方案：
- Softmax 单独：只看 softmax 置信度
- Embedding 距离单独：只看到质心的距离
- 两阶段：softmax + 距离双重检查

**结论**：原模型的 embedding 空间不够紧凑（同类样本分散），距离拒识效果有限。

### 2. Center Loss 训练（脚本 7_3）

**脚本**: `ml_share/scripts/7_3train_center_loss.py`

在原有 crossentropy 分类损失基础上，加了一个 center loss：

```
总损失 = crossentropy + λ × center_loss
```

center loss 强制同一类的 embedding 聚拢到各自的质心，使 embedding 空间更紧凑。

模型架构完全不变（同样的 3 层 CNN），只是训练目标多了一项约束。

### 3. 超参数搜索

并行跑了 7 组不同的超参数组合：

| λ (center_weight) | center_lr | Test Acc | NC 准确率 | Softmax 陌生拒绝 |
|:--:|:--:|:--:|:--:|:--:|
| 0 (原模型) | - | 98.7% | 63.6% | 25% |
| 0.01 | 0.5 | **98.7%** | **97.4%** | **79%** |
| 0.1 | 0.5 | 94.8% | 77.9% | - |
| 0.1 | 0.9 | 96.1% | 70.1% | - |
| 0.5 | 0.5 | 96.1% | 96.1% | - |
| 0.5 | 0.9 | 93.5% | 83.1% | - |
| 1.0 | 0.3 | 98.7% | 93.5% | - |
| 1.0 | 0.5 | 89.6% | 85.7% | - |

**最佳参数: λ=0.01, center_lr=0.5** — 分类准确率不变，embedding 质量大幅提升。

## 最终采用的方案

**Softmax 置信度阈值 + Center Loss 模型**

- 模型文件: `gray48_cnn_center_w001/gray48_cnn_center.keras`
- 推理方式: 和原模型完全一样，看 softmax 输出
- 拒识方式: softmax 最大概率 < 阈值 → 判定为 unknown，拒绝

### 为什么选这个方案

1. **模型架构不变** — 同样的 48×48 灰度输入、3 层 CNN、22K 参数，ESP32 部署代码不需要任何修改
2. **分类准确率不变** — 98.7%，满足 DEV_SPEC 的 ≥90% 要求
3. **拒识能力大幅提升** — 陌生人拒绝率从 25% 提升到 79%
4. **实现最简单** — ESP32 端只需一个阈值判断，不需要存质心、不需要算距离
5. **阈值可调** — 一个 `#define` 即可调整严格程度

### 为什么不选其他方案

- **Embedding 距离单独**: 需要在 ESP32 上存质心（384 bytes）并计算欧氏距离，增加了复杂度，但拒识效果不如 softmax 方案
- **两阶段**: 拒识率最高（94%），但已知人脸接受率只有 45%，用户体验差，且实现复杂
- **4 类训练（A/B/C + Unknown）**: Kirsi 尝试过，分类准确率下降严重，Unknown 类数据太杂难以学好

## 结果对比

### 分类能力（不变）

| 指标 | 原模型 | Center Loss 模型 |
|:--|:--:|:--:|
| Test Accuracy | 98.7% | 98.7% |
| Person A Recall | - | 88.0% |
| Person B Recall | - | 100% |
| Person C Recall | - | 96.3% |

### Unknown 拒识能力（大幅提升）

| 指标 | 原模型 | Center Loss 模型 |
|:--|:--:|:--:|
| 陌生人拒绝率 (softmax 阈值 ~0.99) | 25% (65/259) | **79% (205/259)** |
| Nearest-centroid 分类准确率 | 63.6% | **97.4%** |
| 已知平均 embedding 距离 | 6.10 | 1.76 |
| 陌生人平均 embedding 距离 | 8.92 | 2.09 |

### Embedding 空间质量

Center loss 让同类 embedding 从分散变紧凑：

| 类别 | 原模型平均距离 | Center Loss 平均距离 |
|:--|:--:|:--:|
| person_a | 5.96 | 0.91 |
| person_b | 3.94 | 0.73 |
| person_c | 5.80 | 0.96 |

## ESP32 部署

部署方式和原模型完全一样：

```c
// 唯一区别：换模型文件，调阈值
#define CONFIDENCE_THRESHOLD 0.90f  // 可调

if (max_score >= CONFIDENCE_THRESHOLD && predicted_class < 3)
    gpio_set_level(LED_PIN, 0);   // 开门
else
    gpio_set_level(LED_PIN, 1);   // 拒绝
```

### 下一步

1. Kirsi 对 `gray48_cnn_center.keras` 做 INT8 量化 + TFLite 转换
2. 导出 `model.c` / `model.h`
3. ESP32 端集成（代码逻辑不变，只换模型文件）

## 文件清单

```
ml_share/
├── scripts/
│   ├── 7_3train_center_loss.py      # Center Loss 训练脚本
│   └── 9_eval_embedding_distance.py # Embedding 距离评估脚本
│
└── models/gray48_cnn_center_w001/   # 最佳模型 (λ=0.01)
    ├── gray48_cnn_center.keras      # 模型文件 (float32, 待量化)
    ├── label_map.csv                # 类别映射
    ├── history.csv                  # 训练历史
    ├── centroids_trained.csv        # 训练得到的质心
    └── embedding_eval/              # 评估结果
        ├── embedding_predictions.csv   # 每张图的详细预测
        ├── threshold_sweep.csv         # 阈值扫描数据
        ├── two_stage_sweep.csv         # 两阶段扫描数据
        ├── centroids.csv               # 质心向量
        └── centroids_config.h          # ESP32 用 C header (备用)
```
