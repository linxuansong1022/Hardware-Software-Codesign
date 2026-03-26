# 人脸数据采集指南

## 概述

- **目标**: 4个类别，每类 200-250 张，共 800-1000 张
- **设备**: XIAO ESP32S3 Sense（OV2640 摄像头）
- **工具**: `face/python/collect.py`

| 按键 | 类别 | 文件夹 |
|:----:|:-----|:------|
| 0 | 成员A | `data/person_a/` |
| 1 | 成员B | `data/person_b/` |
| 2 | 成员C | `data/person_c/` |
| 3 | Unknown | `data/unknown/` |

---

## 前置准备

### 1. 烧录固件

使用原版 camera 固件，不需要修改。

```bash
cd camera/esp32
idf.py build
idf.py flash monitor
```

看到以下输出后停止 monitor（`Ctrl+]`）：

```
I (1262) CAMERA: Camera initialized: 320x240 RGB565.
Send 'S' to start.
```

> **注意**: 停止 monitor 是为了释放串口给 Python 程序用。不要在终端里打 `S`。

### 2. 安装 Python 依赖

```bash
cd face/python
pip install pygame pyserial
```

### 3. 启动采集工具

```bash
# Mac/Linux
python collect.py --port /dev/ttyACM0 --output-path ../data/

# Windows
python collect.py --port COM3 --output-path ../data/
```

弹出窗口后，上方是摄像头实时画面，底部显示各类已采集数量。

---

## 采集任务（每个成员）

每人需采集自己的脸约 220 张，按以下任务分批完成。

### 任务 A — 正面 + 不同距离（50 张）

| 距离 | 张数 | 操作 |
|:----:|:----:|:-----|
| 30 cm | ~17 | 正面看摄像头，中性表情，连续按键 |
| 60 cm | ~17 | 后退一步，重复 |
| 100 cm | ~16 | 再后退，重复 |

### 任务 B — 侧面 + 不同表情（40 张）

- 头慢慢左转约30°，按键几张
- 头慢慢右转约30°，按键几张
- 微微低头、抬头，按键几张
- 微笑、说话时按键

### 任务 C — 不同光照（40 张）

| 光照条件 | 张数 | 操作 |
|:--------|:----:|:-----|
| 窗边自然光 | ~10 | 面朝窗户 |
| 顶部荧光灯 | ~10 | 站在灯下 |
| 弱光（台灯） | ~10 | 关大灯，只开台灯 |
| 逆光 | ~10 | 背对窗户 |

### 任务 D — 配饰（30 张）

- 戴眼镜：~15 张
- 戴墨镜：~15 张（如有）

### 任务 E — 不同背景（30 张）

- 实验室桌面：~10 张
- 白墙：~10 张
- 走廊或户外：~10 张

### 任务 F — 动态表情（30 张）

边说话边按键，约 30 张。

---

## Unknown 类采集（约 220 张）

由一人统筹，三人协作完成。

| 内容 | 张数 | 操作 |
|:-----|:----:|:-----|
| 陌生人正面照 | 90 | 请同学/路人站到摄像头前，**拍摄前取得口头同意** |
| 屏幕显示人脸 | 35 | 手机/平板全屏显示网络人脸照片，对着摄像头 |
| 部分遮挡人脸 | 35 | 用手遮住半边脸，或只露半张脸在画面边缘 |
| 空背景 | 35 | 摄像头对着桌面、墙壁、户外，画面里没有人 |
| 非人脸物体 | 25 | 手、书本、杯子等对着摄像头 |

---

## 远程协作方案

如果三人不在同一地点，没有板子的成员可以：

1. **用手机录制 10 分钟自拍视频**（覆盖上述任务 A-F 的各种条件）
2. 发送给有板子的成员
3. 有板子的成员将手机全屏播放视频，放在 ESP32 摄像头前翻拍
4. 翻拍时按对应的数字键保存

> 建议：如果有条件线下见面，至少用板子直接拍每人 50-100 张作为核心数据。

---

## 标注规则

### 保存条件

- 人脸占图像面积 ≥ 30%，清晰可辨
- 包含戴眼镜、轻微侧头（< 45°）、轻度遮挡的情况

### 标注为 Unknown 的情况

- 非团队成员的脸
- 空背景、物体
- 打印/屏幕上的照片
- 侧头 > 45°
- 身份模糊

### 丢弃（不保存）

- 运动模糊严重，无法辨认面部特征
- 人脸占图像面积 < 15%

### 边缘情况

- 画面中同时出现两人 → 标注为 Unknown
- 除非一个团队成员明显占据画面中心

---

## 质量检查

### 数量平衡

```bash
ls data/person_a/ | wc -l
ls data/person_b/ | wc -l
ls data/person_c/ | wc -l
ls data/unknown/  | wc -l
```

各类差距不超过 20%，最少的类不低于 200 张。

### 覆盖度检查

用 `session_log.csv` 确认每人的数据覆盖了所有条件因素：

```
           正面  侧面  自然光  荧光灯  弱光  逆光  眼镜  不同背景
person_a    ✓     ✓     ✓      ✓      ✓     ✓    ✓      ✓
person_b    ✓     ✓     ✓      ✓      ✓     ✓    ✓      ✓
person_c    ✓     ✓     ✓      ✓      ✓     ✓    ✓      ✓
```

每个格子都要有数据。

---

## 采集日志

每次换采集条件时，在 `data/session_log.csv` 中补填一行：

```csv
session_id,timestamp,class,collector,lighting,accessories,background,task,count,notes
1,2026-03-26 10:00,person_a,Alice,daylight,none,lab,A-frontal,50,30/60/100cm
2,2026-03-26 10:15,person_a,Alice,daylight,none,lab,B-angle,40,左右转头+俯仰
3,2026-03-26 10:25,person_a,Alice,low,none,lab,C-lighting,40,四种光照
4,2026-03-26 10:35,person_a,Alice,daylight,glasses,lab,D-accessories,30,眼镜+墨镜
5,2026-03-26 10:45,person_a,Alice,daylight,none,wall,E-background,30,换背景
6,2026-03-26 10:55,person_a,Alice,daylight,none,lab,F-expression,30,说话
```

> 这个日志很重要：后续按 session 分割训练/测试集时需要它，保证同一次连拍的帧不会同时出现在训练集和测试集中。

---

## 采集完成后

确认以下文件夹结构：

```
face/data/
├── session_log.csv
├── person_a/   (200+ 张 .png)
├── person_b/   (200+ 张 .png)
├── person_c/   (200+ 张 .png)
└── unknown/    (200+ 张 .png)
```

下一步：运行 `python preprocess.py` 进行预处理和数据集分割。
