# display.py — 实时推理显示
# 来源: 改编自 camera/python/main.py
# 功能: 接收ESP32串口帧 + 推理结果，显示叠加层（姓名 + 置信度条 + 绿/红边框）
# 状态: TODO — Phase 4 集成联调时实现
#
# 串口协议:
#   ===FRAME===  + 153600字节 RGB565
#   ===RESULT=== + 1字节类别 + 16字节(4×float32)分数
