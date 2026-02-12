import numpy as np  # 导入 NumPy，用于处理高性能数组运算
import tensorflow as tf  # 导入 TensorFlow，用于加载 TFLite 模型并进行推理
from scipy.io import wavfile  # 导入 scipy 的音频模块，用于读取 .wav 文件
from preprocess import preprocess_audio, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT, NUM_CLASSES  # 从预处理脚本中导入算法和参数

# 定义用于生成测试用例的音频文件路径，你可以根据需要修改这个路径
TEST_AUDIO_FILE = '../data/other/audio_noise_1354.wav' 


def generate_test_case(test_case_h_path: str):
    """
    加载音频文件，提取特征，通过 TFLite 模型运行预测，并将结果保存为 C++ 头文件
    """
    # 1. 读取音频文件，返回采样率 (sample_rate) 和原始波形数据 (audio_data)
    sample_rate, audio_data = wavfile.read(TEST_AUDIO_FILE)

    # 2. 调用 Python 的预处理函数，将原始波形转换为 62x64 的浮点声谱图特征 (x_test)
    x_test = preprocess_audio(audio_data)
    
    # 检查生成的特征形状是否正确 (必须是 62x64)，如果不正确则抛出错误
    if x_test.shape[0] != SPECTROGRAM_WIDTH or x_test.shape[1] != SPECTROGRAM_HEIGHT:
        raise ValueError(f'Expected preprocessed data shape ({SPECTROGRAM_WIDTH}, {SPECTROGRAM_HEIGHT}), but got {x_test.shape}')

    # 3. 加载已经生成的 TFLite 量化模型
    interpreter = tf.lite.Interpreter(model_path='gen/model.tflite')
    # 为模型分配内存（这一步是必须的，否则无法运行推理）
    interpreter.allocate_tensors()
    
    # 获取模型的输入层和输出层的详细信息（包括量化参数）
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 提取输入层和输出层的量化参数：Scale（比例）和 Zero Point（偏移）
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    # 4. 手动模拟量化过程：将浮点特征转换为 int8 整数
    # 公式：整数 = 浮点数 / Scale + ZeroPoint
    x_test_quantized = x_test / input_scale + input_zero_point
    # 强制将数值限制在 int8 的合法范围 [-128, 127] 之内
    x_test_quantized = np.clip(x_test_quantized, -128, 127)
    # 将数组类型正式转换为 8 位有符号整数
    x_test_quantized_int = x_test_quantized.astype(np.int8)

    # 5. 使用量化模型进行推理预测
    # 将处理好的 int8 特征放入模型的输入张量中（reshape 是为了匹配 [Batch, Width, Height] 的格式）
    interpreter.set_tensor(input_details[0]['index'], x_test_quantized_int.reshape(1, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT))
    # 执行推理计算（模拟单片机运行过程）
    interpreter.invoke()
    # 从输出张量中获取结果（这是 int8 格式的分类概率）
    y_pred_quantized = interpreter.get_tensor(output_details[0]['index'])[0]

    # 6. 反量化：将模型输出的 int8 结果转回 32 位浮点数，方便人类阅读
    y_pred = (y_pred_quantized.astype(np.float32) - output_zero_point) * output_scale

    # 7. 开始将这些“标准答案”写入 C++ 头文件
    with open(test_case_h_path, 'w') as cpp_file:
        # 写入防止重复包含的宏定义
        cpp_file.write('#ifndef TEST_CASE_H\n')
        cpp_file.write('#define TEST_CASE_H\n\n')

        # 包含必要的 C 标准头文件
        cpp_file.write('#include <stdint.h>\n\n')

        # 写入音频数据的长度
        cpp_file.write('#define TEST_LENGTH {}\n'.format(len(audio_data)))

        # 写入原始音频波形数组，供 ESP32 的预处理算法测试
        cpp_file.write('const int32_t raw_audio[TEST_LENGTH] = {\n')
        for i in range(0, len(audio_data), 12):
            chunk = audio_data[i:i + 12]
            cpp_file.write('    ' + ', '.join(map(str, chunk)) + ',\n')
        cpp_file.write('};\n\n')

        # 写入 Python 计算出的浮点声谱图，用于校验 ESP32 的预处理结果
        cpp_file.write('const float test_x[{}] = {{\n'.format(SPECTROGRAM_WIDTH * SPECTROGRAM_HEIGHT))
        for row in x_test:
            cpp_file.write('    ' + ', '.join(map(str, row)) + ',\n')
        cpp_file.write('};\n\n')

        # 写入量化后的 int8 声谱图，用于校验 ESP32 的量化公式
        cpp_file.write('const int8_t test_xq[{}] = {{\n'.format(SPECTROGRAM_WIDTH * SPECTROGRAM_HEIGHT))
        for row in x_test_quantized_int:
            cpp_file.write('    ' + ', '.join(map(str, row)) + ',\n')
        cpp_file.write('};\n\n')

        # 写入最终的模型预测概率（标准答案），用于校验 ESP32 的模型运行结果
        cpp_file.write('const float test_prediction[{}] = {{ {} }};\n\n'.format(NUM_CLASSES, ', '.join(map(str, y_pred))))

        # 闭合宏定义
        cpp_file.write('#endif // TEST_CASE_H\n')


if __name__ == '__main__':
    # 如果直接运行此脚本，默认生成 test_case.h
    # 注意：在实际项目 main.py 调用时会传入正确的路径
    generate_test_case('test_case.h')