import os
import numpy as np
from scipy.io import wavfile

# Preprocessing parameters
NUM_CLASSES = 3 #yes no other
FRAME_SIZE = 256 #每一帧处理的采样点数
FRAME_STRIDE = 256 #滑动窗口 256表示没有重叠
SPECTROGRAM_HEIGHT = 64 # 声谱图的高度 保留64个频率分量
SAMPLE_RATE = 16000 #采样率16kHz
SPECTROGRAM_WIDTH = SAMPLE_RATE // FRAME_STRIDE  # 62 声谱图的宽度 时间轴上的帧数 16000/256 相当于62个时间片
SPECTRUM_MEAN = 6.305 #用于归一化 预先计算好的数据集均值
SPECTRUM_STD = 2.493 # 标准差


def preprocess_all(data_dir: str, out_dir: str):
    # Load and preprocess all directories
    other_x, other_y = _preprocess_directory(os.path.join(data_dir, 'other'), class_index=0)
    yes_x, yes_y = _preprocess_directory(os.path.join(data_dir, 'yes'), class_index=1)
    no_x, no_y = _preprocess_directory(os.path.join(data_dir, 'no'), class_index=2)

    # Concatenate and shuffle 合并并打乱
    x_all = np.concatenate([other_x, yes_x, no_x])#合并数据
    y_all = np.concatenate([other_y, yes_y, no_y])
    indices = np.arange(len(x_all))
    np.random.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]

    # Print mean and std of the entire dataset
    print('Mean of the dataset:', np.mean(x_all))#平均差
    print('Standard deviation of the dataset:', np.std(x_all))#标准差

    # Split into training, validation and test sets (60% train, 20% val, 20% test)
    num_samples = len(x_all)
    num_train = int(0.6 * num_samples) #60% train
    num_val = int(0.2 * num_samples) #20% val
    x_train = x_all[:num_train] # 取前60%
    y_train = y_all[:num_train]
    x_val = x_all[num_train:num_train + num_val]
    y_val = y_all[num_train:num_train + num_val]
    x_test = x_all[num_train + num_val:]
    y_test = y_all[num_train + num_val:]

    # Save to files
    os.makedirs(out_dir, exist_ok=True)
    np.save(out_dir + 'x_train.npy', x_train)
    np.save(out_dir + 'y_train.npy', y_train)
    np.save(out_dir + 'x_val.npy', x_val)
    np.save(out_dir + 'y_val.npy', y_val)
    np.save(out_dir + 'x_test.npy', x_test)
    np.save(out_dir + 'y_test.npy', y_test)

    # Clean up to reduce memory usage 手动删除不需要的庞大变量 音频处理非常占ram 
    del other_x, other_y, yes_x, yes_y, no_x, no_y, x_all, y_all 


def _preprocess_directory(data_dir: str, class_index: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess all audio files in a directory for a specific class.
    :param data_dir: Path to the directory containing audio files.
    :param class_index: Class index to be assigned to all files in this directory.
    :return: All spectrograms and their corresponding labels as numpy arrays.
    """
    # Load and preprocess all recordings in the data folder for a specific class
    print('Preprocessing directory: ', data_dir)
    spectrograms = []#用来暂时存放处理好的声谱图
    for wav_file in os.listdir(data_dir):
        if wav_file.endswith('.wav'):
            # Load wav file
            sample_rate, sound_data = wavfile.read(os.path.join(data_dir, wav_file))#读取采样率和波形数据
            if sample_rate != SAMPLE_RATE:
                raise ValueError(f'Expected sample rate of {SAMPLE_RATE}, but got {sample_rate}.')

            # Make it exactly 1 second long
            if len(sound_data) < SAMPLE_RATE:
                padding = np.zeros(SAMPLE_RATE - len(sound_data)) # 如果不够1秒，计算差多少，生成对应长度的 0 (静音)
                sound_data = np.concatenate((sound_data, padding))# 拼接到后面
            else:
                sound_data = sound_data[:SAMPLE_RATE]# 如果超过1秒，直接把多余的切掉

            # Preprocess audio 调用函数转换成 声谱图
            spectrogram = preprocess_audio(sound_data)

            # Add it to list
            spectrograms.append(spectrogram)

    return np.stack(spectrograms), np.full(len(spectrograms), class_index)
#把列表里的几百个 (62, 64) 的数组，堆叠成一个 (几百, 62, 64) 的大数组。
#成一个同样长度的数组，里面全是 class_index（比如全是 1）。这就是标签。

#DSP
def preprocess_audio(sound_data: np.ndarray) -> np.ndarray:#接受一个原始波形 返回一个新的数组 声谱图
    """
    Preprocess raw audio data into a spectrogram.
    :param sound_data: Raw audio data as a numpy array.
    :return: Preprocessed spectrogram as a numpy array of shape (SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT).
    """
    # Preprocess data with hamming window and fourier transform
    spectral_frames = []
    for j in range(0, len(sound_data) - FRAME_SIZE, FRAME_STRIDE): #len(sound_data) - FRAME_SIZE防止越界 剩下的直接丢了
        frame = sound_data[j:j + FRAME_SIZE]
        frame = frame - np.average(frame) #去直流偏置
        frame = frame * np.hamming(FRAME_SIZE) #加汉明窗 防止频谱泄漏
        spectral_frame = np.abs(np.fft.rfft(frame))#快速傅立叶变换 只关心能量大小 也就是模长 不关心相位,0对应0Hz
        spectral_frame = np.log1p(spectral_frame)#取对数
        spectral_frames.append(spectral_frame)#把这一帧处理好的频谱存进列表

    # Convert to numpy array
    spectrogram = np.array(spectral_frames)#列表转二维数组
    if spectrogram.shape[0] != SPECTROGRAM_WIDTH:
        raise ValueError(f'Expected spectrogram width of {SPECTROGRAM_WIDTH}, but got {spectrogram.shape[0]}.')

    # Keep the most relevant frequency bins
    spectrogram = spectrogram[:, 1:SPECTROGRAM_HEIGHT + 1]#只需要1到65这些频率点

    # Normalize data
    spectrogram = (spectrogram - SPECTRUM_MEAN) / SPECTRUM_STD #归一化

    return spectrogram


if __name__ == '__main__':
    preprocess_all('../data/', 'gen/')
