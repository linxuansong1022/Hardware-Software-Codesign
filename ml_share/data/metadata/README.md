# 1 image_index.csv
自动生成一个图片索引表,把每个文件夹里的图片按文件名顺序编号成 1、2、3、4……

# 2 range_manifest.csv 记录
这是一段范围一行的中间表,即把拍摄顺序笔记整理成范围表(结构化数据描述文件),之后可以再用Python script把范围表自动展开成一张图一行的 image_manifest.csv

task_group
A-frontal：正面
B-angle：角度变化
C-lighting：光照变化
D-accessories：眼镜/墨镜
E-background：背景变化
F-expression：表情/说话
H-holdout：17:29 之后先单独放着

<!-- # 3 validate_range_manifest.py
检查range_manifest.csv是否有错误 -->

# 4 image_manifest.csv
把range_manifest.csv自动展开成image_manifest.csv

# 5 split_holdout_val_test.py
把holdout分成val和test,生成image_manifest_split.csv

<!-- # 6 check_dataset_ready
数据就绪检查,确认这三件事：
image_manifest_split.csv 里的每条路径都真的存在
每个split里的类别数量正确
没有奇怪的空值或坏数据 -->

# 7_1 train_gray48_cnn
把图片预处理成：灰度,48×48,shape = (48, 48, 1),训练一个CNN,生成模型gray48_cnn
运行后发现val_acc = 0.6579,test_acc = 0.5325,过拟合

# 7_2 train_gray48_cnn_aug.py
在第一版48×48灰度3类CNN的基础上,加入轻量数据增强,尽量缓解过拟合.
增加了 3 个训练时增强：
水平翻转:RandomFlip("horizontal")
轻微旋转:RandomRotation(0.05)
轻微缩放:RandomZoom(0.05)
运行后val_acc = 1.0000,test_acc = 0.9870

# 8 test_unknown_rejection.py
1.读取unknown文件夹里所有图片
2.用当前best model对每张图预测
3.记录每张图的：三类概率,最大概率,预测类别
4.统计：最大概率整体分布,有多少张低于某个阈值,哪些图最可疑、最容易被误认
