# main 组件的 CMakeLists.txt
# SRCS: 列出所有要编译的源文件
# INCLUDE_DIRS: 头文件搜索路径（"." 表示当前目录）
#
# 对比 keywords 版本的改动：
#   去掉了 audio.cpp, test.cpp（暂时不需要）
#   model.c → model_data.c（Kirsi 导出的文件名）
#   加了 camera.cpp（摄像头驱动）
idf_component_register(
    SRCS "main.cpp" "camera.cpp" "preprocess.cpp" "inference.cpp" "model_data.c"
    INCLUDE_DIRS "."
)
