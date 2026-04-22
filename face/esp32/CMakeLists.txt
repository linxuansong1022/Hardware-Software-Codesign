# 项目根 CMakeLists.txt — ESP-IDF 构建系统要求的固定格式
# 只需要改 project() 里的名字
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(face-recognition)
