cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 在相应的构建目录中查找头文件
set(CMAKE_INCLUDE_CURRENT_DIR ON)
project(test)
IF (WIN32)
 ADD_DEFINITIONS(-D Windows)
ELSEIF (UNIX)
 ADD_DEFINITIONS(-D Linux)
ENDIF ()
# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_target_properties(${the_target} PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                        OUTPUT_NAME "${the_target}")
# Add sub directories
add_subdirectory(app)
add_subdirectory(module/qtgui)
add_subdirectory(module/core)
add_subdirectory(module/log)
add_subdirectory(module/algorithm)
add_subdirectory(module/AutoFoucs)
