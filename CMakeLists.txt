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

# ===================== 核心：全局定义 Windows 架构宏 =====================
if(WIN32)
    # 方式1：自动检测架构（推荐，适配 32/64 位）
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        # 64位 Windows：全局定义 _M_X64
        add_compile_definitions(_M_X64)
        message(STATUS "Windows 64位架构：已全局定义 _M_X64")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        # 32位 Windows：全局定义 _M_IX86
        add_compile_definitions(_M_IX86)
        message(STATUS "Windows 32位架构：已全局定义 _M_IX86")
    else()
        message(FATAL_ERROR "无法检测 Windows 目标架构！")
    endif()
	
	# 2. 定义 Windows 版本宏（兼容 Windows 10/11）
    # _WIN32_WINNT_WIN10 = 0x0A00，对应 Windows 10
    add_compile_definitions(
        _WIN32_WINNT=0x0A00
        NTDDI_VERSION=0x0A000006
        WIN32_LEAN_AND_MEAN  # 精简 Windows 头文件，减少冲突
    )
endif()

# ===================== 1. 手动指定 Basler SDK 路径 =====================
set(pylon_DIR "${CMAKE_SOURCE_DIR}/3rdparty/Basler/cmake/pylon")

# 设置输出目录
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set_target_properties(${the_target} PROPERTIES
                        # RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                        # OUTPUT_NAME "${the_target}")
						
# 设置输出目录						
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)  # 可执行文件输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)  # 静态库/导入库输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)  # 动态库输出目录（Linux/Mac）

# ===================== 关键：给所有模块设置分组，按业务层级排序 =====================
# 定义一个函数，批量给模块设置分组（避免重复代码）
function(set_module_folder target_name folder_name)
    if(TARGET ${target_name})
        set_target_properties(${target_name} PROPERTIES
            FOLDER "${folder_name}"  # 设置VS中的显示分组
        )
    endif()
endfunction()

# Add sub directories
add_subdirectory(app)
add_subdirectory(module/core)
add_subdirectory(module/camera)
add_subdirectory(module/qtgui)
add_subdirectory(module/log)
add_subdirectory(module/algorithm)
add_subdirectory(module/AutoFoucs)
add_subdirectory(module/process)
