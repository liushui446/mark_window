project (camera)

# 1. 先正确查找 Basler 依赖（假设你已修复 Basler_DIR 路径）
#set(pylon_DIR "${CMAKE_SOURCE_DIR}/3rdparty/Basler/cmake/pylon")

find_package(OpenCV REQUIRED)
# Locate pylon.
find_package(pylon 8.0.1 REQUIRED)

# 3. 收集源文件（建议用 GLOB_RECURSE 或显式列举，GLOB 可能漏文件）
file(GLOB SRC_FILES 
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

# 4. 定义库目标：使用项目名作为目标名，创建别名
add_library(${PROJECT_NAME} SHARED ${SRC_FILES} )
add_library(sub::camera ALIAS ${PROJECT_NAME})

# 5. 关键：设置库输出路径，匹配 qtgui 模块的查找路径
set_target_properties(${PROJECT_NAME} PROPERTIES
    # 静态库/导入库（.lib）输出路径
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib/$<CONFIG>"
    # 动态库（.dll）输出路径
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/$<CONFIG>"
    # 确保 Release/Debug 路径分开
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/lib/Release"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/lib/Debug"
)


# 6. 设置头文件路径：
target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include  # 自身头文件（对外暴露）
    PRIVATE 
        ${OpenCV_INCLUDE_DIRS}            # OpenCV 头文件（仅内部使用）
        ${pylon_INCLUDE_DIR}              # pylon 头文件（替换错误的 Basler_INCLUDE_DIRS）
)

# 7. 链接依赖库：增加目标存在性检查（可选但更健壮）
target_link_libraries(${PROJECT_NAME}
    PRIVATE 
		pylon::pylon  # 链接 Basler 库
		${OpenCV_LIBS}
		sub::core
		sub::log
)


# 打印库信息
