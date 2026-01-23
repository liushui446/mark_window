# Set the project name
project (qtgui)
# 设置CMake在需要时自动运行moc
set(CMAKE_AUTOMOC ON)
# 从Qt设计师文件列表中创建代码
set(CMAKE_AUTOUIC ON)
# 从Qt资源文件列表中创建代码
set(CMAKE_AUTORCC ON)
find_package(
	Qt5 COMPONENTS
		Widgets
		UiTools
		Sql
	REQUIRED
)
# Add a library with the above sources
add_library(${PROJECT_NAME} SHARED src/gui.cpp src/mainwindow.cpp src/ImageWindow.cpp src/selectablegraphicsview.cpp )
add_library(sub::qtgui ALIAS ${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(${PROJECT_NAME}
    sub::core
	sub::log
	sub::algorithm
	sub::AutoFoucs
	Qt5::Widgets
	Qt5::UiTools
	Qt5::Sql
)
# 查找OpenCV包
find_package(OpenCV REQUIRED)
target_include_directories(${PROJECT_NAME}
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
