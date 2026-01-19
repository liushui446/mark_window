# Set the project name
project (algorithm)

find_package(
	OpenCV
	REQUIRED
)
set(Eigen3_DIR "${CMAKE_SOURCE_DIR}/3rdparty/Eigen/share/eigen3/cmake")

find_package(Eigen3 REQUIRED)
# Add a library with the above sources
#add_library(${PROJECT_NAME} SHARED src/CannyEdge.cpp src/CannyEdgeDetector.cpp src/edge_scene.cpp src/facet_model.cpp src/icp.cpp src/precise_offset_solver.cpp src/sim_template.cpp )
file(GLOB SRC_FILES src/*.cpp)
# 添加库
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
add_library(sub::algorithm ALIAS ${PROJECT_NAME})
target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

# 链接 OpenCV 库
target_link_libraries(${PROJECT_NAME}
   ${OpenCV_LIBS}
    Eigen3::Eigen
    sub::core
)

# 查找 Eigen3
# 打印库信息
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found (version: ${Eigen3_VERSION})")
endif()
