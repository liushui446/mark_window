# Set the project name
project (log)
find_package(
	Qt5 COMPONENTS
		Core
	REQUIRED
)
# Add a library with the above sources
add_library(${PROJECT_NAME} SHARED src/log.cpp)
add_library(sub::log ALIAS ${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(${PROJECT_NAME}
    sub::core
	Qt5::Core
)