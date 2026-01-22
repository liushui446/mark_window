# Set the project name
project (process)
# Add a library with the above sources
add_library(${PROJECT_NAME} SHARED src/process.cpp)
add_library(sub::process ALIAS ${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(${PROJECT_NAME}
    sub::core
)