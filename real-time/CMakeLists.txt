
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CompilerWarnings.cmake")

set(
    SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
)

set(TARGET_NAME "real-time")

add_executable(${TARGET_NAME} ${SRC_FILES})

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")

set_project_warnings(${TARGET_NAME} OFF)

set_target_properties(
    ${TARGET_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
