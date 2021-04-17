
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CompilerWarnings.cmake")

set(
    SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/trainer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/trainer.hpp"
)

set(TARGET_NAME "trainer")

add_executable(${TARGET_NAME} ${SRC_FILES})

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")

set_project_warnings(${TARGET_NAME} OFF)

set_target_properties(
    ${TARGET_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

    target_compile_definitions(${TARGET_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS=1)

endif()

