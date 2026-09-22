cmake_minimum_required(VERSION 3.16)

project(nanodet_object_tracking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc videoio)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of an ONNX Runtime C++ release")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
        "${ONNXRUNTIME_ROOT}/include"
        "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64")

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime C++ was not found. Set ONNXRUNTIME_ROOT to the extracted "
        "onnxruntime-linux-x64-<version> directory.")
endif()

add_executable(object_tracking_cpp
    src/object_tracking.cpp
)

target_compile_options(object_tracking_cpp PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

target_link_libraries(object_tracking_cpp PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})

target_include_directories(object_tracking_cpp PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

set_target_properties(object_tracking_cpp PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${ONNXRUNTIME_ROOT}/lib64"
)