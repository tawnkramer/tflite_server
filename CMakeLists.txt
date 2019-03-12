cmake_minimum_required(VERSION 2.8)

project(tflite_serve)

set (SRCS
	src/main.cpp
    src/tflite_model.cpp
)

include_directories(../tensorflow ../tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)
add_executable(tflite_serve ${SRCS})
target_link_libraries(tflite_serve zmq tensorflow-lite pthread)

