cmake_minimum_required(VERSION 2.8)

project(tflite_serve)

set (SRCS
	src/main.cpp
)

add_executable(tflite_serve ${SRCS})
target_link_libraries(tflite_serve zmq)

