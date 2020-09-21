cmake_minimum_required(VERSION 3.17)
project(opencl_demo)

set(CMAKE_CXX_STANDARD 14)

configure_file(HelloWorld.cl HelloWorld.cl COPYONLY)
configure_file(ProcessArray.cl ProcessArray.cl COPYONLY)

add_executable(opencl_demo main.cpp)

find_package(OpenCL REQUIRED)
target_link_libraries(opencl_demo ${OpenCL_LIBRARY})
