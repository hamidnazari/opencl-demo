cmake_minimum_required(VERSION 3.17)
project(opencl_demo)

set(CMAKE_CXX_STANDARD 11)

configure_file(HelloWorld.cl HelloWorld.cl COPYONLY)
configure_file(ProcessArray.cl ProcessArray.cl COPYONLY)
configure_file(ProcessMultiArray.cl ProcessMultiArray.cl COPYONLY)
configure_file(NumericalReduction.cl NumericalReduction.cl COPYONLY)
configure_file(LargeLoops.cl LargeLoops.cl COPYONLY)

add_executable(opencl_demo main.cpp)

find_package(OpenCL REQUIRED)
target_link_libraries(opencl_demo ${OpenCL_LIBRARY})

add_compile_definitions(CL_HPP_TARGET_OPENCL_VERSION=120)
add_compile_definitions(CL_HPP_MINIMUM_OPENCL_VERSION=120)
