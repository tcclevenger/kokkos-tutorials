cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialMultiGpu)
include(../../common.cmake)

add_executable(ExerciseMultiGPU multi_gpu.cpp)
target_link_libraries(ExerciseMultiGPU Kokkos::kokkos)

