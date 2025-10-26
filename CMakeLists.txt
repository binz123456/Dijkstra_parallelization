cmake_minimum_required(VERSION 3.18)
project(parallel_pathfinding LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")

# CPU binary
add_executable(path_cpu src/main.cpp)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(path_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

# CUDA binary
enable_language(CUDA)
add_library(gpu_dijkstra STATIC src/gpu_dijkstra.cu)
set_target_properties(gpu_dijkstra PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(gpu_dijkstra PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(path_gpu src/main.cpp)
target_link_libraries(path_gpu PRIVATE gpu_dijkstra)
if(OpenMP_CXX_FOUND)
  target_link_libraries(path_gpu PUBLIC OpenMP::OpenMP_CXX)
endif()
