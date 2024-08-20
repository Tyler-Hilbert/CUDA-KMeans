cmake_minimum_required(VERSION 3.5)
project(ArrayFire-Iris-KMeans
  VERSION 3.5.0
  LANGUAGES CXX)

# Add the ArrayFire installation path to CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH "/content/ArrayFire-3.9.0-Linux" ${CMAKE_PREFIX_PATH})

find_package(ArrayFire REQUIRED)

add_executable(iris Iris_ArrayFire.cpp KMeans_ArrayFire.h KMeans_ArrayFire.cpp)
target_link_libraries(iris ArrayFire::afcuda)