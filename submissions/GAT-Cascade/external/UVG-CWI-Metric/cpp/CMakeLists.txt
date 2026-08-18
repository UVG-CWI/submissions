cmake_minimum_required(VERSION 3.16)
project(PointCloudEval_CPP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

include_directories(include)

add_executable(evaluate_points src/main.cpp)
target_link_libraries(evaluate_points PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(evaluate_points PRIVATE /W4 /O2 /arch:AVX2)
else()
    target_compile_options(evaluate_points PRIVATE -Wall -Wextra -O3 -march=native -mavx2)
endif()
