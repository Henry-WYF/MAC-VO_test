cmake_minimum_required(VERSION 3.16)
project(macvo_dbow2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV 4 REQUIRED COMPONENTS core)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(macvo_dbow2
    binding.cpp
    DBoW2/BowVector.cpp
    DBoW2/FORB.cpp
    DBoW2/FeatureVector.cpp
    DBoW2/ScoringObject.cpp
    DUtils/Random.cpp
    DUtils/Timestamp.cpp
)

target_include_directories(macvo_dbow2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(macvo_dbow2 PRIVATE ${OpenCV_LIBS})
target_compile_options(macvo_dbow2 PRIVATE -Wall -Wextra -O3)

