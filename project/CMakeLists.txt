cmake_minimum_required(VERSION 3.12)

project(
    JForecast
    VERSION 1.0.0
    DESCRIPTION "Jetson Forecast"
    LANGUAGES CXX CUDA
)

enable_testing()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FindPkgConfig)

find_package(OpenCV REQUIRED)
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

SET(CMAKE_CXX_STANDARD 17)

CONFIGURE_FILE(
    ${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
    ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
@ONLY)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(json)

