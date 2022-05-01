cmake_minimum_required(VERSION 3.12)

project(
    JForecast
    VERSION 1.0.0
    Languages CXX CUDA
    DESCRIPTION "Jetson Forecast"
)

enable_testing()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FindPkgConfig)

SET(CMAKE_CXX_STANDARD 17)

CONFIGURE_FILE(
    ${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
    ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
@ONLY)
