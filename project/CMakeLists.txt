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
# include(${CMAKE_SOURCE_DIR}/cmake/FindThrust.cmake)

find_package(OpenCV REQUIRED COMPONENTS)

SET(CMAKE_CXX_STANDARD 17)

CONFIGURE_FILE(
    ${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
    ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
@ONLY)

add_custom_target(
    uninstall
    ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(json)

