add_executable(test_forecast
    test_forecast.cc
)

target_include_directories(
    test_forecast PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    # ${THRUST_INCLUDE_DIR}
)

target_link_libraries(
    test_forecast PUBLIC
    # nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    # Thrust
)

install(
    TARGETS test_forecast
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)
