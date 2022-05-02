add_library(JForecast
    weather_forecast.cc
)

set_target_properties(
    JForecast PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION}
)

target_include_directories(
    JForecast PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${THRUST_INCLUDE_DIR}
)

target_link_libraries(
    JForecast PUBLIC
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    Thrust
)
