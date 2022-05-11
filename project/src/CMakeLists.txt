add_library(JForecast SHARED
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
    # ${THRUST_INCLUDE_DIR}
)

target_link_libraries(
    JForecast PUBLIC
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    stdc++fs
    # Thrust
)

install(
    TARGETS JForecast
    DESTINATION ${CMAKE_INSTALL_LIBDIR}}
)

add_executable(generate_features
    generate_features.cc
)

# add_executable(generate_cache
#     generate_cache.cc
# )

# add_executable(weather_forecast
#     weather_forecast_main.cc
# )

target_link_libraries(
    generate_features PUBLIC
    JForecast
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    stdc++fs
)

# target_link_libraries(
#     generate_cache PUBLIC
#     JForecast
#     stdc++fs
# )

# target_link_libraries(
#     weather_forecast PUBLIC
#     JForecast
#     stdc++fs
# )

# install(
#     TARGETS
#         generate_features
#         generate_cache
#         weather_forecast
#     DESTINATION ${CMAKE_INSTALL_BINDIR}
# )
