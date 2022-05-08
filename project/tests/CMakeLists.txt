add_executable(test_forecast
    test_forecast.cc
)

install(
    TARGETS test_forecast
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)
