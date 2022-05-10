#include <benchmarking.h>
#include <weather_forecast.h>

// img, training file, cache file
int main(int argc, char * argv[]) {
    std::string img_file(argv[1]);
    std::string cache_file(argv[2]);

    JForecast forecaster(2592, 1944);
    forecaster.forecast(img_file, cache_file);

    return 0;
}