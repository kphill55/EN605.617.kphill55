#include <benchmarking.h>
#include <weather_forecast.h>

// img, training file, cache file
int main(int argc, char * argv[]) {
    std::string training_file(argv[1]);
    std::string cache_file(argv[2]);
    
    // JForecast forecaster(2592, 1944);
    JForecast forecaster;
    forecaster.generate_cache(training_file, cache_file);

    return 0;
}
