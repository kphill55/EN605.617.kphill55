#include <benchmarking.h>
#include <weather_forecast.h>

// img, training file, cache file
int main(int argc, char * argv[]) {
    std::string output_file(argv[1]);
    std::string jpg_dir(argv[2]);
    std::string weather(argv[3]);
    
    JForecast forecaster(2592, 1944);
    std::cout << forecaster.generate_features(output_file, jpg_dir, weather) << "\n";

    return 0;
}
