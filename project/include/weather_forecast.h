#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <future>
#include <mutex>
#include <fstream>
// #include <filesystem>
#include <experimental/filesystem>
#include <list>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/cudaimgproc.hpp>

#include <nlohmann/json.hpp>

#ifndef FORECAST_H

// Struct to treat a single image as a set of Gaussian features
struct Forecast_Feature {
    // The weather classification
    std::string weather;
    // Means
    double bmean;
    double gmean;
    double rmean;
    // Variances
    double bvar;
    double gvar;
    double rvar;
};

// Jetson Forecast
class JForecast {
    public:
        JForecast(const unsigned int pixel_rows, const unsigned int pixel_cols);
        ~JForecast() = default;
        // Take a single picture using the jetson camera
        void take_picture(unsigned int pixel_rows, unsigned int pixel_cols, const std::string & output_filename);
        // Take a collection of pictures and write a json file containing all their features and what class these features are
        void generate_features(const std::string & classification, const std::string & output_file);
        // Read a JSON file of training features and spit out an averaged representation of the training features
        void generate_cache(const std::string & dir);
        // Load the cached training features and compute the closest distance to a feature classify the image
        std::string forecast(const std::string & weather_image_file, const std::string & cache_file);
        static nlohmann::json ff2json(const Forecast_Feature & ff);
        static Forecast_Feature json2ff(const nlohmann::json & j);

    private:
        void read_img(const std::string & image_file);
        void thrust_gauss_mean_MLE(const cv::Mat & im, Forecast_Feature & ff);
        void kernel_gauss_mean_MLE(const cv::Mat & im, Forecast_Feature & ff);
        
        void thrust_gauss_var_MLE(const cv::Mat & im, Forecast_Feature & ff);
        void kernel_gauss_var_MLE(const cv::Mat & im, Forecast_Feature & ff);
        
        thrust::host_vector<unsigned char> b_host;
        thrust::host_vector<unsigned char> g_host;
        thrust::host_vector<unsigned char> r_host;

        thrust::device_vector<unsigned char> b_dev;
        thrust::device_vector<unsigned char> g_dev;
        thrust::device_vector<unsigned char> r_dev;

        cv::Mat _img_buffer;

        std::list<Forecast_Feature> _feature_condenser;

        std::mutex _mut;

};

#endif
