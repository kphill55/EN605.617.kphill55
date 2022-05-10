#ifndef FORECAST_H

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
#include <cmath>
// #include <filesystem>
#include <experimental/filesystem>
#include <list>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>

#include <nlohmann/json.hpp>

template<typename T>
inline void print_vector(const std::vector<T> & vec) {
	std::cout << "[";
    for (auto i : vec) {
        std::cout << +i << ",";
    }
	std::cout << "]\n";
}

inline void print_cvmat(const cv::Mat & img) {
    std::cout << "[\n";
    for (int i = 0; i < img.rows; ++i) {
        std::cout << "[";
        for (int j = 0; i < img.cols; ++j) {
            std::cout << static_cast<int>(img.at<unsigned char>(i,j)) << ",";
        }
        std::cout << "]";
    }
    std::cout << "]\n";
}

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
        // Take a collection of pictures and write a json file containing all their features and what class these features are
        void generate_features(const std::string & classification, const std::string & output_file);
        // Read a JSON file of training features and spit out an averaged representation of the training features
        void generate_cache(const std::string & training_file, const std::string & dir);
        // Load the cached training features and compute the closest distance to a feature classify the image
        std::string forecast(const std::string & weather_image_file, const std::string & cache_file);
        // static nlohmann::json ff2json(const Forecast_Feature & ff);
        // static Forecast_Feature json2ff(const nlohmann::json & j);

    private:
        void read_image(const std::string & image_file);
        void populate_gmle_means(Forecast_Feature & ff, const cv::Mat & m);
        void populate_gmle_vars(Forecast_Feature & ff, const cv::Mat & m);
        template<typename T>
        inline T calc_distance(T x1, T x2, T y1, T y2);
        cv::Mat _img_buf;
        std::mutex _mut;

};

#endif
