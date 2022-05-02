#include <iostream>
#include <chrono>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <async>
#include <fstream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <nlohmann/json>

#ifndef FORECAST_H

// Feature to treat an image as a set of Gaussian 
struct Forecast_Feature {
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

class JForecast {
    public:
        JForecast(unsigned int res_rows, unsigned int res_cols);
        // Take a single picture using the jetson camera
        void take_picture();
        // Take a collection of pictures and write a json file containing all their features
        void train_model();
        // Read a JSON file of training features and spit out an averaged representation of the training features
        void generate_cache();
        // Load the cached training features and compute the closest distance to a feature classify the image
        void forecast();
        ~JForecast = default;
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

        cv::Mat img_buffer;

};

#endif
