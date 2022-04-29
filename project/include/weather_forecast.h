#include <iostream>
#include <chrono>
#include <string>
#include <cstdlib>
#include <cstdint>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>

#ifndef FORECAST_H

class JForecast {
    public:
        // Take a single picture using the jetson camera
        void take_picture();
        // Take a collection of pictures
        void train_model();
        void forecast();
    private:
        void read_img(const std::string & image_file);
        void thrust_gauss_mean_MLE(const cv::Mat & im);
        void kernel_gauss_mean_MLE(const cv::Mat & im);
        void thrust_gauss_var_MLE(const cv::Mat & im, double e);
        void kernel_gauss_var_MLE(const cv::Mat & im, double e);
        
        thrust::host_vector<unsigned char> b_host;
        thrust::host_vector<unsigned char> g_host;
        thrust::host_vector<unsigned char> r_host;

        thrust::device_vector<unsigned char> b_dev;
        thrust::device_vector<unsigned char> g_dev;
        thrust::device_vector<unsigned char> r_dev;

        cv::Mat img_buffer;

};

#endif
