#include <iostream>
#include <chrono>
#include <string>
#include <cstdlib>
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
        void img_to_rgb(const std::string & image_file);
        void thrust_gauss_mean_MLE();
        // Optimized reduction
        void kernel_gauss_mean_MLE();
};

#endif
