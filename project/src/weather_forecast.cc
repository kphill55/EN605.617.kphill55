#include "weather_forecast.h"

using uchar = unsigned char;
using Pixel = cv::Point_3<uchar>

JForecast::JForecast()
:
b_host(), g_host(),
{

}

JForecast::read_image(const std::string & image_file) {
    img_buffer = imread(image_file, cv::IMREAD_COLOR);
}

// The MLE of a Gaussian Mean is the sum of the samples / n
JForecast::thrust_gauss_mean_MLE(const cv::Mat & img) {
    // pixels (x,y,z) = (1,2,3) is (b,g,r) = (1,2,3).
    img.forEach<Pixel>([](const Pixel & pixel, const int position[]) -> void {
        pixel.x = position[0];
        pixel.y = position[1];
        pixel.z = position[2];
    });
}

JForecast::~JForecast() {

}
