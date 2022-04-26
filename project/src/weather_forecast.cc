#include "weather_forecast.h"

JForecast::JForecast() {

}

JForecast::img_to_rgb(const std::string & image_file) {
    std::string image_path = samples::findFile("starry_night.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("starry_night.png", img);
    }
    cv::Mat image = image.at<cv::Vec3b>(y,x);
    image.at<cv::Vec3b>(y,x)[0] = newval[0];
    image.at<cv::Vec3b>(y,x)[1] = newval[1];
    image.at<cv::Vec3b>(y,x)[2] = newval[2];
}

JForecast::~JForecast() {

}
