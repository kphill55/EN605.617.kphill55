// Tests the general processing for some image in one main script
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

//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>

//#include <nlohmann/json.hpp>

using uchar = unsigned char;

int main(int argc, char * argv[]) {
    cv::Mat img = cv::imread("jforecast_18324_s00_00000.jpg", cv::IMREAD_COLOR);
    cv::cuda::GpuMat dst, src;
    src.upload(img);

    // Split the image into BGR
    std::vector<cv::cuda::GpuMat> channels(3);
    cv::cuda::split(src, channels);
    dst = src;

    cv::cuda::GpuMat b = channels[1];
    cv::cuda::reduce(b, b, 0, cv::REDUCE_AVG);

    cv::Mat result;
    b.download(result);

    std::cout << (int)result.at<char>(0,0) << "\n";
    std::cout << result.size() << "\n";
    std::vector<uchar> array(result.begin<uchar>(), result.end<uchar>());

    //cv::imshow("Pic", result);
    //cv::waitKey(0);
    return 0;
}
