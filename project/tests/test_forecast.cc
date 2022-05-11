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
#include <numeric>

//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>

//#include <nlohmann/json.hpp>

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

using uchar = unsigned char;

int main(int argc, char * argv[]) {
    const cv::Mat img = cv::imread("jforecast_18324_s00_00000.jpg", cv::IMREAD_COLOR);
    cv::cuda::GpuMat dst;
    cv::cuda::GpuMat src;
    src.upload(img);

    src.convertTo(src, CV_32S);
    // cv::cuda::subtract(src, 3, src);
    // cv::cuda::sqr(src, src);

    // Split the image into BGR
    std::vector<cv::cuda::GpuMat> channels(3);
    cv::cuda::split(src, channels);

    // Take a single channel and get the average
    cv::cuda::GpuMat b = channels[0];
    cv::cuda::reduce(b, b, 0, cv::REDUCE_AVG);

    // Copy the result to the cpu
    cv::Mat result;
    b.download(result);

    // Accumulate the shortened array and find the average
    std::cout << (int)result.at<char>(0,0) << "\n";
    std::cout << result.size() << "\n";
    std::vector<uchar> array(result.begin<int>(), result.end<int>());

    double avg = std::accumulate(result.begin<int>(), result.end<int>(), 0) / result.total();
    std::cout << avg << "\n";
    // print_vector(array);

    //cv::imshow("Pic", result);
    //cv::waitKey(0);
    return 0;
}
