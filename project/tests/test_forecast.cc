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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>

#include <nlohmann/json.hpp>

int main(int argc, char * argv[]) {
    cv::Mat img = cv::imread("image.png", IMREAD_GRAYSCALE);
    cv::cuda::GpuMat dst, src;
    src.upload(img);

    dst = src;

    cv::Mat result;
    dst.download(result);


    return 0;
}