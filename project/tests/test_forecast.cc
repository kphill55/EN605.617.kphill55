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

// The MLE of a Gaussian Mean is the sum of the samples / n
void populate_gmle_means(Forecast_Feature & ff, const cv::Mat & m) {
    // Upload the image to the GPU
    // cv::cuda::GpuMat device_mat(m.rows, m.cols, CV_32S);
    cv::cuda::GpuMat device_mat;
    device_mat.upload(m);
    device_mat.convertTo(device_mat, CV_32S);
    
    // Split into BGR channels
    std::vector<cv::cuda::GpuMat> channels(3);
    cv::cuda::split(device_mat, channels);

    cv::cuda::GpuMat b = channels[0];
    cv::cuda::GpuMat g = channels[1];
    cv::cuda::GpuMat r = channels[2];
    
    // Reduce the channels to their average, this returns one row (0 dimension)
    cv::cuda::reduce(b, b, 0, cv::REDUCE_AVG);
    cv::cuda::reduce(g, g, 0, cv::REDUCE_AVG);
    cv::cuda::reduce(r, r, 0, cv::REDUCE_AVG);

    cv::Mat b_result;
    cv::Mat g_result;
    cv::Mat r_result;

    b.download(b_result);
    g.download(g_result);
    r.download(r_result);

    // Store the average of each channel in the feature
    ff.bmean = std::accumulate(b_result.begin<int>(), b_result.end<int>(), 0) / b_result.total();
    ff.gmean = std::accumulate(g_result.begin<int>(), g_result.end<int>(), 0) / g_result.total();
    ff.rmean = std::accumulate(r_result.begin<int>(), r_result.end<int>(), 0) / r_result.total();
}

int main(int argc, char * argv[]) {
    const cv::Mat img = cv::imread("jforecast_18324_s00_00000.jpg", cv::IMREAD_COLOR);
    // // Accumulate the shortened array and find the average
    // std::cout << (int)result.at<char>(0,0) << "\n";
    // std::cout << result.size() << "\n";
    // std::vector<uchar> array(result.begin<int>(), result.end<int>());

    // double avg = std::accumulate(result.begin<int>(), result.end<int>(), 0) / result.total();
    // std::cout << avg << "\n";
    // print_vector(array);

    Forecast_Feature ff;
    Forecast_Feature feature{
                "sunny",
                -1,-1,-1,-1,-1,-1
    };
    populate_gmle_means(ff, img);
    std::cout << ff.bmean << "\n";
    std::cout << ff.gmean << "\n";
    std::cout << ff.rmean << "\n";
    //cv::imshow("Pic", result);
    //cv::waitKey(0);
    return 0;
}
