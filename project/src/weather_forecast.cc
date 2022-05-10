#include <weather_forecast.h>

using uchar = unsigned char;
using uint = unsigned int;
// using Pixel = cv::Point3_<uchar>;
// using fs = std::filesystem;
// using fs = std::experimental::filesystem;
// using json = nlohmann::json;

JForecast::JForecast(const unsigned int pixel_rows, const unsigned int pixel_cols)
{

}

void JForecast::read_image(const std::string & image_file) {
    _img_buf = cv::imread(image_file, cv::IMREAD_COLOR);
}

// The MLE of a Gaussian Mean is the sum of the samples / n
void JForecast::populate_gmle_means(Forecast_Feature & ff, cv::Mat & m) {
    // Upload the image to the GPU
    static cv::cuda::GpuMat device_mat(m.rows, m.cols, CV_32S);
    device_mat.upload(m);
    
    // Split into BGR channels
    static std::vector<cv::cuda::GpuMat> channels(3);
    cv::cuda::split(device_mat, channels);

    static cv::cuda::GpuMat b;
    static cv::cuda::GpuMat g;
    static cv::cuda::GpuMat r;
    
    // Reduce the channels to their average, this returns one row (0 dimension)
    cv::cuda::reduce(b, channels[0], 0, cv::REDUCE_AVG);
    cv::cuda::reduce(g, channels[1], 0, cv::REDUCE_AVG);
    cv::cuda::reduce(r, channels[2], 0, cv::REDUCE_AVG);

    static cv::Mat b_result;
    static cv::Mat g_result;
    static cv::Mat r_result;

    b.download(b_result);
    g.download(b_result);
    r.download(b_result);

    // Store the average of each channel in the feature
    ff.bmean = static_cast<double>(std::accumulate(b_result.begin<uint>(), b_result.end<uint>(), 0)) / static_cast<double>(b_result.total());
    ff.gmean = static_cast<double>(std::accumulate(g_result.begin<uint>(), g_result.end<uint>(), 0)) / static_cast<double>(g_result.total());
    ff.rmean = static_cast<double>(std::accumulate(r_result.begin<uint>(), r_result.end<uint>(), 0)) / static_cast<double>(r_result.total());
}

// The MLE of a Gaussian Variance is the sum of the (samples - mean)^2
void JForecast::populate_gmle_vars(Forecast_Feature & ff, const cv::Mat & m) {
    // Upload the image to the GPU
    static cv::cuda::GpuMat device_mat(m.rows, m.cols, CV_32S);
    device_mat.upload(m);

    // Split into BGR channels
    static std::vector<cv::cuda::GpuMat> channels(3);
    cv::cuda::split(device_mat, channels);

    static cv::cuda::GpuMat b;
    static cv::cuda::GpuMat g;
    static cv::cuda::GpuMat r;

    // Subtract the means
    cv::cuda::subtract(b, ff.bmean, channels[0]);
    cv::cuda::subtract(g, ff.bmean, channels[1]);
    cv::cuda::subtract(r, ff.bmean, channels[2]);

    // Square the results
    cv::cuda::sqr(b, b);
    cv::cuda::sqr(g, g);
    cv::cuda::sqr(r, r);

    // Reduce the channels to their average, this returns one row (0 dimension)
    cv::cuda::reduce(b, channels[0], 0, cv::REDUCE_AVG);
    cv::cuda::reduce(g, channels[1], 0, cv::REDUCE_AVG);
    cv::cuda::reduce(r, channels[2], 0, cv::REDUCE_AVG);

    static cv::Mat b_result;
    static cv::Mat g_result;
    static cv::Mat r_result;

    b.download(b_result);
    g.download(b_result);
    r.download(b_result);

    // Store the average of each channel in the feature
    ff.bvar = static_cast<double>(std::accumulate(b_result.begin<uint>(), b_result.end<uint>(), 0)) / static_cast<double>(b_result.total());
    ff.gvar = static_cast<double>(std::accumulate(g_result.begin<uint>(), g_result.end<uint>(), 0)) / static_cast<double>(g_result.total());
    ff.rvar = static_cast<double>(std::accumulate(r_result.begin<uint>(), r_result.end<uint>(), 0)) / static_cast<double>(r_result.total());
    
}

template<typename T>
T JForecast::calc_distance(T x1, T x2, T y1, T y2) {
    return std::sqrt(std::pow(x2-x1, 2.0) + std::pow(y2-y1, 2.0));
}

// void JForecast::generate_features(const std::string & output_file, const std::string & pic_dir, const std::string & classification) {
//     std::ofstream of(output_file, std::ios_base::app);
//     fs::path path{pic_dir};

//     for (const auto & pic : fs::directory_iterator{path}) {
//         std::async(std::Launch::async, featurize_pic);
//     }

//     // Take the newly filled container of features and write each feature to the output json file
//     for (auto feature : _feature_condenser) {

//     }

// }

