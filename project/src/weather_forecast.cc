#include <weather_forecast.h>

using uchar = unsigned char;
using uint = unsigned int;
// using Pixel = cv::Point3_<uchar>;
// using fs = std::filesystem;
// using fs = std::experimental::filesystem;
using json = nlohmann::json;

JForecast::JForecast(const unsigned int pixel_rows, const unsigned int pixel_cols)
{

}

void JForecast::read_image(const std::string & image_file) {
    _img_buf = cv::imread(image_file, cv::IMREAD_COLOR);
}

// The MLE of a Gaussian Mean is the sum of the samples / n
void JForecast::populate_gmle_means(Forecast_Feature & ff, const cv::Mat & m) {
    // Upload the image to the GPU
    static cv::cuda::GpuMat device_mat(m.rows, m.cols, CV_32S);
    // static cv::cuda::GpuMat device_mat;
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
    ff.bmean = std::accumulate(b_result.begin<int>(), b_result.end<int>(), 0) / b_result.total();
    ff.gmean = std::accumulate(g_result.begin<int>(), g_result.end<int>(), 0) / g_result.total();
    ff.rmean = std::accumulate(r_result.begin<int>(), r_result.end<int>(), 0) / r_result.total();
}

// The MLE of a Gaussian Variance is the sum of the (samples - mean)^2
void JForecast::populate_gmle_vars(Forecast_Feature & ff, const cv::Mat & m) {
    // Upload the image to the GPU
    static cv::cuda::GpuMat device_mat(m.rows, m.cols, CV_32S);
    // static cv::cuda::GpuMat device_mat;
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
    ff.bvar = std::accumulate(b_result.begin<int>(), b_result.end<int>(), 0) / b_result.total();
    ff.gvar = std::accumulate(g_result.begin<int>(), g_result.end<int>(), 0) / g_result.total();
    ff.rvar = std::accumulate(r_result.begin<int>(), r_result.end<int>(), 0) / r_result.total();
    
}

template<typename T>
T JForecast::calc_distance(T x1, T x2, T y1, T y2) {
    return std::sqrt(std::pow(x2-x1, 2.0) + std::pow(y2-y1, 2.0));
}

void JForecast::generate_features(const std::string & output_file, const std::string & pic_dir, const std::string & classification) {
    std::list<Forecast_Feature> feature_condenser;
    std::ofstream of(output_file, std::ios_base::app);
    of.exceptions(std::ifstream::failbit|std::ifstream::badbit);
    if (of.is_open() && of.good()) {
        std::experimental::filesystem::path path{pic_dir};
        for (const auto & pic : std::experimental::filesystem::directory_iterator{path}) {
            Forecast_Feature feature;
            this->read_image(pic.path());
            this->populate_gmle_means(feature, _img_buf);
            this->populate_gmle_vars(feature, _img_buf);
            json j = feature;
            of << j;
        }
    }
    



}

// // void JForecast::generate_cache() {
//             std::ifstream input_feature(pic.path());
//             json jfeature;
//             input_feature >> jfeature;
//             json::parse<;
        // Take the newly filled container of features and write each feature to the output json file
//         for (auto feature : feature_condenser) {

//         }
// // // }
