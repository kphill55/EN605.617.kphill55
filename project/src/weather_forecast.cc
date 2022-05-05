#include <weather_forecast.h>

using uchar = unsigned char;
using Pixel = cv::Point3_<uchar>;
// using fs = std::filesystem;
using fs = std::experimental::filesystem;
using json = nlohmann::json;

JForecast::JForecast(const unsigned int pixel_rows, const unsigned int pixel_cols)
// :
// b_host(), g_host(), r_host(),
// b_dev(), g_dev(), r_dev()
{

}

JForecast::read_image(const std::string & image_file) {
    _img_buffer = imread(image_file, cv::IMREAD_COLOR);
}

static void featurize_pic(std::list<Forecast_Feature> & l, const std::string & pic) {
    // Read picture file


    // Save to feature list
    std::lock_guard(_mut);
    _feature_condenser.push_back

}

JForecast::generate_features(const std::string & output_file, const std::string & pic_dir, const std::string & classification) {
    std::ofstream of(output_file, std::ios_base::app);
    fs::path path{pic_dir};

    for (const auto & pic : fs::directory_iterator{path}) {
        std::async(std::Launch::async, featurize_pic);
    }

    // Take the newly filled container of features and write each feature to the output json file
    for (auto feature : _feature_condenser) {

    }

}

// The MLE of a Gaussian Mean is the sum of the samples / n
JForecast::thrust_gauss_mean_MLE(const cv::Mat & img, Forecast_Feature & ff) {
    // pixels (x,y,z) = (1,2,3) is (b,g,r) = (1,2,3).
    img.forEach<Pixel>([](const Pixel & pixel, const int position[]) -> void {
        // Copy pixel colors to host vectors
        b_host[position[0]] = pixel.x;
        g_host[position[1]] = pixel.y;
        b_host[position[2]] = pixel.z;
    });

    // Copy host vector to device vector
    b_dev = b_host;
    g_dev = g_host;
    r_dev = r_host;

    // Perform the accumulation
    double b_sum = thrust::reduce(b_dev.begin(), b_dev.end());
    double g_sum = thrust::reduce(g_dev.begin(), g_dev.end());
    double r_sum = thrust::reduce(r_dev.begin(), r_dev.end());

    // Divide by n to get the mean
    ff.bmean = b_sum / img.size();
    ff.gmean = g_sum / img.size();
    ff.rmean = r_sum / img.size();
}

// The MLE of a Gaussian Variance is the sum of the (samples - mean)^2
JForecast::thrust_gauss_var_MLE(const cv::Mat & img, Forecast_Feature & ff) {
    // pixels (x,y,z) = (1,2,3) is (b,g,r) = (1,2,3).
    img.forEach<Pixel>([](const Pixel & pixel, const int position[]) -> void {
        // Copy pixel colors to host vectors
        b_host[position[0]] = pixel.x;
        g_host[position[1]] = pixel.y;
        b_host[position[2]] = pixel.z;
    });

    // Copy host vector to device vector
    b_dev = b_host;
    g_dev = g_host;
    r_dev = r_host;

    // Subtract the mean from each element
    double b_sum = thrust::transform(b_dev.begin(), b_dev.end(), );
    double g_sum = thrust::transform(g_dev.begin(), g_dev.end(), );
    double r_sum = thrust::transform(r_dev.begin(), r_dev.end(), );

    // Square each element
    thrust::transform(b_dev.begin(), b_dev.end(), );

    // Reduce/accumulate the result
    thrust::reduce(b_dev.begin(), b_dev.end());

    // Divide by n
}
