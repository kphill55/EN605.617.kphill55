#include <weather_forecast.h>
#include <cassert>

using uchar = unsigned char;
using uint = unsigned int;
// using Pixel = cv::Point3_<uchar>;
// using fs = std::filesystem;
// using fs = std::experimental::filesystem;
using json = nlohmann::json;

void from_json(const json & j, Forecast_Feature & ff) {
    j.at("weather").get_to(ff.weather);
    j.at("bmean").get_to(ff.bmean);
    j.at("gmean").get_to(ff.gmean);
    j.at("rmean").get_to(ff.rmean);
    j.at("bvar").get_to(ff.bvar);
    j.at("gvar").get_to(ff.gvar);
    j.at("rvar").get_to(ff.rvar);
}

void to_json(json & j, const Forecast_Feature & ff) {
    j = json{
        {"weather", ff.weather},
        {"bmean", ff.bmean},
        {"gmean", ff.gmean},
        {"rmean", ff.rmean},
        {"bvar", ff.bvar},
        {"gvar", ff.gvar},
        {"rvar", ff.rvar}
    };
}

void JForecast::read_image(const std::string & image_file) {
    _img_buf = cv::imread(image_file, cv::IMREAD_COLOR);
}

// The MLE of a Gaussian Mean is the sum of the samples / n
void JForecast::populate_gmle_means(Forecast_Feature & ff, const cv::Mat & m) {
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

// The MLE of a Gaussian Variance is the sum of the (samples - mean)^2
void JForecast::populate_gmle_vars(Forecast_Feature & ff, const cv::Mat & m) {
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

    // Subtract the means
    cv::cuda::subtract(b, ff.bmean, b);
    cv::cuda::subtract(g, ff.bmean, g);
    cv::cuda::subtract(r, ff.bmean, r);

    // Square the results
    cv::cuda::sqr(b, b);
    cv::cuda::sqr(g, g);
    cv::cuda::sqr(r, r);

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
    ff.bvar = std::accumulate(b_result.begin<int>(), b_result.end<int>(), 0) / b_result.total();
    ff.gvar = std::accumulate(g_result.begin<int>(), g_result.end<int>(), 0) / g_result.total();
    ff.rvar = std::accumulate(r_result.begin<int>(), r_result.end<int>(), 0) / r_result.total();
    
}

template<typename T>
T JForecast::calc_distance(T x1, T y1, T z1, T x2, T y2, T z2) {
    return std::sqrt(std::pow(x2-x1, 2.0) + std::pow(y2-y1, 2.0) + std::pow(z2-z1, 2.0));
}

// Reads a directory full of jpeg files to condense them into a json array in a training file
void JForecast::generate_features(const std::string & output_file, const std::string & pic_dir, const std::string & classification) {
    // Open up the output file to write the features to
    std::ofstream feature_list_file(output_file, std::ios_base::app);
    feature_list_file.exceptions(std::ofstream::failbit|std::ofstream::badbit);
    if (feature_list_file.is_open() && feature_list_file.good()) {
        std::experimental::filesystem::path path{pic_dir};
        // Write a json array of features
        auto jsonObjects = json::array();
        for (const auto & pic : std::experimental::filesystem::directory_iterator{path}) {
            Forecast_Feature feature{
                classification,
                -1,-1,-1,-1,-1,-1
            };
            // Condense the jpeg into a feature and push back
            this->read_image(pic.path().string());
            this->populate_gmle_means(feature, _img_buf);
            this->populate_gmle_vars(feature, _img_buf);
            json j = feature;
            jsonObjects.push_back(j);
        }
        feature_list_file << jsonObjects;
    }
}

// Takes a training file full of features and generates a weather feature in the cache file
void JForecast::generate_cache(const std::string & training_file, const std::string & cache_file) {
    std::ifstream input_features(training_file);
    std::ofstream cf(cache_file, std::ios_base::app);
    input_features.exceptions(std::ifstream::failbit|std::ifstream::badbit);
    cf.exceptions(std::ofstream::failbit|std::ofstream::badbit);
    if (input_features.is_open() && input_features.good()) {
        // Parse the json file
        std::vector<Forecast_Feature> feature_condenser = json::parse(input_features).get<std::vector<Forecast_Feature>>();
        // Initialize a feature
        Forecast_Feature f{
            feature_condenser.front().weather,
            -1,-1,-1,-1,-1,-1
        };
        // Take the newly filled container of features and write each feature to the output json file
        for (const Forecast_Feature & feature : feature_condenser) {
            assert(f.weather == feature.weather);
            f.weather == feature.weather;
            f.bmean += feature.bmean / feature_condenser.size();
            f.gmean += feature.gmean / feature_condenser.size();
            f.rmean += feature.rmean / feature_condenser.size();
            f.bvar += feature.bvar / feature_condenser.size();
            f.gvar += feature.gvar / feature_condenser.size();
            f.rvar += feature.rvar / feature_condenser.size();
        }
        // Write the condensed feature to file
        json j = f;
        cf << j;
    }
}

// Read the cache
std::string JForecast::forecast(const std::string & weather_image_file, const std::string & cache_file) {
    // Process the image
    Forecast_Feature feature{
        "Unknown",
        -1,-1,-1,-1,-1,-1
    };
    // Condense the jpeg into a feature
    this->read_image(weather_image_file);
    this->populate_gmle_means(feature, _img_buf);
    this->populate_gmle_vars(feature, _img_buf);

    // Read the cache
    std::ifstream cf(cache_file);
    cf.exceptions(std::ifstream::failbit|std::ifstream::badbit);
    std::vector<Forecast_Feature> cached_features = json::parse(cf).get<std::vector<Forecast_Feature>>();

    std::vector<double> means(3);
    std::vector<double> vars(3);

    // Find the distances
    for (int i = 0; i < cached_features.size(); ++i) {
        means[i] = calc_distance(
            feature.bmean,
            feature.gmean,
            feature.rmean,
            cached_features[i].bmean,
            cached_features[i].gmean,
            cached_features[i].rmean
        );
        vars[i] = calc_distance(
            feature.bvar,
            feature.gvar,
            feature.rvar,
            cached_features[i].bvar,
            cached_features[i].gvar,
            cached_features[i].rvar
        );
    }

    // Find the minimum mean
    auto minm = std::min_element(means.begin(), means.end());
    int dm = std::distance(means.begin(), minm);
    std::cout << "Closest mean: " << *minm << " for weather: " << cached_features[dm].weather << "\n";

    // Find the minimum var
    auto minv = std::min_element(vars.begin(), vars.end());
    int dv = std::distance(vars.begin(), minv);
    std::cout << "Closest variance: " << *minv << " for weather: " << cached_features[dv].weather << "\n";

    if (dm == dv) {
        std::cout << "Decision is " << cached_features[dv].weather << "\n";
        return cached_features[dv].weather;
    }
    else if (dm != dv) {
        std::cout << "Decision is split between " << cached_features[dv].weather << " and " << cached_features[dm].weather << "\n";
        return std::string("Split decision!");
    }
    return std::string("Error, no decision!");
}
