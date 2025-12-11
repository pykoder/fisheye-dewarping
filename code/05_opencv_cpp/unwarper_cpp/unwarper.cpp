/*
 * OpenCV Fisheye Unwarper - C++ Implementation
 * 
 * A C++ program that uses OpenCV's fisheye features to dewarp fisheye images
 * into multiple perspective views, maintaining compatibility with the Python
 * unwrapper implementations.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <ranges>

namespace fs = std::filesystem;

class OpenCVDewarper {
private:
    int width{};
    int height{};
    int output_width{};
    int output_height{};
    int zones{};
    cv::Mat K;  // Camera intrinsic matrix
    cv::Mat D;  // Distortion coefficients
    std::vector<cv::Mat> remap_tables;

public:
    explicit OpenCVDewarper(const int w, const int h, const int z = 5)
        : width(w), height(h), zones(z) {
        output_width = width / 2;
        output_height = height / 2;
        
        // Camera intrinsic parameters for fisheye lens
        K = cv::Mat::zeros(3, 3, CV_32F);
        K.at<float>(0, 0) = width / 2.0f;
        K.at<float>(1, 1) = height / 2.0f;
        K.at<float>(0, 2) = width / 2.0f;
        K.at<float>(1, 2) = height / 2.0f;
        K.at<float>(2, 2) = 1.0f;
        
        // Distortion coefficients (assuming no distortion for simplicity)
        D = cv::Mat::zeros(4, 1, CV_32F);
        
        // Create remapping tables for all zones
        createDewarpMapping();
    }
    
    void createDewarpMapping() {
        remap_tables.clear();
        
        for (int zone_id = 0; zone_id < zones; ++zone_id) {
            // Calculate rotation angles for each zone
            constexpr double yaw = 0.0;      // No rotation around Y axis
            constexpr double pitch = 45.0;   // Look up 45 degrees
            const double roll = 360.0 * zone_id / zones;  // Different for each zone
            
            // Convert to radians
            constexpr double yaw_rad = yaw * CV_PI / 180.0;
            constexpr double pitch_rad = pitch * CV_PI / 180.0;
            const double roll_rad = roll * CV_PI / 180.0;
            
            // Calculate rotation matrices
            cv::Mat R_x = cv::Mat::zeros(3, 3, CV_32F);
            R_x.at<float>(0, 0) = 1.0f;
            R_x.at<float>(1, 1) = cos(pitch_rad);
            R_x.at<float>(1, 2) = -sin(pitch_rad);
            R_x.at<float>(2, 1) = sin(pitch_rad);
            R_x.at<float>(2, 2) = cos(pitch_rad);
            
            cv::Mat R_y = cv::Mat::zeros(3, 3, CV_32F);
            R_y.at<float>(0, 0) = cos(yaw_rad);
            R_y.at<float>(0, 2) = sin(yaw_rad);
            R_y.at<float>(1, 1) = 1.0f;
            R_y.at<float>(2, 0) = -sin(yaw_rad);
            R_y.at<float>(2, 2) = cos(yaw_rad);
            
            cv::Mat R_z = cv::Mat::zeros(3, 3, CV_32F);
            R_z.at<float>(0, 0) = cos(roll_rad);
            R_z.at<float>(0, 1) = -sin(roll_rad);
            R_z.at<float>(1, 0) = sin(roll_rad);
            R_z.at<float>(1, 1) = cos(roll_rad);
            R_z.at<float>(2, 2) = 1.0f;
            
            // Combined rotation matrix
            cv::Mat R = R_z * R_y * R_x;
            
            // New camera matrix for the perspective view           
            cv::Mat K_new = cv::Mat::zeros(3, 3, CV_32F);
            K_new.at<float>(0, 0) = output_width / 2.0f;
            K_new.at<float>(1, 1) = output_height / 2.0f;
            K_new.at<float>(0, 2) = output_width / 2.0f;
            K_new.at<float>(1, 2) = output_height / 2.0f;
            K_new.at<float>(2, 2) = 1.0f;
            
            // Create remapping tables
            cv::Mat map_x(output_height, output_width, CV_32F);
            cv::Mat map_y(output_height, output_width, CV_32F);
            
            for (int y = 0; y < output_height; ++y) {
                for (int x = 0; x < output_width; ++x) {
                    // Convert to homogeneous coordinates
                    const cv::Mat point = (cv::Mat_<float>(3, 1) << x, y, 1.0f);
                    
                    // Project to 3D ray directions
                    const cv::Mat ray = K_new.inv() * point;
                    
                    // Normalize ray
                    const float norm = cv::norm(ray);
                    const cv::Mat normalized_ray = ray / norm;
                    
                    // Transform ray to fisheye camera coordinate system
                    const cv::Mat ray_fisheye = R * normalized_ray;
                    
                    // Project fisheye ray to image coordinates
                    // For fisheye, we use spherical projection
                    float z = ray_fisheye.at<float>(2, 0);
                    z = std::clamp(z, -1.0f, 1.0f);  // Clamp to avoid domain errors
                    const float theta = std::acos(z);  // Angle from optical axis
                    const float phi = std::atan2(ray_fisheye.at<float>(1, 0), ray_fisheye.at<float>(0, 0));  // Azimuthal angle
                    
                    // Map to fisheye image coordinates
                    // Assuming equidistant fisheye projection model
                    const float r = theta * width / CV_PI;  // Radius in fisheye image
                    
                    // Convert to Cartesian coordinates
                    const float src_x = r * std::cos(phi) + width / 2.0f;
                    const float src_y = r * std::sin(phi) + height / 2.0f;
                    
                    // Clip to valid range using std::clamp (C++17)
                    const float clamped_src_x = std::clamp(src_x, 0.0f, static_cast<float>(width - 1));
                    const float clamped_src_y = std::clamp(src_y, 0.0f, static_cast<float>(height - 1));
                    
                    map_x.at<float>(y, x) = clamped_src_x;
                    map_y.at<float>(y, x) = clamped_src_y;
                }
            }
            
            remap_tables.push_back(map_x);
            remap_tables.push_back(map_y);
        }
    }
    
    cv::Mat dewarpFrame(const cv::Mat& image, const int zone_id) const {
        if (zone_id < 0 || zone_id >= zones) {
            throw std::invalid_argument("Invalid zone_id");
        }
        
        cv::Mat output;
        cv::remap(image, output,
                 remap_tables[zone_id * 2],
                 remap_tables[zone_id * 2 + 1],
                 cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT,
                 cv::Scalar(0, 0, 0));
        
        return output;
    }
    
    int getZones() const { return zones; }
};

void printUsage(const char* programName) noexcept {
    std::cout << "Usage: " << programName << " <input_image> [options]\n"
              << "Options:\n"
              << "  --output-dir <dir>    Output directory (default: current directory)\n"
              << "  --prefix <prefix>     Output filename prefix (default: input filename)\n"
              << "  --repeat-dewarp <n>   Number of repetition of dewarping for performance testing (default: 1)\n"
              << "  --help                Show this help message\n";
}


int main(const int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return -1;
    }
    
    const std::string input_image = argv[1];
    std::string output_dir = ".";
    std::string prefix = "";
    int repeat_dewarp = 1;
    
    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--prefix" && i + 1 < argc) {
            prefix = argv[++i];
        } else if (arg == "--repeat-dewarp" && i + 1 < argc) {
            repeat_dewarp = std::stoi(argv[++i]);
            if (repeat_dewarp < 1) repeat_dewarp = 1;  // Ensure at least 1 repetition
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    // Validate input file
    if (!fs::exists(input_image)) {
        std::cerr << "Error: Input file '" << input_image << "' does not exist\n";
        return -1;
    }
    
    // Create output directory if needed
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }
    
    // Set default prefix if not provided
    if (prefix.empty()) {
        prefix = fs::path(input_image).stem().string();
    }
    
    try {
        // Load input image
        std::cout << "Loading image: " << input_image << std::endl;
        cv::Mat image = cv::imread(input_image);
        if (image.empty()) {
            std::cerr << "Error: Could not load image: " << input_image << std::endl;
            return -1;
        }
        
        const int height = image.rows;
        const int width = image.cols;
        
        // Initialize dewarper
        std::cout << "Initializing dewarper for " << width << "x" << height << " image" << std::endl;
        const OpenCVDewarper dewarper(width, height, 5);
        
        // Repeat dewarping for performance testing
        std::cout << "Repeat dewarp " << (repeat_dewarp) << std::endl;
        // Process each zone using range-based loop
        for (const auto zone_id : std::views::iota(0, dewarper.getZones())) {
            cv::Mat zone_image;
            
            for (int i = 0; i < repeat_dewarp; ++i) {
                zone_image = dewarper.dewarpFrame(image, zone_id);
                image.at<cv::Vec3b>(i&0xFF, 0) = zone_image.at<cv::Vec3b>(i&0xFF,i&0xFF); // Prevent optimization
                //std::cout << "Dewarping zone " << (zone_id + 1) << ", iteration " << (i + 1) << std::endl;
            }
            const std::string output_path = output_dir + "/" + prefix + "_" +
                                            std::to_string(zone_id + 1) + "_cpp.jpg";
                
            cv::imwrite(output_path, zone_image);
            std::cout << "Saved view " << (zone_id + 1) << " to: " << output_path << std::endl;
        }
        
        std::cout << "Dewarping complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}