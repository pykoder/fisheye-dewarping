/*
 * C++ Library for Fisheye Unwarping - ctypes Integration
 * 
 * A C++ library that provides dewarping functionality to be called from Python
 * using ctypes. This implementation uses the same algorithm as the numpy version
 * but executes the dewarping in C++ for better performance.
 */

#include <cstdint>
#include <cmath>
#include <vector>
#include <iostream>
#include <memory>
#include <array>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <ranges>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


extern "C" {

// Structure to hold dewarping context
struct DewarpContext {
    int width{};
    int height{};
    int output_width{};
    int output_height{};
    int zones{};
    std::vector<int16_t> remap_data;  // Flattened remapping data
};

// Quaternion multiplication
constexpr std::array<double,4> multiply_quaternion(const std::array<double,4> a, const std::array<double,4> b) noexcept {
    return {
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    };
}

// Create dewarping context
DewarpContext* create_dewarp_context(const int width, const int height, const int zones) {
    auto* ctx = new DewarpContext{
        width, height,
        width / 2, height / 2,
        zones, {}
    };
    
    // Calculate total remap data size: zones * output_height * output_width * 2
    const auto total_size = zones * ctx->output_height * ctx->output_width * 2;
    ctx->remap_data.reserve(total_size);
    ctx->remap_data.resize(total_size);

    // Build spherical projection matrix from quaternions
    constexpr double expand = 1.269;  // Experimental expansion factor
    constexpr double offset = 0.25;   // Experimental offset

    // Yaw quaternion, rotate view around Y axis
    const double yaw = 0 * M_PI / 360.0;
    const std::array<double,4> yaw_q = {std::cos(yaw), 0.0, std::sin(yaw), 0.0};
    
    // Pitch quaternion, rotate view around X axis (look up 45 degrees)
    const double pitch = 45 * M_PI / 360.0;
    const std::array<double,4> pitch_q = {std::cos(pitch), std::sin(pitch), 0.0, 0.0};

    const auto temp_q = multiply_quaternion(pitch_q, yaw_q);

    // Generate remapping tables for all zones
    for (int zone_id = 0; zone_id < zones; ++zone_id) {
        
        // Roll quaternion, rotate view around Z axis (look in different direction)
        const double roll = (360.0 * zone_id / zones) * M_PI / 360.0;
        const std::array<double,4> roll_q = {std::cos(roll), 0.0, 0.0, std::sin(roll)};
        
        // Combine quaternions
        const auto rq = multiply_quaternion(roll_q, temp_q);
        
        // Store quaternion components in named variables for clarity
        const double w = rq[0];
        const double x = rq[1];
        const double y = rq[2];
        const double z = rq[3];

        std::array<std::array<double, 3>, 3> m = {{
            {{
                expand * 4.0 * (w*w + x*x - y*y - z*z) / width,
                expand * 8.0 * (x*y - w*z) / height,
                8.0 * (w*y + x*z) / M_PI
            }},
            {{
                expand * 8.0 * (x*y + w*z) / width,
                expand * 4.0 * (w*w - x*x + y*y - z*z) / height,
                8.0 * (y*z - w*x) / M_PI
            }},
            {{
                expand * 8.0 * (x*z - w*y) / width,
                expand * 8.0 * (w*x + y*z) / height,
                4.0 * (w*w - x*x - y*y + z*z) / M_PI
            }}
        }};

        // Build remapping table
        const double offset_width = offset * width;
        const double offset_height = offset * height;
        
        // Use traditional loops with index storage
        for (int j = 0; j < ctx->output_height; ++j) {
            for (int i = 0; i < ctx->output_width; ++i) {
                const double x = i - offset_width;
                const double y = j - offset_height;
                
                // Apply transformation matrix
                const std::array<double, 3> xyz = {
                    m[0][0] * x + m[0][1] * y + m[0][2],
                    m[1][0] * x + m[1][1] * y + m[1][2],
                    m[2][0] * x + m[2][1] * y + m[2][2]
                };
                
                const double hs = std::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);
                const double phi = std::atan2(hs, xyz[2]);
                
                // Calculate source coordinates
                int16_t src_x = static_cast<int16_t>(width * (xyz[0] * phi / (M_PI * hs) + 0.5));
                int16_t src_y = static_cast<int16_t>(height * (xyz[1] * phi / (M_PI * hs) + 0.5));
                
                // Clip to valid range using std::clamp (C++17)
                src_x = std::clamp(src_x, static_cast<int16_t>(0), static_cast<int16_t>(width - 1));
                src_y = std::clamp(src_y, static_cast<int16_t>(0), static_cast<int16_t>(height - 1));
                
                // Store in flattened array
                const int zone_offset = zone_id * ctx->output_height * ctx->output_width * 2;
                const int pixel_offset = (j * ctx->output_width + i) * 2;
                ctx->remap_data[zone_offset + pixel_offset] = src_x;
                ctx->remap_data[zone_offset + pixel_offset + 1] = src_y;
            }
        }
    }
    
    return ctx;
}

// Get remap data pointer for a specific zone
const int16_t* get_zone_remap_data(const DewarpContext* ctx, const int zone_id) noexcept {
    if (!ctx || zone_id < 0 || zone_id >= ctx->zones) {
        return nullptr;
    }
    
    const auto zone_offset = zone_id * ctx->output_height * ctx->output_width * 2;
    return ctx->remap_data.data() + zone_offset;
}

// Apply dewarping to image data
void dewarp_frame(const DewarpContext* ctx, const uint8_t* input_data, uint8_t* output_data, const int zone_id) noexcept {
    if (zone_id < 0 || zone_id >= ctx->zones) {
        return;
    }
    
    const auto* remap_ptr = get_zone_remap_data(ctx, zone_id);
    if (!remap_ptr) {
        return;
    }
          
    // Apply remapping
    for (int j = 0; j < ctx->output_height; ++j) {
        uint8_t buffer[4096];
        //__builtin_prefetch(input_data + j * ctx->width * 3, 0, 3);
        for (int i = 0; i < ctx->output_width; ++i) {
            const int remap_offset = (j * ctx->output_width + i);

            const int16_t src_x = remap_ptr[remap_offset + remap_offset];
            const int16_t src_y = remap_ptr[remap_offset + remap_offset + 1];
            
            const int src_offset = (src_y * ctx->width + src_x) * 3;
            
            // Copy RGB values
            buffer[i*3] = input_data[src_offset];
            buffer[i*3+1] = input_data[src_offset + 1];
            buffer[i*3+2] = input_data[src_offset + 2];
        }
        memcpy(output_data + j * ctx->output_width * 3, buffer, ctx->output_width * 3);
    }
}

// Get context information
int get_width(const DewarpContext* ctx) noexcept { return ctx ? ctx->width : 0; }
int get_height(const DewarpContext* ctx) noexcept { return ctx ? ctx->height : 0; }
int get_output_width(const DewarpContext* ctx) noexcept { return ctx ? ctx->output_width : 0; }
int get_output_height(const DewarpContext* ctx) noexcept { return ctx ? ctx->output_height : 0; }
int get_zones(const DewarpContext* ctx) noexcept { return ctx ? ctx->zones : 0; }

// Free context
void free_dewarp_context(DewarpContext* ctx) noexcept {
    delete ctx;
}

} // extern "C"