# Optimizing Fisheye Dewarping: From OpenCV to Custom C++ (256× Less CPU)

*Part 2 of 2 - In Part 1, we explored FFmpeg, pure Python, and vectorized NumPy implementations. Now we dive into native code: OpenCV Python with multi-threading, OpenCV C++ for reduced overhead, and a custom C++ library that achieves stunning performance.*

---

## Where We Left Off

In Part 1, we compared three approaches for fisheye dewarping:

| Implementation | Wall Time | CPU Time | Cores |  Memory | ms/view |
|----------------|-----------|----------|-------|---------|---------|
| **FFmpeg CLI** | 208.64s   | 1411.84s |   6.8 | 1784 MB |    40.7 |
| **Pure Python** | 1889.36s | 1889.34s |   1.0 |  647 MB |   369.0 |
| **NumPy vectorized** | 110.11s | 113.83s | 1.0 | 256 MB |    21.5 |

**Key findings**:
- FFmpeg wins in wall time thanks to 6.8 cores of parallelism
- NumPy wins in CPU efficiency: **12.4× less CPU** than FFmpeg
- Pure Python is surprisingly efficient per core (only 1.3× worse than FFmpeg)

**The question**: Can we combine NumPy's efficiency with FFmpeg's parallelism? And can native C++ push performance even further?

Let's find out with three more implementations.

---

## Implementation 4: OpenCV Python—Dedicated Image Processing

### The Approach

OpenCV is **the** reference library for computer vision. It offers a specialized function for geometric transformations: `cv2.remap()`, designed specifically for applying pixel-to-pixel lookup tables.

We keep our algorithm for mapping calculation (quaternions, spherical projection) but delegate the application phase to OpenCV.

### Key Modifications

#### Classical Projection

In the previous article we have shown our custom projection formulas. Below is the classical projection forumla, based on ray tracing from Fisheye dome to flat Point of View.

```python
# Camera matrix for perspective view
K = np.array([
    [self.output_width / 2, 0, self.output_width / 2],
    [0, self.output_height / 2, self.output_height / 2],
    [0, 0, 1]
], dtype=np.float32)

# Inverse projection: pixels to 3D rays
rays = np.linalg.inv(K) @ xyz

# Normalization and rotation
rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)
rays_fisheye = R @ rays

# Equidistant fisheye projection
theta = np.arccos(np.clip(rays_fisheye[2, :], -1, 1))
phi = np.arctan2(rays_fisheye[1, :], rays_fisheye[0, :])
r = theta * self.width / np.pi

x = r * np.cos(phi) + self.width / 2
y = r * np.sin(phi) + self.height / 2
```

Standard perspective + equidistant fisheye projection formulas. No more custom trigonometry. Our custom version was slightly faster, but at this point we know that the key of performance for our use case lies in the mapping stage where we apply the same precomputed formula to all images.

#### Application with cv2.remap()

**NumPy vectorized**:
```python
src_x = remap_table[:, :, 0]
src_y = remap_table[:, :, 1]
output_buffer = image[src_y, src_x]
```

**OpenCV**:
```python
map_x = remap_table[:, :, 0].astype(np.float32)
map_y = remap_table[:, :, 1].astype(np.float32)

output = cv2.remap(
    image, 
    map_x, 
    map_y, 
    cv2.INTER_NEAREST,              # Nearest neighbor
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0)
)
```

**Notable differences**:

1. **`float32` type required**: OpenCV requires maps in `float32` (vs `int32` for NumPy). Conversion necessary.

2. **Automatic clipping**: `cv2.remap()` automatically handles out-of-bounds pixels via `borderMode`. Can't disable it. OpenCV systematically checks bounds. With `BORDER_CONSTANT`, out-of-bounds pixels become black.

3. **Explicit interpolation**: `INTER_NEAREST` for maximum performance. `INTER_LINEAR` or `INTER_CUBIC` available for better quality. Using float coordinates allows better interpolation. Of course no effect on NEAREST strategy.

### Benchmarks

**OpenCV Python with INTER_NEAREST**:
```
Command: python3 unwarper_opencv.py fisheye.jpg --repeat-dewarp 1024

======================================================================
BENCHMARK RESULTS
======================================================================
Wall time:              22.21s
CPU time (user+sys):     105.70s
  - User time:           70.10s
  - System time:         35.60s
CPU utilization:        475%
Cores used:             ~4.8
Peak memory:            288.06 MB (294976 KB)
Page faults:            53982 minor, 0 major
Context switches:       34946 vol, 614194 invol
======================================================================

Parallel speedup:       4.76x
(CPU time / Wall time = 105.70s / 22.21s)
```

**OpenCV Python with INTER_LINEAR** (for comparison):
```
Command: python3 unwarper_opencv2.py fisheye.jpg --repeat-dewarp 1024

======================================================================
BENCHMARK RESULTS
======================================================================
Wall time:              31.80s
CPU time (user+sys):     160.71s
  - User time:           123.99s
  - System time:         36.72s
CPU utilization:        505%
Cores used:             ~5.0
Peak memory:            288.20 MB (295120 KB)
======================================================================

Parallel speedup:       5.05x
```

**Results**:
- INTER_NEAREST: 22.21s wall, 105.70s CPU → **4.3 ms/view** wall, **20.6 ms CPU/view**
- INTER_LINEAR: 31.80s wall, 160.71s CPU → **6.2 ms/view** wall, **31.4 ms CPU/view**

Interpolation choice has significant impact: **1.5× more CPU** for linear vs nearest.

### Analysis

**Comparison with previous implementations** (all with nearest neighbor):
- FFmpeg: 208.64s wall, 1411.84s CPU (6.8 cores) → **40.7 ms/view** wall, **276 ms CPU/view**
- NumPy: 110.11s wall, 113.83s CPU (1.0 core) → **21.5 ms/view** wall, **22.2 ms CPU/view**
- **OpenCV Python: 22.21s wall, 105.70s CPU (4.8 cores) → 4.3 ms/view wall, 20.6 ms CPU/view**

**Wall time vs CPU time analysis**:

**Wall time**: OpenCV is **5× faster** than NumPy (110.11s → 22.21s) thanks to 4.8 cores of parallelism.

**CPU time**: OpenCV uses **7% less CPU** than NumPy (113.83s → 105.70s). The gain is modest because both spend most time in optimized C code.

**Why such a small CPU gain?**

In both cases (NumPy and OpenCV Python), we spend **the majority of time in `cv::remap()`**, which is optimized C++ code. The Python overhead represents only a fraction of total time:
- Python function calls to C (via Python's C API)
- Python reference counting and GC
- Type conversions NumPy to cv::Mat

The ratio 22.21s / 110.11s ≈ 5× comes primarily from multièthread/core parallelization, not from more efficient code.

**Multi-threading efficiency**:

OpenCV uses 4.8 cores with 4.76× speedup → **efficiency = 4.76/4.8 = 99%**. Excellent parallelization with minimal overhead!

But compared to NumPy's single-core approach:
- NumPy: 113.83s CPU for 5,120 views = **22.2 ms CPU/view**
- OpenCV: 105.70s CPU for 5,120 views = **20.6 ms CPU/view**

Only **7% CPU savings** while using **4.8× more cores**. For a shared server processing N streams, this might not be optimal.

**✅ Strengths**

- **Excellent wall time**: 5× faster than NumPy thanks to multi-threading. Ideal for interactive pipelines.
- **Automatic multi-threading**: ~4.8 cores used without a single line of threading code.
- **High parallel efficiency**: 99% efficiency (4.76× speedup on 4.8 cores).
- **Clean Python code**: No C++ compilation, just `pip install opencv-python`.
- **Stable memory**: 288 MB, similar to NumPy, much better than FFmpeg (1784 MB).

**❌ Weaknesses**

- **Marginal CPU gain**: Only 7% CPU savings vs NumPy, actually it is in the precision range of our benchmark and is not significant. But if the machine is loaded, monopolizing 4.8 cores may be counter-productive.
- **OpenCV dependency**: ~90 MB to install (vs ~20 MB for NumPy). Installation sometimes finicky.
- **Mandatory clipping**: `cv2.remap()` systematically checks bounds. Can't disable even if coordinates are pre-clipped.
- **Type conversion**: Maps must be `float32`. Conversion from `int32` on every call.
- **Less flexible**: Black box. Impossible to tweak `remap()` implementation.

### Verdict

OpenCV Python brings a **5× wall time gain** thanks to multi-threading, but consumes **almost as much total CPU** as NumPy (only 7% savings).

**This implementation is suitable for**:
- Interactive pipelines where latency matters (wall time critical)
- Dedicated machines where monopolizing 4-5 cores isn't a problem
- Applications already using OpenCV (avoids extra dependency)

**Not suitable for**:
- Shared servers with high CPU load (monopolizes too many cores for small gain)
- Batch processing where total CPU load matters more than wall time
- Environments where OpenCV is difficult to deploy

**Question**: Can we do better by leaving Python entirely? The next section explores OpenCV C++ with native compiled code.

---

## Implementation 5: OpenCV C++—Native Compiled Code

### The Approach

After exploring Python's limits, let's move to **native C++**. Same algorithm, same `cv::remap()`, but this time compiled directly to binary without Python interpreter.

Objective: measure the real cost of Python interpretation and see if C++ brings significant gains.

### The Code (Key Sections)

```cpp
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

class Dewarper {
private:
    int width, height;
    int output_width, output_height;
    int zones;
    std::vector<cv::Mat> remap_x, remap_y;

public:
    Dewarper(int w, int h, int z = 5) 
        : width(w), height(h), zones(z),
          output_width(w/2), output_height(h/2) {
        
        // Pre-calculate all mapping tables
        calculateMappings();
    }
    
    void calculateMappings() {
        for (int zone_id = 0; zone_id < zones; ++zone_id) {
            // Build rotation matrix (quaternions → matrix)
            cv::Mat R = getRotationMatrix(0, 45, zone_id * 360.0 / zones);
            
            // Camera intrinsics
            cv::Mat K = (cv::Mat_<float>(3, 3) << 
                output_width/2, 0, output_width/2,
                0, output_height/2, output_height/2,
                0, 0, 1);
            
            cv::Mat map_x(output_height, output_width, CV_32F);
            cv::Mat map_y(output_height, output_width, CV_32F);
            
            // Generate mapping
            for (int y = 0; y < output_height; ++y) {
                for (int x = 0; x < output_width; ++x) {
                    // Project to 3D ray
                    cv::Vec3f ray = projectToRay(x, y, K, R);
                    
                    // Convert to fisheye coordinates
                    auto [fx, fy] = rayToFisheye(ray);
                    
                    map_x.at<float>(y, x) = fx;
                    map_y.at<float>(y, x) = fy;
                }
            }
            
            remap_x.push_back(map_x);
            remap_y.push_back(map_y);
        }
    }
    
    cv::Mat dewarpFrame(const cv::Mat& image, int zone_id) {
        cv::Mat output;
        cv::remap(image, output, 
                  remap_x[zone_id], remap_y[zone_id],
                  cv::INTER_NEAREST,
                  cv::BORDER_CONSTANT);
        return output;
    }
};

int main(int argc, char** argv) {
    // Load image
    cv::Mat image = cv::imread(argv[1]);
    
    // Initialize dewarper
    Dewarper dewarper(image.cols, image.rows, 5);
    
    // Dewarp all zones (with repetitions for benchmarking)
    int repeat = std::stoi(argv[3]);
    for (int zone_id = 0; zone_id < 5; ++zone_id) {
        cv::Mat result;
        for (int i = 0; i < repeat; ++i) {
            result = dewarper.dewarpFrame(image, zone_id);
        }
        cv::imwrite(output_path, result);
    }
    
    return 0;
}
```

### Implementation Details

**Impact of interpolation: Python vs C++ difference**

Interpolation choice has **very different impact** depending on Python or C++:

**OpenCV C++**:
- `INTER_NEAREST`: 7.34s wall, 23.85s CPU
- `INTER_LINEAR`: 11.04s wall, 53.46s CPU
- **Impact: ~2.3× more CPU, 1.5× slower**

**OpenCV Python**:
- `INTER_NEAREST`: 22.21s wall, 105.70s CPU
- `INTER_LINEAR`: 31.80s wall, 160.71s CPU
- **Impact: 1.5× more CPU, 1.4× slower**

**Explanation**: In Python, interpreter overhead and NumPy/OpenCV conversions "drown out" part of interpolation cost. Time spent in Python layers is incompressible and masks the impact of interpolation choice.

In pure C++, **100% of time is in critical code** (`cv::remap()`). Every CPU cycle counts. The overhead of linear interpolation (4 memory accesses + calculations vs 1 access) becomes dominant.

**For benchmarks**, we use `INTER_NEAREST` to maximize performance and have fair comparison.

### Benchmark

**Configuration: `cv::INTER_NEAREST`**
```
Command: ./unwarper fisheye.jpg --repeat-dewarp 1024

======================================================================
BENCHMARK RESULTS
======================================================================
Wall time:              7.34s
CPU time (user+sys):     23.85s
CPU utilization:        324%
Cores used:             ~3.2
Peak memory:            110.01 MB (112652 KB)
======================================================================
```

**Comparisons** (all with 1024 frames × 5 views, `INTER_NEAREST`):
- NumPy: 110.11s wall, 113.83s CPU, 1.0 core, 256 MB
- OpenCV Python: 22.21s wall, 105.70s CPU, 4.8 cores, 288 MB
- **OpenCV C++: 7.34s wall, 23.85s CPU, 3.2 cores, 110 MB**

### Analysis

**Comparison with OpenCV Python**:
- **Wall time**: 7.34s vs 22.21s → **3× faster**
- **CPU time**: 23s vs 105.70s → **4.5× less CPU consumed**
- **Memory**: 110 MB vs 288 MB → **2.6× less RAM**
- **Cores**: 3.2 vs 4.8 → slightly less parallelism

**Comparison with NumPy**:
- **Wall time**: 7.34s vs 110.11s → **15× faster**
- **CPU time**: 23.85s vs 113.83s → **5× less CPU**

**Is the C++ gain significant?**

Compared to OpenCV Python, the gain is **moderate but real**:
- 3× in wall time (not spectacular between Python and C++)
- 4.5× in CPU time (better, significant savings on shared server)
- 2.6× in memory (most notable gain)

**Why so little difference from Python?**

In both cases (Python and C++), we spend **the majority of time in `cv::remap()`**, which is optimized C++ code. Python overhead represents only a fraction:
- Python function call to C (via Python C API)
- Python reference management and GC
- NumPy type conversions to cv::Mat

The 3× ratio represents this Python overhead of two third of total time.

**Multi-threading differs slightly**: OpenCV Python (4.8 cores) vs OpenCV C++ (3.8 cores). Similar configuration, likely due to some OpenMP settings or Python binding forcing more parallelism.

**The real gains: CPU and memory**:
- **CPU time**: 4.5× less load (105.70s → 23.85s). On a shared server processing N streams in parallel, this savings is significant.
- **Memory**: 2.6× less (288 MB → 110 MB). Python objects (NumPy wrappers, reference counters, attribute dictionaries) add ~178 MB overhead. In C++, `cv::Mat` are compact structures.

**✅ Strengths**

- **Reduced CPU time**: 23.85s vs 105.70s (OpenCV Python). **4.5× gain** on total CPU load—critical for shared servers.
- **Minimal memory**: 110 MB, 2.6× less than OpenCV Python. Excellent for constrained environments or processing many streams simultaneously.
- **Efficient multi-threading**: 3.2 cores, 3.2× speedup → 101% parallel efficiency.
- **No Python overhead**: No GC, no interpreter, no type conversions. Direct native code.
- **Predictable performance**: Deterministic behavior, no random GC pauses that can cause latency spikes.
- **Clear sensitivity to optimizations**: Interpolation impact is measurable (2.3×), allowing precise performance/quality trade-off choices.

**❌ Weaknesses**

- **Compilation required**: Make, C++ toolchain (GCC/Clang), OpenCV dev headers. Much more complex than `pip install opencv-python`.
- **Moderate gain over Python**: Only 3× in wall time. If OpenCV Python already suffices, C++ may not be worth the complexity.
- **Less flexible**: Any code modification = full recompilation. Slower development cycle than Python.
- **Limited portability**: Platform-specific binaries (Linux x64, Windows, macOS, ARM). Python is portable out-of-the-box.
- **Build dependencies**: Requires OpenCV compiled with correct options (OpenMP, optimizations). More complex dependency management.

### Verdict

OpenCV C++ brings **moderate but real gains**: **3× less CPU, 2.6× less RAM, 2.2× faster**.

**The gain isn't spectacular** because OpenCV Python already spent most time in C++ code. Python overhead represented about 66% of total time, now eliminated.

**This implementation is suitable for**:
- **Shared servers** where saving 2.2× CPU on each stream matters (processing dozens of simultaneous streams)
- **Memory-constrained applications** (110 MB vs 288 MB allows processing more parallel streams)
- **Embedded or edge environments** where Python is difficult to deploy or too heavy
- **Need for deterministic performance** (no GC that randomly pauses)
- **Fine optimization**: clear impact of choices (interpolation) allows precise tuning

**Not necessary if**:
- OpenCV Python already suffices (2.2× wall time gain may not justify complexity)
- Deployment simplicity and maintenance matter
- Python development agility is critical (rapid prototyping, frequent modifications)
- No strong CPU or memory constraints

**Can we do even better?** In our use case, this implementation wouldn't be suitable as is, as the AI pipeline is implemented in Python, hence would need to get back to images data. However this can be easily solved with minimal overhead making it a ctypes shared library called from Python. The next section explores a **custom C++ library** written from scratch with fine memory management and algorithms specific to our use case (fixed cameras, static views). Can we still go significantly faster? At what cost in complexity and maintainability?

This is the final option we will consider in this article and the way the feature was originaly coded for production.

---

## Implementation 6: Custom C++ Library — The Ultimate Optimization

### The Approach

After exploring OpenCV, let's move to a **C++ library written from scratch** and optimized specifically for our use case. No more OpenCV dependency—just pure C++ with fine grain memory management and minimal algorithm.

The idea: keep Python call simplicity (via ctypes) while exploiting optimizations impossible with OpenCV:
- Minimal code without generic library overhead
- Optimal memory management (reusable buffers)
- Specialized algorithm for our case (fixed cameras, no recalibration)
- No multi-threading (avoids contention, optimal for single stream)

### Architecture

**C++ side**: A shared library (.so) exposing a simple API:
```cpp
extern "C" {
    // Create dewarping context (calculates mapping once)
    DewarpContext* create_dewarp_context(int width, int height, int zones);
    
    // Apply dewarping (fast, called in loop)
    void dewarp_frame(DewarpContext* ctx, uint8_t* input, uint8_t* output, int zone_id);
    
    // Free context
    void free_dewarp_context(DewarpContext* ctx);
}
```

**Python side**: Minimal ctypes wrapper:
```python
import ctypes
import numpy as np

# Load library
lib = ctypes.CDLL('libunwarper_ctypes.so')

# Configure signatures
lib.create_dewarp_context.argtypes = [c_int, c_int, c_int]
lib.create_dewarp_context.restype = c_void_p

lib.dewarp_frame.argtypes = [c_void_p, POINTER(c_uint8), POINTER(c_uint8), c_int]
lib.dewarp_frame.restype = None

# Usage
ctx = lib.create_dewarp_context(1920, 1920, 5)
lib.dewarp_frame(ctx, input_ptr, output_ptr, zone_id)
```

### Key Optimizations

**1. Compact mapping table**: Storage in `int16_t` instead of `float32` (OpenCV). Memory savings and better cache locality.

**2. Optimized remapping loop**:
```cpp
void dewarp_frame(const DewarpContext* ctx, const uint8_t* input_data, 
                  uint8_t* output_data, const int zone_id) {
    const auto* remap_ptr = get_zone_remap_data(ctx, zone_id);
    
    // Line-by-line processing with local buffer
    for (int j = 0; j < ctx->output_height; ++j) {
        uint8_t buffer[4096];  // Stack buffer, ultra-fast
        
        for (int i = 0; i < ctx->output_width; ++i) {
            const int remap_offset = (j * ctx->output_width + i);
            const int16_t src_x = remap_ptr[remap_offset * 2];
            const int16_t src_y = remap_ptr[remap_offset * 2 + 1];
            
            const int src_offset = (src_y * ctx->width + src_x) * 3;
            
            // Direct RGB copy into local buffer
            buffer[i*3]     = input_data[src_offset];
            buffer[i*3 + 1] = input_data[src_offset + 1];
            buffer[i*3 + 2] = input_data[src_offset + 2];
        }
        // Grouped copy from buffer to output
        memcpy(output_data + j * ctx->output_width * 3, buffer, ctx->output_width * 3);
    }
}
```

**Why is it fast?**:
- **Local stack buffer**: Avoids repeated dynamic allocations
- **Sequential accesses**: Maximizes CPU cache usage
- **No bounds checking**: Coordinates pre-clipped in mapping
- **No multi-threading**: Zero synchronization or contention overhead

**3. No interpolation**: Nearest neighbor only. Sufficient for object detection.

### Benchmark
```
Command: python3 unwarper_ctypes.py fisheye.jpg --repeat-dewarp 1024

======================================================================
BENCHMARK RESULTS
======================================================================
Wall time:              4.91s
CPU time (user+sys):     5.52s
  - User time:           5.48s
  - System time:         0.04s
CPU utilization:        112%
Cores used:             ~1.1
Peak memory:            80.14 MB (82068 KB)
======================================================================

Parallel speedup:       1.12x
(CPU time / Wall time = 5.52s / 4.91s)
```

**Comparisons** (all on 1024 frames × 5 views = 5,120 images):
- FFmpeg: 208.64s wall, 1411.84s CPU, 6.8 cores, 1784 MB
- Pure Python: 1889.36s wall, 1889.34s CPU, 1.0 core, 647 MB
- NumPy: 110.11s wall, 113.83s CPU, 1.0 core, 256 MB
- OpenCV Python: 22.21s wall, 105.70s CPU, 4.8 cores, 288 MB
- OpenCV C++: 7.34s wall, 23.85s CPU, 4.8 cores, 110 MB
- **Custom C++ lib: 4.91s wall, 5.52s CPU, 1.1 core, 80 MB**

### Analysis

**Spectacular gains on all metrics**:

**vs OpenCV C++:**
- **1.5× faster** in wall time (7.34s → 4.91s)
- **4.5× less CPU** (23.85s → 5.52s)
- **1.4× less memory** (110 MB → 80 MB)

**vs OpenCV Python:**
- **4.5× faster** in wall time
- **19× less CPU**
- **3.6× less memory**

**vs FFmpeg:**
- **42.5× faster** in wall time
- **256× less CPU**
- **22× less memory**

**vs NumPy:**
- **22.4× faster** in wall time
- **20.6× less CPU**
- **3.2× less memory**

**Where do these massive gains come from?**

The paradox: we use **1.1 core** vs 3.2 for OpenCV C++, but we're **1.5× faster** in wall time.

But the best indicator of efficiency is **total CPU time**:
- OpenCV C++: **23.85s CPU consumed**
- Custom lib: **5.52s CPU consumed**

We consume **4.3× less CPU resources** for the same work.

**1. No multi-threading = maximum efficiency**

Explanation: OpenCV's multi-threading has **hidden costs**:
- Thread synchronization (mutexes, barriers)
- Cache contention (false sharing)
- Frequent context switches (3000 for OpenCV vs 140 for us)
- Thread creation/destruction overhead

Our single-thread code avoids all this. One thread running at full speed, sequential memory accesses, optimal CPU cache.

**2. Minimal code without overhead**

OpenCV `cv::remap()` is a generic function handling:
- Multiple interpolation types
- Multiple border types
- Optional GPU support
- Validity checks
- cv::Mat abstraction with reference counting

Our code does **exactly what we need, nothing more**:
- Nearest neighbor interpolation only
- Pre-handled borders (clipping in mapping)
- No abstraction, just raw pointers
- No checks in critical phase

**3. Optimal memory management**

- **80 MB** vs 110 MB (OpenCV C++): 30 MB savings
- Mapping table in `int16_t`: 2× more compact than `float32`
- No OpenCV structures (`cv::Mat` with headers, refcounting)
- Temporary buffer on stack (no malloc)

**4. Good memory locality**

Line-by-line local buffer maximizes L1/L2 cache usage. All accesses are within ~4KB of data (one line), which fits entirely in L1 cache (32KB on modern CPUs).

**✅ Strengths**

- **Absolute performance**: Fastest of all implementations, on all metrics.
- **Exceptional CPU efficiency**: 5.52s CPU for 5,120 views. Unbeatable (Well, maybe, I still have some optimisations ideas and custom cuda or openMP could be an option, but it's fast enough).
- **Minimal memory footprint**: 80 MB only. Allows massive parallel processing.
- **Simple Python calling**: Trivial ctypes wrapper, no need to compile complex bindings.
- **No dependencies**: Just C++17 stdlib. No OpenCV, no third-party libs.
- **Optimal single-thread**: No contention, no synchronization. Ideal for processing N streams in parallel.
- **Simple deployment**: Single .so to compile, no dynamic dependencies.

**❌ Weaknesses**

- **C++ code to maintain**: Any modification requires recompilation.
- **No multi-threading**: If processing ONE SINGLE very heavy stream, we don't exploit multi-core. But our use case = N streams in parallel.
- **Fixed interpolation**: Nearest neighbor only. No linear/cubic. Acceptable for detection, not for photographic quality.
- **Compilation required**: Make, C++ toolchain. More complex than `pip install`.
- **Specialized code**: Optimized for our precise use case (fixed cameras, static views). Not generic.

### Verdict

The custom C++ library represents **the ultimate optimization**: **256× less CPU than FFmpeg, 19× less than OpenCV Python, 4.5× less than OpenCV C++**.

**This implementation is perfect for**:
- **High-performance production**: Processing dozens of simultaneous streams with maximum efficiency
- **Shared servers**: Minimizes total CPU load (5.52s vs 23.85s for OpenCV C++)
- **Memory-constrained environments**: 80 MB only
- **Applications needing predictable performance**: Single-thread, no GC, no contention

**Assumed trade-offs**:
- No multi-threading (intentionally)
- No flexibility (fixed interpolation)
- Specialized code (no OpenCV genericity)

**These trade-offs are acceptable** because our use case allows it:
- We process N streams in parallel (no need for multi-thread per stream)

---

## Conclusion:

**From 1411 seconds to 5.52 seconds.**

That's the journey we took: from FFmpeg's multi-threaded approach to a custom C++ library optimized for one specific task. A 256× improvement in CPU efficiency.

But here's the thing: you probably don't need to go that far.

For most use cases:

- FFmpeg is perfect for prototypes and one-off scripts
- NumPy delivers excellent efficiency if you're staying in Python
- OpenCV Python balances performance and convenience beautifully

We only built the custom C++ library because we had specific constraints: processing dozens of simultaneous camera streams on shared servers where every megabyte and every CPU cycle mattered.

**The real lesson isn't about the fastest implementation.**

It's about understanding your constraints, measuring your actual bottlenecks, and choosing the right tool for your specific problem.

Sometimes that tool is FFmpeg. Sometimes it's NumPy. Sometimes it's a hand-crafted C++ library that took two weeks to build and three years to prove its worth.

**Know your problem. Measure your performance. Optimize what matters.**

Everything else is just premature optimization.

Thanks for reading! If you found this useful, consider sharing it with someone who's struggling with their own performance challenges.

---

**Full code for all implementations**: [github.com/pykoder/fisheye-dewarping](https://github.com/pykoder/fisheye-dewarping)

*Thanks to Damien for the maths behind the algorithm (quaternion and spherical projection) and to all the Veesion team for their support all these years.

*Article written in December 2025. Benchmarks performed on a Lenovo ThinkPad P14s - Ubuntu 25.04, Intel Core i7-1185G7 (4 physical cores, 8 threads), 16GB RAM. All tests process 1024 frames × 5 views = 5,120 dewarped images.*