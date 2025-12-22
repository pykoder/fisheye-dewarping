# Fisheye Ceiling view, 6 Dewarping Implementations: A Technical Journey

*TL;DR: Faced with a critical production bug, I had to recode a fisheye dewarper from scratch. Three years later, I revisit this problem to compare 6 different approaches—from FFmpeg command line to custom optimized C++ library.*

---

## The Fisheye Incident 

**January 2022.** Our Docker container crashes in production after a few hours on certain sites. The diagnosis comes: memory leak in the proprietary library handling fisheye 360° camera dewarping.

Quick context: our AI analyzes video streams to detect gestures. Our models are trained on flat views, not on distorted circular views. Without dewarping, **zero detection** on fisheye cameras, merely 1% of the fleet, but important clients.

The leaking library? Provided by the camera manufacturer. We can't patch it, can't wait for a fix.

**Verdict**: we code our own version.

Fifteen days diving into projective geometry books, a C++ implementation based on quaternions and spherical projections, a few weeks of optimization... and we had our solution in production.

**Three years later**, with some hindsight and free time, a question keeps nagging me: **what if we had done it differently?**

Which approach would have been most effective given our constraints:
- **Fixed** ceiling-mounted fisheye cameras
- Need for **5 flat views** per frame
- **Static** virtual camera viewpoints (no dynamic rotation)
- **Critical performance**: real-time processing of multiple video streams
- "Sufficient" quality for detection (no need for photographic perfection)

This article compares **6 different implementations** for this specific use case, with benchmarks to back it up.

---

## What You'll Discover

1. **Theoretical basics** of fisheye dewarping (short version, I promise)
2. **Three implementations** of the same algorithm:
   - FFmpeg command line
   - Pure Python (loops, no libs)
   - Vectorized NumPy (two variants)
3. **Comparative benchmarks**: execution time, RAM, implementation complexity
4. A teaser for **Part 2** where we'll explore OpenCV and custom C++ implementations

**Important**: This comparison is specific to *our* use case (fixed cameras, static views, real-time performance). Your needs may differ: mobile cameras, dynamic recalibration, maximum quality, etc. Adapt accordingly.

---

## A Bit of Theory (Just Enough)

### The Problem in Images

A 360° fisheye camera on the ceiling captures the entire space in a distorted circular image.

For our detection algorithms to work, we transform this into several flat rectangular views.

![Principle Diagram](https://github.com/pykoder/fisheye-dewarping/blob/main/images/schema.png?raw=true)

### How It Works (Simplified Version)

Dewarping breaks down into two phases:

**Phase 1: Mapping Calculation (once at startup)**

We create a correspondence table: for each pixel in our 5 output flat views, we calculate which pixel from the fisheye image to fetch.

Concretely:
1. Project each point from the fisheye image onto a virtual hemisphere centered on the camera
2. Define 5 "virtual cameras" with their fixed positions and orientations
3. For each virtual camera, calculate which portion of the sphere it "sees"
4. Store everything in a lookup table

This phase uses spherical projections (and in our initial case, quaternions for rotations). It's fairly heavy computation, but we only do it **once**.

**Phase 2: Mapping Application (for each video frame)**

For each new image from the fisheye camera:
1. Iterate through our 5 output views
2. For each pixel, look up in the lookup table where it comes from
3. Copy the source pixel color (with optional interpolation for quality)

This is the phase we need to optimize heavily. It runs in a loop on every frame.

### Note on Calibration

Fisheye lenses vary from model to model. Precise calibration allows obtaining perfectly rectilinear views. In our case, we skip it: our detection algorithms tolerate slight residual distortions. This simplifies the code and boosts performance.

---

## Implementation 1: FFmpeg CLI - The Quick Parallel Solution

### The Approach

FFmpeg natively supports this type of dewarping via its `v360` filter. The implementation is straightforward but reveals some interesting subtleties.

### The Code
```bash
#!/bin/bash
# unwarper_ffmpeg.sh

INPUT_IMAGE="$1"
OUTPUT_DIR="${2:-.}"
BASENAME=$(basename "$INPUT_IMAGE" | cut -d. -f1)
INTERP=near

# Process all 5 zones in a single FFmpeg pass
ffmpeg -y -i "fisheye.mp4" \
-vf "crop=1920:1920,v360=input=fisheye:output=flat:interp=near:pitch=45:yaw=0:roll0:v_fov=90:w=960:h=960" "unwarped_1.mp4" \
-vf "rotate=4*72*PI/180,crop=1920:1920,v360=input=fisheye:output=flat:interp=near:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "unwarped_2.mp4" \
-vf "rotate=3*72*PI/180,crop=1920:1920,v360=input=fisheye:output=flat:interp=near:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "unwarped_3.mp4" \
-vf "rotate=2*72*PI/180,crop=1920:1920,v360=input=fisheye:output=flat:interp=near:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "unwarped_4.mp4" \
-vf "rotate=72*PI/180,crop=1920:1920,v360=input=fisheye:output=flat:interp=near:yaw=0:pitch=45:roll=0:v_fov=90:w=960:h=960" "unwarped_5.mp4"
```

### Implementation Details

**v360 filter parameters:**
- `yaw`, `pitch`, `roll`: rotations describing the fisheye camera orientation (ceiling dome)
- `v_fov=90`: vertical field of view of the output view
- `w=960:h=960`: resolution of dewarped views (half of 1920×1920 source)
- `interp=near`: nearest neighbor interpolation (vs default `linear`)

**Optimization choices:**

1. **Minimal interpolation**: We force `interp=near` instead of default linear interpolation. Image quality is slightly degraded but performance is better. For object detection, it's largely sufficient.

2. **Source image rotation**: Control of viewpoint via `yaw/pitch/roll` is limited and only produces the desired result in one direction. We work around this by applying a preliminary rotation to the fisheye image (multiples of 72° to cover 360°).

3. **Single source file read**: All views are generated in a single FFmpeg pass (one command with 5 outputs). Essential for performance.

4. **1024 frame video**: Benchmarks use a video long enough that mapping application time largely dominates initial mapping calculation time (the phase we're really interested in). We will also repeat dewarping 1024 times on other solutions.

### Benchmark
```
Command: ./unwarper_ffmpeg.sh fisheye_video.mp4

======================================================================
BENCHMARK RESULTS
======================================================================
Wall time:              208.64s
CPU time (user+sys):     1411.84s
  - User time:           1401.84s
  - System time:         10.00s
CPU utilization:        676%
Cores used:             ~6.8
Peak memory:            1784.20 MB (1827016 KB)
Page faults:            387273 minor, 0 major
Context switches:       328327 vol, 712451 invol
Exit status:            0

Parallel speedup:       6.77x
(CPU time / Wall time = 1411.84s / 208.64s)
======================================================================
```

**Results:** 208.64 seconds to process 1024 frames and generate 5 views = 5,120 dewarped images total. That's about **40.7 ms per dewarped view**.

FFmpeg exploits 6.8 cores with excellent parallel efficiency (6.77× speedup). However, memory usage climbs to **1.78 GB**, which is significant.

FFmpeg is slightly penalized, because it also encode movies for each views. Which is not necessary for our use case.

### Analysis

**✅ Strengths**
- **Immediate setup**: a single command, no libraries to install (FFmpeg suffices)
- **Native parallelization**: optimal multi-core usage without effort (6.8 cores)
- **Excellent parallel efficiency**: 6.77× speedup from parallelization
- **Robust**: FFmpeg is battle-tested in production everywhere
- **No maintenance**: stable external dependency, bugs already fixed by the community

**❌ Weaknesses**
- **Total black box**: impossible to audit or modify the dewarping algorithm
- **Limited flexibility**: stuck with parameters exposed by `v360`
- **Not finely integrable**: requires spawning an external process, impossible to call directly as a Python function
- **High memory consumption**: 1.78 GB for a 1920×1920 video, potentially problematic at scale
- **Limited optimizations**: can't optimize the mapping phase specifically for our use case (fixed cameras, static views)

### Verdict

FFmpeg is **the ideal weapon for a quick POC** or when you need a result that works *now* without questions. Perfect for:
- Testing if dewarping solves your business problem
- One-shot scripts or occasional batch processing
- Situations where RAM isn't a constraint

**But unsuitable if:**
- You need to integrate dewarping into a complex Python pipeline
- You want to optimize finely (pre-compute mapping once, reuse)
- Memory consumption is critical
- You need to understand or tweak the underlying algorithm

In our case (proprietary library memory leak), FFmpeg could have been an acceptable fallback solution... but we would have quickly been limited for optimization and integration.

---

## Implementation 2: Pure Python - For Understanding the Math

### The Approach

Now that we've seen the "black box" solution with FFmpeg, let's dive into the algorithm's guts. This **pure Python** implementation uses only standard libraries and NumPy for array manipulation, but **without any vectorization**.

The goal here isn't performance, but **understanding**. Each mathematical step is explicit, documented, comprehensible. It's the pedagogical reference that will serve as a baseline for all future optimizations.

### The Code (Key Sections)

```python
def multiply_quaternion(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions using Python primitives.
    
    Args:
        a: First quaternion [w, x, y, z]
        b: Second quaternion [w, x, y, z]
        
    Returns:
        Result quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z], dtype=np.float64)

def get_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Generate rotation matrix from yaw, pitch, roll angles (in degrees).
    
    Args:
        yaw: Rotation around Y axis in degrees
        pitch: Rotation around X axis in degrees
        roll: Rotation around Z axis in degrees
    Returns:
        3x3 rotation matrix as numpy array
    """
    # Yaw quaternion, rotate view around Y axis
    yaw = np.deg2rad(0)
    yaw_q = np.array([np.cos(yaw/2.0), 0.0, np.sin(yaw/2.0), 0.0], dtype=np.float64)
    # Pitch quaternion, rotate view around X axis (look up 45 degrees)
    pitch = np.deg2rad(45)             
    pitch_q = np.array([np.cos(pitch/2.0), np.sin(pitch/2.0), 0.0, 0.0], dtype=np.float64)
    # Roll quaternion, rotate view around Z axis (look in different direction)
    roll = np.deg2rad(roll)
    roll_q = np.array([np.cos(roll/2.0), 0.0, 0.0, np.sin(roll/2.0)], dtype=np.float64)

    rq = multiply_quaternion(roll_q, multiply_quaternion(pitch_q, yaw_q))

    # Build spherical projection matrix from quaternions
    w, x, y, z = rq
    return np.array([
        [ (w*w + x*x - y*y - z*z),  2.0 * (x*y - z*w), 2.0 * (w*y + x*z)],
        [ 2.0 * (w*z + x*y), (w*w - x*x + y*y - z*z), 2.0 * (y*z - w*x)],
        [ 2.0 * (x*z - y*w), 2.0 * (w*x + y*z), (w*w - x*x - y*y + z*z)]], dtype=np.float64)


def project2D(xyz: np.ndarray) -> Tuple[int, int]:
    hs = np.hypot(xyz[0],xyz[1])
    phi = np.arctan2(hs, xyz[2])
    coeff = phi / (hs * np.pi)
    src_x = xyz[0] * coeff + 0.5
    src_y = xyz[1] * coeff + 0.5
    return src_x, src_y


class PythonDewarper:
    """
    Pure Python implementation of the fisheye dewarper.    
    """
    
    def __init__(self, width: int, height: int, zones: int = 3):
        """
        Initialize dewarper with image dimensions.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
        """
        self.width = width
        self.height = height
        self.output_width = self.width // 2
        self.output_height = self.height // 2
        self.zones = zones
        
        # Remapping tables for each view
        self.remap = self._dewarp_mapping()
        self.output_buffer = np.zeros((self.zones, self.output_height, self.output_width, 3), dtype=np.uint8)
    

    def _dewarp_mapping(self) -> List[List[List[Tuple[int, int]]]]:
        """
        Create pixel remapping table for specific view.
        
        This is the core dewarping algorithm using spherical projection.
        """
       
        remap = []
        for zone_id in range(self.zones):
    
            # Get rotation matrix for this zone
            R = get_rotation_matrix(0, 45, zone_id * (360.0 / self.zones))

            remap_zone = []
            for j in range(self.output_height):
                line = []
                for i in range(self.output_width):
                    v = np.array([i / (0.25 * self.width) - 1.0, j / (0.25 * self.height) - 1.0, 1.0])
                    xyz = R @ v.T
                    src_x, src_y = project2D(xyz)
                    map_y = int(src_y * self.height)
                    map_x = int(src_x * self.width)
                    if 0 <= map_y < self.height and 0 <= map_x < self.width:    
                        line.append((map_y, map_x))
                    else:
                        line.append((0, 0))
                remap_zone.append(line)
            remap.append(remap_zone)

        return remap
    
    def dewarp_frame(self, image: np.ndarray, zone_id: int = -1):
        """
        Apply dewarping transformation to image.
        
        Args:
            image: Input image as NumPy array (H, W, 3)
        """
        
        remap_table = self.remap[zone_id]
        output_buffer = self.output_buffer[zone_id]

        for i in range(self.output_height):
            for j in range(self.output_width):
                # Note: never out of bound as it is ensured when building remapping
                output_buffer[i, j] = image[remap_table[i][j]]

        return output_buffer.reshape((self.output_height, self.output_width, 3))
```

### Detailed Mathematical Explanation

#### Why Quaternions?

**Brief history**: Quaternions were invented in 1843 by William Hamilton to represent rotations in 3D space. They've since conquered 3D graphics, robotics, aerospace, and... fisheye dewarping.

**Why not classic Euler angles?**

Euler angles (yaw/pitch/roll) are intuitive (well, not really for me, I should probably say "more intuitive") but have several flaws:
- **Gimbal lock**: certain rotation combinations cause a loss of degree of freedom
- **Non-linear interpolation**: difficult to interpolate cleanly between two orientations
- **Heavier computation**: composing rotations requires 3 matrix multiplications (3×3)

Quaternions solve all this:
- **No gimbal lock**: all orientations are representable without singularities
- **Efficient composition**: multiplying two quaternions = 16 multiplications + 12 additions (vs 27 mult + 18 add for 3×3 matrices)
- **Compact**: 4 numbers instead of 9 (3×3 matrix)
- **Easily normalizable**: avoids accumulation of numerical errors

**Quaternion structure**: `[w, x, y, z]` where:
- `w`: scalar part (cosine of half-rotation)
- `(x, y, z)`: vector part (direction of rotation axis)

#### The Three Rotations

Our virtual camera must be able to look in any direction. We decompose this into 3 rotations:

**1. Yaw (rotation around Y)**: camera rotation on its optical axis (here: `yaw=0°`)

**2. Pitch (rotation around X)**: tilt of gaze (here: `pitch=45°` - looking down at 45° from horizontal, ideal for seeing floor + portion of ceiling)

**3. Roll (rotation around Z)**: turning head left/right, sweeping the horizon. This is the rotation we vary: `roll = 360° × zone_id / 5` to cover 360° in 5 views (0°, 72°, 144°, 216°, 288°)

**Composition**: We combine the three rotations by multiplying quaternions:
```python
rq = roll_q × pitch_q × yaw_q
```

Order matters! The rightmost rotation is applied first.

#### From quaternions to projection matrix

Quaternions give us camera orientation, to project pixels we also need to perform a spherical projection, using a spherical projection matrix.

Fo each point `(x, y)` of output image we get a point `(X, Y, Z)` of the virtual half-sphere centered on fisheye camera.

And using a bit of trigonometry, we can relate this `(X, Y, Z)` point to some original point of the fisheye image.


### Benchmark
```
Command: python3 unwarper_python.py fisheye.jpg --repeat-dewarp 1024

======================================================================
BENCHMARK RESULTS
======================================================================
Wall time:              1889.36s
CPU time (user+sys):     1889.34s
  - User time:           1888.94s
  - System time:         0.40s
CPU utilization:        99%
Cores used:             ~1.0
Peak memory:            646.63 MB (662148 KB)
======================================================================

Parallel speedup:       1.00x
(CPU time / Wall time = 1889.36s / 1889.34s)
```

**Results**: 1889 seconds (31 minutes 29 seconds) to process 1 image × 1024 repetitions × 5 views = 5,120 dewarped images. About **369 ms per dewarped view**.

### Analysis

**Performance compared to FFmpeg**:
- FFmpeg: 208.64s for 5,120 views → **40.7 ms/view**
- Pure Python: 1889.36s for 5,120 views → **369 ms/view**
- **Ratio: Pure Python is 9.1× slower than FFmpeg**

But remember FFmpeg uses 6.8 cores while Python uses only 1 core. In terms of **total CPU consumption**:
- FFmpeg: 1411.84s CPU time → **276 ms CPU time/view**
- Pure Python: 1889.34s CPU time → **369 ms CPU time/view**
- **Ratio: Pure Python uses only 1.3× more CPU than FFmpeg!**

This is surprisingly efficient for interpreted Python! The difference comes from FFmpeg's parallel overhead and video decoding/encoding costs.

**✅ Strengths**

- **Readable and understandable code**: Each mathematical step is explicit. Ideal for understanding the algorithm, debugging, or adapting it.
- **Moderate memory consumption**: 647 MB vs 1784 MB for FFmpeg = **2.8× less**. We pre-calculate the lookup table (~9 MB for 5 views) and process one frame at a time.
- **Solid reference baseline**: Correct and verified implementation for comparing optimizations.
- **Easily modifiable**: No compilation needed.
- **CPU efficient**: Only 1.3× more CPU than FFmpeg despite being interpreted Python!

**❌ Weaknesses**

- **Very slow in wall time**: 9.1× slower than FFmpeg - not usable for real-time.
- **Catastrophic Python loops**: 960 × 960 × 5 = 4.6 million Python loop iterations per frame. Each iteration involves interpreted bytecode, which is orders of magnitude slower than native code.
- **No vectorization**: NumPy is only used as a data container. We exploit **none** of the SIMD optimizations.
- **Single-core only**: Python's GIL prevents parallelism (though this keeps CPU consumption low).

### Verdict

Pure Python is **9.1× slower** than FFmpeg in wall time, but only **1.3× worse in CPU consumption**. It's a **pedagogical tool**, not a production solution for latency-critical applications.

**This code is perfect for**:
- Understanding exactly how fisheye dewarping works
- Serving as a reference to verify correctness of optimized implementations
- Quickly prototyping algorithm variations
- Learning the math behind quaternions and spherical projections

**Unsuitable for**:
- Real-time processing (too slow in wall time)
- High-throughput batch processing
- Any use case where latency matters

**The question now**: can we keep Python's simplicity while catching up to FFmpeg? The next section explores NumPy vectorization.

---

## Implementation 3: Vectorized NumPy—The First Performance Boost

### The Approach

We keep exactly the same algorithm as pure Python, but **eliminate all Python loops** by using NumPy's vectorized operations. The idea: let NumPy (written in optimized C) handle the millions of iterations.

The mapping calculation remains identical (quaternions, projection matrix), but the application phase becomes massively parallel thanks to broadcasting and vectorization.

### Two Variants: Ad-hoc vs. Camera Model

I implemented **two variants** of the vectorized approach to explore different projection methods:

**Variant 1:**
- Direct port of the pure Python version with vectorization
- Simpler code, fewer operations

**Variant 2: Camera intrinsics model**
- Uses standard camera matrix `K` with focal length and principal point
- Generates 3D rays from perspective view, transforms to fisheye coordinates
- More "theoretically correct" but slightly more complex
- More familiar for experts of the domain

Both produce identical output and similar performance. I'll show Variant 1 (ad-hoc) as it's shorter and match pure python loop.

### Key Modifications

#### Phase 1: Vectorized Mapping Calculation

**Before (Pure Python)**:
```python
remap_zone = []
for j in range(self.output_height):
    line = []
    for i in range(self.output_width):
        # Calculations for this pixel...
        line.append((src_y, src_x))
    remap_zone.append(line)
```

**After (Vectorized NumPy)**:
```python

# Create grid of ALL coordinates at once
i_coords, j_coords = np.meshgrid(
    np.arange(self.output_width),
    np.arange(self.output_height),
    indexing='xy')

# Globaly fix coordinates from i,j to x,y (Vectorized)
x_coords = i_coords.flatten() * inv_width  - 1.0
y_coords = j_coords.flatten() * inv_height - 1.0
coords = np.column_stack([x_coords, y_coords, np.ones_like(x_coords)]).T

# Apply rotation matrix to ALL coordinates vectors
xyz = R @ coords

# Map 3D point to 2D fisheye point
# Source coordinate calculation for ALL pixels
hs = np.hypot(xyz[0, :],xyz[1, :])
phi = np.arctan2(hs, xyz[2, :])
coeff = phi / (hs * np.pi)
src_x = (self.width * (xyz[0, :] * coeff + 0.5)).astype(np.int32)
src_y = (self.height * (xyz[1, :] * coeff + 0.5)).astype(np.int32)

# Clip to valid range
src_x_int = np.clip(src_x, 0, self.width - 1)
src_y_int = np.clip(src_y, 0, self.height - 1)

# Reshape to 2D arrays and store
src_x_2d = src_x_int.reshape((self.output_height, self.output_width))
src_y_2d = src_y_int.reshape((self.output_height, self.output_width))           
zone_mapping = np.stack([src_x_2d, src_y_2d], axis=-1)
```

**Gain**: Instead of 921,600 Python loop iterations, we have **one** optimized matrix multiplication in C + vectorized operations on all pixels at once. NumPy uses CPU SIMD instructions (SSE, AVX) to process multiple values simultaneously.

It's not really marginal at it allows to reduce computing time by several seconds... but it's a minor gain anyway at is will be done only once for all frames.

#### Phase 2: Vectorized Mapping Application

**Before (Pure Python)**:
```python
for i in range(self.output_height):
    for j in range(self.output_width):
        output[i, j] = image[remap_table[i][j]]
```

**After (Vectorized NumPy)**:
```python
# Extract source coordinates
src_x = remap_table[:, :, 0]
src_y = remap_table[:, :, 1]

# NumPy advanced indexing: copy ALL pixels at once
output_buffer = image[src_y, src_x]
```

**Critical detail**: Why we clip coordinates in Phase 1 instead of using a validity mask.

A naive approach would use a boolean mask:
```python
valid_mask = ((src_y >= 0) & (src_y < height) & (src_x >= 0) & (src_x < width))
output[valid_mask] = image[src_y[valid_mask], src_x[valid_mask]]
```

**Disastrous impact**: This slows down by 2-3×! Why?

The boolean mask breaks **memory locality**. Without the mask, NumPy accesses pixels sequentially, exploiting CPU caches. With the mask, accesses become random and scattered. The CPU spends time waiting for data from RAM instead of computing.

**Elegant solution**: Clip coordinates in Phase 1:
```python
src_x = np.clip(src_x, 0, self.width - 1)
src_y = np.clip(src_y, 0, self.height - 1)
```

Out-of-bounds pixels now point to the image edge (negligible visual artifact) but **all memory accesses remain valid and sequential**. NumPy can aggressively optimize indexing.

### Benchmarks

**Variant 1: Ad-hoc constants**
```
Command: python3 unwarper_numpy.py fisheye.jpg --repeat-dewarp 1024

======================================================================
BENCHMARK RESULTS
======================================================================
Wall time:              110.11s
CPU time (user+sys):     113.83s
  - User time:           113.14s
  - System time:         0.69s
CPU utilization:        103%
Cores used:             ~1.0
Peak memory:            255.73 MB (261868 KB)
======================================================================

Parallel speedup:       1.03x
```

**Variant 2: Camera intrinsics model**
```
Command: python3 unwarper_numpy2.py fisheye.jpg --repeat-dewarp 1024

======================================================================
BENCHMARK RESULTS
======================================================================
Wall time:              115.39s
CPU time (user+sys):     119.93s
  - User time:           119.35s
  - System time:         0.58s
CPU utilization:        103%
Cores used:             ~1.0
Peak memory:            296.11 MB (303212 KB)
======================================================================

Parallel speedup:       1.04x
```

**Results**: 
- Variant 1: 110.11s for 5,120 views → **21.5 ms/view**
- Variant 2: 115.39s for 5,120 views → **22.5 ms/view**

Both variants perform nearly identically. The camera model variant is slightly slower due to extra matrix operations but more "theoretically sound."

### Analysis

**Comparison with previous implementations**:
- FFmpeg: 208.64s wall, 1411.84s CPU (6.8 cores) → **40.7 ms/view** wall, **276 ms CPU/view**
- Pure Python: 1889.36s wall, 1889.34s CPU (1.0 core) → **369 ms/view**
- **NumPy vectorized: 110.11s wall, 113.83s CPU (1.0 core) → 21.5 ms/view wall, 22.2 ms CPU/view**

**Massive gains**:
- **17.2× faster** than pure Python in wall time (1889.36s → 110.11s)
- **17.0× less CPU** than pure Python (1889.34s → 113.83s)
- **1.9× faster** than FFmpeg in wall time (208.64s → 110.11s)
- **12.4× less CPU** than FFmpeg (1411.84s → 113.83s)

NumPy vectorization **crushes both pure Python AND FFmpeg** in terms of efficiency!

**Why is NumPy so efficient?**

✅ **Optimized C code**: NumPy operations are implemented in highly optimized C with `-O3` compilation.

✅ **SIMD vectorization**: Mathematical calculations (`hypot`, `arctan2`) exploit AVX instructions to process 4-8 float64 simultaneously.

✅ **Memory locality**: Vectorized operations access memory sequentially, maximizing cache efficiency.

✅ **Single-core efficiency**: No parallelization overhead. All computation is pure calculation, no thread management.

**Why isn't it faster in wall time than FFmpeg?**

FFmpeg uses 6.8 cores (6.77× speedup) while NumPy uses 1.0 core. FFmpeg's wall time advantage comes purely from parallelization, **not from more efficient code**. In fact, NumPy uses **12.4× less total CPU** than FFmpeg!

If you run multiple dewarping processes in parallel (realistic for processing multiple camera streams), NumPy would scale much better than FFmpeg because:
- Lower CPU per stream (113.83s vs 1411.84s)
- Lower memory per stream (256 MB vs 1784 MB)
- No thread contention between processes

**✅ Strengths**

- **Excellent performance**: 1.9× faster wall time than FFmpeg, **12.4× less CPU consumption**
- **Still pure Python**: No compilation, easy to modify and debug
- **Very low memory**: 256 MB vs 1784 MB (FFmpeg) = **7× less**
- **CPU efficient**: Single-core design means zero parallelization overhead
- **Scales better**: For processing N streams in parallel, NumPy's low CPU footprint wins

**❌ Weaknesses**

- **Single-core only**: Can't exploit multi-core for a single stream (but this is actually an advantage for multi-stream scenarios)
- **NumPy dependency**: Requires NumPy + BLAS backend (OpenBLAS or MKL)
- **Learning curve**: Broadcasting, advanced indexing, performance traps (validity mask issue)

### Verdict

Vectorized NumPy delivers **stunning results**: 17× faster than pure Python, and **more efficient than FFmpeg** (1.9× faster in wall time, 12.4× less CPU).

The two variants (ad-hoc vs. camera model) perform nearly identically, showing that the projection method matters less than the vectorization itself.

**This implementation is ideal for**:
- Production environments processing multiple camera streams in parallel
- CPU-constrained servers where total CPU consumption matters more than single-stream latency
- Python-based pipelines where adding C++ would be complicated
- Applications where 21 ms/view is fast enough (which it is for most real-time scenarios)

**Limitations**:
- Single-stream latency is higher than a parallelized solution
- Can't leverage multi-core for individual stream processing

**But wait—can we go even faster?** Part 2 explores three more implementations: OpenCV Python (with multi-core `cv2.remap()`), OpenCV C++ (native compiled code), and a custom C++ library that achieves **42× faster** than FFmpeg. How? Read on in Part 2!

---

## What's Next in Part 2

In the next article, we'll explore three more implementations that push performance even further:

1. **OpenCV Python**: Using `cv2.remap()` with multi-core parallelization (~4.8 cores)
2. **OpenCV C++**: Native compiled code eliminating Python overhead
3. **Custom C++ Library**: The ultimate optimization—42× faster than FFmpeg with only 80 MB RAM

We'll discover:
- How OpenCV's multi-threading compares to NumPy's single-core efficiency
- Whether native C++ provides meaningful gains over Python bindings
- What optimizations enable the custom library to achieve 4.91s for 5,120 views (0.96 ms/view!)

**Key insights from Part 1**:
- FFmpeg: Fast but memory-hungry (1.78 GB), high CPU consumption (1411s)
- Pure Python: Slow but surprisingly CPU-efficient (only 1.3× worse than FFmpeg per core)
- NumPy: The efficiency champion—12.4× less CPU than FFmpeg, 7× less memory

**Spoiler alert**: The custom C++ library will process the same workload in just **4.91 seconds** using **1.1 cores** and **80 MB RAM**. That's:
- **42.5× faster** than FFmpeg in wall time
- **256× less CPU** consumption than FFmpeg
- **22× less memory** than FFmpeg

How is this possible? Find out in Part 2!

---

**Full code for all implementations**: [github.com/pykoder/fisheye-dewarping](https://github.com/pykoder/fisheye-dewarping)

*Article written in December 2025. Benchmarks performed on a Lenovo ThinkPad P14s - Ubuntu 25.04, Intel Core i7-1185G7 (4 physical cores, 8 threads), 16GB RAM. All tests process 1024 frames × 5 views = 5,120 dewarped images.*