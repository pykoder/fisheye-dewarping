
# ffmpeg CLI

🔍 Commande: 01_ffmpeg/unwarper_ffmpeg.sh

======================================================================
📈 RÉSULTATS BENCHMARK
======================================================================
⏱️  Wall time:              208.64s
⚙️  CPU time (user+sys):     1411.84s
    ├─ User time:           1401.84s
    └─ System time:         10.00s
🔥 CPU utilization:        676%
💻 Cores utilisés:         ~6.8
🧠 Mémoire pic:            1784.20 MB (1827016 KB)
📄 Page faults:            387273 minor, 0 major
🔄 Context switches:       328327 vol, 712451 invol
✅ Exit status:            0
======================================================================

💡 Speedup parallèle:      6.77x
   (CPU time / Wall time = 1411.84s / 208.64s)


# Pur python avec boucles


🔍 Commande: uv run 02_python/unwarper_python.py ../images/fisheye.jpg --repeat-dewarp 1024

======================================================================
📈 RÉSULTATS BENCHMARK
======================================================================
⏱️  Wall time:              1889.36s
⚙️  CPU time (user+sys):     1889.34s
    ├─ User time:           1888.94s
    └─ System time:         0.40s
🔥 CPU utilization:        99%
💻 Cores utilisés:         ~1.0
🧠 Mémoire pic:            646.63 MB (662148 KB)
📄 Page faults:            172286 minor, 1 major
🔄 Context switches:       36 vol, 40265 invol
✅ Exit status:            0
======================================================================

# numpy avec vectorisation

## numpy avec constantes expérimentales + trigo 
🚀 Lancement du benchmark...

🔍 Commande: uv run 03_numpy/unwarper_numpy.py ../images/fisheye.jpg --repeat-dewarp 1024

======================================================================
📈 RÉSULTATS BENCHMARK
======================================================================
⏱️  Wall time:              110.11s
⚙️  CPU time (user+sys):     113.83s
    ├─ User time:           113.14s
    └─ System time:         0.69s
🔥 CPU utilization:        103%
💻 Cores utilisés:         ~1.0
🧠 Mémoire pic:            255.73 MB (261868 KB)
📄 Page faults:            51669 minor, 0 major
🔄 Context switches:       87 vol, 3731 invol
✅ Exit status:            0
======================================================================

💡 Speedup parallèle:      1.03x
   (CPU time / Wall time = 113.83s / 110.11s)

## numpy avec calcul de rayons pour le pov

🚀 Lancement du benchmark...

🔍 Commande: uv run 03_numpy/unwarper_numpy2.py ../images/fisheye.jpg --repeat-dewarp 1024

======================================================================
📈 RÉSULTATS BENCHMARK
======================================================================
⏱️  Wall time:              115.39s
⚙️  CPU time (user+sys):     119.93s
    ├─ User time:           119.35s
    └─ System time:         0.58s
🔥 CPU utilization:        103%
💻 Cores utilisés:         ~1.0
🧠 Mémoire pic:            296.11 MB (303212 KB)
📄 Page faults:            74156 minor, 8 major
🔄 Context switches:       84 vol, 4101 invol
✅ Exit status:            0
======================================================================

💡 Speedup parallèle:      1.04x
   (CPU time / Wall time = 119.93s / 115.39s)

# Version avec fonction remap de opencv 

## Version opencv interpolation au point le plus proche

🚀 Lancement du benchmark...

🔍 Commande: uv run 04_opencv/unwarper_opencv.py ../images/fisheye.jpg --repeat-dewarp 1024

======================================================================
📈 RÉSULTATS BENCHMARK
======================================================================
⏱️  Wall time:              22.21s
⚙️  CPU time (user+sys):     105.70s
    ├─ User time:           70.10s
    └─ System time:         35.60s
🔥 CPU utilization:        475%
💻 Cores utilisés:         ~4.8
🧠 Mémoire pic:            288.06 MB (294976 KB)
📄 Page faults:            53982 minor, 0 major
🔄 Context switches:       34946 vol, 614194 invol
✅ Exit status:            0
======================================================================

💡 Speedup parallèle:      4.76x
   (CPU time / Wall time = 105.70s / 22.21s)

## Version opencv interpolation linéaire

🚀 Lancement du benchmark...

🔍 Commande: uv run 04_opencv/unwarper_opencv2.py ../images/fisheye.jpg --repeat-dewarp 1024

INTERP_LINEAR : interpolation linéaire

======================================================================
📈 RÉSULTATS BENCHMARK
======================================================================
⏱️  Wall time:              31.80s
⚙️  CPU time (user+sys):     160.71s
    ├─ User time:           123.99s
    └─ System time:         36.72s
🔥 CPU utilization:        505%
💻 Cores utilisés:         ~5.0
🧠 Mémoire pic:            288.20 MB (295120 KB)
📄 Page faults:            54040 minor, 0 major
🔄 Context switches:       35566 vol, 404702 invol
✅ Exit status:            0
======================================================================

💡 Speedup parallèle:      5.05x
   (CPU time / Wall time = 160.71s / 31.80s)

## Opencv full C++

🚀 Lancement du benchmark...

🔍 Commande: 05_opencv_cpp/unwarper_cpp/unwarper ../images/fisheye.jpg --repeat-dewarp 1024

======================================================================
📈 RÉSULTATS BENCHMARK
======================================================================
⏱️  Wall time:              10.09s
⚙️  CPU time (user+sys):     48.90s
    ├─ User time:           44.30s
    └─ System time:         4.60s
🔥 CPU utilization:        484%
💻 Cores utilisés:         ~4.8
🧠 Mémoire pic:            110.01 MB (112652 KB)
📄 Page faults:            23966 minor, 0 major
🔄 Context switches:       138 vol, 2888 invol
✅ Exit status:            0
======================================================================

💡 Speedup parallèle:      4.85x
   (CPU time / Wall time = 48.90s / 10.09s)

## Manually optimized C++ library called via ctypes

🚀 Lancement du benchmark...

🔍 Commande: uv run 06_unwarper_ctypes/unwarper_ctypes.py ../images/fisheye.jpg --repeat-dewarp 1024

======================================================================
📈 RÉSULTATS BENCHMARK
======================================================================
⏱️  Wall time:              4.91s
⚙️  CPU time (user+sys):     5.52s
    ├─ User time:           5.48s
    └─ System time:         0.04s
🔥 CPU utilization:        112%
💻 Cores utilisés:         ~1.1
🧠 Mémoire pic:            80.14 MB (82068 KB)
📄 Page faults:            27204 minor, 0 major
🔄 Context switches:       31 vol, 112 invol
✅ Exit status:            0
======================================================================

💡 Speedup parallèle:      1.12x
   (CPU time / Wall time = 5.52s / 4.91s)
