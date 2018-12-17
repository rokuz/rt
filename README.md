# Overview
Simple C++ ray tracing framework and demos.

# Features
* Multithreading support;
* Supersampling support;
* CUDA support (Windows only).

# Demos
## 1. Pretty spheres
![Pretty spheres Demo](screenshots/pretty_spheres.png?raw=true "Pretty spheres Demo")
Here we render a bunch of spheres with random color, roughness and refraction index.
* Global illumination - scattering on the matte and reflective surfaces;
* Directional light source - light (shadow) rays;
* Specularity - Cook-Torrance approximation.
## 2. Glass spheres
![Glass spheres Demo](screenshots/glass_spheres.png?raw=true "Glass spheres Demo")
Everything as in Pretty spheres demo + refraction on glass surfaces.

## Command line arguments
-w - window's width (default = 1024);

-h - window's height (default = 768);

-s - supersampling samples (in row) count (default = 2, 2x2 samples here);

-t - ray tracing threads count (default = 4);

-d - demo index (default = 1);

-c - enable CUDA acceleration (only Windows supported).

## Controls
* Esc - exit;
* T - rerender scene;
* Q - rerender scene in high quality (at least 10x10 samples).

# Supported platforms
* Windows (msvc in Visual Studio 2017);
* MacOS (clang).

# Requirements
* CMake 3.12+;
* CUDA compatible device (for Windows with CUDA_ENABLE flag in CMake, tested on CUDA 10.0);
* OpenGL 4.1 support.

# License
You can find it [here](LICENSE)

Copyright (c) 2018 Roman Kuznetsov aka rokuz
