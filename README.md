# Overview
Simple C++ ray tracing framework and demos.

# Features
* Multithreading support;
* Supersampling support.

# Demos
## Pretty spheres
![Pretty spheres Demo](screenshots/pretty_spheres.png?raw=true "Pretty spheres Demo")
![Pretty spheres Demo](screenshots/pretty_spheres2.png?raw=true "Pretty spheres Demo")
Here we render a bunch of spheres with random color, roughness and refraction index.
* Global illumination - scattering on the matte and reflective surfaces;
* Directional light source - light (shadow) rays;
* Specularity - Cook-Torrance approximation.

## Command line arguments
-w - window's width (default = 1024);

-h - window's height (default = 768);

-s - supersampling samples (in row) count (default = 2, 2x2 samples here);

-t - ray tracing threads count (default = 4).
## Controls
* Esc - exit;
* T - rerender scene;
* Q - rerender scene in high quality (at least 10x10 samples).

# Supported platforms
* Windows (msvc in Visual Studio 2017);
* MacOS (clang).

# Requirements
* CMake 3.12+;
* OpenGL 4.1 support.

# License
You can find it [here](LICENSE)

Copyright (c) 2018 Roman Kuznetsov aka rokuz
