# HIPRT-Path-Tracer

![HIPRT path tracer cover](README_data/img/McLaren_P1_Render.jpg)

Physically based Monte Carlo path tracer written with the [HIP RT](https://gpuopen.com/hiprt/) and [Orochi](https://gpuopen.com/orochi/) libraries.

HIPRT is AMD's equivalent to [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix). It allows the use of the ray tracing accelerators of RDNA2+ AMD GPUs and can run on NVIDIA devices as well (although it wouldn't take advatange of RT cores) as it is not AMD specific. 

The Orochi library allows device kernels to be compiled at run time and the application doesn't have to be recompiled to be used on a different device (unlike HIP which would require a recompilation).

# System requirements

- AMD RDNA1 GPU or newer (RX 5000 or newer) **or** NVIDIA Maxwell GPU or newer (GTX 700 & GTX 900 Series or newer)
- Visual Studio 2022 (only version tested but older versions might work as well)

# Building
## Windows
### - AMD GPUs
Building should be straightforward:
``` sh
git clone https://github.com/TomClabault/HIPRT-Path-Tracer.git
cd HIPRT-Path-Tracer
mkdir build
cd build
cmake ..
```
### - NVIDIA GPUs
To build the project on NVIDIA hardware, you will need to install the NVIDIA CUDA SDK v12.2. It can be downloaded and installed from [here](https://developer.nvidia.com/cuda-12-2-0-download-archive).
The CMake build then expects the CUDA_PATH environment variable to be defined. This should automatically be the case after installing the CUDA Toolkit but just in case, you can define it yourself such that CUDA_PATH/include/cuda.h is a valid file path.

### ---
For both NVIDIA and AMD, a Visual Studio solution will be generated in the build/ folder that you can open and compile the project with.

## Linux

Not yet supported.
# License

GNU General Public License v3.0 or later

See [COPYING](https://github.com/TomClabault/HIPRT-Path-Tracer/blob/main/COPYING) to see the full text.