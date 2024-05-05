# HIPRT-Path-Tracer

![HIPRT path tracer cover](README_data/img/McLaren_P1_Render.jpg)

Physically based Monte Carlo path tracer written with the [HIP RT](https://gpuopen.com/hiprt/) and [Orochi](https://gpuopen.com/orochi/) libraries.

HIPRT is AMD's equivalent to [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix). It allows the use of the ray tracing accelerators of RDNA2+ AMD GPUs and can run on NVIDIA devices as well (although it wouldn't take advatange of RT cores) as it is not AMD specific. 

The Orochi library allows the loading of HIP and CUDA libraries at runtime meaning that the application doesn't have to be recompiled to be used on a GPU from a different vendor (unlike HIP which would require a recompilation + linking).

# System requirements

- AMD RDNA1 GPU or newer (RX 5000 or newer) **or** NVIDIA Maxwell GPU or newer (GTX 700 & GTX 900 Series or newer)
- Visual Studio 2022 (only version tested but older versions might work as well)

# Building
## Preliminary setup
### Windows
#### - AMD GPUs
Nothing to do, go to the building step.
#### - NVIDIA GPUs
To build the project on NVIDIA hardware, you will need to install the NVIDIA CUDA SDK v12.2. It can be downloaded and installed from [here](https://developer.nvidia.com/cuda-12-2-0-download-archive).

The CMake build then expects the CUDA_PATH environment variable to be defined. This should automatically be the case after installing the CUDA Toolkit but just in case, you can define it yourself such that CUDA_PATH/include/cuda.h is a valid file path.

### Linux

All that follows has only been tested on Ubuntu 22.04.
#### - AMD GPUs

1) Install OpenGL, GLFW and glew dependencies:

```sh
sudo apt install freeglut3-dev
sudo apt install libglfw3-dev
sudo apt install libglew-dev
```

2) Install AMD HIP (if you already have ROCm installed, you should have a `/opt/rocm` folder on your system and you can skip this step):

Download `amdgpu-install` package: https://www.amd.com/en/support/linux-drivers
Install the package: 

```sh
sudo apt install ./amdgpu-install_xxxx.deb
```

Install HIP: 

```sh
sudo amdgpu-install --usecase=hip
```

3) To access GPU hardware without having to run your app as sudo everytime, add the user to the `render` group and **reboot** :

```sh
sudo usermod -a -G render $LOGNAME
```
#### - NVIDIA GPUs

Not yet tested. If you have an NVIDIA GPU and want to try and compile the project on your Linux installation, please feel free to do so. You could then open a merge request to add installation instructions to the README, that would be greatly appreciated!
## Compiling

With the preliminary setup done, you now just have to run the CMake:

``` sh
git clone https://github.com/TomClabault/HIPRT-Path-Tracer.git
cd HIPRT-Path-Tracer
mkdir build
cd build
cmake ..
```

On Windows, a Visual Studio solution will be generated in the `build` folder that you can open and compile the project with.

On Linux, the executable will be generated in the `build` folder.
# License

GNU General Public License v3.0 or later

See [COPYING](https://github.com/TomClabault/HIPRT-Path-Tracer/blob/main/COPYING) to see the full text.