# HIPRT-Path-Tracer

![HIPRT path tracer cover](README_data/img/McLaren_P1_Render.jpg)

Physically based Monte Carlo path tracer written with the [HIP RT](https://gpuopen.com/hiprt/) and [Orochi](https://gpuopen.com/orochi/) libraries.
HIPRT is AMD's equivalent to [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix). It allows the use of the ray tracing accelerators of RDNA2+ AMD GPUs and can run on NVIDIA devices as well (although it wouldn't take advatange of RT cores) as it is not AMD specific. 
Thanks to the Orochi library, device kernels are compiled at run time and the application doesn't have to be recompiled to be used on a different device.

# Requires Git LFS! (for now)

# License

GNU General Public License v3.0 or later

See [COPYING](https://github.com/TomClabault/HIPRT-Path-Tracer/blob/main/COPYING) to see the full text.