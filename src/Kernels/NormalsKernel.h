#include "Kernels/HIPRTMaths.h"

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

// TODO put this in a Utils header

// The HIPRT_KERNEL_SIGNATURE is only useful to help Visual Studio's Intellisense
// Without this macro, all kernel functions would be declared as:
// extern "C" void __global__ my_function(......)
// but Visual Studio doesn't like the 'extern "C" void __global__' part and it
// breaks code coloration and autocompletion. It is however required for the shader
// compiler
// To circumvent this problem, we're only declaring the functions 'void' when in the text editor
// (when __KERNELCC__ is not defined) and we're correctly declaring the function with the full
// attributes when it's the shader compiler processing the function (when __KERNELCC__ is defined)
#ifdef __KERNELCC__
#define HIPRT_KERNEL_SIGNATURE extern "C" void __global__
#else
#define HIPRT_KERNEL_SIGNATURE void
#endif

HIPRT_KERNEL_SIGNATURE NormalsKernel(hiprtGeometry geom, float* pixels, int2 res)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	float3 o = {x / static_cast<float>(res.x), y / static_cast<float>(res.y), 4.0f};
	float3 d = { 0.0f, 0.0f, -1.0f };

	hiprtRay ray;
	ray.origin = o;
	ray.direction = d;

	hiprtGeomTraversalClosest tr(geom, ray);
	hiprtHit				  hit = tr.getNextHit();

	hiprtFloat3 normal = normalize(hit.normal);

	pixels[index * 4 + 0] = hit.hasHit() ? normal.x : 0.0f;
	pixels[index * 4 + 1] = hit.hasHit() ? normal.y : 0.0f;
	pixels[index * 4 + 2] = hit.hasHit() ? normal.z : 0.0f;
	pixels[index * 4 + 3] = 1.0f;
}