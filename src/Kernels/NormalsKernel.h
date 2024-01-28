#include "Kernels/HIPRTMaths.h"

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

extern "C"
void __global__ NormalsKernel(hiprtGeometry geom, float* pixels, int2 res)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	float3 o = {x / static_cast<float>(res.x), y / static_cast<float>(res.y), 10.0f};
	float3 d = { 0.0f, 0.0f, -1.0f };

	hiprtRay ray;
	ray.origin = o;
	ray.direction = d;

	hiprtGeomTraversalClosest tr(geom, ray);
	hiprtHit				  hit = tr.getNextHit();

	pixels[index * 4 + 0] = (static_cast<float>(x) / res.x) * 255.0f;//hit.hasHit() ? (static_cast<float>(x) / res.x) * 255.0f : 0.0f;
	pixels[index * 4 + 1] = (static_cast<float>(y) / res.y) * 255.0f;// hit.hasHit() ? (static_cast<float>(y) / res.y) * 255.0f : 0.0f;
	pixels[index * 4 + 2] = 0.0f;
	pixels[index * 4 + 3] = 255.0f;
}
