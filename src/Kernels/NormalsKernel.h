#include "Kernels/HIPRTMaths.h"
#include "Kernels/HIPRTCommon.h"

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>


GLOBAL_KERNEL_SIGNATURE(void) NormalsKernel(hiprtGeometry geom, float* pixels, int2 res, Camera camera)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * res.x;

	hiprtRay ray = get_camera_ray(camera, x, y, res.x, res.y);

	hiprtGeomTraversalClosest tr(geom, ray);
	hiprtHit				  hit = tr.getNextHit();

	hiprtFloat3 normal = normalize(hit.normal);

	pixels[index * 4 + 0] = hit.hasHit() ? normal.x : 0.0f;
	pixels[index * 4 + 1] = hit.hasHit() ? normal.y : 0.0f;
	pixels[index * 4 + 2] = hit.hasHit() ? normal.z : 0.0f;
	pixels[index * 4 + 3] = 1.0f;
}