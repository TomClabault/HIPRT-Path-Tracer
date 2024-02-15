#include "Kernels/includes/HIPRT_camera.h"
#include "Kernels/includes/HIPRT_common.h"
#include "Kernels/includes/HIPRT_maths.h"
#include "Kernels/includes/hiprt_render_data.h"

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>


GLOBAL_KERNEL_SIGNATURE(void) NormalsKernel(hiprtGeometry geom, HIPRTRenderData scene_geometry, HIPRTColor* pixels, int2 res, HIPRTCamera camera)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = (x + y * res.x);

	if (index >= res.x * res.y)
		return;

	hiprtRay ray = camera.get_camera_ray(x, y, res);

	hiprtGeomTraversalClosest tr(geom, ray);
	hiprtHit				  hit = tr.getNextHit();

	int index_A = scene_geometry.triangles_indices[hit.primID * 3 + 0];
	int index_B = scene_geometry.triangles_indices[hit.primID * 3 + 1];
	int index_C = scene_geometry.triangles_indices[hit.primID * 3 + 2];

	hiprtFloat3 vertex_A = scene_geometry.triangles_vertices[index_A];
	hiprtFloat3 vertex_B = scene_geometry.triangles_vertices[index_B];
	hiprtFloat3 vertex_C = scene_geometry.triangles_vertices[index_C];

	hiprtFloat3 normal = hiprtFloat3{ 0.5f, 0.5f, 0.5f } * normalize(cross(vertex_B - vertex_A, vertex_C - vertex_A)) + hiprtFloat3{ 0.5f, 0.5f, 0.5f };

	HIPRTColor color{ hit.hasHit() ? normal.x : 0.0f, hit.hasHit() ? normal.y : 0.0f, hit.hasHit() ? normal.z : 0.0f };
	pixels[index] = color * (scene_geometry.render_settings.frame_number + 1);
}