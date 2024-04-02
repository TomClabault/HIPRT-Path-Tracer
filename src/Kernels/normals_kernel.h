#include "Kernels/includes/HIPRT_camera.h"
#include "Kernels/includes/HIPRT_common.h"
#include "Kernels/includes/HIPRT_maths.h"
#include "Kernels/includes/hiprt_render_data.h"
#include "Kernels/includes/hiprt_sampling.h"

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>


GLOBAL_KERNEL_SIGNATURE(void) NormalsKernel(hiprtGeometry geom, HIPRTRenderData render_data, int2 res, HIPRTCamera camera)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = (x + y * res.x);

	if (index >= res.x * res.y)
		return;

	hiprtRay ray = camera.get_camera_ray(x, y, res);

	hiprtGeomTraversalClosest tr(geom, ray);
	hiprtHit				  hit = tr.getNextHit();

	int index_A = render_data.triangles_indices[hit.primID * 3 + 0];
	int index_B = render_data.triangles_indices[hit.primID * 3 + 1];
	int index_C = render_data.triangles_indices[hit.primID * 3 + 2];

	hiprtFloat3 vertex_A = render_data.triangles_vertices[index_A];
	hiprtFloat3 vertex_B = render_data.triangles_vertices[index_B];
	hiprtFloat3 vertex_C = render_data.triangles_vertices[index_C];

	hiprtFloat3 normal;
	if (render_data.normals_present[index_A])
	{
		// Smooth normal
		hiprtFloat3 smooth_normal = render_data.vertex_normals[index_B] * hit.uv.x
			+ render_data.vertex_normals[index_C] * hit.uv.y
			+ render_data.vertex_normals[index_A] * (1.0f - hit.uv.x - hit.uv.y);

		normal = normalize(smooth_normal);
	}
	else
		normal = normalize(cross(vertex_B - vertex_A, vertex_C - vertex_A));

	Color final_color(hit.hasHit() ? abs(normal) : hiprtFloat3{ 0.0f, 0.0f, 0.0f });
	render_data.pixels[index] = final_color * (render_data.m_render_settings.sample_number + 1);
}