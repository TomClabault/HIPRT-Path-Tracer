/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include <Device/includes/fix_intellisense.h>
#include <Device/includes/sampling.h>
#include <HostDeviceCommon/camera.h>
#include <HostDeviceCommon/math.h>
#include <HostDeviceCommon/render_data.h>

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

	int index_A = render_data.buffers.triangles_indices[hit.primID * 3 + 0];
	int index_B = render_data.buffers.triangles_indices[hit.primID * 3 + 1];
	int index_C = render_data.buffers.triangles_indices[hit.primID * 3 + 2];

	float3 vertex_A = render_data.buffers.triangles_vertices[index_A];
	float3 vertex_B = render_data.buffers.triangles_vertices[index_B];
	float3 vertex_C = render_data.buffers.triangles_vertices[index_C];

	float3 normal;
	if (render_data.buffers.normals_present[index_A])
	{
		// Smooth normal
		float3 smooth_normal = render_data.buffers.vertex_normals[index_B] * hit.uv.x
			+ render_data.buffers.vertex_normals[index_C] * hit.uv.y
			+ render_data.buffers.vertex_normals[index_A] * (1.0f - hit.uv.x - hit.uv.y);

		normal = hiprtpt::normalize(smooth_normal);
	}
	else
		normal = hiprtpt::normalize(hiprtpt::cross(vertex_B - vertex_A, vertex_C - vertex_A));

	Color final_color(hit.hasHit() ? hiprtpt::abs(normal) : Color(0.0f, 0.0f, 0.0f));
	render_data.buffers.pixels[index] = final_color * (render_data.render_settings.sample_number + 1);
}