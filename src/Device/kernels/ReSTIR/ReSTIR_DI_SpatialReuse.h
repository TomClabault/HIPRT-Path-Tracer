/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_SPATIAL_REUSE_H
#define DEVICE_RESTIR_DI_SPATIAL_REUSE_H 

#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Sampling.h"

#include "HostDeviceCommon/Camera.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

 /** References:
 *
 * [1] [Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting] https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal/
 * [2] [A Gentle Introduction to ReSTIR: Path Reuse in Real-time] https://intro-to-restir.cwyman.org/
 * [3] [A Gentle Introduction to ReSTIR: Path Reuse in Real-time - SIGGRAPH 2023 Presentation Video] https://dl.acm.org/doi/10.1145/3587423.3595511#sec-supp
 * [4] [NVIDIA RTX DI SDK - Github] https://github.com/NVIDIAGameWorks/RTXDI
 * [5] [Uniform disk sampling] https://rh8liuqy.github.io/Uniform_Disk.html
 */

#define SPATIAL_REUSE_PASSES_COUNT 1
#define NEIGHBOR_REUSE_COUNT 5
#define REUSE_RADIUS 30

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReSTIR_DI_SpatialReuse(HIPRTRenderData render_data, int2 res, HIPRTCamera camera)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_DI_SpatialReuse(HIPRTRenderData render_data, int2 res, HIPRTCamera camera, int x, int y)
#endif
{
#ifdef __KERNELCC__
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
	uint32_t pixel_index = (x + y * res.x);
	if (pixel_index >= res.x * res.y)
		return;

	unsigned int seed;
	if (render_data.render_settings.freeze_random)
		seed = wang_hash(pixel_index + 1);
	else
		seed = wang_hash((pixel_index + 1) * (hippt::min(1u, render_data.random_seed)));

	Xorshift32Generator random_number_generator(seed);

	Reservoir new_reservoir;

	int2 neighbor_offsets[NEIGHBOR_REUSE_COUNT];
	int2 pixel_coords = make_int2(x, y);
	float3 inter_point = render_data.g_buffer.first_hits[pixel_index];
	float3 shading_normal = render_data.g_buffer.shading_normals[pixel_index];
	float3 evaluated_point = inter_point + shading_normal * 1.0e-4f;
	float3 view_direction = render_data.g_buffer.view_directions[pixel_index];
	RendererMaterial material = RendererMaterial(render_data.g_buffer.materials[pixel_index]);

	for (int i = 0; i < NEIGHBOR_REUSE_COUNT; i++)
		neighbor_offsets[i] = uniform_sample_in_disk(REUSE_RADIUS, random_number_generator);

	unsigned int total_sample_count_mis_weight = 0;
	for (int i = 0; i < NEIGHBOR_REUSE_COUNT + 1; i++)
	{
		int neighbor_pixel_index;

		if (i == NEIGHBOR_REUSE_COUNT)
			neighbor_pixel_index = pixel_coords.x + pixel_coords.y * res.x;
		else
		{
			int2 neighbor_offset = neighbor_offsets[i];
			int2 neighbor_pixel_coords = pixel_coords + neighbor_offset;
			if (neighbor_pixel_coords.x < 0 || neighbor_pixel_coords.x >= res.x || neighbor_pixel_coords.y < 0 || neighbor_pixel_coords.y >= res.y)
				// Rejecting the sample if it's outside of the viewport
				continue;

			neighbor_pixel_index = neighbor_pixel_coords.x + neighbor_pixel_coords.y * res.x;
		}

		Reservoir neighbor_reservoir = render_data.aux_buffers.initial_reservoirs[neighbor_pixel_index];
		total_sample_count_mis_weight += neighbor_reservoir.M;
	}

	for (int i = 0; i < NEIGHBOR_REUSE_COUNT + 1; i++)
	{
		int neighbor_pixel_index;

		if (i == NEIGHBOR_REUSE_COUNT)
			neighbor_pixel_index = pixel_coords.x + pixel_coords.y * res.x;
		else
		{
			int2 neighbor_offset = neighbor_offsets[i];
			int2 neighbor_pixel_coords = pixel_coords + neighbor_offset;
			if (neighbor_pixel_coords.x < 0 || neighbor_pixel_coords.x >= res.x || neighbor_pixel_coords.y < 0 || neighbor_pixel_coords.y >= res.y)
				// Rejecting the sample if it's outside of the viewport
				continue;

			neighbor_pixel_index = neighbor_pixel_coords.x + neighbor_pixel_coords.y * res.x;
		}

		Reservoir neighbor_reservoir = render_data.aux_buffers.initial_reservoirs[neighbor_pixel_index];
		SimplifiedRendererMaterial neighbor_mat = render_data.g_buffer.materials[neighbor_pixel_index];

		float bsdf_pdf;
		float distance_to_light;
		float3 to_light_direction = neighbor_reservoir.sample.point_on_light_source - evaluated_point;
		float3 sample_direction = to_light_direction / (distance_to_light = hippt::length(to_light_direction));

		RayVolumeState trash_volume_state;
		ColorRGB bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, shading_normal, sample_direction, bsdf_pdf);
		float target_function = bsdf_color.length() * neighbor_reservoir.sample.emission.length() * hippt::max(0.0f, hippt::dot(shading_normal, sample_direction));

		new_reservoir.combine_with(neighbor_reservoir, 1.0f / total_sample_count_mis_weight, target_function, random_number_generator);
	}

	new_reservoir.end();
	
	render_data.aux_buffers.spatial_reservoirs[pixel_index] = new_reservoir;
}

#endif
