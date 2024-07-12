/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_SPATIAL_REUSE_H
#define DEVICE_RESTIR_DI_SPATIAL_REUSE_H 

#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Intersect.h"
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
#define NEIGHBOR_REUSE_COUNT 1
#define REUSE_RADIUS 30

/**
 * Target function is BSDF * Le * V
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_DI_evaluate_target_function(const HIPRTRenderData& render_data, const ReservoirSample& sample, const RendererMaterial& material, float3 view_direction, float3 shading_point, float3 shading_normal)
{
	RayVolumeState trash_volume_state;

	float bsdf_pdf;
	float distance_to_light;
	float3 sample_direction = sample.point_on_light_source - shading_point;
	distance_to_light = hippt::length(sample_direction);
	sample_direction = sample_direction / distance_to_light;

	ColorRGB bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, shading_normal, sample_direction, bsdf_pdf);
	float cosine_term = hippt::max(0.0f, hippt::dot(shading_normal, sample_direction));
	float target_function = (bsdf_color * sample.emission * cosine_term).luminance();
	if (target_function == 0.0f)
		// Quick exit because computing the visiblity that follows isn't going
		// to change anything to the fact that we have 0.0f target function here
		return 0.0f;

#if RISUseVisiblityTargetFunction == RIS_USE_VISIBILITY_TRUE
	hiprtRay shadow_ray;
	shadow_ray.origin = shading_point;
	shadow_ray.direction = sample_direction;

	bool visible = !evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);

	target_function *= visible;
#endif

	return target_function;
}

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
		seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1));
	
	Xorshift32Generator random_number_generator(seed);

	Reservoir new_reservoir;
	int2 neighbor_offsets[NEIGHBOR_REUSE_COUNT];
	int2 pixel_coords = make_int2(x, y);

	for (int i = 0; i < NEIGHBOR_REUSE_COUNT; i++)
	{
		// Generating neighbor screen space position and storing in an array. Inefficient but easy implementation for now.
		neighbor_offsets[i] = integer_sample_in_disk(REUSE_RADIUS, random_number_generator);
		neighbor_offsets[i].x = 10; //hippt::abs(neighbor_offsets[i].x) + 2;
		neighbor_offsets[i].y = 0;
	}

	RendererMaterial current_pixel_material = render_data.g_buffer.materials[pixel_index];
	float3 current_pixel_view_direction = render_data.g_buffer.view_directions[pixel_index];
	float3 current_pixel_shading_normal = render_data.g_buffer.shading_normals[pixel_index];
	float3 current_pixel_shading_point = render_data.g_buffer.first_hits[pixel_index] + current_pixel_shading_normal * 1.0e-4f;

	for (int neighbor = 0; neighbor < NEIGHBOR_REUSE_COUNT; neighbor++)
	{
		int neighbor_pixel_index;

		if (neighbor == NEIGHBOR_REUSE_COUNT)
			neighbor_pixel_index = pixel_coords.x + pixel_coords.y * res.x;
		else
		{
			int2 neighbor_offset = neighbor_offsets[neighbor];
			int2 neighbor_pixel_coords = pixel_coords + neighbor_offset;
			if (neighbor_pixel_coords.x < 0 || neighbor_pixel_coords.x >= res.x || neighbor_pixel_coords.y < 0 || neighbor_pixel_coords.y >= res.y)
				// Rejecting the sample if it's outside of the viewport
				continue;

			neighbor_pixel_index = neighbor_pixel_coords.x + neighbor_pixel_coords.y * res.x;
		}

		Reservoir neighbor_reservoir = render_data.aux_buffers.initial_reservoirs[neighbor_pixel_index];

		float target_function = ReSTIR_DI_evaluate_target_function(render_data, neighbor_reservoir.sample, current_pixel_material, current_pixel_view_direction, current_pixel_shading_point, current_pixel_shading_normal);
		if (target_function > 0.0f)
			new_reservoir.combine_with(neighbor_reservoir, neighbor_reservoir.M, target_function, random_number_generator);
	}

	// Now checking how many of our neighbors could have produced the sample that we just picked
	float valid_neighbor_count = 0.0f;
	if (new_reservoir.weight_sum > 0.0f)
	{
		for (int neighbor = 0; neighbor < NEIGHBOR_REUSE_COUNT; neighbor++)
		{
			int neighbor_pixel_index;

			if (neighbor == NEIGHBOR_REUSE_COUNT)
				neighbor_pixel_index = pixel_coords.x + pixel_coords.y * res.x;
			else
			{
				int2 neighbor_offset = neighbor_offsets[neighbor];
				int2 neighbor_pixel_coords = pixel_coords + neighbor_offset;
				if (neighbor_pixel_coords.x < 0 || neighbor_pixel_coords.x >= res.x || neighbor_pixel_coords.y < 0 || neighbor_pixel_coords.y >= res.y)
					// Rejecting the sample if it's outside of the viewport
					continue;

				neighbor_pixel_index = neighbor_pixel_coords.x + neighbor_pixel_coords.y * res.x;
			}

			RendererMaterial neighbor_material = render_data.g_buffer.materials[neighbor_pixel_index];
			float3 neighbor_view_direction = render_data.g_buffer.view_directions[neighbor_pixel_index];
			float3 neighbor_shading_normal = render_data.g_buffer.shading_normals[neighbor_pixel_index];
			float3 neighbor_shading_point = render_data.g_buffer.first_hits[neighbor_pixel_index] + neighbor_shading_normal * 1.0e-4f;

			float target_function_at_neighbor = ReSTIR_DI_evaluate_target_function(render_data, new_reservoir.sample, neighbor_material, neighbor_view_direction, neighbor_shading_point, neighbor_shading_normal);
			if (target_function_at_neighbor > 0.0f)
			{
				Reservoir neighbor_reservoir = render_data.aux_buffers.initial_reservoirs[neighbor_pixel_index];

				valid_neighbor_count += neighbor_reservoir.M;
			}
		}
	}

	new_reservoir.end_Z(valid_neighbor_count);
	render_data.aux_buffers.spatial_reservoirs[pixel_index] = new_reservoir;
}

#endif
