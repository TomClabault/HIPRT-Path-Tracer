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

#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

 /** References:
 *
 * [1] [Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting] https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal/
 * [2] [A Gentle Introduction to ReSTIR: Path Reuse in Real-time] https://intro-to-restir.cwyman.org/
 * [3] [A Gentle Introduction to ReSTIR: Path Reuse in Real-time - SIGGRAPH 2023 Presentation Video] https://dl.acm.org/doi/10.1145/3587423.3595511#sec-supp
 * [4] [NVIDIA RTX DI SDK - Github] https://github.com/NVIDIAGameWorks/RTXDI
 * [5] [Generalized Resampled Importance Sampling Foundations of ReSTIR] https://research.nvidia.com/publication/2022-07_generalized-resampled-importance-sampling-foundations-restir
 * [6] [Uniform disk sampling] https://rh8liuqy.github.io/Uniform_Disk.html
 */

#define SPATIAL_REUSE_PASSES_COUNT 1
#define NEIGHBOR_REUSE_COUNT 1
#define REUSE_RADIUS 30

#define USE_BALANCE_HEURISTICS 0
#define MIS_LIKE_WEIGHTS 0

/**
 * Target function is BSDF * Le * cos(theta) * V
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_DI_evaluate_target_function(const HIPRTRenderData& render_data, const ReservoirSample& sample, const RendererMaterial& material, const RayVolumeState& volume_state, float3 view_direction, float3 shading_point, float3 shading_normal)
{
	float bsdf_pdf;
	float distance_to_light;
	float3 sample_direction;
	sample_direction = sample.point_on_light_source - shading_point;
	sample_direction = sample_direction / (distance_to_light = hippt::length(sample_direction));

	RayVolumeState trash_volume_state = volume_state;
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, shading_normal, sample_direction, bsdf_pdf);
	float cosine_term = hippt::max(0.0f, hippt::dot(shading_normal, sample_direction));

	// TODO include geometry term? benchmark variance with and wxithout (ImGui option)
	//float geometry_term = 1.0f / distance_to_light / distance_to_light;
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

HIPRT_HOST_DEVICE HIPRT_INLINE int get_neighbor_pixel_index(int neighbor_number, int2* neighbor_offsets, int2 center_pixel_coords, int2 res)
{
	int neighbor_pixel_index;

	if (neighbor_number == NEIGHBOR_REUSE_COUNT)
		// If this is the last neighbor, we set it to ourselves
		// This is why our loop on the neighbors goes up to 'i < NEIGHBOR_REUSE_COUNT + 1'
		// It's so that when i == NEIGHBOR_REUSE_COUNT, we resample ourselves
		neighbor_pixel_index = center_pixel_coords.x + center_pixel_coords.y * res.x;
	else
	{
		int2 neighbor_offset = neighbor_offsets[neighbor_number];
		int2 neighbor_pixel_coords = center_pixel_coords + neighbor_offset;
		if (neighbor_pixel_coords.x < 0 || neighbor_pixel_coords.x >= res.x || neighbor_pixel_coords.y < 0 || neighbor_pixel_coords.y >= res.y)
			// Rejecting the sample if it's outside of the viewport
			return -1;

		neighbor_pixel_index = neighbor_pixel_coords.x + neighbor_pixel_coords.y * res.x;
	}

	return neighbor_pixel_index;
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

	// Initializing the random generator
	unsigned int seed;
	if (render_data.render_settings.freeze_random)
		seed = wang_hash(pixel_index + 1);
	else
		seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1));
	
	Xorshift32Generator random_number_generator(seed);

	Reservoir new_reservoir;
	// The neighbor screen space offsets will be in an array. Inefficient but easy implementation for debugging.
	int2 neighbor_offsets[NEIGHBOR_REUSE_COUNT];
	// Center pixel coordinates
	int2 center_pixel_coords = make_int2(x, y);

	for (int i = 0; i < NEIGHBOR_REUSE_COUNT; i++)
	{
		// TODO low discrepancy sequence here
		int2 neighbor_offset = integer_sample_in_disk(REUSE_RADIUS, random_number_generator);

		neighbor_offsets[i].x = neighbor_offset.x;
		neighbor_offsets[i].y = neighbor_offset.y;
	}

	// Surface data of the center pixel
	RendererMaterial center_pixel_material = render_data.g_buffer.materials[pixel_index];
	RayVolumeState center_volume_state = render_data.g_buffer.ray_volume_states[pixel_index];
	float3 center_pixel_view_direction = render_data.g_buffer.view_directions[pixel_index];
	float3 center_pixel_shading_normal = render_data.g_buffer.shading_normals[pixel_index];
	float3 center_pixel_shading_point = render_data.g_buffer.first_hits[pixel_index] + center_pixel_shading_normal * 1.0e-4f;

	// Resampling the neighbors. Using neighbors + 1 here so that
	// we can use the last iteration of the loop to resample ourselves (the center pixel)
	// 
	// See the implementation of get_neighbor_pixel_index() earlier in this file
	int selected_neighbor = 0;
	for (int neighbor = 0; neighbor < NEIGHBOR_REUSE_COUNT + 1; neighbor++)
	{
		int neighbor_pixel_index = get_neighbor_pixel_index(neighbor, neighbor_offsets, center_pixel_coords, res);
		if (neighbor_pixel_index == -1)
			// Neighbor out of the viewport
			continue;

		Reservoir neighbor_reservoir = render_data.aux_buffers.initial_reservoirs[neighbor_pixel_index];

		float target_function_at_center = ReSTIR_DI_evaluate_target_function(render_data, neighbor_reservoir.sample, center_pixel_material, center_volume_state, center_pixel_view_direction, center_pixel_shading_point, center_pixel_shading_normal);

		float jacobian_determinant = 1.0f;
		if (neighbor_reservoir.UCW != 0.0f)
		{
			float distance_to_light_at_center;
			float distance_to_light_at_neighbor;
			float3 to_light_direction_at_center = neighbor_reservoir.sample.point_on_light_source - center_pixel_shading_point;
			float3 to_light_direction_at_neighbor = neighbor_reservoir.sample.point_on_light_source - render_data.g_buffer.first_hits[neighbor_pixel_index];
			to_light_direction_at_center /= (distance_to_light_at_center = hippt::length(to_light_direction_at_center));
			to_light_direction_at_neighbor /= (distance_to_light_at_neighbor = hippt::length(to_light_direction_at_neighbor));

			float cosine_ratio = hippt::abs(hippt::dot(-to_light_direction_at_center, neighbor_reservoir.sample.light_source_normal)) / hippt::abs(hippt::dot(-to_light_direction_at_neighbor, neighbor_reservoir.sample.light_source_normal));
			float distance_squared_ratio = (distance_to_light_at_neighbor * distance_to_light_at_neighbor) / (distance_to_light_at_center * distance_to_light_at_center);

			jacobian_determinant = cosine_ratio * distance_squared_ratio;
		}

		float mis_weight = 1.0f;
#if USE_BALANCE_HEURISTICS
		float nume = 0.0f;
		// We already have the target function at the center pixel, adding it to the denom
		float denom = 0.0f;

		for (int j = 0; j < NEIGHBOR_REUSE_COUNT + 1; j++)
		{
			int neighbor_index_j = get_neighbor_pixel_index(j, neighbor_offsets, center_pixel_coords, res);
			if (neighbor_index_j == -1)
				continue;

			float3 neighbor_shading_normal = render_data.g_buffer.shading_normals[neighbor_index_j];
			float3 neighbor_shading_point = render_data.g_buffer.first_hits[neighbor_index_j] + neighbor_shading_normal * 1.0e-4f;
			float3 neighbor_view_direction = render_data.g_buffer.view_directions[neighbor_index_j];
			SimplifiedRendererMaterial neighbor_material = render_data.g_buffer.materials[neighbor_index_j];
			RayVolumeState neighbor_ray_volume_state = render_data.g_buffer.ray_volume_states[neighbor_index_j];

			float target_function_at_j = ReSTIR_DI_evaluate_target_function(render_data, neighbor_reservoir.sample, neighbor_material, neighbor_ray_volume_state, neighbor_view_direction, neighbor_shading_point, neighbor_shading_normal);
			denom += target_function_at_j;
			if (j == neighbor)
				nume = target_function_at_j;
		}

		if (denom == 0.0f)
			mis_weight = 0.0f;
		else
			mis_weight = nume / denom;
#elif MIS_LIKE_WEIGHTS
		mis_weight = 1.0f;
#else
		mis_weight = neighbor_reservoir.M;
#endif

		// Combining as in Alg. 6 of the paper
		if (new_reservoir.combine_with(neighbor_reservoir, mis_weight, target_function_at_center, jacobian_determinant, random_number_generator))
			selected_neighbor = neighbor;
	}


	// Unbiased normalization term as in ReSTIR 2019 Alg. 6
	float Z = 0.0f;
	float mis_like_denom = 0.0f;
	float mis_like_nume = 0.0f;

	// Now checking how many of our neighbors could have produced the sample that we just picked
	if (new_reservoir.weight_sum > 0.0f)
	{
		for (int neighbor = 0; neighbor < NEIGHBOR_REUSE_COUNT + 1; neighbor++)
		{
			int neighbor_pixel_index = get_neighbor_pixel_index(neighbor, neighbor_offsets, center_pixel_coords, res);
			if (neighbor_pixel_index == -1)
				// Neighbor out of the viewport
				continue;

			// Getting the surface data at the neighbor
			RendererMaterial neighbor_material = render_data.g_buffer.materials[neighbor_pixel_index];
			RayVolumeState neighbor_ray_volume_state = render_data.g_buffer.ray_volume_states[neighbor_pixel_index];
			float3 neighbor_view_direction = render_data.g_buffer.view_directions[neighbor_pixel_index];
			float3 neighbor_shading_normal = render_data.g_buffer.shading_normals[neighbor_pixel_index];
			float3 neighbor_shading_point = render_data.g_buffer.first_hits[neighbor_pixel_index] + neighbor_shading_normal * 1.0e-4f;

			float target_function_at_neighbor = ReSTIR_DI_evaluate_target_function(render_data, new_reservoir.sample, neighbor_material, neighbor_ray_volume_state, neighbor_view_direction, neighbor_shading_point, neighbor_shading_normal);

			if (target_function_at_neighbor > 0.0f)
			{
				// If the neighbor could have produced this sample...
				Reservoir neighbor_reservoir = render_data.aux_buffers.initial_reservoirs[neighbor_pixel_index];

				// ... adding M to the Z normalization term
				// TODO add the possibility through ImGui to choose whether we're using confidence
				// weights MIS weights (adding M to the numerator and Z denom) or just 1 weight for
				// each sample (+ 1 to numerator each time, still Z denom but only +1 for each valid
				// sample. Basically the exact same as confidence weights but forcing M=1 for every
				// reused reservoir)
				Z += neighbor_reservoir.M;

				if (neighbor == selected_neighbor)
					mis_like_nume += target_function_at_neighbor;
				mis_like_denom += target_function_at_neighbor;
			}
		}
	}


	// Compute the unbiased contribution weight using 1/Z normalization weight as in ReSTIR 2019 Alg. 6
#if USE_BALANCE_HEURISTICS
	new_reservoir.end();
#elif MIS_LIKE_WEIGHTS
	new_reservoir.end_normalized(mis_like_denom / mis_like_nume);
#else
	new_reservoir.end_normalized(Z);
#endif

	render_data.aux_buffers.spatial_reservoirs[pixel_index] = new_reservoir;
}

#endif
