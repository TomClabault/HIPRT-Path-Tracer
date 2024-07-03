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

// Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// This is a very general form, allowing input parameters to specfiy normalization and targetPdf
// rather than computing them from `newReservoir`.  Named "internal" since these parameters take
// different meanings (e.g., in RTXDI_CombineDIReservoirs() or RTXDI_StreamNeighborWithPairwiseMIS())
HIPRT_HOST_DEVICE HIPRT_INLINE bool RTXDI_InternalSimpleResample(
	Reservoir& reservoir,
	Reservoir newReservoir,
	float random,
	float targetPdf,            // Usually closely related to the sample normalization, 
	float sampleNormalization,  //     typically off by some multiplicative factor 
	float sampleM               // In its most basic form, should be newReservoir.M
)
{
	// What's the current weight (times any prior-step RIS normalization factor)
	float risWeight = targetPdf * sampleNormalization;

	// Our *effective* candidate pool is the sum of our candidates plus those of our neighbors
	reservoir.M += sampleM;

	// Update the weight sum
	reservoir.weight_sum += risWeight;

	// Decide if we will randomly pick this sample
	bool selectSample = (random * reservoir.weight_sum < risWeight);

	// If we did select this sample, update the relevant data
	if (selectSample)
	{
		reservoir.sample = newReservoir.sample;
		reservoir.sample.target_function = targetPdf;
	}

	return selectSample;
}

// Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// Algorithm (4) from the ReSTIR paper, Combining the streams of multiple reservoirs.
// Normalization - Equation (6) - is postponed until all reservoirs are combined.
HIPRT_HOST_DEVICE HIPRT_INLINE bool RTXDI_CombineDIReservoirs(
	Reservoir& reservoir,
	Reservoir newReservoir,
	float random,
	float targetPdf)
{
	return RTXDI_InternalSimpleResample(
		reservoir,
		newReservoir,
		random,
		targetPdf,
		newReservoir.weight_sum * newReservoir.M,
		newReservoir.M
	);
}

// Performs normalization of the reservoir after streaming. Equation (6) from the ReSTIR paper.
HIPRT_HOST_DEVICE HIPRT_INLINE void RTXDI_FinalizeResampling(
	Reservoir& reservoir,
	float normalizationNumerator,
	float normalizationDenominator)
{
	float denominator = reservoir.sample.target_function * normalizationDenominator;

	reservoir.UCW = (denominator == 0.0) ? 0.0 : (reservoir.weight_sum * normalizationNumerator) / denominator;
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

	//for (int neighbor = 0; neighbor < NEIGHBOR_REUSE_COUNT + 1; neighbor++)
	//{
	//	int neighbor_pixel_index;

	//	if (neighbor == NEIGHBOR_REUSE_COUNT)
	//		neighbor_pixel_index = pixel_coords.x + pixel_coords.y * res.x;
	//	else
	//	{
	//		int2 neighbor_offset = neighbor_offsets[neighbor];
	//		int2 neighbor_pixel_coords = pixel_coords + neighbor_offset;
	//		if (neighbor_pixel_coords.x < 0 || neighbor_pixel_coords.x >= res.x || neighbor_pixel_coords.y < 0 || neighbor_pixel_coords.y >= res.y)
	//			// Rejecting the sample if it's outside of the viewport
	//			continue;

	//		neighbor_pixel_index = neighbor_pixel_coords.x + neighbor_pixel_coords.y * res.x;
	//	}

	//	// Getting the reservoir of the neighbor being resampled
	//	Reservoir neighbor_reservoir = render_data.aux_buffers.initial_reservoirs[neighbor_pixel_index];
	//	if (neighbor_reservoir.UCW == 0.0f)
	//		continue;

	//	// Computing the MIS weight by iterating over all the neighbors because
	//	// we need to take into account all the sampling techniques (one neighbor
	//	// is one sampling technique) for each sample (which is the outer for loop
	//	// above)
	//	//float balance_heuristic_nume = 0.0f;
	//	//float balance_heuristic_denom = 0.0f;
	//	//for (int j = 0; j < NEIGHBOR_REUSE_COUNT + 1; j++)
	//	//{	
	//	//	int2 neighbor_mis_pixel_coords;

	//	//	if (j == NEIGHBOR_REUSE_COUNT)
	//	//		neighbor_mis_pixel_coords = pixel_coords;
	//	//	else
	//	//	{
	//	//		int2 neighbor_mis_offset = neighbor_offsets[j];
	//	//		neighbor_mis_pixel_coords = pixel_coords + neighbor_mis_offset;
	//	//	}

	//	//	if (neighbor_mis_pixel_coords.x < 0 || neighbor_mis_pixel_coords.x >= res.x || neighbor_mis_pixel_coords.y < 0 || neighbor_mis_pixel_coords.y >= res.y)
	//	//		// Rejecting the sample if it's outside of the viewport
	//	//		continue;

	//	//	int neighbor_mis_pixel_index = neighbor_mis_pixel_coords.x + neighbor_mis_pixel_coords.y * res.x;

	//	//	float bsdf_pdf;
	//	//	float neighbor_distance_to_light;
	//	//	float3 neighbor_view_direction = render_data.g_buffer.view_directions[neighbor_mis_pixel_index];
	//	//	float3 neighbor_shading_normal = render_data.g_buffer.shading_normals[neighbor_mis_pixel_index];
	//	//	float3 neighbor_shading_point = render_data.g_buffer.first_hits[neighbor_mis_pixel_index] + neighbor_shading_normal * 1.0e-4f;
	//	//	float3 neighbor_to_light_direction = neighbor_reservoir.sample.point_on_light_source - neighbor_shading_point;
	//	//	float3 neighbor_sample_direction = neighbor_to_light_direction / (neighbor_distance_to_light = hippt::length(neighbor_to_light_direction));
	//	//	SimplifiedRendererMaterial neighbor_material = render_data.g_buffer.materials[neighbor_mis_pixel_index];

	//	//	RayVolumeState trash_volume_state;

	//	//	// Evaluating the target function at the neighbor
	//	//	float target_function_at_neighbor = ReSTIR_DI_evaluate_target_function(render_data, neighbor_reservoir.sample, neighbor_material, neighbor_view_direction, neighbor_shading_point, neighbor_shading_normal);

	//	//	balance_heuristic_denom += target_function_at_neighbor;
	//	//	if (j == neighbor)
	//	//		// For the current sample, we want to evaluate it at the current
	//	//		// pixel, not its neighbor so we're going to do this outside of this loop
	//	//		balance_heuristic_nume = target_function_at_neighbor;
	//	//}

	//	float3 inter_point = render_data.g_buffer.first_hits[pixel_index];
	//	float3 shading_normal = render_data.g_buffer.shading_normals[pixel_index];
	//	float3 shading_point = inter_point + shading_normal * 1.0e-4f;
	//	float3 view_direction = render_data.g_buffer.view_directions[pixel_index];
	//	RendererMaterial material = RendererMaterial(render_data.g_buffer.materials[pixel_index]);

	//	float target_function = ReSTIR_DI_evaluate_target_function(render_data, neighbor_reservoir.sample, material, view_direction, shading_point, shading_normal);
	//	if (target_function == 0.0f)
	//		continue;

	//	//float mis_weight = balance_heuristic_nume / balance_heuristic_denom;
	//	new_reservoir.combine_with(neighbor_reservoir, 1.0f, target_function, random_number_generator);
	//}














	RendererMaterial current_pixel_material = render_data.g_buffer.materials[pixel_index];
	float3 current_pixel_view_direction = render_data.g_buffer.view_directions[pixel_index];
	float3 current_pixel_shading_normal = render_data.g_buffer.shading_normals[pixel_index];
	float3 current_pixel_shading_point = render_data.g_buffer.first_hits[pixel_index] + current_pixel_shading_normal * 1.0e-4f;

	for (int neighbor = 0; neighbor < NEIGHBOR_REUSE_COUNT + 1; neighbor++)
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

	float valid_neighbor_count = 0.0f;
	if (new_reservoir.weight_sum > 0.0f)
	{
		for (int neighbor = 0; neighbor < NEIGHBOR_REUSE_COUNT + 1; neighbor++)
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

			RendererMaterial neighbor_material = render_data.g_buffer.materials[neighbor_pixel_index];
			float3 neighbor_view_direction = render_data.g_buffer.view_directions[neighbor_pixel_index];
			float3 neighbor_shading_normal = render_data.g_buffer.shading_normals[neighbor_pixel_index];
			float3 neighbor_shading_point = render_data.g_buffer.first_hits[neighbor_pixel_index] + neighbor_shading_normal * 1.0e-4f;

			float target_function_at_neighbor = ReSTIR_DI_evaluate_target_function(render_data, new_reservoir.sample, neighbor_material, neighbor_view_direction, neighbor_shading_point, neighbor_shading_normal);
			if (target_function_at_neighbor > 0.0f)
				valid_neighbor_count += neighbor_reservoir.M;
		}
	}

	new_reservoir.end_Z(valid_neighbor_count);
	render_data.aux_buffers.spatial_reservoirs[pixel_index] = new_reservoir;














	//Reservoir state;

	//// Since we're using our bias correction scheme, we need to remember which light selection we made
	//int selected = -1;

	//Reservoir centerReservoir = render_data.aux_buffers.initial_reservoirs[pixel_index];

	//RTXDI_CombineDIReservoirs(state, centerReservoir, /* random = */ 0.5f, centerReservoir.sample.target_function);

	//// We loop through neighbors twice.  Cache the validity / edge-stopping function
	////   results for the 2nd time through.
	//unsigned int cachedResult = 0;

	//RendererMaterial center_material = render_data.g_buffer.materials[pixel_index];
	//float3 center_view_direction = render_data.g_buffer.view_directions[pixel_index];
	//float3 center_shading_normal = render_data.g_buffer.shading_normals[pixel_index];
	//float3 center_shading_point  = render_data.g_buffer.first_hits[pixel_index] + center_shading_normal * 1.0e-4f;

	//// Walk the specified number of neighbors, resampling using RIS
	//for (int i = 0; i < NEIGHBOR_REUSE_COUNT; ++i)
	//{
	//	// Get screen-space location of neighbor
	//	int2 neighborReservoirPos = neighbor_offsets[i];
	//	neighborReservoirPos += pixel_coords;
	//	if (neighborReservoirPos.x < 0 || neighborReservoirPos.x >= res.x || neighborReservoirPos.y < 0 || neighborReservoirPos.y >= res.y)
	//		// Rejecting the sample if it's outside of the viewport
	//		continue;

	//	int neighborReservoirIndex = neighborReservoirPos.x + neighborReservoirPos.y * res.x;

	//	Reservoir neighborSample = render_data.aux_buffers.initial_reservoirs[neighborReservoirIndex];

	//	cachedResult |= (1u << i);

	//	// Load that neighbor's RIS state, do resampling
	//	float neighborWeight = 0;
	//	if (neighborSample.weight_sum > 0.0f)
	//		neighborWeight = ReSTIR_DI_evaluate_target_function(render_data, neighborSample.sample, center_material, center_view_direction, center_shading_point, center_shading_normal);

	//	RTXDI_CombineDIReservoirs(state, neighborSample, random_number_generator(), neighborWeight);
	//}

	//if (state.weight_sum > 0.0f)
	//{
	//	// Compute the unbiased normalization term (instead of using 1/M)
	//	float pi = state.sample.target_function;
	//	float piSum = state.sample.target_function * centerReservoir.M;

	//	// To do this, we need to walk our neighbors again
	//	for (int i = 0; i < NEIGHBOR_REUSE_COUNT; ++i)
	//	{
	//		// If we skipped this neighbor above, do so again.
	//		if ((cachedResult & (1u << i)) == 0) continue;

	//		// Get screen-space location of neighbor
	//		int2 neighborReservoirPos = neighbor_offsets[i];
	//		neighborReservoirPos += pixel_coords;
	//		if (neighborReservoirPos.x < 0 || neighborReservoirPos.x >= res.x || neighborReservoirPos.y < 0 || neighborReservoirPos.y >= res.y)
	//			// Rejecting the sample if it's outside of the viewport
	//			continue;

	//		int neighborReservoirIndex = neighborReservoirPos.x + neighborReservoirPos.y * res.x;

	//		// Load our neighbor's G-buffer
	//		Reservoir neighborSample = render_data.aux_buffers.initial_reservoirs[neighborReservoirIndex];

	//		float distance_to_light;
	//		RendererMaterial neighbor_material = render_data.g_buffer.materials[neighborReservoirIndex];
	//		float3 neighbor_view_direction = render_data.g_buffer.view_directions[neighborReservoirIndex];
	//		float3 neighbor_shading_normal = render_data.g_buffer.shading_normals[neighborReservoirIndex];
	//		float3 neighbor_shading_point = render_data.g_buffer.first_hits[neighborReservoirIndex] + neighbor_shading_normal * 1.0e-4f;
	//		float3 neighbor_to_light_direction = neighborSample.sample.point_on_light_source - neighbor_shading_point;
	//		neighbor_to_light_direction /= (distance_to_light = hippt::length(neighbor_to_light_direction));

	//		float ps = ReSTIR_DI_evaluate_target_function(render_data, neighborSample.sample, neighbor_material, neighbor_view_direction, neighbor_shading_point, neighbor_shading_normal);

	//		hiprtRay shadow_ray;
	//		shadow_ray.origin = neighbor_shading_point;
	//		shadow_ray.direction = neighbor_to_light_direction;

	//		if (!evaluate_shadow_ray(render_data, shadow_ray, distance_to_light))
	//			ps = 0;

	//		// Select this sample for the (normalization) numerator if this particular neighbor pixel
	//		//     was the one we selected via RIS in the first loop, above.
	//		pi = selected == i ? ps : pi;

	//		// Add to the sums of weights for the (normalization) denominator
	//		piSum += ps * neighborSample.M;
	//	}

	//	// Use "MIS-like" normalization
	//	RTXDI_FinalizeResampling(state, pi, piSum);
	//}

	//render_data.aux_buffers.spatial_reservoirs[pixel_index] = state;
}

#endif
