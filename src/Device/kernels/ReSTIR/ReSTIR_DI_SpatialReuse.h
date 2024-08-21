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
#include "Device/includes/LightUtils.h"
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

#define USE_BALANCE_HEURISTICS 0
#define MIS_LIKE_WEIGHTS 1

template <bool withVisiblity>
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_DI_evaluate_target_function(const HIPRTRenderData& render_data, const ReSTIRDISample& sample, const RendererMaterial& material, const RayVolumeState& volume_state, float3 view_direction, float3 shading_point, float3 shading_normal)
{
	std::cerr << "ReSTIR_DI_evaluate_target_function() wrong specialization called: " << withVisiblity << std::endl;
	Utils::debugbreak();
}

template <>
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_DI_evaluate_target_function<KERNEL_OPTION_FALSE>(const HIPRTRenderData& render_data, const ReSTIRDISample& sample, const RendererMaterial& material, const RayVolumeState& volume_state, float3 view_direction, float3 shading_point, float3 shading_normal)
{
	float bsdf_pdf;
	float distance_to_light;
	float3 sample_direction;
	sample_direction = sample.point_on_light_source - shading_point;
	sample_direction = sample_direction / (distance_to_light = hippt::length(sample_direction));

	RayVolumeState trash_volume_state = volume_state;
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, shading_normal, sample_direction, bsdf_pdf);
	float cosine_term = hippt::max(0.0f, hippt::dot(shading_normal, sample_direction));

	float geometry_term = 1.0f;
	if (render_data.render_settings.restir_di_settings.target_function.geometry_term_in_target_function)
	{
		float3 light_source_normal = hippt::normalize(get_triangle_normal_non_normalized(render_data, sample.emissive_triangle_index));
		float cosine_at_light_source = hippt::abs(hippt::dot(sample_direction, light_source_normal));

		geometry_term = cosine_at_light_source / (distance_to_light * distance_to_light);
	}

	int material_index = render_data.buffers.material_indices[sample.emissive_triangle_index];
	ColorRGB32F sample_emission = render_data.buffers.materials_buffer[material_index].emission;

	float target_function = (bsdf_color * sample_emission * cosine_term * geometry_term).luminance();
	if (target_function == 0.0f)
		// Quick exit because computing the visiblity that follows isn't going
		// to change anything to the fact that we have 0.0f target function here
		return 0.0f;

	return target_function;
}

template <>
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_DI_evaluate_target_function<KERNEL_OPTION_TRUE>(const HIPRTRenderData& render_data, const ReSTIRDISample& sample, const RendererMaterial& material, const RayVolumeState& volume_state, float3 view_direction, float3 shading_point, float3 shading_normal)
{
	float bsdf_pdf;
	float distance_to_light;
	float3 sample_direction;
	sample_direction = sample.point_on_light_source - shading_point;
	sample_direction = sample_direction / (distance_to_light = hippt::length(sample_direction));

	RayVolumeState trash_volume_state = volume_state;
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, shading_normal, sample_direction, bsdf_pdf);
	float cosine_term = hippt::max(0.0f, hippt::dot(shading_normal, sample_direction));

	float geometry_term = 1.0f;
	if (render_data.render_settings.restir_di_settings.target_function.geometry_term_in_target_function)
	{
		float3 light_source_normal = hippt::normalize(get_triangle_normal_non_normalized(render_data, sample.emissive_triangle_index));
		float cosine_at_light_source = hippt::abs(hippt::dot(sample_direction, light_source_normal));

		geometry_term = cosine_at_light_source / (distance_to_light * distance_to_light);
	}

	int material_index = render_data.buffers.material_indices[sample.emissive_triangle_index];
	ColorRGB32F sample_emission = render_data.buffers.materials_buffer[material_index].emission;

	float target_function = (bsdf_color * sample_emission * cosine_term * geometry_term).luminance();
	if (target_function == 0.0f)
		// Quick exit because computing the visiblity that follows isn't going
		// to change anything to the fact that we have 0.0f target function here
		return 0.0f;

	hiprtRay shadow_ray;
	shadow_ray.origin = shading_point;
	shadow_ray.direction = sample_direction;

	bool visible = !evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);

	target_function *= visible;

	return target_function;
}

/**
 * Returns the linear index that can be used directly to index a buffer
 * of render_data of the 'neighbor_number'th neighbor that we're going
 * to spatially reuse from
 * 
 * 'neighbor_number' is in [0, neighbor_reuse_count]
 * 'neighbor_reuse_count' is in [1, ReSTIR_DI_Settings.spatial_reuse_neighbor_count]
 * 'neighbor_reuse_radius' is the radius of the disk within which the neighbors are sampled
 * 'center_pixel_coords' is the coordinates of the center pixel that is currently 
 *	 doing the resampling of its neighbors
 * 'res' is the resolution of the viewport. This is used to check whether the generated
 *	 neighbor location is outside of the viewport or not
 * 'cos_sin_theta_rotation' is a pair of float [x, y] with x = cos(random_rotation) and
 *	 y = sin(random_rotation). This is used to rotate the points generated by the Hammersley
 *	 sampler so that not each pixel on the image resample the exact same neighbors (and so
 *	 that a given pixel P resamples different neighbors accros different frame, otherwise
 *	 the Hammersley sampler would always generate the exact same points
 */
HIPRT_HOST_DEVICE HIPRT_INLINE int get_neighbor_pixel_index(int neighbor_number, int neighbor_reuse_count, int neighbor_reuse_radius, int2 center_pixel_coords, int2 res, float2 cos_sin_theta_rotation, Xorshift32Generator& random_number_generator)
{
	int neighbor_pixel_index;

	if (neighbor_number == neighbor_reuse_count)
	{
		// If this is the last neighbor, we set it to ourselves
		// This is why our loop on the neighbors goes up to 'i < NEIGHBOR_REUSE_COUNT + 1'
		// It's so that when i == NEIGHBOR_REUSE_COUNT, we resample ourselves
		neighbor_pixel_index = center_pixel_coords.x + center_pixel_coords.y * res.x;
	}
	else
	{
		// +1 and +1 here because we want to skip the frist point as it is always (0, 0)
		// which means that we would be resampling ourselves (the center pixel) --> 
		// pointless because we already resample ourselves "manually" (that's why there's that
		// "if (neighbor_number == neighbor_reuse_count)" above)
		float2 uv = sample_hammersley_2D(neighbor_reuse_count + 1, neighbor_number + 1);
		// TODO first sample of hammersley is always in the center of the disk so we're reusing ourselves?
		float2 neighbor_offset_in_disk = sample_in_disk_uv(neighbor_reuse_radius, uv);

		// 2D rotation matrix: https://en.wikipedia.org/wiki/Rotation_matrix
		float cos_theta = cos_sin_theta_rotation.x;
		float sin_theta = cos_sin_theta_rotation.y;
		float2 neighbor_offset_rotated = make_float2(neighbor_offset_in_disk.x * cos_theta - neighbor_offset_in_disk.y * sin_theta, neighbor_offset_in_disk.x * sin_theta + neighbor_offset_in_disk.y * cos_theta);
		int2 neighbor_offset_int = make_int2(static_cast<int>(neighbor_offset_rotated.x), static_cast<int>(neighbor_offset_rotated.y));

		int2 neighbor_pixel_coords = center_pixel_coords + neighbor_offset_int;
		if (neighbor_pixel_coords.x < 0 || neighbor_pixel_coords.x >= res.x || neighbor_pixel_coords.y < 0 || neighbor_pixel_coords.y >= res.y)
			// Rejecting the sample if it's outside of the viewport
			return -1;

		neighbor_pixel_index = neighbor_pixel_coords.x + neighbor_pixel_coords.y * res.x;
	}

	return neighbor_pixel_index;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_resampling_MIS_weight(const HIPRTRenderData& render_data, const ReSTIRDIReservoir& neighbor_reservoir, int current_neighbor, int reused_neighbors_count, int2 center_pixel_coords, int2 res, float2 cos_sin_theta_rotation, Xorshift32Generator& random_number_generator)
{
#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_M || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
	return neighbor_reservoir.M;
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE_CONFIDENCE_WEIGHTS
	// No resampling MIS weights for this. Everything is computed in the last step where
	// we check which neighbors could have produced the sample that we picked
	return 1.0f;
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH_CONFIDENCE_WEIGHTS
	float nume = 0.0f;
	// We already have the target function at the center pixel, adding it to the denom
	float denom = 0.0f;

	for (int j = 0; j < reused_neighbors_count + 1; j++)
	{
		int neighbor_index_j = get_neighbor_pixel_index(j, reused_neighbors_count, render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_radius, center_pixel_coords, res, cos_sin_theta_rotation, random_number_generator);
		if (neighbor_index_j == -1)
			continue;

		float3 neighbor_shading_normal = render_data.g_buffer.shading_normals[neighbor_index_j];
		float3 neighbor_shading_point = render_data.g_buffer.first_hits[neighbor_index_j] + neighbor_shading_normal * 1.0e-4f;
		float3 neighbor_view_direction = render_data.g_buffer.view_directions[neighbor_index_j];
		SimplifiedRendererMaterial neighbor_material = render_data.g_buffer.materials[neighbor_index_j];
		RayVolumeState neighbor_ray_volume_state = render_data.g_buffer.ray_volume_states[neighbor_index_j];

		float target_function_at_j = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_SpatialReuseBiasUseVisiblity>(render_data, neighbor_reservoir.sample, neighbor_material, neighbor_ray_volume_state, neighbor_view_direction, neighbor_shading_point, neighbor_shading_normal);

		unsigned int M = 1.0f;
#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH_CONFIDENCE_WEIGHTS
		M = render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs[neighbor_index_j].M;
#endif

		denom += target_function_at_j * M;
		if (j == current_neighbor)
			nume = target_function_at_j * M;
}

	if (denom == 0.0f)
		return 0.0f;
	else
		return nume / denom;
#else
#error "Unsupported bias correction mode in ReSTIR DI spatial reuse get_resampling_MIS_weight"
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE void get_normalization_denominator_numerator(float& out_normalization_nume, float& out_normalization_denom, const HIPRTRenderData& render_data, const ReSTIRDIReservoir& new_reservoir, int selected_neighbor, int reused_neighbors_count, int2 center_pixel_coords, int2 res, float2 cos_sin_theta_rotation, Xorshift32Generator& random_number_generator)
{
	if (new_reservoir.weight_sum <= 0)
	{
		// Invalid reservoir, returning directly
		out_normalization_nume = 1.0;
		out_normalization_denom = 1.0f;

		return;
	}

#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_M
	// 1/M MIS weights are basically confidence weights only i.e. c_i / sum(c_j) with
	// c_i = r_i.M

	out_normalization_nume = 1.0f;
	// We're simply going to divide by the sum of all the M values of all the neighbors we resampled (including the center pixel)
	// so we're only going to set the denominator to that and the numerator isn't going to change
	out_normalization_denom = 0.0f;
	for (int neighbor = 0; neighbor < reused_neighbors_count + 1; neighbor++)
	{
		int neighbor_pixel_index = get_neighbor_pixel_index(neighbor, reused_neighbors_count, render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_radius, center_pixel_coords, res, cos_sin_theta_rotation, random_number_generator);
		if (neighbor_pixel_index == -1)
			// Neighbor out of the viewport
			continue;

		ReSTIRDIReservoir neighbor_reservoir = render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs[neighbor_pixel_index];
		out_normalization_denom += neighbor_reservoir.M;
	}
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE_CONFIDENCE_WEIGHTS
	// Checking how many of our neighbors could have produced the sample that we just picked
	// and we're going to divide by the sum of M values of those neighbors
	out_normalization_denom = 0.0f;
#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
	out_normalization_nume = 1.0f;
#else
	out_normalization_nume = 0.0f;
#endif

	for (int neighbor = 0; neighbor < reused_neighbors_count + 1; neighbor++)
	{
		int neighbor_pixel_index = get_neighbor_pixel_index(neighbor, reused_neighbors_count, render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_radius, center_pixel_coords, res, cos_sin_theta_rotation, random_number_generator);
		if (neighbor_pixel_index == -1)
			// Neighbor out of the viewport
			continue;

		float target_function_at_neighbor = 0.0f;
		if (neighbor == reused_neighbors_count && ReSTIR_DI_SpatialReuseBiasUseVisiblity == ReSTIR_DI_TargetFunctionVisibility)
		{
			// The last neighbor is the center pixel.
			// We want to evaluate the target function of the picked sample at the center pixel.
			// We already have the value of the target function in the reservoir sample
			// 
			// Note that we can only reuse the target function value if the visibility is used in both
			// cases (or not used in both cases), otherwise, the tharget function aren't going to be the same.
			// Also, for the first pass, the visibility has already been included if we've done the visibility
			// reuse pass
			target_function_at_neighbor = new_reservoir.sample.target_function;
		}
		else
		{
			// Getting the surface data at the neighbor
			RendererMaterial neighbor_material = render_data.g_buffer.materials[neighbor_pixel_index];
			RayVolumeState neighbor_ray_volume_state = render_data.g_buffer.ray_volume_states[neighbor_pixel_index];
			float3 neighbor_view_direction = render_data.g_buffer.view_directions[neighbor_pixel_index];
			float3 neighbor_shading_normal = render_data.g_buffer.shading_normals[neighbor_pixel_index];
			float3 neighbor_shading_point = render_data.g_buffer.first_hits[neighbor_pixel_index] + neighbor_shading_normal * 1.0e-4f;

			target_function_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_SpatialReuseBiasUseVisiblity>(render_data, new_reservoir.sample, neighbor_material, neighbor_ray_volume_state, neighbor_view_direction, neighbor_shading_point, neighbor_shading_normal);
		}

		if (target_function_at_neighbor > 0.0f)
		{
			// If the neighbor could have produced this sample...
			ReSTIRDIReservoir neighbor_reservoir = render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs[neighbor_pixel_index];

#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
			out_normalization_denom += neighbor_reservoir.M;
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE
			if (neighbor == selected_neighbor)
				out_normalization_nume += target_function_at_neighbor;
			out_normalization_denom += target_function_at_neighbor;
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE_CONFIDENCE_WEIGHTS
			if (neighbor == selected_neighbor)
				out_normalization_nume += target_function_at_neighbor * neighbor_reservoir.M;
			out_normalization_denom += target_function_at_neighbor * neighbor_reservoir.M;
#endif
		}
	}
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH_CONFIDENCE_WEIGHTS
	// Nothing more to normalize, everything is already handled when resampling the neighbors with balance heuristic MIS weights in the m_i terms
	out_normalization_nume = 1.0f;
	out_normalization_denom = 1.0f;
#else
#error "Unsupported bias correction mode in ReSTIR DI spatial reuse get_normalization_denominator_numerator()"
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_jacobian_determinant_reconnection_shift(const HIPRTRenderData& render_data, const ReSTIRDIReservoir& neighbor_reservoir, float3 center_pixel_shading_point, int neighbor_pixel_index)
{
	float distance_to_light_at_center;
	float distance_to_light_at_neighbor;
	float3 to_light_direction_at_center = neighbor_reservoir.sample.point_on_light_source - center_pixel_shading_point;
	float3 to_light_direction_at_neighbor = neighbor_reservoir.sample.point_on_light_source - render_data.g_buffer.first_hits[neighbor_pixel_index];
	to_light_direction_at_center /= (distance_to_light_at_center = hippt::length(to_light_direction_at_center));
	to_light_direction_at_neighbor /= (distance_to_light_at_neighbor = hippt::length(to_light_direction_at_neighbor));

	float3 light_source_normal = hippt::normalize(get_triangle_normal_non_normalized(render_data, neighbor_reservoir.sample.emissive_triangle_index));
	float cosine_ratio = hippt::abs(hippt::dot(-to_light_direction_at_center, light_source_normal)) / hippt::abs(hippt::dot(-to_light_direction_at_neighbor, light_source_normal));
	float distance_squared_ratio = (distance_to_light_at_neighbor * distance_to_light_at_neighbor) / (distance_to_light_at_center * distance_to_light_at_center);

	return cosine_ratio * distance_squared_ratio;
}

HIPRT_HOST_DEVICE HIPRT_INLINE void spatial_visibility_reuse(const HIPRTRenderData& render_data, ReSTIRDIReservoir& reservoir, float3 shading_point)
{
	if (reservoir.UCW == 0.0f)
		return;

	float distance_to_light;
	float3 sample_direction = reservoir.sample.point_on_light_source - shading_point;
	sample_direction /= (distance_to_light = hippt::length(sample_direction));

	hiprtRay shadow_ray;
	shadow_ray.origin = shading_point;
	shadow_ray.direction = sample_direction;

	bool visible = !evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);
	if (!visible)
		reservoir.UCW = 0.0f;
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
	uint32_t center_pixel_index = (x + y * res.x);
	if (center_pixel_index >= res.x * res.y)
		return;

	// Initializing the random generator
	unsigned int seed;
	if (render_data.render_settings.freeze_random)
		seed = wang_hash(center_pixel_index + 1);
	else
		seed = wang_hash((center_pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_seed);
	
	Xorshift32Generator random_number_generator(seed);

	ReSTIRDIReservoir* input_reservoir_buffer = render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs;

	ReSTIRDIReservoir new_reservoir;
	// Center pixel coordinates
	int2 center_pixel_coords = make_int2(x, y);

	// Surface data of the center pixel
	RendererMaterial center_pixel_material = render_data.g_buffer.materials[center_pixel_index];
	RayVolumeState center_volume_state = render_data.g_buffer.ray_volume_states[center_pixel_index];
	float3 center_pixel_view_direction = render_data.g_buffer.view_directions[center_pixel_index];
	float3 center_pixel_shading_normal = render_data.g_buffer.shading_normals[center_pixel_index];
	float3 center_pixel_shading_point = render_data.g_buffer.first_hits[center_pixel_index] + center_pixel_shading_normal * 1.0e-4f;

	// Rotation that is going to be used to rotate the points generated by the Hammersley sampler
	// for generating the neighbors location to resample
	float rotation_theta = 2.0f * M_PI * random_number_generator();
	float2 cos_sin_theta_rotation = make_float2(cos(rotation_theta), sin(rotation_theta));

	int selected_neighbor = 0;
	int reused_neighbors_count = render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_neighbor_count;
	// Resampling the neighbors. Using neighbors + 1 here so that
	// we can use the last iteration of the loop to resample ourselves (the center pixel)
	// 
	// See the implementation of get_neighbor_pixel_index() earlier in this file
	for (int neighbor = 0; neighbor < reused_neighbors_count + 1; neighbor++)
	{
		int neighbor_pixel_index = get_neighbor_pixel_index(neighbor, reused_neighbors_count, render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_radius, center_pixel_coords, res, cos_sin_theta_rotation, random_number_generator);
		if (neighbor_pixel_index == -1)
			// Neighbor out of the viewport
			continue;

		ReSTIRDIReservoir neighbor_reservoir = input_reservoir_buffer[neighbor_pixel_index];
		if (neighbor_reservoir.UCW == 0.0f)
		{
			// Nothing to do here, just take the M of the resampled neighbor into account.
			// This is basically euiqvalent to combining the reservoir with the
			// new_reservoir.combine_with() function knowing that the target function will
			// be 0.0f (because there's no neighbor reservoir sample)
			new_reservoir.M += neighbor_reservoir.M;

			continue;
		}

		float target_function_at_center = 0.0f;
		if (neighbor == reused_neighbors_count)
			// No need to evaluate the center sample at the center pixel, that's exactly
			// the target function of the center reservoir
			target_function_at_center = neighbor_reservoir.sample.target_function;
		else
			target_function_at_center = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_TargetFunctionVisibility>(render_data, neighbor_reservoir.sample, center_pixel_material, center_volume_state, center_pixel_view_direction, center_pixel_shading_point, center_pixel_shading_normal);

		float jacobian_determinant = 1.0f;
		// If the neighbor reservoir is invalid, do not compute the jacobian
		// Also, if this is the last neighbor resample (meaning that it is the sample pixel), 
		// the jacobian is going to be 1.0f so no need to compute
		if (target_function_at_center > 0.0f && neighbor_reservoir.UCW != 0.0f && neighbor != reused_neighbors_count)
			// The reconnection shift is what is implicitely used in ReSTIR DI. We need this because
			// the initial light sample candidates that we generate on the area of the lights have an
			// area measure PDF. This area measure PDF is converted to solid angle in the initial candidates
			// sampling routine by multiplying by the distance squared and dividing by the cosine
			// angle at the light source. However, a PDF in solid angle measure is only viable at a
			// given point. We say "solid angle with respect to the shading point". This means that
			// reusing a light sample with PDF (the UCW of the neighbor reservoir) in solid angle
			// from a neighbor is invalid since that PDF is only valid at the neighbor point, not
			// at the point we're resampling from (the center pixel). We thus need to convert from the
			// "solid angle PDF at the neighbor" to the solid angle at the center pixel and we do
			// that by multiplying by the jacobian determinant of the reconnection shift in solid
			// angle, Eq. 52 of 2022, "Generalized Resampled Importance Sampling".
			jacobian_determinant = get_jacobian_determinant_reconnection_shift(render_data, neighbor_reservoir, center_pixel_shading_point, neighbor_pixel_index);

		float mis_weight = 1.0f;
		if (target_function_at_center > 0.0f)
			// No need to compute the MIS weight if the target function is 0.0f because we're never going to pick
			// that sample anyway when combining the reservoir since the resampling weight will be 0.0f because of
			// the multiplication by the target function that is 0.0f
			mis_weight = get_resampling_MIS_weight(render_data, neighbor_reservoir, neighbor, reused_neighbors_count, center_pixel_coords, res, cos_sin_theta_rotation, random_number_generator);

		// Combining as in Alg. 6 of the paper
		if (new_reservoir.combine_with(neighbor_reservoir, mis_weight, target_function_at_center, jacobian_determinant, random_number_generator))
			selected_neighbor = neighbor;
	}

	float normalization_numerator = 1.0f;
	float normalization_denominator = 1.0f;

	get_normalization_denominator_numerator(normalization_numerator, normalization_denominator, render_data, new_reservoir, selected_neighbor, reused_neighbors_count, center_pixel_coords, res, cos_sin_theta_rotation, random_number_generator);

	new_reservoir.end_normalized(normalization_numerator, normalization_denominator);

#if ReSTIR_DI_DoVisibilityReuse && ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
	// Why is this needed?
	//
	// Picture the case where we have visibility reuse (at the end of the initial candidates sampling pass),
	// visibility term in the bias correction target function (when counting the neighbors that could
	// have produced the picked sample) and 2 spatial reuse passes.
	//
	// The first spatial reuse pass reuses from samples that were produced with visibility in mind
	// (because of the visibility reuse pass that discards occluded samples). This means that we need
	// the visibility in the target function used when counting the neighbors that could have produced
	// the picked sample otherwise we may think that our neighbor could have produced the picked
	// sample where actually it couldn't because the sample is occluded at the neighbor. We would
	// then have a Z denominator (with 1/Z weights) that is too large and we'll end up with darkening.
	//
	// Now at the end of the first spatial reuse pass, the center pixel ends up with a sample that may
	// or may not be occluded from the center's pixel point of view. We didn't include the visibility
	// in the target function when resampling the neighbors (only when counting the "corect" neighbors
	// but that's all) so we are not giving a 0 weight to occluded resampled neighbors --> it is possible
	// that we picked an occluded sample.
	//
	// In the second spatial reuse pass, we are now going to resample from our neighbors and get some
	// samples that were not generated with occlusion in mind (because resampling target function of
	// the first spatial reuse doesn't include visibility). Yet, we are going to weight them with occlusion
	// in mind. This means that we are probably going to discard samples because of occlusion that could
	// have been generated because they are generated without occlusion test. We end up discarding too many
	// samples --> brightening bias.
	//
	// With the visibility reuse at the end of each spatial pass, we force samples at the end of each
	// spatial reuse to take visibility into account so that when we weight them with visibility testing,
	// everything goes well
	if (render_data.render_settings.restir_di_settings.spatial_pass.number_of_passes > 1)
		spatial_visibility_reuse(render_data, new_reservoir, center_pixel_shading_point);
#endif

	render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs[center_pixel_index] = new_reservoir;
}

#endif
