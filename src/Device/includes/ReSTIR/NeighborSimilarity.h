/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_NEIGHBOR_SIMILARITY_H
#define DEVICE_RESTIR_NEIGHBOR_SIMILARITY_H
 
#include "Device/includes/ReSTIR/Jacobian.h"

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/ReSTIRSettingsHelper.h"

/**
 * Returns true if the two given points pass the plane distance check, false otherwise
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool plane_distance_heuristic(const ReSTIRCommonNeighborSimiliaritySettings& neighbor_similarity_settings, const float3& temporal_world_space_point, const float3& current_point, const float3& current_surface_normal, float plane_distance_threshold)
{
	if (!neighbor_similarity_settings.use_plane_distance_heuristic)
		return true;

	float3 direction_between_points = temporal_world_space_point - current_point;
	float distance_to_plane = hippt::abs(hippt::dot(direction_between_points, current_surface_normal));

	return distance_to_plane < plane_distance_threshold;
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool normal_similarity_heuristic(const ReSTIRCommonNeighborSimiliaritySettings& neighbor_similarity_settings, const float3& current_normal, const float3& neighbor_normal, float threshold)
{
	if (!neighbor_similarity_settings.use_normal_similarity_heuristic)
		return true;

	return hippt::dot(current_normal, neighbor_normal) > threshold;
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool roughness_similarity_heuristic(const ReSTIRCommonNeighborSimiliaritySettings& neighbor_similarity_settings, float neighbor_roughness, float center_pixel_roughness, float threshold)
{
	if (!neighbor_similarity_settings.use_roughness_similarity_heuristic)
		return true;

	// We don't want to temporally reuse on materials smoother than 0.075f because this
	// causes near-specular/glossy reflections to darken when camera ray jittering is used.
	// 
	// This glossy reflections darkening only happens with confidence weights and 
	// ray jittering but I'm not sure why. Probably because samples from one pixel (or sub-pixel location)
	// cannot efficiently be reused at another pixel (or sub-pixel location through jittering)
	// but confidence weights overweight these bad neighbor samples --> you end up using these
	// bad samples --> the shading loses in energy since we're now shading with samples that
	// don't align well with the glossy reflection direction
	return hippt::abs(neighbor_roughness - center_pixel_roughness) < threshold;
}

//HIPRT_HOST_DEVICE HIPRT_INLINE bool jacobian_similarity_heuristic(const HIPRTRenderData& render_data, int neighbor_pixel_index, int center_pixel_index, float3 current_shading_point, float3 neighbor_shading_point)
//{
//	if (render_data.render_settings.restir_gi_settings.spatial_pass.input_reservoirs[neighbor_pixel_index].UCW == 0.0f)
//		return true;
//
//	float3 reconnection_normal = render_data.render_settings.restir_gi_settings.spatial_pass.input_reservoirs[neighbor_pixel_index].sample.sample_point_geometric_normal;
//	if (ReSTIRGISample::is_envmap_path(reconnection_normal))
//		return true;
//
//	float3 reconnection_point = render_data.render_settings.restir_gi_settings.spatial_pass.input_reservoirs[neighbor_pixel_index].sample.sample_point;
//	float jacobian = get_jacobian_determinant_reconnection_shift(reconnection_point, reconnection_normal, current_shading_point, neighbor_shading_point, render_data.render_settings.restir_gi_settings.get_jacobian_heuristic_threshold());
//	return jacobian != -1.0f;
//}

template <bool IsReSTIRGI>
HIPRT_HOST_DEVICE HIPRT_INLINE bool check_neighbor_similarity_heuristics(const HIPRTRenderData& render_data,
																		 int neighbor_pixel_index, int center_pixel_index, 
																		 const float3& current_shading_point, const float3& current_normal, bool previous_frame = false)
{
	if (neighbor_pixel_index == center_pixel_index)
		// A pixel always passes the similarity test with itself
		return true;

	if (previous_frame)
	{
		if (render_data.g_buffer_prev_frame.first_hit_prim_index[neighbor_pixel_index] == -1)
			// Cannot reuse from a neighbor that doesn't have a primary hit (direct miss into the envmap)
			return false;
	}
	else
	{
		if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] == -1)
			// Cannot reuse from a neighbor that doesn't have a primary hit (direct miss into the envmap)
			return false;
	}

	const ReSTIRCommonNeighborSimiliaritySettings& neighbor_similarity_settings = ReSTIRSettingsHelper::get_restir_neighbor_similarity_settings<IsReSTIRGI>(render_data);

	float3 neighbor_world_space_point;
	float neighbor_roughness = 0.0f;
	float current_material_roughness = 0.0f;

	if (previous_frame)
	{
		if (neighbor_similarity_settings.use_plane_distance_heuristic)
			// Only getting the point plane distance heuristic, otherwise it's never used
			neighbor_world_space_point = render_data.g_buffer_prev_frame.primary_hit_position[neighbor_pixel_index];

		if (neighbor_similarity_settings.use_roughness_similarity_heuristic)
			// Only getting the roughness for the roughness heuristic otherwise it's not going to be used
			neighbor_roughness = render_data.g_buffer_prev_frame.materials[neighbor_pixel_index].get_roughness();
	}
	else
	{
		neighbor_world_space_point = render_data.g_buffer.primary_hit_position[neighbor_pixel_index];
		neighbor_roughness = render_data.g_buffer.materials[neighbor_pixel_index].get_roughness();
	}

	if (neighbor_similarity_settings.use_roughness_similarity_heuristic)
		// Getting the roughness at the current point
		current_material_roughness = render_data.g_buffer.materials[center_pixel_index].get_roughness();

	bool plane_distance_passed = plane_distance_heuristic(neighbor_similarity_settings, neighbor_world_space_point, current_shading_point, current_normal, neighbor_similarity_settings.plane_distance_threshold);
	bool normal_similarity_passed = normal_similarity_heuristic(neighbor_similarity_settings, current_normal, render_data.g_buffer.shading_normals[neighbor_pixel_index].unpack(), neighbor_similarity_settings.normal_similarity_angle_precomp);
	bool roughness_similarity_passed = roughness_similarity_heuristic(neighbor_similarity_settings, neighbor_roughness, current_material_roughness, neighbor_similarity_settings.roughness_similarity_threshold);

	return plane_distance_passed && normal_similarity_passed && roughness_similarity_passed;
}

#endif
