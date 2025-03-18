/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_JACOBIAN_H
#define DEVICE_RESTIR_JACOBIAN_H

#include "Device/includes/LightUtils.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE HIPRT_INLINE float get_jacobian_determinant_reconnection_shift(const float3& reconnection_point, const float3& reconnection_point_surface_normal, const float3& center_pixel_visible_point, const float3& neighbor_visible_point, float jacobian_threshold)
{
	float3 direction_to_reconnection_point_from_center = reconnection_point - center_pixel_visible_point;
	float3 direction_to_reconnection_point_from_neighbor = reconnection_point - neighbor_visible_point;
	float distance_to_reconnection_point_from_center = hippt::length(direction_to_reconnection_point_from_center);
	float distance_to_reconnection_point_from_neighbor = hippt::length(direction_to_reconnection_point_from_neighbor);
	direction_to_reconnection_point_from_center /= distance_to_reconnection_point_from_center;
	direction_to_reconnection_point_from_neighbor /= distance_to_reconnection_point_from_neighbor;

	float cosine_at_reconnection_point_from_center = hippt::abs(hippt::dot(-direction_to_reconnection_point_from_center, reconnection_point_surface_normal));
	float cosine_at_reconnection_point_from_neighbor = hippt::abs(hippt::dot(-direction_to_reconnection_point_from_neighbor, reconnection_point_surface_normal));

	float cosine_ratio = cosine_at_reconnection_point_from_center / cosine_at_reconnection_point_from_neighbor;
	float distance_squared_ratio = (distance_to_reconnection_point_from_neighbor * distance_to_reconnection_point_from_neighbor) / (distance_to_reconnection_point_from_center * distance_to_reconnection_point_from_center);

	float jacobian_determinant = cosine_ratio * distance_squared_ratio;

	if (jacobian_determinant == 0.0f)
		return 0.0f;

	if (jacobian_determinant > jacobian_threshold || jacobian_determinant < 1.0f / jacobian_threshold || hippt::is_nan(jacobian_determinant))
		// Samples are too dissimilar, returning -1 to indicate that we must reject the sample
		return -1.0f;
	else
		return jacobian_determinant;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_jacobian_determinant_reconnection_shift(const HIPRTRenderData& render_data, const ReSTIRDIReservoir& neighbor_reservoir, const float3& center_pixel_shading_point, const float3& neighbor_shading_point)
{
	float3 reconnection_point = neighbor_reservoir.sample.point_on_light_source;
	float3 light_source_normal = hippt::normalize(get_triangle_normal_not_normalized(render_data, neighbor_reservoir.sample.emissive_triangle_index));

	return get_jacobian_determinant_reconnection_shift(reconnection_point, light_source_normal, center_pixel_shading_point, neighbor_shading_point, render_data.render_settings.restir_gi_settings.get_jacobian_heuristic_threshold());
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_jacobian_determinant_reconnection_shift(const HIPRTRenderData& render_data, const ReSTIRDIReservoir& neighbor_reservoir, const float3& center_pixel_shading_point, int neighbor_pixel_index)
{
	return get_jacobian_determinant_reconnection_shift(render_data, neighbor_reservoir, center_pixel_shading_point, render_data.g_buffer.primary_hit_position[neighbor_pixel_index]);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRGIReservoir shift_sample_reconnection_shift(const ReSTIRGIReservoir& input_sample, float& out_shift_jacobian_determinant, float3 reconnected_point_from_target_domain, float jacobian_threshold)
{
	ReSTIRGIReservoir out_shifted_sample = input_sample;

	// Eq. 11 of the ReSTIR GI paper
	out_shift_jacobian_determinant = get_jacobian_determinant_reconnection_shift(input_sample.sample.sample_point, input_sample.sample.sample_point_geometric_normal, reconnected_point_from_target_domain, input_sample.sample.visible_point, jacobian_threshold);

	return input_sample;
}

#endif
