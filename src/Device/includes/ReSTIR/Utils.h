/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_UTILS_H
#define DEVICE_RESTIR_DI_UTILS_H 

#include "Device/includes/Dispatcher.h"
#include "Device/includes/Envmap.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/ReSTIR/Surface.h"
#include "Device/includes/ReSTIR/NeighborSimilarity.h"
#include "Device/includes/ReSTIR/GI/Utils.h"

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/ReSTIRSettingsHelper.h"

/**
 * 'last_primitive_hit_index' is the index of the triangle we're currently sitting 
 * on and that we're shooting a ray from. This is used to avoid self intersections.
 */
HIPRT_HOST_DEVICE HIPRT_INLINE void ReSTIR_DI_visibility_reuse(const HIPRTRenderData& render_data, ReSTIRDIReservoir& reservoir, float3 shading_point, int last_primitive_hit_index, Xorshift32Generator& random_number_generator)
{
	if (reservoir.UCW <= 0.0f)
		return;
	else if (reservoir.sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED)
		// The sample is already unoccluded, no need to test for visibility
		return;

	float distance_to_light;
	float3 sample_direction;
	if (reservoir.sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE)
	{
		sample_direction = matrix_X_vec(render_data.world_settings.envmap_to_world_matrix, reservoir.sample.point_on_light_source);
		distance_to_light = 1.0e35f;
	}
	else
	{
		sample_direction = reservoir.sample.point_on_light_source - shading_point;
		sample_direction /= (distance_to_light = hippt::length(sample_direction));
	}

	hiprtRay shadow_ray;
	shadow_ray.origin = shading_point;
	shadow_ray.direction = sample_direction;

	bool visible = !evaluate_shadow_ray(render_data, shadow_ray, distance_to_light, last_primitive_hit_index, /* bounce. Always 0 for ReSTIR DI*/ 0, random_number_generator);
	if (!visible)
		// Setting to -1 here so that we know when debugging that this is because of visibility reuse
		reservoir.UCW = -1.0f;
	else
		// Visible so the sample is unoccluded
		reservoir.sample.flags |= RESTIR_DI_FLAGS_UNOCCLUDED;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_jacobian_determinant_reconnection_shift(const HIPRTRenderData& render_data, const float3& reconnection_point, const float3& reconnection_point_surface_normal, const float3& center_pixel_shading_point, const float3& neighbor_shading_point)
{
	float distance_to_reconnection_point_from_center;
	float distance_to_reconnection_point_from_neighbor;
	float3 direction_to_reconnection_point_from_center = reconnection_point - center_pixel_shading_point;
	float3 direction_to_reconnection_point_from_neighbor = reconnection_point - neighbor_shading_point;
	direction_to_reconnection_point_from_center /= (distance_to_reconnection_point_from_center = hippt::length(direction_to_reconnection_point_from_center));
	direction_to_reconnection_point_from_neighbor /= (distance_to_reconnection_point_from_neighbor = hippt::length(direction_to_reconnection_point_from_neighbor));

	float cosine_at_reconnection_point_from_center = hippt::abs(hippt::dot(-direction_to_reconnection_point_from_center, reconnection_point_surface_normal));
	float cosine_at_reconnection_point_from_neighbor = hippt::abs(hippt::dot(-direction_to_reconnection_point_from_neighbor, reconnection_point_surface_normal));

	float cosine_ratio = cosine_at_reconnection_point_from_center / cosine_at_reconnection_point_from_neighbor;
	float distance_squared_ratio = (distance_to_reconnection_point_from_neighbor * distance_to_reconnection_point_from_neighbor) / (distance_to_reconnection_point_from_center * distance_to_reconnection_point_from_center);

	float jacobian = cosine_ratio * distance_squared_ratio;

	constexpr float jacobian_clamp = 2000.0f;
	if (jacobian > jacobian_clamp || jacobian < 1.0f / jacobian_clamp || hippt::is_NaN(jacobian))
		// Samples are too dissimilar, returning -1 to indicate that we must reject the sample
		return -1;
	else
		return jacobian;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_jacobian_determinant_reconnection_shift(const HIPRTRenderData& render_data, const ReSTIRDIReservoir& neighbor_reservoir, const float3& center_pixel_shading_point, const float3& neighbor_shading_point)
{
	float3 reconnection_point = neighbor_reservoir.sample.point_on_light_source;
	float3 light_source_normal = hippt::normalize(get_triangle_normal_non_normalized(render_data, neighbor_reservoir.sample.emissive_triangle_index));

	return get_jacobian_determinant_reconnection_shift(render_data, reconnection_point, light_source_normal, center_pixel_shading_point, neighbor_shading_point);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_jacobian_determinant_reconnection_shift(const HIPRTRenderData& render_data, const ReSTIRDIReservoir& neighbor_reservoir, const float3& center_pixel_shading_point, int neighbor_pixel_index)
{
	return get_jacobian_determinant_reconnection_shift(render_data, neighbor_reservoir, center_pixel_shading_point, render_data.g_buffer.primary_hit_position[neighbor_pixel_index]);
}

#endif
