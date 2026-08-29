/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_JACOBIAN_H
#define DEVICE_RESTIR_JACOBIAN_H

#include "Device/includes/LightSampling/LightClamping.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE float get_jacobian_determinant_reconnection_shift(const float3_t& reconnection_point,
															   const float3_t& reconnection_point_surface_normal,
															   const float3_t& point_being_reconnected,
															   const float3_t& vertex_before_reconnection_original_path,
															   float jacobian_threshold)
{
	float3_t direction_to_reconnection_point_from_center   = reconnection_point - point_being_reconnected;
	float3_t direction_to_reconnection_point_from_neighbor = reconnection_point - vertex_before_reconnection_original_path;
	float distance_to_reconnection_point_from_center	   = hippt::length(direction_to_reconnection_point_from_center);
	float distance_to_reconnection_point_from_neighbor	   = hippt::length(direction_to_reconnection_point_from_neighbor);
	direction_to_reconnection_point_from_center /= distance_to_reconnection_point_from_center;
	direction_to_reconnection_point_from_neighbor /= distance_to_reconnection_point_from_neighbor;

	float cosine_at_reconnection_point_from_center = hippt::abs(hippt::dot(-direction_to_reconnection_point_from_center, reconnection_point_surface_normal));
	float cosine_at_reconnection_point_from_neighbor =
							hippt::abs(hippt::dot(-direction_to_reconnection_point_from_neighbor, reconnection_point_surface_normal));

	float cosine_ratio			 = cosine_at_reconnection_point_from_center / cosine_at_reconnection_point_from_neighbor;
	float distance_squared_ratio = (distance_to_reconnection_point_from_neighbor * distance_to_reconnection_point_from_neighbor) /
								   (distance_to_reconnection_point_from_center * distance_to_reconnection_point_from_center);

	float jacobian_determinant = cosine_ratio * distance_squared_ratio;

	if (jacobian_determinant > jacobian_threshold || jacobian_determinant < 1.0f / jacobian_threshold || hippt::is_nan(jacobian_determinant) ||
		hippt::is_inf(jacobian_determinant))
		// Samples are too dissimilar, returning 0 to indicate that we must reject the sample
		return 0.0f;
	else
		return jacobian_determinant;
}

#endif // #ifndef DEVICE_RESTIR_JACOBIAN_H
