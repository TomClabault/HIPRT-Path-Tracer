/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/ReSTIRSettingsHelper.h"

/**
 * 'last_primitive_hit_index' is the index of the triangle we're currently sitting 
 * on and that we're shooting a ray from. This is used to avoid self intersections.
 * 
 * Returns true if the reservoir was killed, false otherwise
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool ReSTIR_DI_visibility_test_kill_reservoir(const HIPRTRenderData& render_data, ReSTIRDIReservoir& reservoir, float3 shading_point, int last_primitive_hit_index, Xorshift32Generator& random_number_generator)
{
	if (reservoir.UCW <= 0.0f && reservoir.weight_sum <= 0.0f)
		return false;
	else if (reservoir.sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED)
		// The sample is already unoccluded, no need to test for visibility
		return false;

	float distance_to_light;
	float3 sample_direction;
	if (reservoir.sample.is_envmap_sample())
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
	{
		// Setting to -1 here so that we know when debugging that this is because of visibility reuse
		reservoir.UCW = -1.0f;

		return true;
	}
	else
	{
		// Visible so the sample is unoccluded
		reservoir.sample.flags |= RESTIR_DI_FLAGS_UNOCCLUDED;

		return false;
	}
}

/**
 * Tests the visibility of the sample containde in 'reservoir' from the given shading point and kills the reservoir
 * if the visibility is occluded
 * 
 * Returns true if the reservoir was killed, false otherwise
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool ReSTIR_GI_visibility_validation(const HIPRTRenderData& render_data, ReSTIRGIReservoir& reservoir, float3 shading_point, int last_hit_primitive_index, Xorshift32Generator& random_number_generator)
{
	if (reservoir.UCW <= 0.0f && reservoir.weight_sum <= 0.0f)
		return false;

	float distance_to_sample_point;
	float3 sample_direction;
	if (reservoir.sample.is_envmap_path())
	{
		// For envmap path, the direction is stored in the 'sample_point' value
		sample_direction = reservoir.sample.sample_point;
		distance_to_sample_point = 1.0e35f;
	}
	else
	{
		// Not an envmap path, the direction is the difference between the current shading
		// point and the reconnection point
		sample_direction = reservoir.sample.sample_point - shading_point;
		distance_to_sample_point = hippt::length(sample_direction);
		if (distance_to_sample_point <= 1.0e-6f)
		{
			// To avoid numerical instabilities, killing the reservoir
			reservoir.UCW = 0.0f;

			return true;
		}

		sample_direction /= distance_to_sample_point;
	}

	hiprtRay shadow_ray;
	shadow_ray.origin = shading_point;
	shadow_ray.direction = sample_direction;

	bool visible = !evaluate_shadow_ray(render_data, shadow_ray, distance_to_sample_point, last_hit_primitive_index, 
		/* bounce. Always 1 for ReSTIR GI from visible point to sample point */ 1, random_number_generator);

	if (!visible)
	{
		// Setting to -1 here so that we know when debugging that this is because of visibility reuse
		reservoir.UCW = 0.0f;

		return true;
	}

	return false;
}

#endif
