/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_UTILS_H
#define DEVICE_RESTIR_DI_UTILS_H

#include "Device/includes/BSDFs/Dispatcher.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/LightSampling/LightClamping.h"
#include "Device/includes/ReSTIR/NeighborSimilarity.h"
#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRSettingsHelper.h"

/**
 * 'last_primitive_hit_index' is the index of the triangle we're currently sitting
 * on and that we're shooting a ray from. This is used to avoid self intersections.
 *
 * Returns true if the reservoir was killed, false otherwise
 */
HIPRT_DEVICE bool ReSTIR_DI_visibility_test_kill_reservoir(const HIPRTRenderData& render_data,
														   ReSTIRDIReservoir& reservoir,
														   float3_t shading_point,
														   int last_primitive_hit_index,
														   Xorshift32Generator& random_number_generator)
{
	if (reservoir.UCW <= 0.0f && reservoir.weight_sum <= 0.0f)
		return false;
	else if (reservoir.sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED)
		// The sample is already unoccluded, no need to test for visibility
		return false;

	float distance_to_light;
	float3_t sample_direction;
	if (reservoir.sample.is_envmap_sample())
	{
		sample_direction  = matrix_X_vec(render_data.world_settings.envmap_to_world_matrix, reservoir.sample.point_on_light_source);
		distance_to_light = 1.0e35f;
	}
	else
	{
		sample_direction = reservoir.sample.point_on_light_source - shading_point;
		sample_direction /= (distance_to_light = hippt::length(sample_direction));
	}

	hiprtRay shadow_ray;
	shadow_ray.origin	 = shading_point;
	shadow_ray.direction = sample_direction;

	bool visible = !evaluate_shadow_ray_occluded(render_data, shadow_ray, distance_to_light, last_primitive_hit_index, random_number_generator);
	if (!visible)
	{
		// Setting to -1 here so that we know when debugging that this is because of visibility reuse
		reservoir.UCW					 = ReSTIRDIReservoir::VISIBILITY_REUSE_KILLED_UCW;
		reservoir.sample.target_function = 0.0f;

		return true;
	}
	else
	{
		// Visible so the sample is unoccluded
		reservoir.sample.flags |= RESTIR_DI_FLAGS_UNOCCLUDED;

		return false;
	}
}

#endif // #ifndef DEVICE_RESTIR_DI_UTILS_H
