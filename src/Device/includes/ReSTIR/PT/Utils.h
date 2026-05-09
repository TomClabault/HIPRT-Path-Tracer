/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_UTILS_H
#define DEVICE_RESTIR_PT_UTILS_H

#include "Device/includes/BSDFs/Dispatcher.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/LightSampling/LightClamping.h"
#include "Device/includes/ReSTIR/NeighborSimilarity.h"
#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRSettingsHelper.h"

/**
 * Tests the visibility of the sample containde in 'reservoir' from the given shading point and kills the reservoir
 * if the visibility is occluded
 *
 * Returns true if the reservoir was killed, false otherwise
 */
HIPRT_DEVICE bool ReSTIR_PT_visibility_validation(const HIPRTRenderData& render_data,
												  ReSTIRPTReservoir& reservoir,
												  float3_t shading_point,
												  int last_hit_primitive_index,
												  Xorshift32Generator& random_number_generator)
{
	if (reservoir.UCW <= 0.0f && reservoir.weight_sum <= 0.0f)
		return false;

	float distance_to_sample_point;
	float3_t sample_direction;
	if (reservoir.sample.is_envmap_path())
	{
		// For envmap path, the direction is stored in the 'rc_vertex' value
		sample_direction		 = reservoir.sample.rc_vertex;
		distance_to_sample_point = 1.0e35f;
	}
	else
	{
		// Not an envmap path, the direction is the difference between the current shading
		// point and the reconnection point

		sample_direction		 = reservoir.sample.rc_vertex - shading_point;
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
	shadow_ray.origin	 = shading_point;
	shadow_ray.direction = sample_direction;

	bool visible = !evaluate_shadow_ray_occluded(render_data, shadow_ray, distance_to_sample_point, last_hit_primitive_index,
												 /* bounce. Always 0 for ReSTIR PT from visible point to reconnection vertex */ 0, random_number_generator);

	if (!visible)
	{
		reservoir.UCW = 0.0f;

		return true;
	}

	return false;
}

#endif
