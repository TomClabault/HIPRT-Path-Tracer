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

#endif
