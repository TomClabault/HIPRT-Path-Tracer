/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_VISIBILITY_TEST_H
#define DEVICE_KERNELS_REGIR_VISIBILITY_TEST_H

#include "Device/includes/Intersect.h"
#include "Device/includes/ReSTIR/ReGIR/Representative.h"

#include "HostDeviceCommon/RenderData.h"

/**
 * Returns false if the given 'point_on_light' is occluded from the point of view of the
 * representative point of the grid cell given by 'hash_grid_cell_index'
 *
 * Returns true if unoccluded
 */
HIPRT_DEVICE bool ReGIR_grid_cell_visibility_test(
	const HIPRTRenderData& render_data, float3_t representative_point, int representative_primitive_index, float3_t point_on_light, Xorshift32Generator& rng)
{
	float3_t to_light_direction = point_on_light - representative_point;
	float distance_to_light		= hippt::length(to_light_direction);
	to_light_direction /= distance_to_light;

	hiprtRay shadow_ray;
	shadow_ray.origin	 = representative_point;
	shadow_ray.direction = to_light_direction;

	return !evaluate_shadow_ray_occluded(render_data, shadow_ray, distance_to_light, representative_primitive_index, rng);
}

HIPRT_DEVICE bool ReGIR_grid_cell_visibility_test(
	const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit, float3_t point_on_light, Xorshift32Generator& rng)
{
	int representative_primitive_index = ReGIR_get_cell_primitive_index(render_data, hash_grid_cell_index, primary_hit);
	float3_t representative_point	   = ReGIR_get_cell_world_point(render_data, hash_grid_cell_index, primary_hit);

	return ReGIR_grid_cell_visibility_test(render_data, representative_point, representative_primitive_index, point_on_light, rng);
}

#endif // #ifndef DEVICE_KERNELS_REGIR_VISIBILITY_TEST_H
