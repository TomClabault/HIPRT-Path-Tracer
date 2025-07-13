/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_FUNCTIONS_FILTER_FUNCTION_H
#define DEVICE_FUNCTIONS_FILTER_FUNCTION_H

#include "Device/functions/FilterFunctionPayload.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Material.h"

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

/**
 * This filter function handles self intersection avoidance and alpha testing
 * 
 * return FALSE if the intersection is ACCEPTED
 * return true if the intersection is rejected
 */
HIPRT_DEVICE HIPRT_INLINE bool filter_function(const hiprtRay&, const void*, void* payld, const hiprtHit& hit)
{
	FilterFunctionPayload* payload = reinterpret_cast<FilterFunctionPayload*>(payld);

	int global_triangle_index_hit;
	if (payload->simplified_light_ray)
		// If the ray is shot in the BVH containg only the emissive triangles, the hit.primID is the index of the emissive triangle in that BVH,
		// not the index of the emissive triangle in the whole scene, which 'payload->last_hit_primitive_index' is
		//
		// So we need to 'convert' the hit index in the light BVH to a hit index in the whole scene BVH and do
		// the comparison against that
		global_triangle_index_hit = payload->render_data->buffers.emissive_triangles_indices[hit.primID];
	else
		global_triangle_index_hit = hit.primID;

	if (global_triangle_index_hit == payload->last_hit_primitive_index)
		// This is a self-intersection, filtering it out
		//
		// Triangles are planar so one given triangle can
		// never be intersect twice in a row (unless we're absolutely
		// perfectly parallel to the triangle but let's ignore that...)
		//
		// This self-intersection avoidance only works for planar primitives
		return true;

	if (!payload->render_data->render_settings.do_alpha_testing)
		// No alpha testing
		return false;

	if (payload->bounce >= payload->render_data->render_settings.alpha_testing_indirect_bounce)
		// Alpha testing is disabled at the current bounce
		// 
		// Returning false to indicate an intersection
		return false;

	int material_index = payload->render_data->buffers.material_indices[global_triangle_index_hit];
	if (payload->render_data->buffers.material_opaque[material_index])
		// The material is fully opaque, no need to test further, accept the intersection
		return false;

	// Composition both the alpha of the base color texture and the material
	unsigned short int base_color_texture_index = payload->render_data->buffers.materials_buffer.get_base_color_texture_index(material_index);
	float base_color_alpha = get_hit_base_color_alpha(*payload->render_data, base_color_texture_index, global_triangle_index_hit, hit.uv);
	float alpha_opacity = payload->render_data->buffers.materials_buffer.get_alpha_opacity(material_index);
	float composited_alpha = alpha_opacity * base_color_alpha;

	if ((*payload->random_number_generator)() < composited_alpha)
		// Alpha test not passing, the ray is blocked
		return false;

	// No tests stopped the ray, that's not a hit.
	// Returning 'true' to filter out the intersection
	return true;
}

#endif
