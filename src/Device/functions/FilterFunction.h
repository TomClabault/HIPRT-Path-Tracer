/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
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
 * return false if the intersection is accepted
 * return true if the intersection is rejected
 */
HIPRT_DEVICE HIPRT_INLINE bool filter_function(const hiprtRay&, const void*, void* payld, const hiprtHit& hit)
{
	FilterFunctionPayload* payload = reinterpret_cast<FilterFunctionPayload*>(payld);
	if (hit.primID == payload->last_hit_primitive_index)
		// This is a self-intersection, filtering it out
		//
		// Triangles are planar so one given triangle can
		// never be intersect twice in a row (unless we're absolutely
		// perfectly parallel to the triangle but let's ignore that...)
		//
		// This self-intersection avoidance only works for planar primitives
		return true;

	if (!payload->render_data->render_settings.do_alpha_testing)
		return false;

	if (payload->bounce >= payload->render_data->render_settings.alpha_testing_indirect_bounce)
		// Alpha testing is disable at the current bounce
		return false;

	int material_index = payload->render_data->buffers.material_indices[hit.primID];
	if (payload->render_data->buffers.material_opaque[material_index])
		// The material is fully opaque, no need to test further, accept the intersection
		return false;

	// Composition both the alpha of the base color texture and the material
	unsigned short int base_color_texture_index = payload->render_data->buffers.materials_buffer.get_base_color_texture_index(material_index);
	float base_color_alpha = get_hit_base_color_alpha(*payload->render_data, base_color_texture_index, hit);
	float alpha_opacity = payload->render_data->buffers.materials_buffer.get_alpha_opacity(material_index);
	float composited_alpha = alpha_opacity * base_color_alpha;

	if ((*payload->random_number_generator)() < composited_alpha)
		return false;

	// No tests stopped the ray, that's not a hit.
	// Returning 'true' to filter out the intersection
	return true;
}

#endif
