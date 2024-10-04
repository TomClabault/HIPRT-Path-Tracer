/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_FUNCTIONS_ALPHA_TESTING_H
#define DEVICE_FUNCTIONS_ALPHA_TESTING_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Material.h"

#include "HostDeviceCommon/RenderData.h"

struct AlphaTestingPayload
{
	const HIPRTRenderData* render_data;
	Xorshift32Generator* random_number_generator;
};

HIPRT_DEVICE HIPRT_INLINE bool alpha_testing(const hiprtRay&, const void*, void* payld, const hiprtHit& hit)
{
	AlphaTestingPayload* payload = reinterpret_cast<AlphaTestingPayload*>(payld);
	if (!payload->render_data->render_settings.do_alpha_testing)
		return false;

	int material_index = payload->render_data->buffers.material_indices[hit.primID];
	RendererMaterial material = payload->render_data->buffers.materials_buffer[material_index];

	// Composition both the alpha of the base color texture and the material
	float base_color_alpha = get_hit_base_color_alpha(*payload->render_data, material, hit);
	float composited_alpha = material.alpha_opacity * base_color_alpha;

	if ((*payload->random_number_generator)() < composited_alpha)
		return false;

	// No tests stopped the ray, that's not a hit.
	// Returning 'true' to filter out the intersection
	return true;
}

#endif
