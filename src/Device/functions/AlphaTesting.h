/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_FUNCTIONS_ALPHA_TESTING_H
#define DEVICE_FUNCTIONS_ALPHA_TESTING_H

#include "Device/includes/FixIntellisense.h"

#include "HostDeviceCommon/RenderData.h"

__device__ bool alpha_testing(const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit)
{
	HIPRTRenderData* render_data = reinterpret_cast<HIPRTRenderData*>(payload);

	if (render_data->render_settings.do_alpha_testing)
		return get_hit_base_color_alpha(*render_data, hit) < 1.0f;

	return false;
}

#endif