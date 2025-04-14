/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_TARGET_FUNCTION_H
#define DEVICE_INCLUDES_REGIR_TARGET_FUNCTION_H

#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE float ReGIR_grid_fill_evaluate_target_function(HIPRTRenderData& render_data, float3 cell_center, const LightSampleInformation& regir_sample)
{
	return regir_sample.emission.luminance() / hippt::length2(cell_center - regir_sample.point_on_light);
}

#endif