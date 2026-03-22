/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "HostDeviceCommon/RenderData.h"

#ifndef DEVICE_INCLUDES_RANDOM_H
#define DEVICE_INCLUDES_RANDOM_H

HIPRT_DEVICE unsigned int generate_fresh_pixel_random_seed(const HIPRTRenderData& render_data, unsigned int pixel_index)
{
	return wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1) ^ 0xdeadbeef);
}

#endif
