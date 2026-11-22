/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_TRIANGLE_UTILS_H
#define DEVICE_TRIANGLE_UTILS_H

#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE float triangle_load_area(const HIPRTRenderData& render_data, int triangle_index)
{
	return render_data.buffers.triangles_areas[triangle_index];
}

HIPRT_DEVICE ColorRGB32F triangle_load_emission(const HIPRTRenderData& render_data, int triangle_index)
{
	return render_data.buffers.materials_buffer[render_data.buffers.material_indices[triangle_index]].get_raw_emission();
}

#endif // #ifndef DEVICE_TRIANGLE_UTILS_H
