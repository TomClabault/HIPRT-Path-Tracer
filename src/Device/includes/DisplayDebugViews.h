/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_DISPLAY_DEBUG_VIEWS_H
#define DEVICE_INCLUDES_DISPLAY_DEBUG_VIEWS_H

#include "HostDeviceCommon/DisplayPostProcessSettings.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE ColorRGB32F display_view_denoiser_albedo(const HIPRTRenderData& render_data, unsigned int pixel_index)
{
	if (render_data.aux_buffers.denoiser_albedo == nullptr)
		return ColorRGB32F(0.0f);

	return render_data.aux_buffers.denoiser_albedo[pixel_index];
}

HIPRT_DEVICE ColorRGB32F display_view_denoiser_normals(const HIPRTRenderData& render_data, unsigned int pixel_index)
{
	if (render_data.aux_buffers.denoiser_normals == nullptr)
		return ColorRGB32F(0.0f);

	float3_t normal = render_data.aux_buffers.denoiser_normals[pixel_index];
	return ColorRGB32F((normal.x + 1.0f) * 0.5f, (normal.y + 1.0f) * 0.5f, (normal.z + 1.0f) * 0.5f);
}

HIPRT_DEVICE ColorRGB32F display_view_apply_white_furnace_threshold(ColorRGB32F color, const DisplayPostProcessSettings& display_settings)
{
	if (display_settings.white_furnace_use_high_threshold == 1 && (color.r > 0.505f || color.g > 0.505f || color.b > 0.505f))
		return ColorRGB32F(1.0f, 0.0f, 0.0f);

	if (display_settings.white_furnace_use_low_threshold == 1 && (color.r < 0.495f || color.g < 0.495f || color.b < 0.495f))
		return ColorRGB32F(0.0f, 1.0f, 0.0f);

	return color;
}

#endif // #ifndef DEVICE_INCLUDES_DISPLAY_DEBUG_VIEWS_H
