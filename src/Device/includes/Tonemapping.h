/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_TONEMAPPING_H
#define DEVICE_INCLUDES_TONEMAPPING_H

#include "HostDeviceCommon/Color.h"

/**
 * Applies exponential tone mapping followed by gamma correction.
 *
 * The exponential operator maps non-negative linear HDR values to the [0, 1)
 * range before gamma encoding them for display.
 */
HIPRT_DEVICE ColorRGB32F tonemap_exponential(ColorRGB32F linear_color, float exposure, float gamma)
{
	ColorRGB32F tone_mapped = ColorRGB32F(1.0f) - intrin_expf(-linear_color * exposure);
	return intrin_pow(tone_mapped, 1.0f / gamma);
}

#endif // #ifndef DEVICE_INCLUDES_TONEMAPPING_H
