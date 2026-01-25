/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LIGHT_UTILS_H
#define DEVICE_LIGHT_UTILS_H

#include "HostDeviceCommon/Color.h"

 /**
  * 'clamp_condition' is an additional condition that needs to be met
  * for clamping to occur. If the additional condition is not met (the boolean
  * 'clamp_condition' is false, then the 'light_contribution' parameter is returned
  * untouched
  */
HIPRT_DEVICE ColorRGB32F clamp_light_contribution(ColorRGB32F light_contribution, float clamp_max_value, bool clamp_condition)
{
	if (!light_contribution.has_nan() && clamp_max_value > 0.0f && clamp_condition)
		// We don't want to clamp NaNs because that's UB (kind of) and the NaNs get
		// immediately clamped to 'clamp_max_value' in my experience
		//
		// Not clamping the negatives to 0 because
		// spectral rendering (for dispersion for example) may produce negative values
		// and we don't want to clamp those to 0
		light_contribution.clamp(-clamp_max_value, clamp_max_value);

	return light_contribution;
}

#endif
