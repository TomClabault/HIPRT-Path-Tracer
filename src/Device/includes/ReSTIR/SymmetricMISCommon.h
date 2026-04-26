/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_SYMMETRIC_MIS_WEIGHTS_COMMON_H
#define DEVICE_RESTIR_SYMMETRIC_MIS_WEIGHTS_COMMON_H

HIPRT_DEVICE float symmetric_ratio_MIS_weights_difference_function(float target_function_at_center, float target_function_from_i, float exponent)
{
	if (target_function_at_center == 0.0f || target_function_from_i == 0.0f)
		return 0.0f;

	float ratio = hippt::min(target_function_at_center / target_function_from_i, target_function_from_i / target_function_at_center);

	if (exponent == 2.0f)
		return hippt::square(ratio);
	else if (exponent == 3.0f)
		return hippt::pow_3(ratio);
	else if (exponent == 4.0f)
		return hippt::pow_4(ratio);
	else
		return hippt::intrin_pow(ratio, exponent);
}

#endif
