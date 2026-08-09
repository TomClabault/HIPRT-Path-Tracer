/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_INPUT_ENCODINGS_H
#define DEVICE_INCLUDES_NEURAL_INPUT_ENCODINGS_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

/**
 * Encodes each raw input feature with frequency-major sine/cosine pairs.
 *
 * The encoded order is [sin(1*x*pi), cos(1*x*pi), sin(2*x*pi), cos(2*x*pi), ...]
 * for each raw feature before continuing with the next feature.
 */
template <unsigned int InputSizeRaw_, unsigned int FrequencyCount_>
HIPRT_DEVICE void encode_frequency_input(const float* input, float* encoded_input)
{
	for (unsigned int input_raw_index = 0; input_raw_index < InputSizeRaw_; input_raw_index++)
	{
		float input_value = input[input_raw_index];

		for (unsigned int frequency_index = 0; frequency_index < FrequencyCount_; frequency_index++)
		{
			float frequency			   = static_cast<float>(1 << frequency_index);
			unsigned int encoded_index = input_raw_index * FrequencyCount_ * 2 + frequency_index * 2;

			encoded_input[encoded_index + 0] = hippt::intrin_sinf(frequency * input_value * hippt::M_Pi);
			encoded_input[encoded_index + 1] = hippt::intrin_cosf(frequency * input_value * hippt::M_Pi);
		}
	}
}

#endif
