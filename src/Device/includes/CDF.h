/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_CDF_H
#define DEVICE_INCLUDES_CDF_H

#include "HostDeviceCommon/Xorshift.h"

struct CDFDevice
{
	HIPRT_DEVICE unsigned int sample(Xorshift32Generator& rng) const
	{
		float random_value = rng();

		// Binary search
		unsigned int left  = 0;
		unsigned int right = size - 1;

		while (left < right)
		{
			unsigned int mid = (left + right) / 2;
			float cdf_value;
			if (mid == 0)
				// This shortcut avoids one memory load and also allows storing whatever data we want in cdf[0] since it will never be used (such as the total
				// sum of the weights for example)
				cdf_value = 0.0f;
			else
				cdf_value = cdf[mid];
			if (cdf_value < random_value)
				left = mid + 1;
			else
				right = mid;
		}

		return left > 0 ? left - 1 : 0;
	}

	const float* cdf = nullptr;

	unsigned int size = 0;
};

struct CDFDeviceU16
{
	HIPRT_DEVICE unsigned int sample(Xorshift32Generator& rng) const
	{
		unsigned short int random_value = rng() * 65535.0f;
		unsigned short int value_zero	= cdf_u16[0];

		// Binary search
		unsigned int left  = 0;
		unsigned int right = size - 1;

		while (left < right)
		{
			unsigned int mid = (left + right) / 2;
			unsigned short int cdf_value;
			if (mid == 0)
				// This shortcut avoids one memory load and also allows storing whatever data we want in cdf_u16[0] since it will never be used (such as the
				// total sum of the weights for example)
				cdf_value = 0;
			else
				cdf_value = cdf_u16[mid];
			if (cdf_value < random_value)
				left = mid + 1;
			else
				right = mid;
		}

		return left > 0 ? left - 1 : 0;
	}

	const unsigned short int* cdf_u16 = nullptr;

	unsigned int size = 0;
};

#endif
