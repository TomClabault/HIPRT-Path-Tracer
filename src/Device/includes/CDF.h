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
			if (hippt::ldg_load(cdf + mid) < random_value)
				left = mid + 1;
			else
				right = mid;
		}

		return left;
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
			if (hippt::ldg_load(cdf_u16 + mid) < random_value)
				left = mid + 1;
			else
				right = mid;
		}

		return left;
	}

	const unsigned short int* cdf_u16 = nullptr;

	unsigned int size = 0;
};

#endif
