/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_CDF_H
#define DEVICE_INCLUDES_CDF_H

#include "HostDeviceCommon/Xorshift.h"

struct CDFDevice
{
	template <int sampleCount>
	struct Samples
	{
		unsigned int samples[sampleCount];
	};

	HIPRT_DEVICE unsigned int sample(Xorshift32Generator& rng) const
	{
		float random_value = rng();

		// Binary search
		unsigned int left  = 0;
		unsigned int right = size;

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

	// TODO the size here could be a template parameter but then we'd need to write all sorting networks for all supported template parameter sizes
	HIPRT_DEVICE typename CDFDevice::template Samples<8> sample_8(Xorshift32Generator& rng) const
	{
		float random_numbers[8];
		for (int i = 0; i < 8; ++i)
			random_numbers[i] = rng();

		// Sorting the random numbers with a sorting network
		hippt::compare_swap(random_numbers[0], random_numbers[1]);
		hippt::compare_swap(random_numbers[2], random_numbers[3]);
		hippt::compare_swap(random_numbers[4], random_numbers[5]);
		hippt::compare_swap(random_numbers[6], random_numbers[7]);

		hippt::compare_swap(random_numbers[0], random_numbers[2]);
		hippt::compare_swap(random_numbers[1], random_numbers[3]);
		hippt::compare_swap(random_numbers[4], random_numbers[6]);
		hippt::compare_swap(random_numbers[5], random_numbers[7]);

		hippt::compare_swap(random_numbers[1], random_numbers[2]);
		hippt::compare_swap(random_numbers[5], random_numbers[6]);

		hippt::compare_swap(random_numbers[0], random_numbers[4]);
		hippt::compare_swap(random_numbers[1], random_numbers[5]);
		hippt::compare_swap(random_numbers[2], random_numbers[6]);
		hippt::compare_swap(random_numbers[3], random_numbers[7]);

		hippt::compare_swap(random_numbers[0], random_numbers[2]);
		hippt::compare_swap(random_numbers[1], random_numbers[3]);
		hippt::compare_swap(random_numbers[4], random_numbers[6]);
		hippt::compare_swap(random_numbers[5], random_numbers[7]);

		hippt::compare_swap(random_numbers[0], random_numbers[1]);
		hippt::compare_swap(random_numbers[2], random_numbers[3]);
		hippt::compare_swap(random_numbers[4], random_numbers[5]);
		hippt::compare_swap(random_numbers[6], random_numbers[7]);

		hippt::compare_swap(random_numbers[2], random_numbers[4]);
		hippt::compare_swap(random_numbers[3], random_numbers[5]);

		hippt::compare_swap(random_numbers[1], random_numbers[2]);
		hippt::compare_swap(random_numbers[3], random_numbers[4]);
		hippt::compare_swap(random_numbers[5], random_numbers[6]);

		hippt::compare_swap(random_numbers[2], random_numbers[3]);
		hippt::compare_swap(random_numbers[4], random_numbers[5]);

		typename CDFDevice::template Samples<8> samples;

		// Because the random numbers are sorted in ascending order, the left index can be reused from random number to the next random number
		unsigned int current_left = 0;
		for (int i = 0; i < 8; ++i)
		{
			float random_value = random_numbers[i];

			// Binary search
			unsigned int left  = current_left;
			unsigned int right = size;

			while (left < right)
			{
				unsigned int mid = (left + right) / 2;
				float cdf_value;
				if (mid == 0)
					// This shortcut avoids one memory load and also allows storing whatever data we want in cdf[0] since it will never be used (such as the
					// total sum of the weights for example)
					cdf_value = 0.0f;
				else
					cdf_value = cdf[mid];
				if (cdf_value < random_value)
					left = mid + 1;
				else
					right = mid;
			}

			samples.samples[i] = left > 0 ? left - 1 : 0;

			current_left = left;
		}

		return samples;
	}

	const float* cdf = nullptr;

	unsigned int size = 0;
};

template <unsigned int LUTSize, typename LUTIndexType>
struct CDFDeviceWithLUT
{
	HIPRT_DEVICE unsigned int sample(Xorshift32Generator& rng) const
	{
		// Binary search
		float random_value				  = rng();
		unsigned int lut_bin_index		  = hippt::min(LUTSize - 1, static_cast<unsigned int>(random_value * LUTSize));
		LUTIndexType DEBUGcdf_index_start = cdf_lut[lut_bin_index];

		LUTIndexType left  = cdf_lut[lut_bin_index];
		LUTIndexType right = (lut_bin_index < LUTSize - 1 ? cdf_lut[lut_bin_index + 1] : size - 1) + 1;

		while (left < right)
		{
			LUTIndexType mid = (left + right) / 2;
			float cdf_value	 = mid == 0 ? 0.0f : cdf[mid];

			if (cdf_value < random_value)
				left = mid + 1;
			else
				right = mid;
		}

		return left > 0 ? left - 1 : 0;
	}

	const float* cdf  = nullptr;
	unsigned int size = 0;

	LUTIndexType* cdf_lut = nullptr;
};

struct CDFDeviceU16
{
	HIPRT_DEVICE unsigned int sample(Xorshift32Generator& rng) const
	{
		unsigned short int random_value = rng() * 65535.0f;
		unsigned short int value_zero	= cdf_u16[0];

		// Binary search
		unsigned int left  = 0;
		unsigned int right = size;

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
