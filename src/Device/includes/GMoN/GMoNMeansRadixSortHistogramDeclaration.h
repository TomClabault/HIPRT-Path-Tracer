/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_GMON_RADIX_SORT_HISTROGRAM_DECLARATION_H
#define DEVICE_GMON_RADIX_SORT_HISTROGRAM_DECLARATION_H

#include "HostDeviceCommon/KernelOptions/GMoNOptions.h"

// The maximum number of sets allowed is 31
// This means that the values of the histogram will never go above 31
// 
// 31 can be encoded with 5 bits
// 1 unsigned int is 32 bits
// 
// That makes 6 histogram bins of 5 bits per 32bits uint
#define BITS_PER_HISTOGRAM_BIN 5
#define MAX_BINS_PER_HISTOGRAM_UINT 6
#define MAX_BINS_PER_HISTOGRAM_UINT_F 6.0f

/**
 * We're using a class here to compute the histogram for two reasons:
 * 
 *	- Without this class, we would probably use an array unsigned int[HISTOGRAM_SIZE] but arrays like that
 *		behave very poorly with the HIP compiler so we're using simple unsigned int variables instead, not an array
 *		(and that's why we have a big #if, #elif, #endif at the end of the structure to declare the histogram
 *		variables depending	on how many we need)
 * 
 *	- We use this class to do some packing since we allow only a maximum of 31 sets, we can make some assumption
 *		about how many bits we need per histogram bins
 */
struct GMoNRadixSortHistogram
{
	/**
	 * This function adds 'value' to the correct histogram bin
	 */
	HIPRT_HOST_DEVICE HIPRT_INLINE void increment(unsigned int index, unsigned int value)
	{
		unsigned int histogram_variable_index = static_cast<unsigned int>(index / MAX_BINS_PER_HISTOGRAM_UINT_F);
		unsigned int bin_index = index - histogram_variable_index * MAX_BINS_PER_HISTOGRAM_UINT;

		switch (histogram_variable_index)
		{
		case 0:
			histogram0 += value << (bin_index * BITS_PER_HISTOGRAM_BIN);
			break;

#if GMoNSortRadixSize >= 4
		case 1:
			histogram1 += value << (bin_index * BITS_PER_HISTOGRAM_BIN);
			break;

		case 2:
			histogram2 += value << (bin_index * BITS_PER_HISTOGRAM_BIN);
			break;
#endif
		}
	}

	/**
	 * This function adds 'value' to the correct histogram bin
	 */
	HIPRT_HOST_DEVICE HIPRT_INLINE void decrement(unsigned int index, unsigned int value)
	{
		unsigned int histogram_variable_index = static_cast<unsigned int>(index / MAX_BINS_PER_HISTOGRAM_UINT_F);
		unsigned int bin_index = index - histogram_variable_index * MAX_BINS_PER_HISTOGRAM_UINT;

		// Getting the current value of the bin
		unsigned int histogram_current_value = fetch_value(index);

		// Decrementing
		histogram_current_value -= value;

		// Clearing before setting
		clear_bin(index);

		// Adding (to the bin value that is now 0)
		increment(index, histogram_current_value);
	}

	/**
     * Returns
     */
	HIPRT_HOST_DEVICE HIPRT_INLINE unsigned int fetch_value(unsigned int index)
	{
		unsigned int histogram_variable_index = static_cast<unsigned int>(index / MAX_BINS_PER_HISTOGRAM_UINT_F);
		unsigned int bin_index = index - histogram_variable_index * MAX_BINS_PER_HISTOGRAM_UINT;

		switch (histogram_variable_index)
		{
		case 0:
			return (histogram0 >> (bin_index * BITS_PER_HISTOGRAM_BIN)) & 31;

#if GMoNSortRadixSize >= 4
		case 1:
			return (histogram1 >> (bin_index * BITS_PER_HISTOGRAM_BIN)) & 31;

		case 2:
			return (histogram2 >> (bin_index * BITS_PER_HISTOGRAM_BIN)) & 31;
#endif

		default:
			return -1;
		}
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void clear_bin(unsigned int index)
	{
		unsigned int histogram_variable_index = static_cast<unsigned int>(index / MAX_BINS_PER_HISTOGRAM_UINT_F);
		unsigned int bin_index = index - histogram_variable_index * MAX_BINS_PER_HISTOGRAM_UINT;

		switch (histogram_variable_index)
		{
		case 0:
			histogram0 &= ~(31 << (bin_index * BITS_PER_HISTOGRAM_BIN));
			break;

#if GMoNSortRadixSize >= 4
		case 1:
			histogram1 &= ~(31 << (bin_index * BITS_PER_HISTOGRAM_BIN));
			break;

		case 2:
			histogram2 &= ~(31 << (bin_index * BITS_PER_HISTOGRAM_BIN));
			break;
#endif
		}
	}

#if GMoNSortRadixSize == 1
	unsigned int histogram0 = 0;
#elif GMoNSortRadixSize == 2
	unsigned int histogram0 = 0;
#elif GMoNSortRadixSize == 4
	unsigned int histogram0 = 0, histogram1 = 0, histogram2 = 0;
#endif
};

#endif
