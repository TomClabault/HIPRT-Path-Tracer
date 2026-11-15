/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_XORSHIFT_H
#define HOST_DEVICE_COMMON_XORSHIFT_H

#ifdef __KERNELCC__
#include <hiprt/hiprt_device.h>
#endif

#include "HostDeviceCommon/Maths/Math.h"

struct Xorshift32State
{
	unsigned int seed = 42;
};

struct Xorshift32Generator
{
	static const unsigned int XORSHIFT_MAX = 0xffffffff;

	HIPRT_DEVICE Xorshift32Generator()
	{
		m_state.seed = 42;
	}

	HIPRT_DEVICE Xorshift32Generator(unsigned int seed)
	{
		m_state.seed = seed;
	}

	/*
	 * Returns a uniform random number between 0 and
	 * array_size - 1 (included)
	 */
	HIPRT_DEVICE int random_index(int max_index_excluded)
	{
		int random_num = xorshift32() / static_cast<float>(XORSHIFT_MAX) * max_index_excluded;
		return hippt::min(random_num, max_index_excluded - 1);
	}

	/*
	 * Same as random_index(int) but 'max_index_excluded_as_float' is the float value of 'max_index_excluded', computed once on the host.
	 *
	 * On the GPU, converting a uniform int to a float is a vector instruction: when the compiler hoists that conversion out of a loop, the converted
	 * value keeps a vector register for the whole loop. A float read from memory is used directly from a scalar register instead
	 */
	HIPRT_DEVICE int random_index(int max_index_excluded, float max_index_excluded_as_float)
	{
		int random_num = xorshift32() / static_cast<float>(XORSHIFT_MAX) * max_index_excluded_as_float;
		return hippt::min(random_num, max_index_excluded - 1);
	}

	/*
	 * Returns a float int [0, 1.0 - 1.0e-9f]
	 */
	HIPRT_DEVICE float operator()()
	{
		// Float in [0, 1[
		float a = xorshift32() / static_cast<float>(XORSHIFT_MAX);
		return hippt::min(a, 1.0f - 1.0e-7f);
	}

	/**
	 * Returns a random uint
	 */
	HIPRT_DEVICE unsigned int xorshift32()
	{
		/* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
		unsigned int x = m_state.seed;
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;
		return m_state.seed = x;
	}

	Xorshift32State m_state;
};

#endif // #ifndef HOST_DEVICE_COMMON_XORSHIFT_H
