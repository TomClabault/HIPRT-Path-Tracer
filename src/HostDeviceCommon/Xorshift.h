/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef XORSHIFT_H
#define XORSHIFT_H

#include <hiprt/hiprt_device.h>

#include "HostDeviceCommon/Math.h"

struct Xorshift32State {
    unsigned int seed = 42;
};

struct Xorshift32Generator
{
    static const unsigned int XORSHIFT_MAX = 0xffffffff;

    HIPRT_HOST_DEVICE Xorshift32Generator()
    {
        m_state.seed = 42;
    }

    HIPRT_HOST_DEVICE Xorshift32Generator(unsigned int seed)
    {
        m_state.seed = seed;
    }

    /*
     * Returns a uniform random number between 0 and
     * array_size - 1 (included)
     */
    HIPRT_HOST_DEVICE int random_index(int array_size)
    {
        int random_num = xorshift32() / (float)XORSHIFT_MAX * array_size;
        return hippt::min(random_num, array_size - 1);
    }

    /*
     * Returns a float int [0, 1.0 - 1.0e-9f]
     */
    HIPRT_HOST_DEVICE float operator()()
    {
        //Float in [0, 1[
        return hippt::min(xorshift32() / (float)XORSHIFT_MAX, 1.0f - 1.0e-9f);
    }

    /**
     * Returns a random uint
     */
    HIPRT_HOST_DEVICE unsigned int xorshift32()
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

#endif