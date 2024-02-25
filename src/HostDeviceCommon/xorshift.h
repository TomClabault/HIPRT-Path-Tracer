#ifndef XORSHIFT_H
#define XORSHIFT_H

#include <hiprt/hiprt_device.h>
#include "Kernels/includes/HIPRT_maths.h"

struct xorshift32_state {
    unsigned int a = 42;
};

struct xorshift32_generator
{
    __device__ xorshift32_generator(unsigned int seed)
    {
        m_state.a = seed;
    }

    __device__ float operator()()
    {
        //Float in [0, 1[
        return RT_MIN(xorshift32() / (float)UINT_MAX, 1.0f - 1.0e-6f);
    }

    __device__ unsigned int xorshift32()
    {
        /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
        unsigned int x = m_state.a;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return m_state.a = x;
    }

    xorshift32_state m_state;
};

#endif