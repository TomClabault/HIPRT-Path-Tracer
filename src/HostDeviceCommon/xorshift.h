#ifndef XORSHIFT_H
#define XORSHIFT_H

#include <hiprt/hiprt_device.h>
#include "Kernels/includes/HIPRT_maths.h"

struct Xorshift32State {
    unsigned int a = 42;
};

struct Xorshift32Generator
{
    __device__ Xorshift32Generator(unsigned int seed)
    {
        m_state.a = seed;
    }

    /*
     * Returns a uniform random number between 0 and
     * array_size - 1 (included)
     */
    __device__ int random_index(int array_size)
    {
        return RT_MIN((int)(xorshift32() / (float)UINT_MAX * array_size), array_size - 1);
    }

    /*
     * Returns a float int [0, 1.0 - 1.0e-6f]
     */
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

    Xorshift32State m_state;
};

#endif