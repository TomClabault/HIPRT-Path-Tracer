#ifndef XORSHIFT_H
#define XORSHIFT_H

#include <cstdint>

struct xorshift32_state {
    uint32_t a = 42;
};

struct xorshift32_generator
{
    xorshift32_generator(uint32_t seed) : m_state({ seed }) {}

    float operator()()
    {
        //Float in [0, 1[
        return std::min(xorshift32() / (float)std::numeric_limits<unsigned int>::max(), 1.0f - 1.0e-6f);
    }

    uint32_t xorshift32()
    {
        /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
        uint32_t x = m_state.a;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return m_state.a = x;
    }

    xorshift32_state m_state;
};

#endif
