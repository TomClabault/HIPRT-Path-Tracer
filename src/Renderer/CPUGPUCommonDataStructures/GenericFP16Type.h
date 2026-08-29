/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GENERIC_FP16_TYPE_H
#define RENDERER_GENERIC_FP16_TYPE_H

#include <cstdint>
#include <type_traits>
#include <vector>

template <template <typename> class Container>
using GenericFP16Type = typename std::conditional_t<
    std::is_same<Container<float>, std::vector<float>>::value,
    float,     // CPU: 4-byte float, full 32-bit precision
    uint16_t   // GPU: 2-byte type matching __half size for correct GPU allocation
>;

#endif // #ifndef RENDERER_GENERIC_FP16_TYPE_H
