/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_ATOMIC_TYPE_H
#define HOST_DEVICE_COMMON_ATOMIC_TYPE_H

#ifdef __KERNELCC__
template <typename T>
using AtomicType = T;
#else
#include <atomic>

template <typename T>
using AtomicType = std::atomic<T>;
#endif

#endif
