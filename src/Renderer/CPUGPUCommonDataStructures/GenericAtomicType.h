/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GENERIC_ATOMIC_TYPE_H
#define RENDERER_GENERIC_ATOMIC_TYPE_H

#include <type_traits>
#include <vector>

#include "HostDeviceCommon/AtomicType.h"

template <typename T, template <typename> class Container>
using GenericAtomicType = typename std::conditional_t<std::is_same<Container<T>, std::vector<T>>::value, AtomicType<T>, T>;

#endif // #ifndef RENDERER_GENERIC_ATOMIC_TYPE_H
