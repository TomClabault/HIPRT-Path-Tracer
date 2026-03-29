/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/FixIntellisense.h"

#include "HostDeviceCommon/Maths/Math.h"

#include <limits>

#ifndef DEVICE_INCLUDES_COMPUTE_COMMON_OPERATORS_H
#define DEVICE_INCLUDES_COMPUTE_COMMON_OPERATORS_H

template <typename T>
struct OperatorSum
{
	static constexpr T identity = T(0);

	HIPRT_DEVICE static T apply(T a, T b)
	{
		return a + b;
	}
};

template <typename T>
struct OperatorMin
{
	static constexpr T identity = std::numeric_limits<T>::max();

	HIPRT_DEVICE static T apply(T a, T b)
	{
		return hippt::min(a, b);
	}
};

template <typename T>
struct OperatorMax
{
	static constexpr T identity = std::numeric_limits<T>::lowest();

	HIPRT_DEVICE static T apply(T a, T b)
	{
		return hippt::max(a, b);
	}
};

#endif
