/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef FILTER_FUNCTION_H
#define FILTER_FUNCTION_H

#include <hiprt/hiprt_types.h> // for hiprtRay

using FilterFunction = bool(*)(const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit);

#endif