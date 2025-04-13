/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RESTIR_DI_PRESAMPLED_LIGHT_H
#define RESTIR_DI_PRESAMPLED_LIGHT_H

#include "Device/includes/ReSTIR/DI/SampleFlags.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Math.h"

struct ReSTIRDIPresampledLight
{
	// Global primitive index corresponding to the emissive triangle sampled
	int emissive_triangle_index = -1;

	// For envmap samples, this 'point_on_light_source' is the envmap direction in *envmap space*
	// A sample is an envmap sample if 'flags' contains 'RESTIR_DI_FLAGS_ENVMAP_SAMPLE'
	float3 point_on_light_source = { 0, 0, 0 };
	// Only defined if the sample isn't an envmap sample
	float3 light_source_normal = { 0, 0, 0 };

	ColorRGB32F radiance;
	float pdf = 0.0f;

	// Some flags about the sample
	unsigned char flags = RESTIR_DI_FLAGS_NONE;
};

#endif
