/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_FUNCTIONS_FILTER_FUNCTION_PAYLOAD_H
#define DEVICE_FUNCTIONS_FILTER_FUNCTION_PAYLOAD_H

struct HIPRTRenderData;
struct Xorshift32Generator;

struct FilterFunctionPayload
{
	// -- Alpha testing payload --
	const HIPRTRenderData* render_data;
	Xorshift32Generator* random_number_generator;
	// -- Alpha testing payload --

	// What bounce the ray being launched currently is at
	int bounce = 0;

	// -- Self intersection avoidance payload --
	int last_hit_primitive_index;
	// -- Self intersection avoidance payload --
};

#endif
