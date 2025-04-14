/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_RESERVOIR_H
#define DEVICE_KERNELS_REGIR_RESERVOIR_H

#include "HostDeviceCommon/Xorshift.h"
#include "HostDeviceCommon/LightSampleInformation.h"

struct ReGIRSample
{
	// TODO Octahedral
	float3 light_source_normal = { 0.0f, 1.0f, 0.0f };
	float3 point_on_light = make_float3(0.0f, 0.0f, 0.0f);

	int emissive_triangle_index = -1;
	float light_area = 0.0f;

	// TODO maybe not needed
	float target_function = 0.0f;
};

struct ReGIRReservoir
{
	HIPRT_HOST_DEVICE void stream_sample(float mis_weight, float target_function, float source_pdf, const LightSampleInformation& light_sample, Xorshift32Generator& rng)
	{
		float resampling_weight = mis_weight * target_function / source_pdf;

		M++;
		weight_sum += resampling_weight;

		if (rng() < resampling_weight / weight_sum)
		{
			sample.light_source_normal = light_sample.light_source_normal;
			sample.point_on_light = light_sample.point_on_light;
			sample.emissive_triangle_index = light_sample.emissive_triangle_index;
			sample.light_area = light_sample.light_area;

			sample.target_function = target_function;
		}
	}

	HIPRT_HOST_DEVICE void finalize_resampling()
	{
		if (weight_sum == 0.0f)
			UCW = 0.0f;
		else
			UCW = 1.0f / sample.target_function * weight_sum;
	}

	ReGIRSample sample;

	int M = 0;
	// TODO weight sum is never used at the same time as UCW so only one variable can be used for both to save space
	float weight_sum = 0.0f;
	// If the UCW is set to -1, this is because the reservoir was killed by visibility reuse
	float UCW = 0.0f;
};

#endif
