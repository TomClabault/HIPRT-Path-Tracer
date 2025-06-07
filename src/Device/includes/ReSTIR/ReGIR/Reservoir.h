/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_RESERVOIR_H
#define DEVICE_KERNELS_REGIR_RESERVOIR_H

#include "HostDeviceCommon/Xorshift.h"
#include "HostDeviceCommon/LightSampleInformation.h"
#include "HostDeviceCommon/Packing.h"

struct ReGIRSample
{
	int emissive_triangle_index = -1; // Only needed for ReSTIR DI
	// unsigned int random_seed = 0;
	float3 point_on_light;

	// Note: the target function isn't stored in the sample SoA, it's just there during the sampling process
	float target_function = 0.0f;
};

struct ReGIRReservoir
{
	static constexpr float VISIBILITY_REUSE_KILLED_UCW = -42.0f;
	static constexpr float UNDEFINED_UCW = -4242.0f;
	
	HIPRT_DEVICE bool stream_sample(float mis_weight, float target_function, float source_pdf, unsigned int random_seed, const LightSampleInformation& light_sample, Xorshift32Generator& rng)
	{
		float resampling_weight = mis_weight * target_function / source_pdf;

		weight_sum += resampling_weight;

		if (rng() < resampling_weight / weight_sum)
		{
			sample.emissive_triangle_index = light_sample.emissive_triangle_index;
			// sample.random_seed = random_seed;
			sample.point_on_light = light_sample.point_on_light;

			sample.target_function = target_function;

			return true;
		}

		return false;
	}

	HIPRT_DEVICE bool stream_reservoir(float mis_weight, float target_function, const ReGIRReservoir& other_reservoir, Xorshift32Generator& rng)
	{
		float resampling_weight = mis_weight * target_function * other_reservoir.UCW;

		if (resampling_weight <= 0.0f)
			return false;

		weight_sum += resampling_weight;

		if (rng() < resampling_weight / weight_sum)
		{
			sample = other_reservoir.sample;
			sample.target_function = target_function;

			return true;
		}

		return false;
	}

	HIPRT_DEVICE void finalize_resampling(float normalization_weight = 1.0f)
	{
		if (weight_sum <= 0.0f || normalization_weight == 0.0f)
			UCW = 0.0f;
		else
			UCW = 1.0f / sample.target_function * weight_sum / normalization_weight;
	}

	ReGIRSample sample;

	float weight_sum = 0.0f;
	// If the UCW is set to -1, this is because the reservoir was killed by visibility reuse
	float UCW = 0.0f;
};

#endif
