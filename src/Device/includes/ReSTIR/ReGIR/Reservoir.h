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
	Float3xLengthUint10bPacked emission;
	int emissive_triangle_index; // Only needed for ReSTIR DI

	float light_area = 0.0f;
	// TODO maybe not needed during shading?
	float target_function = 0.0f;
	float3 point_on_light = make_float3(0.0f, 0.0f, 0.0f);

	// The light source normal is here for padding reasons :(
	Octahedral24BitNormal light_source_normal;
};

struct ReGIRReservoir
{
	static constexpr float VISIBILITY_REUSE_KILLED_UCW = -42.0f;
	
	HIPRT_HOST_DEVICE bool stream_sample(float mis_weight, float target_function, float source_pdf, const LightSampleInformation& light_sample, Xorshift32Generator& rng)
	{
		float resampling_weight = mis_weight * target_function / source_pdf;

		M++;
		weight_sum += resampling_weight;

		if (rng() < resampling_weight / weight_sum)
		{
			sample.light_source_normal = Octahedral24BitNormal::pack_static(light_sample.light_source_normal);
			sample.point_on_light = light_sample.point_on_light;
			sample.emission.pack(light_sample.emission);
			sample.light_area = light_sample.light_area;
			sample.emissive_triangle_index = light_sample.emissive_triangle_index;

			sample.target_function = target_function;

			return true;
		}

		return false;
	}

	HIPRT_HOST_DEVICE bool stream_reservoir(float mis_weight, float target_function, const ReGIRReservoir& other_reservoir, Xorshift32Generator& rng)
	{
		float resampling_weight = mis_weight * target_function * other_reservoir.UCW;

		M += other_reservoir.M;
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

	HIPRT_HOST_DEVICE void finalize_resampling(float normalization_denom = 1.0f)
	{
		if (weight_sum == 0.0f || normalization_denom == 0.0f)
			UCW = 0.0f;
		else
			UCW = 1.0f / sample.target_function * weight_sum / normalization_denom;
	}

	ReGIRSample sample;

	// TODO weight sum is never used at the same time as UCW so only one variable can be used for both to save space
	float weight_sum = 0.0f;
	// If the UCW is set to -1, this is because the reservoir was killed by visibility reuse
	float UCW = 0.0f;

	unsigned char M = 0;
};

#endif
