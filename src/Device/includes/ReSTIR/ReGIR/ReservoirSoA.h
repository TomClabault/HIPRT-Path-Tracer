/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_RESERVOIR_SOA_H
#define DEVICE_KERNELS_REGIR_RESERVOIR_SOA_H

#include "Device/includes/ReSTIR/ReGIR/Reservoir.h"

struct ReGIRSampleSoADevice
{
	HIPRT_HOST_DEVICE void store_sample(int linear_reservoir_index, const ReGIRSample& sample)
	{
		emission[linear_reservoir_index] = sample.emission;
		emissive_triangle_index[linear_reservoir_index] = sample.emissive_triangle_index;
		light_area[linear_reservoir_index] = sample.light_area;
		point_on_light[linear_reservoir_index] = sample.point_on_light;
		target_function[linear_reservoir_index] = sample.target_function;
		light_source_normal[linear_reservoir_index] = sample.light_source_normal;
	}

	HIPRT_HOST_DEVICE ReGIRSample read_sample(int linear_reservoir_index) const
	{
		ReGIRSample sample;

		sample.emission = emission[linear_reservoir_index];
		sample.emissive_triangle_index = emissive_triangle_index[linear_reservoir_index];
		sample.light_area = light_area[linear_reservoir_index];
		sample.point_on_light = point_on_light[linear_reservoir_index];
		sample.target_function = target_function[linear_reservoir_index];
		sample.light_source_normal = light_source_normal[linear_reservoir_index];

		return sample;
	}

	FP32x3LengthUint10Packed* emission = nullptr;
	// Only needed for ReSTIR DI
	int* emissive_triangle_index = nullptr;

	float* light_area = nullptr;
	float3* point_on_light = nullptr;

	// TODO maybe not needed in the SoA?
	float* target_function = nullptr;

	Octahedral24BitNormal* light_source_normal = nullptr;
};

struct ReGIRReservoirSoADevice
{
	HIPRT_HOST_DEVICE void store_reservoir(int linear_reservoir_index, const ReGIRReservoir& reservoir)
	{
		weight_sum[linear_reservoir_index] = reservoir.weight_sum;
		UCW[linear_reservoir_index] = reservoir.UCW;
		M[linear_reservoir_index] = reservoir.M;
	}

	HIPRT_HOST_DEVICE ReGIRReservoir read_reservoir(int linear_reservoir_index) const
	{
		ReGIRReservoir reservoir;

		reservoir.weight_sum = weight_sum[linear_reservoir_index];
		reservoir.UCW = UCW[linear_reservoir_index];
		reservoir.M = M[linear_reservoir_index];

		return reservoir;
	}

	float* weight_sum = nullptr;
	float* UCW = nullptr;

	unsigned char* M = nullptr;
};

#endif