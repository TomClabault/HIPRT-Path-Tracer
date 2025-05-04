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
		light_source_normal[linear_reservoir_index] = sample.light_source_normal;
	}

	HIPRT_HOST_DEVICE ReGIRSample read_sample(int linear_reservoir_index) const
	{
		ReGIRSample sample;

		sample.emission = emission[linear_reservoir_index];
		sample.emissive_triangle_index = emissive_triangle_index[linear_reservoir_index];
		sample.light_area = light_area[linear_reservoir_index];
		sample.point_on_light = point_on_light[linear_reservoir_index];
		sample.light_source_normal = light_source_normal[linear_reservoir_index];

		return sample;
	}

	Float3xLengthUint10bPacked* emission = nullptr;
	// Only needed for ReSTIR DI
	int* emissive_triangle_index = nullptr;

	float* light_area = nullptr;
	float3* point_on_light = nullptr;

	Octahedral24BitNormalPadded32b* light_source_normal = nullptr;
};

struct ReGIRReservoirSoADevice
{
	HIPRT_HOST_DEVICE void store_reservoir_opt(int linear_reservoir_index, const ReGIRReservoir& reservoir)
	{
		UCW[linear_reservoir_index] = reservoir.UCW;
		M[linear_reservoir_index] = reservoir.M;
	}

	/**
	 * The template parameter can be used to indicate whether or not to read the UCW.
	 * 
	 * This makes sense to pass this parameter as false if you've already read the UCW
	 * of the reservoir by some other means
	 */
	template <bool readUCW = true>
	HIPRT_HOST_DEVICE ReGIRReservoir read_reservoir(int linear_reservoir_index) const
	{
		ReGIRReservoir reservoir;

		if constexpr (readUCW)
			reservoir.UCW = UCW[linear_reservoir_index];
		reservoir.M = M[linear_reservoir_index];

		return reservoir;
	}

	float* UCW = nullptr;
	unsigned char* M = nullptr;

	unsigned int number_of_reservoirs_per_cell = 0;
};

#endif