/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_SPATIAL_SAMPLE_MOMENTS_SOA_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_SPATIAL_SAMPLE_MOMENTS_SOA_DEVICE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeSpatialSampleMoments.h"
#include "HostDeviceCommon/AtomicType.h"

struct IlluminationAwareKDTreeSpatialSampleMomentsSoADevice
{
	// Conversion helpers are necessary because the spatial split code still consumes the
	// compact AoS moments type, while atomic accumulation requires scalar SoA components.
	HIPRT_DEVICE IlluminationAwareKDTreeSpatialSampleMoments read(unsigned int index) const
	{
		IlluminationAwareKDTreeSpatialSampleMoments moments{};

		moments.positive_radiance_sample_count = positive_radiance_sample_count[index];
		moments.position_sum				   = make_float3(position_sum_x[index], position_sum_y[index], position_sum_z[index]);
		moments.position_squared_sum		   = make_float3(position_squared_sum_x[index], position_squared_sum_y[index], position_squared_sum_z[index]);

		return moments;
	}

	HIPRT_DEVICE void write(unsigned int index, const IlluminationAwareKDTreeSpatialSampleMoments& moments)
	{
		positive_radiance_sample_count[index] = moments.positive_radiance_sample_count;
		position_sum_x[index]				  = moments.position_sum.x;
		position_sum_y[index]				  = moments.position_sum.y;
		position_sum_z[index]				  = moments.position_sum.z;
		position_squared_sum_x[index]		  = moments.position_squared_sum.x;
		position_squared_sum_y[index]		  = moments.position_squared_sum.y;
		position_squared_sum_z[index]		  = moments.position_squared_sum.z;
	}

	HIPRT_DEVICE void reset(unsigned int index)
	{
		positive_radiance_sample_count[index] = 0u;
		position_sum_x[index]				  = 0.0f;
		position_sum_y[index]				  = 0.0f;
		position_sum_z[index]				  = 0.0f;
		position_squared_sum_x[index]		  = 0.0f;
		position_squared_sum_y[index]		  = 0.0f;
		position_squared_sum_z[index]		  = 0.0f;
	}

	AtomicType<unsigned int>* positive_radiance_sample_count = nullptr;
	AtomicType<float>* position_sum_x						 = nullptr;
	AtomicType<float>* position_sum_y						 = nullptr;
	AtomicType<float>* position_sum_z						 = nullptr;
	AtomicType<float>* position_squared_sum_x				 = nullptr;
	AtomicType<float>* position_squared_sum_y				 = nullptr;
	AtomicType<float>* position_squared_sum_z				 = nullptr;
};

#endif
