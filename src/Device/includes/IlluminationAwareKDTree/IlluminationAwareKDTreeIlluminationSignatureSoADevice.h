/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_SIGNATURE_SOA_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_SIGNATURE_SOA_DEVICE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeIlluminationSignature.h"
#include "HostDeviceCommon/AtomicType.h"

struct IlluminationAwareKDTreeIlluminationSignatureSoADevice
{
	// Conversion helpers are necessary because the spatial KD-tree code still consumes the
	// compact AoS signature type, while atomic accumulation requires scalar SoA components.
	HIPRT_DEVICE IlluminationAwareKDTreeIlluminationSignature read(unsigned int index) const
	{
		IlluminationAwareKDTreeIlluminationSignature signature{};

		signature.valid_observation_count	  = valid_observation_count[index];
		signature.scalar_radiance_sum		  = scalar_radiance_sum[index];
		signature.squared_scalar_radiance_sum = squared_scalar_radiance_sum[index];
		signature.weighted_direction_sum	  = make_float3(weighted_direction_sum_x[index], weighted_direction_sum_y[index], weighted_direction_sum_z[index]);

		return signature;
	}

	HIPRT_DEVICE void write(unsigned int index, const IlluminationAwareKDTreeIlluminationSignature& signature)
	{
		valid_observation_count[index]	   = signature.valid_observation_count;
		scalar_radiance_sum[index]		   = signature.scalar_radiance_sum;
		squared_scalar_radiance_sum[index] = signature.squared_scalar_radiance_sum;
		weighted_direction_sum_x[index]	   = signature.weighted_direction_sum.x;
		weighted_direction_sum_y[index]	   = signature.weighted_direction_sum.y;
		weighted_direction_sum_z[index]	   = signature.weighted_direction_sum.z;
	}

	HIPRT_DEVICE void reset(unsigned int index)
	{
		valid_observation_count[index]	   = 0u;
		scalar_radiance_sum[index]		   = 0.0f;
		squared_scalar_radiance_sum[index] = 0.0f;
		weighted_direction_sum_x[index]	   = 0.0f;
		weighted_direction_sum_y[index]	   = 0.0f;
		weighted_direction_sum_z[index]	   = 0.0f;
	}

	AtomicType<unsigned int>* valid_observation_count = nullptr;
	AtomicType<float>* scalar_radiance_sum			  = nullptr;
	AtomicType<float>* squared_scalar_radiance_sum	  = nullptr;
	AtomicType<float>* weighted_direction_sum_x		  = nullptr;
	AtomicType<float>* weighted_direction_sum_y		  = nullptr;
	AtomicType<float>* weighted_direction_sum_z		  = nullptr;
};

#endif
