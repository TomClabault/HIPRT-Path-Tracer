/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_SIGNATURE_SOA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_SIGNATURE_SOA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeIlluminationSignatureSoADevice.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeIlluminationSignatureSoAHost
{
	using Data = GenericSoA<DataContainer,
							GenericAtomicType<unsigned int, DataContainer>,
							GenericAtomicType<float, DataContainer>,
							GenericAtomicType<float, DataContainer>,
							GenericAtomicType<float, DataContainer>,
							GenericAtomicType<float, DataContainer>,
							GenericAtomicType<float, DataContainer>,
							GenericAtomicType<float, DataContainer>>;

	enum Buffers
	{
		VALID_OBSERVATION_COUNT,
		SCALAR_RADIANCE_SUM,
		SQUARED_SCALAR_RADIANCE_SUM,
		WEIGHTED_DIRECTION_SUM_X,
		WEIGHTED_DIRECTION_SUM_Y,
		WEIGHTED_DIRECTION_SUM_Z,
	};

	void resize(unsigned int new_size)
	{
		data.resize(new_size);
	}

	void free()
	{
		data.free();
	}

	std::size_t get_byte_size() const
	{
		return data.get_byte_size();
	}

	std::size_t maximum_size() const
	{
		return data.maximum_size();
	}

	IlluminationAwareKDTreeIlluminationSignatureSoADevice to_device()
	{
		IlluminationAwareKDTreeIlluminationSignatureSoADevice device;

		device.valid_observation_count	   = data.template get_buffer_data_atomic_ptr<VALID_OBSERVATION_COUNT>();
		device.scalar_radiance_sum		   = data.template get_buffer_data_atomic_ptr<SCALAR_RADIANCE_SUM>();
		device.squared_scalar_radiance_sum = data.template get_buffer_data_atomic_ptr<SQUARED_SCALAR_RADIANCE_SUM>();
		device.weighted_direction_sum_x	   = data.template get_buffer_data_atomic_ptr<WEIGHTED_DIRECTION_SUM_X>();
		device.weighted_direction_sum_y	   = data.template get_buffer_data_atomic_ptr<WEIGHTED_DIRECTION_SUM_Y>();
		device.weighted_direction_sum_z	   = data.template get_buffer_data_atomic_ptr<WEIGHTED_DIRECTION_SUM_Z>();

		return device;
	}

	Data data;
};

#endif
