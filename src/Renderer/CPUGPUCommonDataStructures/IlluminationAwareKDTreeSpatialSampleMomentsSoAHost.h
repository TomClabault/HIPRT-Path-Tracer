/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_SPATIAL_SAMPLE_MOMENTS_SOA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_SPATIAL_SAMPLE_MOMENTS_SOA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeSpatialSampleMomentsSoADevice.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeSpatialSampleMomentsSoAHost
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
		POSITIVE_RADIANCE_SAMPLE_COUNT,
		POSITION_SUM_X,
		POSITION_SUM_Y,
		POSITION_SUM_Z,
		POSITION_SQUARED_SUM_X,
		POSITION_SQUARED_SUM_Y,
		POSITION_SQUARED_SUM_Z,
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

	IlluminationAwareKDTreeSpatialSampleMomentsSoADevice to_device()
	{
		IlluminationAwareKDTreeSpatialSampleMomentsSoADevice device;

		device.positive_radiance_sample_count = data.template get_buffer_data_atomic_ptr<POSITIVE_RADIANCE_SAMPLE_COUNT>();
		device.position_sum_x				  = data.template get_buffer_data_atomic_ptr<POSITION_SUM_X>();
		device.position_sum_y				  = data.template get_buffer_data_atomic_ptr<POSITION_SUM_Y>();
		device.position_sum_z				  = data.template get_buffer_data_atomic_ptr<POSITION_SUM_Z>();
		device.position_squared_sum_x		  = data.template get_buffer_data_atomic_ptr<POSITION_SQUARED_SUM_X>();
		device.position_squared_sum_y		  = data.template get_buffer_data_atomic_ptr<POSITION_SQUARED_SUM_Y>();
		device.position_squared_sum_z		  = data.template get_buffer_data_atomic_ptr<POSITION_SQUARED_SUM_Z>();

		return device;
	}

	Data data;
};

#endif // #ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_SPATIAL_SAMPLE_MOMENTS_SOA_HOST_H
