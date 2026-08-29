/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_NISML_POSITION_LEARNABLE_DENSE_GRID_DATA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_NISML_POSITION_LEARNABLE_DENSE_GRID_DATA_HOST_H

#include "Device/includes/Neural/NISML/NISMLPositionLearnableDenseGrid.h"
#include "HostDeviceCommon/Xorshift.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

#include <type_traits>
#include <vector>

template <template <typename> typename DataContainer>
using NISMLPositionLearnableDenseGridDataHostInternal =
	GenericSoA<DataContainer, float, GenericFP16Type<DataContainer>, GenericAtomicType<float, DataContainer>, float, float>;

enum NISMLPositionLearnableDenseGridDataHostBuffers
{
	NISML_POSITION_LEARNABLE_DENSE_GRID_FEATURES,
	NISML_POSITION_LEARNABLE_DENSE_GRID_FEATURES_FP16,
	NISML_POSITION_LEARNABLE_DENSE_GRID_GRADIENT_FEATURES,
	NISML_POSITION_LEARNABLE_DENSE_GRID_ADAM_FEATURE_MEANS,
	NISML_POSITION_LEARNABLE_DENSE_GRID_ADAM_FEATURE_VARIANCES
};

template <template <typename> typename DataContainer>
struct NISMLPositionLearnableDenseGridDataHost
{
	void resize()
	{
		m_grid_data.resize(NISML_POSITION_LEARNABLE_DENSE_GRID_TOTAL_PARAMETER_COUNT);
	}

	void initialize()
	{
		if (maximum_size() == 0)
			return;

		Xorshift32Generator random_number_generator(0xdeadbeef);

		std::vector<float> features(NISML_POSITION_LEARNABLE_DENSE_GRID_TOTAL_PARAMETER_COUNT);
		for (float& feature : features)
			feature = random_number_generator() * 2.0e-4f - 1.0e-4f;

		m_grid_data.template upload_to_buffer<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_FEATURES>(features);

		std::vector<GenericFP16Type<DataContainer>> features_fp16(features.size());
		for (unsigned int feature_index = 0; feature_index < features.size(); feature_index++)
		{
			if constexpr (std::is_same_v<GenericFP16Type<DataContainer>, float>)
				features_fp16[feature_index] = features[feature_index];
			else
				features_fp16[feature_index] = hippt::fp32_to_fp16_bits(features[feature_index]);
		}

		m_grid_data.template upload_to_buffer<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_FEATURES_FP16>(features_fp16);
		m_grid_data.template memset_buffer<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_GRADIENT_FEATURES>(0.0f);
		m_grid_data.template memset_buffer<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_ADAM_FEATURE_MEANS>(0.0f);
		m_grid_data.template memset_buffer<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_ADAM_FEATURE_VARIANCES>(0.0f);
	}

	bool free()
	{
		if (maximum_size() == 0)
			return false;

		m_grid_data.free();
		return true;
	}

	std::size_t get_byte_size() const
	{
		return m_grid_data.get_byte_size();
	}

	std::size_t maximum_size() const
	{
		return m_grid_data.maximum_size();
	}

	NISMLPositionLearnableDenseGridDevice to_device(float adam_learning_rate)
	{
		NISMLPositionLearnableDenseGridDevice device;
		if (maximum_size() == 0)
			return device;

		device.features =
			m_grid_data.template get_buffer_data_ptr<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_FEATURES>();
		device.features_fp16 = reinterpret_cast<fp16*>(
			m_grid_data.template get_buffer_data_ptr<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_FEATURES_FP16>());
		device.gradient_features =
			m_grid_data
				.template get_buffer_data_atomic_ptr<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_GRADIENT_FEATURES>();
		device.adam_feature_means =
			m_grid_data.template get_buffer_data_ptr<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_ADAM_FEATURE_MEANS>();
		device.adam_feature_variances =
			m_grid_data
				.template get_buffer_data_ptr<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_ADAM_FEATURE_VARIANCES>();
		device.adam_learning_rate = adam_learning_rate;

		for (unsigned int level = 0; level < NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT; level++)
		{
			device.level_resolutions[level] = NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_RESOLUTIONS[level];
			device.level_offsets[level]		= NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_OFFSETS[level];
		}

		return device;
	}

	NISMLPositionLearnableDenseGridDataHostInternal<DataContainer> m_grid_data;
};

#endif // #ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_NISML_POSITION_LEARNABLE_DENSE_GRID_DATA_HOST_H
