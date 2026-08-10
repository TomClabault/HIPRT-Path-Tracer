/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_NIS_ML_POSITION_GRID_DATA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_NIS_ML_POSITION_GRID_DATA_HOST_H

#include "Device/includes/Neural/NISML/NISMLPositionGrid.h"
#include "HostDeviceCommon/Xorshift.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

#include <cmath>
#include <type_traits>
#include <vector>

template <template <typename> typename DataContainer>
using NISPositionGridDataHostInternal = GenericSoA<DataContainer, float, GenericFP16Type<DataContainer>, GenericAtomicType<float, DataContainer>, float, float>;

enum NISPositionGridDataHostBuffers
{
	NIS_POSITION_GRID_FEATURES,
	NIS_POSITION_GRID_FEATURES_FP16,
	NIS_POSITION_GRID_GRADIENT_FEATURES,
	NIS_POSITION_GRID_ADAM_FEATURE_MEANS,
	NIS_POSITION_GRID_ADAM_FEATURE_VARIANCES
};

template <template <typename> typename DataContainer>
struct NISPositionGridDataHost
{
	void resize()
	{
		m_grid_data.resize(NIS_POSITION_GRID_TOTAL_PARAMETER_COUNT);
	}

	void initialize()
	{
		if (maximum_size() == 0)
			return;

		std::vector<float> features(NIS_POSITION_GRID_TOTAL_PARAMETER_COUNT);
		Xorshift32Generator random_number_generator(0x4E495347u);
		for (float& feature : features)
			feature = random_number_generator() * 2.0e-4f - 1.0e-4f;

		m_grid_data.template upload_to_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_FEATURES>(features);

		std::vector<GenericFP16Type<DataContainer>> features_fp16(features.size());
		for (unsigned int feature_index = 0; feature_index < features.size(); feature_index++)
		{
			if constexpr (std::is_same_v<GenericFP16Type<DataContainer>, float>)
				features_fp16[feature_index] = features[feature_index];
			else
				features_fp16[feature_index] = hippt::fp32_to_fp16_bits(features[feature_index]);
		}

		m_grid_data.template upload_to_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_FEATURES_FP16>(features_fp16);
		m_grid_data.template memset_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_GRADIENT_FEATURES>(0.0f);
		m_grid_data.template memset_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_ADAM_FEATURE_MEANS>(0.0f);
		m_grid_data.template memset_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_ADAM_FEATURE_VARIANCES>(0.0f);
	}

	bool free()
	{
		if (maximum_size() == 0)
			return false;

		m_grid_data.free();
		return true;
	}

	void apply_adam(unsigned int training_sample_count, unsigned int adam_step, float adam_learning_rate)
	{
		if (training_sample_count == 0u || maximum_size() == 0)
			return;

		if constexpr (std::is_same_v<DataContainer<float>, std::vector<float>>)
		{
			float time_step		   = static_cast<float>(adam_step + 1u);
			float beta1_correction = 1.0f - std::pow(NIS_ADAM_BETA1, time_step);
			float beta2_correction = 1.0f - std::pow(NIS_ADAM_BETA2, time_step);

			std::vector<float>& features			  = m_grid_data.template get_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_FEATURES>();
			std::vector<float>& features_fp16		  = m_grid_data.template get_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_FEATURES_FP16>();
			std::vector<AtomicType<float>>& gradients = m_grid_data.template get_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_GRADIENT_FEATURES>();
			std::vector<float>& means				  = m_grid_data.template get_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_ADAM_FEATURE_MEANS>();
			std::vector<float>& variances = m_grid_data.template get_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_ADAM_FEATURE_VARIANCES>();

			for (unsigned int feature_index = 0; feature_index < NIS_POSITION_GRID_TOTAL_PARAMETER_COUNT; feature_index++)
			{
				float gradient = gradients[feature_index].load() / static_cast<float>(training_sample_count);
				gradients[feature_index].store(0.0f);
				float mean				 = NIS_ADAM_BETA1 * means[feature_index] + (1.0f - NIS_ADAM_BETA1) * gradient;
				float variance			 = NIS_ADAM_BETA2 * variances[feature_index] + (1.0f - NIS_ADAM_BETA2) * gradient * gradient;
				means[feature_index]	 = mean;
				variances[feature_index] = variance;

				float corrected_mean	 = mean / beta1_correction;
				float corrected_variance = variance / beta2_correction;
				features[feature_index] -= adam_learning_rate * corrected_mean / (std::sqrt(corrected_variance) + NIS_ADAM_EPSILON);
				features_fp16[feature_index] = features[feature_index];
			}
		}
	}

	std::size_t get_byte_size() const
	{
		return m_grid_data.get_byte_size();
	}

	std::size_t maximum_size() const
	{
		return m_grid_data.maximum_size();
	}

	NISPositionGridDevice to_device(float adam_learning_rate)
	{
		NISPositionGridDevice device;
		if (maximum_size() == 0)
			return device;

		device.features = m_grid_data.template get_buffer_data_ptr<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_FEATURES>();
		device.features_fp16 =
			reinterpret_cast<fp16*>(m_grid_data.template get_buffer_data_ptr<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_FEATURES_FP16>());
		device.gradient_features	  = m_grid_data.template get_buffer_data_atomic_ptr<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_GRADIENT_FEATURES>();
		device.adam_feature_means	  = m_grid_data.template get_buffer_data_ptr<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_ADAM_FEATURE_MEANS>();
		device.adam_feature_variances = m_grid_data.template get_buffer_data_ptr<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_ADAM_FEATURE_VARIANCES>();
		device.adam_learning_rate	  = adam_learning_rate;

		for (unsigned int level = 0; level < NIS_POSITION_GRID_LEVEL_COUNT; level++)
		{
			device.level_resolutions[level] = NIS_POSITION_GRID_LEVEL_RESOLUTIONS[level];
			device.level_offsets[level]		= NIS_POSITION_GRID_LEVEL_OFFSETS[level];
		}

		return device;
	}

	NISPositionGridDataHostInternal<DataContainer> m_grid_data;
};

#endif
