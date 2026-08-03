/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_MLP_DATA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_MLP_DATA_HOST_H

#include "Device/includes/HashGridHash.h"
#include "Device/includes/Neural/MLPFullyFusedDevice.h"
#include "HostDeviceCommon/Xorshift.h"

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using MLPDataHostInternal = GenericSoA<DataContainer,
									   float,										   // Neurons biases
									   GenericAtomicType<float, DataContainer>,		   // Gradient biases
									   float,										   // Connection weights
									   GenericFP16Type<DataContainer>,				   // Connection weights FP16
									   GenericAtomicType<float, DataContainer>,		   // Gradient weights
									   GenericAtomicType<unsigned int, DataContainer>, // Last training sample count
									   float,										   // Adam weights means
									   float,										   // Adam weights variances
									   float,										   // Adam biases means
									   float,										   // Adam biases variances
									   GenericFP16Type<DataContainer>>;				   // Train activations

enum MLPDataHostBuffers
{
	MLP_NEURONS_BIASES,
	MLP_GRADIENT_BIASES,
	MLP_CONNECTION_WEIGHTS,
	MLP_CONNECTION_WEIGHTS_FP16,
	MLP_GRADIENT_WEIGHTS,
	MLP_LAST_TRAINING_SAMPLE_COUNT,
	MLP_ADAM_WEIGHTS_MEANS,
	MLP_ADAM_WEIGHTS_VARIANCES,
	MLP_ADAM_BIASES_MEANS,
	MLP_ADAM_BIASES_VARIANCES,
	MLP_TRAIN_ACTIVATIONS
};

template <template <typename> typename DataContainer, typename MLPType>
struct MLPDataHost
{
	void resize(unsigned int batch_size = 0)
	{
		m_mlp_data.template resize_one_buffer<MLPDataHostBuffers::MLP_CONNECTION_WEIGHTS>(MLPType::CONNECTIONS_COUNT);
		m_mlp_data.template resize_one_buffer<MLPDataHostBuffers::MLP_CONNECTION_WEIGHTS_FP16>(MLPType::CONNECTIONS_COUNT);
		m_mlp_data.template resize_one_buffer<MLPDataHostBuffers::MLP_GRADIENT_WEIGHTS>(MLPType::CONNECTIONS_COUNT);
		m_mlp_data.template resize_one_buffer<MLPDataHostBuffers::MLP_LAST_TRAINING_SAMPLE_COUNT>(1);
		m_mlp_data.template resize_one_buffer<MLPDataHostBuffers::MLP_ADAM_WEIGHTS_MEANS>(MLPType::CONNECTIONS_COUNT);
		m_mlp_data.template resize_one_buffer<MLPDataHostBuffers::MLP_ADAM_WEIGHTS_VARIANCES>(MLPType::CONNECTIONS_COUNT);

		if constexpr (MLPType::USE_BIASES)
		{
			m_mlp_data.template resize_one_buffer<MLPDataHostBuffers::MLP_NEURONS_BIASES>(MLPType::NEURON_COUNT);
			m_mlp_data.template resize_one_buffer<MLPDataHostBuffers::MLP_GRADIENT_BIASES>(MLPType::NEURON_COUNT);
			m_mlp_data.template resize_one_buffer<MLPDataHostBuffers::MLP_ADAM_BIASES_MEANS>(MLPType::NEURON_COUNT);
			m_mlp_data.template resize_one_buffer<MLPDataHostBuffers::MLP_ADAM_BIASES_VARIANCES>(MLPType::NEURON_COUNT);
		}

		if (batch_size > 0)
			m_mlp_data.template resize_one_buffer<MLPDataHostBuffers::MLP_TRAIN_ACTIVATIONS>(batch_size * MLPType::NEURON_COUNT);
	}

	/**
	 * Initializes all weights using Xavier's uniform distribution and all biases to 0
	 */
	void initialize()
	{
		if (maximum_size() == 0)
			return;

		m_mlp_data.template memset_buffer<MLPDataHostBuffers::MLP_LAST_TRAINING_SAMPLE_COUNT>(0);
		m_mlp_data.template memset_buffer<MLPDataHostBuffers::MLP_GRADIENT_WEIGHTS>(0.0f);

		// Initializing weights using Xavier's uniform distribution
		Xorshift32Generator rng(pcg_hash(static_cast<unsigned int>(MLPType::INPUT_SIZE) +
										 pcg_hash(static_cast<unsigned int>(MLPType::OUTPUT_SIZE) +
												  pcg_hash(static_cast<unsigned int>(MLPType::HIDDEN_LAYER_COUNT * MLPType::HIDDEN_LAYER_SIZE)))));

		std::vector<float> weights = m_mlp_data.template download_buffer<MLPDataHostBuffers::MLP_CONNECTION_WEIGHTS>();

		for (unsigned int layer_index = 1; layer_index < MLPType::LAYER_COUNT; layer_index++)
		{
			unsigned int neurons_count_previous_layer = MLPType::get_layer_neuron_count(layer_index - 1);
			unsigned int neurons_count_current_layer  = MLPType::get_layer_neuron_count(layer_index);

			float random_range = hippt::sqrt(6.0f / (neurons_count_previous_layer + neurons_count_current_layer));

			for (unsigned int neuron_index_current_layer = 0; neuron_index_current_layer < neurons_count_current_layer; neuron_index_current_layer++)
			{
				for (unsigned int neuron_index_previous_layer = 0; neuron_index_previous_layer < neurons_count_previous_layer; neuron_index_previous_layer++)
				{
					unsigned int connection_data_index =
						MLPType::get_connection_data_index(layer_index, neuron_index_previous_layer, neuron_index_current_layer);

					weights[connection_data_index] = rng() * 2.0f * random_range - random_range;
				}
			}
		}
		m_mlp_data.template upload_to_buffer<MLPDataHostBuffers::MLP_CONNECTION_WEIGHTS>(weights);

		// Adam state initialized to 0
		m_mlp_data.template memset_buffer<MLPDataHostBuffers::MLP_ADAM_WEIGHTS_MEANS>(0.0f);
		m_mlp_data.template memset_buffer<MLPDataHostBuffers::MLP_ADAM_WEIGHTS_VARIANCES>(0.0f);

		if constexpr (MLPType::USE_BIASES)
		{
			m_mlp_data.template memset_buffer<MLPDataHostBuffers::MLP_GRADIENT_BIASES>(0.0f);
			// All biases are initialized to 0
			m_mlp_data.template memset_buffer<MLPDataHostBuffers::MLP_NEURONS_BIASES>(0.0f);
			m_mlp_data.template memset_buffer<MLPDataHostBuffers::MLP_ADAM_BIASES_MEANS>(0.0f);
			m_mlp_data.template memset_buffer<MLPDataHostBuffers::MLP_ADAM_BIASES_VARIANCES>(0.0f);
		}
	}

	bool free()
	{
		if (maximum_size() > 0)
		{
			m_mlp_data.free();

			return true;
		}

		return false;
	}

	std::size_t get_byte_size() const
	{
		return m_mlp_data.get_byte_size();
	}

	std::size_t maximum_size() const
	{
		return m_mlp_data.maximum_size();
	}

	MLPType to_device()
	{
		MLPType mlp_device;

		if (maximum_size() == 0)
			return mlp_device;

		if constexpr (MLPType::USE_BIASES)
		{
			mlp_device.neurons_biases		 = m_mlp_data.template get_buffer_data_ptr<MLPDataHostBuffers::MLP_NEURONS_BIASES>();
			mlp_device.gradient_biases		 = m_mlp_data.template get_buffer_data_atomic_ptr<MLPDataHostBuffers::MLP_GRADIENT_BIASES>();
			mlp_device.adam_biases_means	 = m_mlp_data.template get_buffer_data_ptr<MLPDataHostBuffers::MLP_ADAM_BIASES_MEANS>();
			mlp_device.adam_biases_variances = m_mlp_data.template get_buffer_data_ptr<MLPDataHostBuffers::MLP_ADAM_BIASES_VARIANCES>();
		}

		mlp_device.connection_weights = m_mlp_data.template get_buffer_data_ptr<MLPDataHostBuffers::MLP_CONNECTION_WEIGHTS>();
		mlp_device.connection_weights_fp16 =
			reinterpret_cast<fp16*>(m_mlp_data.template get_buffer_data_ptr<MLPDataHostBuffers::MLP_CONNECTION_WEIGHTS_FP16>());
		mlp_device.gradient_weights = m_mlp_data.template get_buffer_data_atomic_ptr<MLPDataHostBuffers::MLP_GRADIENT_WEIGHTS>();

		mlp_device.last_training_sample_count = m_mlp_data.template get_buffer_data_atomic_ptr<MLPDataHostBuffers::MLP_LAST_TRAINING_SAMPLE_COUNT>();

		mlp_device.adam_weights_means	  = m_mlp_data.template get_buffer_data_ptr<MLPDataHostBuffers::MLP_ADAM_WEIGHTS_MEANS>();
		mlp_device.adam_weights_variances = m_mlp_data.template get_buffer_data_ptr<MLPDataHostBuffers::MLP_ADAM_WEIGHTS_VARIANCES>();

		return mlp_device;
	}

	MLPDataHostInternal<DataContainer> m_mlp_data;
};

#endif
