/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_MLP_FULLY_FUSED_DEVICE_CPU_H
#define DEVICE_INCLUDES_NEURAL_MLP_FULLY_FUSED_DEVICE_CPU_H

#include "Device/includes/Neural/MLPFullyFusedDeviceCommon.h"

template <unsigned int InputSizeEncoded_,
		  unsigned int HiddenLayerCount_,
		  unsigned int HiddenLayerSize_,
		  unsigned int OutputSize_,
		  unsigned int BlockSize_,
		  bool UseBiases_							= true,
		  MLPActivationFunction ActivationFunction_ = MLPActivationFunction::LEAKY_RELU,
		  bool UseOutputActivation_					= false>
struct MLPFullyFusedDeviceCPU : public MLPFullyFusedDeviceCommon<InputSizeEncoded_,
																 HiddenLayerCount_,
																 HiddenLayerSize_,
																 OutputSize_,
																 BlockSize_,
																 UseBiases_,
																 ActivationFunction_,
																 UseOutputActivation_>
{
	using Common = MLPFullyFusedDeviceCommon<InputSizeEncoded_,
											 HiddenLayerCount_,
											 HiddenLayerSize_,
											 OutputSize_,
											 BlockSize_,
											 UseBiases_,
											 ActivationFunction_,
											 UseOutputActivation_>;

	using InputLayer = typename Common::InputLayer;

	using Common::ACTIVATION_FUNCTION;
	using Common::ACTIVATION_WIDTH;
	using Common::BLOCK_SIZE;
	using Common::CONNECTIONS_COUNT;
	using Common::ERROR_WIDTH;
	using Common::HIDDEN_LAYER_COUNT;
	using Common::HIDDEN_LAYER_SIZE;
	using Common::INPUT_SIZE_ENCODED;
	using Common::INPUT_SIZE_PADDED_WMMA;
	using Common::LAYER_COUNT;
	using Common::NEURON_COUNT;
	using Common::OUTPUT_SIZE;
	using Common::OUTPUT_SIZE_PADDED_WMMA;
	using Common::USE_BIASES;
	using Common::USE_OUTPUT_ACTIVATION;

	using Common::connection_weights;
	using Common::get_connection_data_index;
	using Common::get_layer_neuron_count;
	using Common::get_neuron_data_index;
	using Common::gradient_biases;
	using Common::gradient_weights;
	using Common::last_training_sample_count;
	using Common::load_input_ref;
	using Common::neurons_biases;

	HIPRT_DEVICE void forward_single_thread(const InputLayer& input, float* neurons_activations) const
	{
		load_input_ref(input.input, neurons_activations);

		for (unsigned int layer = 1; layer < LAYER_COUNT; ++layer)
		{
			unsigned int previous_count	 = get_layer_neuron_count(layer - 1);
			unsigned int current_count	 = get_layer_neuron_count(layer);
			unsigned int previous_offset = get_neuron_data_index(layer - 1, 0);
			unsigned int current_offset	 = get_neuron_data_index(layer, 0);

			for (unsigned int neuron = 0; neuron < current_count; ++neuron)
			{
				float value = 0.0f;

				if constexpr (USE_BIASES)
					value += neurons_biases[current_offset + neuron];

				for (unsigned int previous_neuron = 0; previous_neuron < previous_count; ++previous_neuron)
				{
					unsigned int connection = get_connection_data_index(layer, previous_neuron, neuron);

					value += connection_weights[connection] * neurons_activations[previous_offset + previous_neuron];
				}

				if (layer != LAYER_COUNT - 1)
					value = this->activation_function(value);
				else if constexpr (USE_OUTPUT_ACTIVATION)
					value = this->activation_function(value);

				neurons_activations[current_offset + neuron] = value;
			}
		}
	}

	HIPRT_DEVICE void backpropagation_from_output_gradient(float* neurons_activations,
														   const float* output_gradient,
														   float* input_gradients			 = nullptr,
														   unsigned int input_gradient_count = 0) const
	{
		float neurons_errors[NEURON_COUNT];

		unsigned int output_layer_index = LAYER_COUNT - 1;
		for (unsigned int neuron_index_in_output_layer = 0; neuron_index_in_output_layer < get_layer_neuron_count(output_layer_index);
			 neuron_index_in_output_layer++)
		{
			unsigned int current_neuron_data_index = get_neuron_data_index(output_layer_index, neuron_index_in_output_layer);
			float output_error					   = output_gradient[neuron_index_in_output_layer];

			neurons_errors[current_neuron_data_index] = output_error;
			if constexpr (USE_BIASES)
				hippt::atomic_fetch_add(&gradient_biases[current_neuron_data_index], output_error);

			for (unsigned int previous_neuron_index = 0; previous_neuron_index < get_layer_neuron_count(output_layer_index - 1); previous_neuron_index++)
			{
				unsigned int previous_neuron_data_index = get_neuron_data_index(output_layer_index - 1, previous_neuron_index);
				unsigned int connection_data_index		= get_connection_data_index(output_layer_index, previous_neuron_index, neuron_index_in_output_layer);

				float previous_activation = neurons_activations[previous_neuron_data_index];
				hippt::atomic_fetch_add(&gradient_weights[connection_data_index], output_error * previous_activation);
			}
		}

		for (unsigned int layer_index = LAYER_COUNT - 2; layer_index > 0; layer_index--)
		{
			for (unsigned int neuron_index = 0; neuron_index < get_layer_neuron_count(layer_index); neuron_index++)
			{
				unsigned int current_neuron_data_index = get_neuron_data_index(layer_index, neuron_index);
				float d_cost_d_z					   = 0.0f;

				for (unsigned int next_neuron_index = 0; next_neuron_index < get_layer_neuron_count(layer_index + 1); next_neuron_index++)
				{
					unsigned int next_neuron_data_index = get_neuron_data_index(layer_index + 1, next_neuron_index);
					unsigned int connection_data_index	= get_connection_data_index(layer_index + 1, neuron_index, next_neuron_index);

					d_cost_d_z += neurons_errors[next_neuron_data_index] * connection_weights[connection_data_index];
				}

				float current_activation = neurons_activations[current_neuron_data_index];
				d_cost_d_z *= this->activation_function_derivative(current_activation);

				neurons_errors[current_neuron_data_index] = d_cost_d_z;
				if constexpr (USE_BIASES)
					hippt::atomic_fetch_add(&gradient_biases[current_neuron_data_index], d_cost_d_z);

				for (unsigned int previous_neuron_index = 0; previous_neuron_index < get_layer_neuron_count(layer_index - 1); previous_neuron_index++)
				{
					unsigned int previous_neuron_data_index = get_neuron_data_index(layer_index - 1, previous_neuron_index);
					unsigned int connection_data_index		= get_connection_data_index(layer_index, previous_neuron_index, neuron_index);

					float previous_activation = neurons_activations[previous_neuron_data_index];
					hippt::atomic_fetch_add(&gradient_weights[connection_data_index], d_cost_d_z * previous_activation);
				}
			}
		}

		if (input_gradients != nullptr)
		{
			unsigned int first_hidden_layer_offset	   = get_neuron_data_index(1, 0);
			unsigned int first_layer_connection_offset = get_connection_data_index(1, 0, 0);
			for (unsigned int input_index = 0; input_index < input_gradient_count; input_index++)
			{
				float input_gradient = 0.0f;
				for (unsigned int hidden_index = 0; hidden_index < HIDDEN_LAYER_SIZE; hidden_index++)
				{
					unsigned int connection_index = first_layer_connection_offset + hidden_index * INPUT_SIZE_PADDED_WMMA + input_index;
					input_gradient += neurons_errors[first_hidden_layer_offset + hidden_index] * connection_weights[connection_index];
				}

				input_gradients[input_index] = input_gradient;
			}
		}

		hippt::atomic_fetch_add(last_training_sample_count, 1u);
	}
};

#endif // #ifndef DEVICE_INCLUDES_NEURAL_MLP_FULLY_FUSED_DEVICE_CPU_H
