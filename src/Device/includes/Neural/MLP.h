/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_MLP_H
#define DEVICE_INCLUDES_NEURAL_MLP_H

#define LAYER_COUNT		  3
#define NEURONS_PER_LAYER 128

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Maths/Math.h"

struct MLP
{
	HIPRT_DEVICE unsigned int get_neuron_data_index(unsigned int layer_index, unsigned int neuron_index) const
	{
		return neuron_data_base_offset[layer_index] + neuron_index;
	}

	HIPRT_DEVICE unsigned int get_connection_data_index(unsigned int layer_index, unsigned int neuron_from_index, unsigned int neuron_to_index) const
	{
		return connection_data_base_offset[layer_index] + (neuron_to_index * neurons_per_layer[layer_index - 1]) + neuron_from_index;
	}

	HIPRT_DEVICE ColorRGB32F evaluate(float* input) const
	{
		neurons_activations[0] = input[0];
		neurons_activations[1] = input[1];

		for (unsigned int layer_index = 1; layer_index < LAYER_COUNT; layer_index++)
		{
			unsigned int neurons_current_layer	= neurons_per_layer[layer_index];
			unsigned int neurons_previous_layer = neurons_per_layer[layer_index - 1];

			for (unsigned int neuron_index = 0; neuron_index < neurons_current_layer; neuron_index++)
			{
				unsigned int current_neuron_data_index = get_neuron_data_index(layer_index, neuron_index);

				float activation = 0.0f;

				for (unsigned int previous_neuron_index = 0; previous_neuron_index < neurons_previous_layer; previous_neuron_index++)
				{
					unsigned int previous_neuron_data_index = get_neuron_data_index(layer_index - 1, previous_neuron_index);
					unsigned int connection_data_index		= get_connection_data_index(layer_index, previous_neuron_index, neuron_index);

					activation += connection_weights[connection_data_index] * neurons_activations[previous_neuron_data_index];
				}

				activation += neurons_biases[current_neuron_data_index];
				activation = activation_function(activation);

				neurons_activations[current_neuron_data_index] = activation;
			}
		}

		unsigned int output_layer_index		  = LAYER_COUNT - 1;
		unsigned int output_neuron_data_index = get_neuron_data_index(output_layer_index, 0);

		return ColorRGB32F(neurons_activations[output_neuron_data_index + 0], neurons_activations[output_neuron_data_index + 1],
						   neurons_activations[output_neuron_data_index + 2]);
	}

	HIPRT_DEVICE void train(float* input, float* target_output) const
	{
		// Output layer bias and weights gradients
		unsigned int output_layer_index = LAYER_COUNT - 1;
		for (unsigned int neuron_index_in_output_layer = 0; neuron_index_in_output_layer < neurons_per_layer[output_layer_index];
			 neuron_index_in_output_layer++)
		{
			unsigned int current_neuron_data_index = get_neuron_data_index(output_layer_index, neuron_index_in_output_layer);
			float output_activation				   = neurons_activations[current_neuron_data_index];

			// Gradient of bias
			float d_cost_d_Oi = cost_function_derivative(output_activation, target_output[neuron_index_in_output_layer]);
			float d_Oi_d_Zi	  = activation_function_derivative(output_activation);
			float d_cost_d_Zi = d_cost_d_Oi * d_Oi_d_Zi;

			neurons_errors[current_neuron_data_index] = d_cost_d_Zi;
			hippt::atomic_fetch_add(&gradient_biases[current_neuron_data_index], d_cost_d_Zi);

			// Gradient of weights
			for (unsigned int previous_neuron_index = 0; previous_neuron_index < neurons_per_layer[output_layer_index - 1]; previous_neuron_index++)
			{
				unsigned int previous_neuron_data_index = get_neuron_data_index(output_layer_index - 1, previous_neuron_index);
				unsigned int connection_data_index		= get_connection_data_index(output_layer_index, previous_neuron_index, neuron_index_in_output_layer);

				float previous_activation = neurons_activations[previous_neuron_data_index];
				float weight			  = connection_weights[connection_data_index];

				float d_cost_d_Wij = d_cost_d_Zi * previous_activation;

				hippt::atomic_fetch_add(&gradient_weights[connection_data_index], d_cost_d_Wij);
			}
		}

		// Hidden layers bias and weights gradients
		for (unsigned int layer_index = LAYER_COUNT - 2; layer_index > 0; layer_index--)
		{
			for (unsigned int neuron_index = 0; neuron_index < neurons_per_layer[layer_index]; neuron_index++)
			{
				unsigned int current_neuron_data_index = get_neuron_data_index(layer_index, neuron_index);

				float d_cost_d_Zi = 0.0f;

				for (unsigned int next_neuron_index = 0; next_neuron_index < neurons_per_layer[layer_index + 1]; next_neuron_index++)
				{
					unsigned int next_neuron_data_index = get_neuron_data_index(layer_index + 1, next_neuron_index);
					unsigned int connection_data_index	= get_connection_data_index(layer_index + 1, neuron_index, next_neuron_index);

					float weight	  = connection_weights[connection_data_index];
					float d_cost_d_Zj = neurons_errors[next_neuron_data_index];

					d_cost_d_Zi += d_cost_d_Zj * weight;
				}

				float current_activation = neurons_activations[current_neuron_data_index];
				float d_Zi_d_Oi			 = activation_function_derivative(current_activation);
				d_cost_d_Zi *= d_Zi_d_Oi;

				neurons_errors[current_neuron_data_index] = d_cost_d_Zi;
				hippt::atomic_fetch_add(&gradient_biases[current_neuron_data_index], d_cost_d_Zi);

				for (unsigned int previous_neuron_index = 0; previous_neuron_index < neurons_per_layer[layer_index - 1]; previous_neuron_index++)
				{
					unsigned int previous_neuron_data_index = get_neuron_data_index(layer_index - 1, previous_neuron_index);
					unsigned int connection_data_index		= get_connection_data_index(layer_index, previous_neuron_index, neuron_index);

					float previous_activation = neurons_activations[previous_neuron_data_index];
					float d_cost_d_Wij		  = d_cost_d_Zi * previous_activation;

					hippt::atomic_fetch_add(&gradient_weights[connection_data_index], d_cost_d_Wij);
				}
			}
		}
	}

	HIPRT_DEVICE constexpr float leaky_ReLU(float x) const
	{
		return x > 0.0f ? x : 0.01f * x;
	}

	HIPRT_DEVICE constexpr float leaky_ReLU_derivative(float x) const
	{
		return x > 0.0f ? 1.0f : 0.01f;
	}

	HIPRT_DEVICE constexpr float activation_function(float x) const
	{
		return leaky_ReLU(x);
	}

	HIPRT_DEVICE constexpr float activation_function_derivative(float x) const
	{
		return leaky_ReLU_derivative(x);
	}

	HIPRT_DEVICE float cost_function(float* output, ColorRGB32F target_output) const
	{
		float cost = 0.0f;

		cost += hippt::square(output[0] - target_output.r);
		cost += hippt::square(output[1] - target_output.g);
		cost += hippt::square(output[2] - target_output.b);

		return cost / 3.0f;
	}

	HIPRT_DEVICE float cost_function_derivative(float output, float target_output) const
	{
		float d_cost = output - target_output;

		return 2.0f * d_cost / 3.0f;
	}

	unsigned int* neuron_data_base_offset	  = nullptr;
	unsigned int* connection_data_base_offset = nullptr;
	// TODO unsigned char should be enough?
	unsigned int* neurons_per_layer = nullptr;

	float* neurons_biases			   = nullptr;
	float* neurons_activations		   = nullptr;
	float* neurons_errors			   = nullptr;
	AtomicType<float>* gradient_biases = nullptr;

	float* connection_weights			= nullptr;
	AtomicType<float>* gradient_weights = nullptr;

	float learning_rate = 0.01f;
};

#endif
