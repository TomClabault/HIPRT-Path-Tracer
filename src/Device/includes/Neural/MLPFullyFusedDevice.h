/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_MLP_FULLY_FUSED_DEVICE_H
#define DEVICE_INCLUDES_NEURAL_MLP_FULLY_FUSED_DEVICE_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Maths/Math.h"

#define MLP_FULLY_FUSED_PREDICT_THREAD_BLOCK_SIZE 64
#define MLP_FULLY_FUSED_SAMPLES_PER_THREAD_BLOCK  MLP_FULLY_FUSED_PREDICT_THREAD_BLOCK_SIZE

#define FREQUENCY_ENCODING_NUM_FREQUENCIES 8

#define MLP_INPUT_SIZE_RAW	   2
#define MLP_INPUT_SIZE		   (MLP_INPUT_SIZE_RAW * 2 * FREQUENCY_ENCODING_NUM_FREQUENCIES)
#define MLP_OUTPUT_SIZE		   3
#define MLP_HIDDEN_LAYER_COUNT 3
#define MLP_HIDDEN_LAYER_SIZE  64

#define MLP_LAYER_COUNT	 (MLP_HIDDEN_LAYER_COUNT + 2)
#define MLP_NEURON_COUNT (MLP_INPUT_SIZE + MLP_OUTPUT_SIZE + (MLP_HIDDEN_LAYER_COUNT * MLP_HIDDEN_LAYER_SIZE))
#define MLP_CONNECTIONS_COUNT                                                                                                                                  \
	((MLP_INPUT_SIZE * MLP_HIDDEN_LAYER_SIZE) + ((MLP_HIDDEN_LAYER_COUNT - 1) * MLP_HIDDEN_LAYER_SIZE * MLP_HIDDEN_LAYER_SIZE) +                               \
	 (MLP_OUTPUT_SIZE * MLP_HIDDEN_LAYER_SIZE))

#define PING_PONG_ACTIVATIONS_OFFSET(layer) (((layer) & 1) * MLP_HIDDEN_LAYER_SIZE)

struct MLPFullyFusedDevice
{
	struct OutputLayer
	{
		float output[MLP_OUTPUT_SIZE];
	};

	struct InputLayer
	{
		float input[MLP_INPUT_SIZE_RAW];
	};

	HIPRT_DEVICE static constexpr unsigned int get_layer_neuron_count(unsigned int layer)
	{
		return layer == 0 ? MLP_INPUT_SIZE : (layer == MLP_HIDDEN_LAYER_COUNT + 1 ? MLP_OUTPUT_SIZE : MLP_HIDDEN_LAYER_SIZE);
	}

	HIPRT_DEVICE static constexpr unsigned int get_neuron_data_index(unsigned int layer, unsigned int neuron)
	{
		unsigned int offset = 0;
		for (unsigned int l = 0; l < layer; l++)
			offset += get_layer_neuron_count(l);
		return offset + neuron;
	}

	HIPRT_DEVICE static constexpr unsigned int get_connection_data_index(unsigned int layer, unsigned int neuron_from, unsigned int neuron_to)
	{
		unsigned int offset = 0;
		for (unsigned int l = 1; l < layer; l++)
			offset += get_layer_neuron_count(l) * get_layer_neuron_count(l - 1);
		return offset + (neuron_to * get_layer_neuron_count(layer - 1)) + neuron_from;
	}

	HIPRT_DEVICE void encode_input(float* input, float out_activations[MLP_HIDDEN_LAYER_SIZE * 2][MLP_FULLY_FUSED_SAMPLES_PER_THREAD_BLOCK]) const
	{
		unsigned int sample_in_chunk = threadIdx.x;
		for (unsigned int input_raw_index = 0; input_raw_index < MLP_INPUT_SIZE_RAW; input_raw_index++)
		{
			float input_value = input[input_raw_index];

			for (unsigned int frequency_index = 0; frequency_index < FREQUENCY_ENCODING_NUM_FREQUENCIES; frequency_index++)
			{
				float frequency = static_cast<float>(1 << frequency_index);
				out_activations[input_raw_index * FREQUENCY_ENCODING_NUM_FREQUENCIES * 2 + frequency_index * 2 + 0][sample_in_chunk] =
					sinf(frequency * input_value * hippt::M_Pi);
				out_activations[input_raw_index * FREQUENCY_ENCODING_NUM_FREQUENCIES * 2 + frequency_index * 2 + 1][sample_in_chunk] =
					cosf(frequency * input_value * hippt::M_Pi);
			}
		}
	}

	HIPRT_DEVICE void encode_input_ref(float* input, float* out_activations) const
	{
		for (unsigned int input_raw_index = 0; input_raw_index < MLP_INPUT_SIZE_RAW; input_raw_index++)
		{
			float input_value = input[input_raw_index];

			for (unsigned int frequency_index = 0; frequency_index < FREQUENCY_ENCODING_NUM_FREQUENCIES; frequency_index++)
			{
				float frequency = static_cast<float>(1 << frequency_index);
				out_activations[input_raw_index * FREQUENCY_ENCODING_NUM_FREQUENCIES * 2 + frequency_index * 2 + 0] =
					sinf(frequency * input_value * hippt::M_Pi);
				out_activations[input_raw_index * FREQUENCY_ENCODING_NUM_FREQUENCIES * 2 + frequency_index * 2 + 1] =
					cosf(frequency * input_value * hippt::M_Pi);
			}
		}
	}

	HIPRT_DEVICE OutputLayer get_output_layer(float activations_buffer[MLP_HIDDEN_LAYER_SIZE * 2][MLP_FULLY_FUSED_SAMPLES_PER_THREAD_BLOCK]) const
	{
		OutputLayer output;

		constexpr unsigned int output_layer_index = MLP_LAYER_COUNT - 1;
		unsigned int shared_mem_ping_pong_offset  = PING_PONG_ACTIVATIONS_OFFSET(output_layer_index);

		for (unsigned int neuron_index_in_output_layer = 0; neuron_index_in_output_layer < MLP_OUTPUT_SIZE; neuron_index_in_output_layer++)
			output.output[neuron_index_in_output_layer] = activations_buffer[shared_mem_ping_pong_offset + neuron_index_in_output_layer][threadIdx.x];

		return output;
	}

	HIPRT_DEVICE void inference(InputLayer input, float activations_buffer[MLP_HIDDEN_LAYER_SIZE * 2][MLP_FULLY_FUSED_SAMPLES_PER_THREAD_BLOCK]) const
	{
		encode_input(input.input, activations_buffer);

		unsigned int warp_id = threadIdx.x / 32;
		unsigned int lane_id = threadIdx.x & 31;

		for (unsigned int layer_index = 1; layer_index < MLP_LAYER_COUNT; layer_index++)
		{
			unsigned int neurons_current_layer	= get_layer_neuron_count(layer_index);
			unsigned int neurons_previous_layer = get_layer_neuron_count(layer_index - 1);

			unsigned int in_shared_mem_ping_pong_offset	 = PING_PONG_ACTIVATIONS_OFFSET(layer_index - 1);
			unsigned int out_shared_mem_ping_pong_offset = PING_PONG_ACTIVATIONS_OFFSET(layer_index);

			// The 4 warps of the thread block will compute 16 neurons each, for a total of 64 neurons per layer
			unsigned int neuron_base = warp_id * 16;

			for (unsigned int n = 0; n < 16 && (n + neuron_base) < neurons_current_layer; n++)
			{
				unsigned int neuron_index = n + neuron_base;

				// Because each warp goes through all samples in the thread block, that's MLP_FULLY_FUSED_SAMPLES_PER_THREAD_BLOCK / 32 samples per lane, 128 /
				// 32 = 4 samples per lane with thread block size 128 for example
				for (unsigned int s = 0; s < MLP_FULLY_FUSED_SAMPLES_PER_THREAD_BLOCK / 32; s++)
				{
					float activation = 0.0f;
					for (unsigned int previous_neuron_index = 0; previous_neuron_index < neurons_previous_layer; previous_neuron_index++)
					{
						unsigned int connection_data_index = get_connection_data_index(layer_index, previous_neuron_index, neuron_index);
						activation += connection_weights[connection_data_index] *
									  activations_buffer[in_shared_mem_ping_pong_offset + previous_neuron_index][lane_id + s * 32];
					}

					activation += neurons_biases[get_neuron_data_index(layer_index, neuron_index)];
					activation = activation_function(activation);

					activations_buffer[out_shared_mem_ping_pong_offset + neuron_index][lane_id + s * 32] = activation;
				}
			}

			__syncthreads();
		}
	}

	HIPRT_DEVICE void forward_pass(InputLayer input, float* out_activations) const
	{
		encode_input_ref(input.input, out_activations);

		for (unsigned int layer_index = 1; layer_index < MLP_LAYER_COUNT; layer_index++)
		{
			unsigned int neurons_current_layer	= get_layer_neuron_count(layer_index);
			unsigned int neurons_previous_layer = get_layer_neuron_count(layer_index - 1);

			for (unsigned int neuron_index = 0; neuron_index < neurons_current_layer; neuron_index++)
			{
				unsigned int current_neuron_data_index = get_neuron_data_index(layer_index, neuron_index);

				float activation = 0.0f;

				for (unsigned int previous_neuron_index = 0; previous_neuron_index < neurons_previous_layer; previous_neuron_index++)
				{
					unsigned int previous_neuron_data_index = get_neuron_data_index(layer_index - 1, previous_neuron_index);
					unsigned int connection_data_index		= get_connection_data_index(layer_index, previous_neuron_index, neuron_index);

					activation += connection_weights[connection_data_index] * out_activations[previous_neuron_data_index];
				}

				activation += neurons_biases[current_neuron_data_index];
				activation = activation_function(activation);

				out_activations[current_neuron_data_index] = activation;
			}
		}
	}

	HIPRT_DEVICE void backpropagation(float* neurons_activations, float* target_output) const
	{
		float neurons_errors[MLP_NEURON_COUNT];

		// Output layer bias and weights gradients
		unsigned int output_layer_index = MLP_LAYER_COUNT - 1;
		for (unsigned int neuron_index_in_output_layer = 0; neuron_index_in_output_layer < get_layer_neuron_count(output_layer_index);
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
			for (unsigned int previous_neuron_index = 0; previous_neuron_index < get_layer_neuron_count(output_layer_index - 1); previous_neuron_index++)
			{
				unsigned int previous_neuron_data_index = get_neuron_data_index(output_layer_index - 1, previous_neuron_index);
				unsigned int connection_data_index		= get_connection_data_index(output_layer_index, previous_neuron_index, neuron_index_in_output_layer);

				float previous_activation = neurons_activations[previous_neuron_data_index];

				float d_cost_d_Wij = d_cost_d_Zi * previous_activation;

				hippt::atomic_fetch_add(&gradient_weights[connection_data_index], d_cost_d_Wij);
			}
		}

		// Hidden layers bias and weights gradients
		for (unsigned int layer_index = MLP_LAYER_COUNT - 2; layer_index > 0; layer_index--)
		{
			for (unsigned int neuron_index = 0; neuron_index < get_layer_neuron_count(layer_index); neuron_index++)
			{
				unsigned int current_neuron_data_index = get_neuron_data_index(layer_index, neuron_index);

				float d_cost_d_Zi = 0.0f;

				for (unsigned int next_neuron_index = 0; next_neuron_index < get_layer_neuron_count(layer_index + 1); next_neuron_index++)
				{
					unsigned int next_neuron_data_index = get_neuron_data_index(layer_index + 1, next_neuron_index);
					unsigned int connection_data_index	= get_connection_data_index(layer_index + 1, neuron_index, next_neuron_index);

					float d_cost_d_Zj = neurons_errors[next_neuron_data_index];

					d_cost_d_Zi += d_cost_d_Zj * connection_weights[connection_data_index];
				}

				float current_activation = neurons_activations[current_neuron_data_index];
				float d_Zi_d_Oi			 = activation_function_derivative(current_activation);
				d_cost_d_Zi *= d_Zi_d_Oi;

				neurons_errors[current_neuron_data_index] = d_cost_d_Zi;
				hippt::atomic_fetch_add(&gradient_biases[current_neuron_data_index], d_cost_d_Zi);

				for (unsigned int previous_neuron_index = 0; previous_neuron_index < get_layer_neuron_count(layer_index - 1); previous_neuron_index++)
				{
					unsigned int previous_neuron_data_index = get_neuron_data_index(layer_index - 1, previous_neuron_index);
					unsigned int connection_data_index		= get_connection_data_index(layer_index, previous_neuron_index, neuron_index);

					float previous_activation = neurons_activations[previous_neuron_data_index];
					float d_cost_d_Wij		  = d_cost_d_Zi * previous_activation;

					hippt::atomic_fetch_add(&gradient_weights[connection_data_index], d_cost_d_Wij);
				}
			}
		}

		hippt::atomic_fetch_add(last_training_sample_count, 1u);
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

		return 2.0f * d_cost;
	}

	float* neurons_biases			   = nullptr;
	AtomicType<float>* gradient_biases = nullptr;

	float* connection_weights			= nullptr;
	AtomicType<float>* gradient_weights = nullptr;

	AtomicType<unsigned int>* last_training_sample_count = nullptr;

	float* adam_weights_means	  = nullptr;
	float* adam_weights_variances = nullptr;
	float* adam_biases_means	  = nullptr;
	float* adam_biases_variances  = nullptr;

	unsigned int training_step = 0;

	// Adam optimizer parameters
	float adam_learning_rate = 0.001f;
};

#endif
