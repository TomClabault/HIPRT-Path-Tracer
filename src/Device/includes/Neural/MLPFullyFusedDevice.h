/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_MLP_FULLY_FUSED_DEVICE_H
#define DEVICE_INCLUDES_NEURAL_MLP_FULLY_FUSED_DEVICE_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Maths/Math.h"

#define PAD_SIZE_WMMA(size) ((size + 15) / 16 * 16)

template <unsigned int InputSizeRaw_,
		  unsigned int FreqEncodingFreqs_,
		  unsigned int HiddenLayerCount_,
		  unsigned int HiddenLayerSize_,
		  unsigned int OutputSize_,
		  unsigned int BlockSize_>
struct MLPFullyFusedDevice
{
	static constexpr unsigned int INPUT_SIZE			  = InputSizeRaw_ * 2 * FreqEncodingFreqs_;
	static constexpr unsigned int OUTPUT_SIZE_PADDED_WMMA = (OutputSize_ + 15) / 16 * 16;
	static constexpr unsigned int LAYER_COUNT			  = HiddenLayerCount_ + 2;
	static constexpr unsigned int NEURON_COUNT			  = INPUT_SIZE + OUTPUT_SIZE_PADDED_WMMA + (HiddenLayerCount_ * HiddenLayerSize_);
	static constexpr unsigned int CONNECTIONS_COUNT =
		(INPUT_SIZE * HiddenLayerSize_) + ((HiddenLayerCount_ - 1) * HiddenLayerSize_ * HiddenLayerSize_) + (OUTPUT_SIZE_PADDED_WMMA * HiddenLayerSize_);
	static constexpr unsigned int SAMPLES_PER_BLOCK = BlockSize_;

	static constexpr unsigned int INPUT_SIZE_RAW				= InputSizeRaw_;
	static constexpr unsigned int FREQ_ENCODING_NUM_FREQUENCIES = FreqEncodingFreqs_;
	static constexpr unsigned int HIDDEN_LAYER_COUNT			= HiddenLayerCount_;
	static constexpr unsigned int HIDDEN_LAYER_SIZE				= HiddenLayerSize_;
	static constexpr unsigned int OUTPUT_SIZE					= OutputSize_;
	static constexpr unsigned int BLOCK_SIZE					= BlockSize_;

	struct OutputLayer
	{
		float output[OutputSize_];
	};

	struct InputLayer
	{
		float input[InputSizeRaw_];
	};

	HIPRT_DEVICE static constexpr unsigned int get_layer_neuron_count(unsigned int layer)
	{
		return layer == 0 ? INPUT_SIZE : (layer == LAYER_COUNT - 1 ? OUTPUT_SIZE : HIDDEN_LAYER_SIZE);
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

	HIPRT_DEVICE void encode_input(float* input, fp16 out_activations[HiddenLayerSize_ * 2][BlockSize_]) const
	{
		unsigned int sample_in_chunk = threadIdx.x;
		for (unsigned int input_raw_index = 0; input_raw_index < InputSizeRaw_; input_raw_index++)
		{
			float input_value = input[input_raw_index];

			for (unsigned int frequency_index = 0; frequency_index < FreqEncodingFreqs_; frequency_index++)
			{
				float frequency = static_cast<float>(1 << frequency_index);
				out_activations[input_raw_index * FreqEncodingFreqs_ * 2 + frequency_index * 2 + 0][sample_in_chunk] =
					static_cast<fp16>(hippt::intrin_sinf(frequency * input_value * hippt::M_Pi));
				out_activations[input_raw_index * FreqEncodingFreqs_ * 2 + frequency_index * 2 + 1][sample_in_chunk] =
					static_cast<fp16>(hippt::intrin_cosf(frequency * input_value * hippt::M_Pi));
			}
		}
	}

	HIPRT_DEVICE void encode_input_ref(float* input, float* out_activations) const
	{
		for (unsigned int input_raw_index = 0; input_raw_index < InputSizeRaw_; input_raw_index++)
		{
			float input_value = input[input_raw_index];

			for (unsigned int frequency_index = 0; frequency_index < FreqEncodingFreqs_; frequency_index++)
			{
				float frequency																		= static_cast<float>(1 << frequency_index);
				out_activations[input_raw_index * FreqEncodingFreqs_ * 2 + frequency_index * 2 + 0] = hippt::intrin_sinf(frequency * input_value * hippt::M_Pi);
				out_activations[input_raw_index * FreqEncodingFreqs_ * 2 + frequency_index * 2 + 1] = hippt::intrin_cosf(frequency * input_value * hippt::M_Pi);
			}
		}
	}

	HIPRT_DEVICE OutputLayer get_output_layer(fp16 activations_buffer[HiddenLayerSize_ * 2][BlockSize_]) const
	{
		OutputLayer output;

		constexpr unsigned int output_layer_index = LAYER_COUNT - 1;
		unsigned int shared_mem_ping_pong_offset  = (output_layer_index & 1) * HiddenLayerSize_;

		for (unsigned int neuron_index_in_output_layer = 0; neuron_index_in_output_layer < OutputSize_; neuron_index_in_output_layer++)
			output.output[neuron_index_in_output_layer] =
				static_cast<float>(activations_buffer[shared_mem_ping_pong_offset + neuron_index_in_output_layer][threadIdx.x]);

		return output;
	}

	HIPRT_DEVICE void inference(InputLayer input, fp16 activations_buffer[HiddenLayerSize_ * 2][BlockSize_]) const
	{
		static_assert(INPUT_SIZE % 16 == 0, "INPUT_SIZE must be a multiple of 16 for WMMA");
		static_assert(HiddenLayerSize_ % 16 == 0, "HiddenLayerSize_ must be a multiple of 16 for WMMA");
		static_assert(OUTPUT_SIZE_PADDED_WMMA % 16 == 0, "OUTPUT_SIZE_PADDED_WMMA must be a multiple of 16 for WMMA");

		encode_input(input.input, activations_buffer);
		__syncthreads();

		for (unsigned int layer_index = 1; layer_index < LAYER_COUNT; layer_index++)
		{
			unsigned int neurons_current_layer				= get_layer_neuron_count(layer_index);
			unsigned int neurons_current_layer_padded_wmma	= PAD_SIZE_WMMA(neurons_current_layer);
			unsigned int neurons_previous_layer				= get_layer_neuron_count(layer_index - 1);
			unsigned int neurons_previous_layer_padded_wmma = PAD_SIZE_WMMA(neurons_previous_layer);

			unsigned int layer_neuron_offset	 = get_neuron_data_index(layer_index, 0);
			unsigned int layer_connection_offset = get_connection_data_index(layer_index, 0, 0);

			unsigned int in_shared_mem_ping_pong_offset	 = ((layer_index - 1) & 1) * HiddenLayerSize_;
			unsigned int out_shared_mem_ping_pong_offset = (layer_index & 1) * HiddenLayerSize_;

			// Each warp computes TILES_PER_WARP tiles of 16 neurons. With block size 64
			// (2 warps), each warp handles 2 tiles to cover 64 hidden neurons for example.
			constexpr unsigned int TILES_PER_WARP = HiddenLayerSize_ / 16 / (BlockSize_ / 32);
			static_assert(BlockSize_ % 32 == 0, "BlockSize_ must be a multiple of 32 for MLP");
			static_assert(HiddenLayerSize_ % 16 == 0, "HiddenLayerSize_ must be a multiple of 16 for WMMA");

			unsigned int warp_id = threadIdx.x / 32;
			unsigned int lane_id = threadIdx.x & 31;

			unsigned int previous_neurons_tile_count = neurons_previous_layer_padded_wmma / 16;
			unsigned int current_neuron_tile_count	 = neurons_current_layer_padded_wmma / 16;
			unsigned int warp_first_tile			 = warp_id * TILES_PER_WARP;
			constexpr unsigned int sample_tile_count = BlockSize_ / 16;

			// WMMA compatible architectures #if guard
#if __gfx1100__ || __gfx1101__ || __gfx1102__ || __gfx1200__ || __gfx1201__
			unsigned int lane_id_wmma = lane_id & 15; // Lane [0 - 15] need to be duplicated in [16 - 31] for WMMA on RDNA3
			unsigned int lane_high	  = lane_id / 16;

			if (warp_first_tile < current_neuron_tile_count)
			{
				for (unsigned int tile = 0; tile < TILES_PER_WARP && (warp_first_tile + tile) < current_neuron_tile_count; tile++)
				{
					fp16x16 activation_tiles[BlockSize_ / 16] = {};

					unsigned int current_neuron_base = (warp_first_tile + tile) * 16;
					for (unsigned int previous_neuron_tile_index = 0; previous_neuron_tile_index < previous_neurons_tile_count; previous_neuron_tile_index++)
					{
						fp16x16 neuron_weights_fragment;

						unsigned int previous_neuron_base = previous_neuron_tile_index * 16;
						for (unsigned int w = 0; w < 16; w++)
							neuron_weights_fragment[w] =
								connection_weights_fp16[layer_connection_offset + (current_neuron_base + lane_id_wmma) * neurons_previous_layer +
														previous_neuron_base + w];

						for (unsigned int sample_tile_index = 0; sample_tile_index < sample_tile_count; sample_tile_index++)
						{
							fp16x16 previous_activations_fragment;

							unsigned int sample_base = sample_tile_index * 16;
							for (unsigned int a = 0; a < 16; a++)
								previous_activations_fragment[a] =
									activations_buffer[in_shared_mem_ping_pong_offset + previous_neuron_base + a][sample_base + lane_id_wmma];

							activation_tiles[sample_tile_index] = hippt::amdgcn_wmma_f16_16x16x16_f16_w32(
								neuron_weights_fragment, previous_activations_fragment, activation_tiles[sample_tile_index]);
						}
					}

					// Add the bias and apply the activation function
					for (unsigned int n_tile = 0; n_tile < sample_tile_count; n_tile++)
					{
						for (int ele = 0; ele < 8; ++ele)
						{
							unsigned int r = ele * 2 + lane_high;
							unsigned int m = current_neuron_base + r;
							unsigned int n = n_tile * 16 + lane_id_wmma;

							float val = activation_tiles[n_tile][ele * 2] + neurons_biases[layer_neuron_offset + m];
							val		  = activation_function(val);

							activations_buffer[out_shared_mem_ping_pong_offset + m][n] = static_cast<fp16>(val);
						}
					}
				}
			}
#else
			// Non WMMA codepath, slow AF

			if (warp_first_tile < current_neuron_tile_count)
			{
				for (unsigned int tile = 0; tile < TILES_PER_WARP && (warp_first_tile + tile) < current_neuron_tile_count; tile++)
				{
					unsigned int neuron_base = (warp_first_tile + tile) * 16;
					for (unsigned int n = 0; n < 16 && (n + neuron_base) < neurons_current_layer; n++)
					{
						unsigned int neuron_index = n + neuron_base;

						for (unsigned int s = 0; s < BlockSize_ / 32; s++)
						{
							float activation = 0.0f;
							for (unsigned int previous_neuron_index = 0; previous_neuron_index < neurons_previous_layer; previous_neuron_index++)
							{
								unsigned int connection_data_index = layer_connection_offset + neuron_index * neurons_previous_layer + previous_neuron_index;
								activation += connection_weights[connection_data_index] *
											  activations_buffer[in_shared_mem_ping_pong_offset + previous_neuron_index][lane_id + s * 32];
							}

							activation += neurons_biases[layer_neuron_offset + neuron_index];
							activation = activation_function(activation);

							activations_buffer[out_shared_mem_ping_pong_offset + neuron_index][lane_id + s * 32] = activation;
						}
					}
				}
			}
#endif

			__syncthreads();
		}
	}

	HIPRT_DEVICE void forward_pass(InputLayer input, float* out_activations) const
	{
		encode_input_ref(input.input, out_activations);

		for (unsigned int layer_index = 1; layer_index < LAYER_COUNT; layer_index++)
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
		float neurons_errors[NEURON_COUNT];

		// Output layer bias and weights gradients
		unsigned int output_layer_index = LAYER_COUNT - 1;
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
		for (unsigned int layer_index = LAYER_COUNT - 2; layer_index > 0; layer_index--)
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
	fp16* connection_weights_fp16		= nullptr;
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
