/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_MLP_FULLY_FUSED_DEVICE_GPU_H
#define DEVICE_INCLUDES_NEURAL_MLP_FULLY_FUSED_DEVICE_GPU_H

#include "Device/includes/Neural/MLPFullyFusedDeviceCommon.h"
#include "HostDeviceCommon/Neural/NISMLTrainProfiling.h"

#ifndef NISML_HAS_WMMA
#define NISML_HAS_WMMA (__gfx1100__ || __gfx1101__ || __gfx1102__ || __gfx1200__ || __gfx1201__)
#endif

template <unsigned int InputSizeEncoded_,
		  unsigned int HiddenLayerCount_,
		  unsigned int HiddenLayerSize_,
		  unsigned int OutputSize_,
		  unsigned int BlockSize_,
		  bool UseBiases_							= true,
		  MLPActivationFunction ActivationFunction_ = MLPActivationFunction::LEAKY_RELU,
		  bool UseOutputActivation_					= false>
struct MLPFullyFusedDeviceGPU : public MLPFullyFusedDeviceCommon<InputSizeEncoded_,
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

	using Common::ACTIVATION_FUNCTION;
	using Common::ACTIVATION_WIDTH;
	using Common::BLOCK_SIZE;
	using Common::connection_weights;
	using Common::connection_weights_fp16;
	using Common::CONNECTIONS_COUNT;
	using Common::ERROR_WIDTH;
	using Common::get_connection_data_index;
	using Common::get_layer_neuron_count;
	using Common::get_neuron_data_index;
	using Common::gradient_biases;
	using Common::gradient_weights;
	using Common::HIDDEN_LAYER_COUNT;
	using Common::HIDDEN_LAYER_SIZE;
	using Common::INPUT_SIZE_ENCODED;
	using Common::INPUT_SIZE_PADDED_WMMA;
	using InputLayer = typename Common::InputLayer;
	using Common::last_training_sample_count;
	using Common::LAYER_COUNT;
	using Common::NEURON_COUNT;
	using Common::neurons_biases;
	using Common::OUTPUT_SIZE;
	using Common::OUTPUT_SIZE_PADDED_WMMA;
	using Common::USE_BIASES;
	using Common::USE_OUTPUT_ACTIVATION;

	HIPRT_DEVICE void load_input(const float* input, fp16 activations_buffer[ACTIVATION_WIDTH * 2][BLOCK_SIZE]) const
	{
		unsigned int sample_in_chunk = threadIdx.x;

		for (unsigned int input_index = 0; input_index < INPUT_SIZE_ENCODED; input_index++)
			activations_buffer[input_index][sample_in_chunk] = static_cast<fp16>(input[input_index]);

		for (unsigned int input_index = INPUT_SIZE_ENCODED; input_index < INPUT_SIZE_PADDED_WMMA; input_index++)
			activations_buffer[input_index][sample_in_chunk] = static_cast<fp16>(0.0f);

		__syncthreads();
	}

	HIPRT_DEVICE void zero_padded_input(const float* input, fp16 activations_buffer[ACTIVATION_WIDTH * 2][BLOCK_SIZE]) const
	{
		unsigned int sample_in_chunk = threadIdx.x;

		for (unsigned int input_index = INPUT_SIZE_ENCODED; input_index < INPUT_SIZE_PADDED_WMMA; input_index++)
			activations_buffer[input_index][sample_in_chunk] = static_cast<fp16>(0.0f);

		__syncthreads();
	}

	HIPRT_DEVICE void forward_train_wmma(fp16 activations_buffer[ACTIVATION_WIDTH * 2][BLOCK_SIZE],
										 fp16* train_activations_global,
										 unsigned int sample_offset) const
	{
		static_assert(INPUT_SIZE_PADDED_WMMA % 16 == 0, "INPUT_SIZE_PADDED_WMMA must be a multiple of 16 for WMMA");
		static_assert(HiddenLayerSize_ % 16 == 0, "HiddenLayerSize_ must be a multiple of 16 for WMMA");
		static_assert(OUTPUT_SIZE_PADDED_WMMA % 16 == 0, "OUTPUT_SIZE_PADDED_WMMA must be a multiple of 16 for WMMA");

#if NISML_HAS_WMMA
		for (unsigned int layer_index = 1; layer_index < LAYER_COUNT; layer_index++)
		{
			unsigned int neurons_current_layer				= get_layer_neuron_count(layer_index);
			unsigned int neurons_current_layer_padded_wmma	= PAD_SIZE_WMMA(neurons_current_layer);
			unsigned int neurons_previous_layer				= get_layer_neuron_count(layer_index - 1);
			unsigned int neurons_previous_layer_padded_wmma = PAD_SIZE_WMMA(neurons_previous_layer);

			unsigned int layer_neuron_offset	 = get_neuron_data_index(layer_index, 0);
			unsigned int layer_connection_offset = get_connection_data_index(layer_index, 0, 0);

			unsigned int in_shared_mem_ping_pong_offset	 = ((layer_index - 1) & 1) * ACTIVATION_WIDTH;
			unsigned int out_shared_mem_ping_pong_offset = (layer_index & 1) * ACTIVATION_WIDTH;

			constexpr unsigned int TILES_PER_WARP = HiddenLayerSize_ / 16 / (BlockSize_ / 32);
			static_assert(BlockSize_ % 32 == 0, "BlockSize_ must be a multiple of 32 for MLP");
			static_assert(HiddenLayerSize_ % 16 == 0, "HiddenLayerSize_ must be a multiple of 16 for WMMA");

			unsigned int warp_id = threadIdx.x / 32;
			unsigned int lane_id = threadIdx.x & 31;

			unsigned int previous_neurons_tile_count = neurons_previous_layer_padded_wmma / 16;
			unsigned int current_neuron_tile_count	 = neurons_current_layer_padded_wmma / 16;
			unsigned int warp_first_tile			 = warp_id * TILES_PER_WARP;
			constexpr unsigned int sample_tile_count = BlockSize_ / 16;

			unsigned int lane_id_wmma = lane_id & 15;
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
						for (unsigned int weight_index = 0; weight_index < 16; weight_index++)
							neuron_weights_fragment[weight_index] =
								connection_weights_fp16[layer_connection_offset + (current_neuron_base + lane_id_wmma) * neurons_previous_layer +
														previous_neuron_base + weight_index];

						for (unsigned int sample_tile_index = 0; sample_tile_index < sample_tile_count; sample_tile_index++)
						{
							fp16x16 previous_activations_fragment;

							unsigned int sample_base = sample_tile_index * 16;
							for (unsigned int activation_index = 0; activation_index < 16; activation_index++)
								previous_activations_fragment[activation_index] =
									activations_buffer[in_shared_mem_ping_pong_offset + previous_neuron_base + activation_index][sample_base + lane_id_wmma];

							activation_tiles[sample_tile_index] = hippt::amdgcn_wmma_f16_16x16x16_f16_w32(
								neuron_weights_fragment, previous_activations_fragment, activation_tiles[sample_tile_index]);
						}
					}

					for (unsigned int neuron_tile = 0; neuron_tile < sample_tile_count; neuron_tile++)
					{
						for (int element = 0; element < 8; ++element)
						{
							unsigned int row	= element * 2 + lane_high;
							unsigned int neuron = current_neuron_base + row;
							unsigned int sample = neuron_tile * 16 + lane_id_wmma;

							float value = activation_tiles[neuron_tile][element * 2];
							if constexpr (USE_BIASES)
								value += neurons_biases[layer_neuron_offset + neuron];
							if (layer_index < LAYER_COUNT - 1)
								value = this->activation_function(value);
							else if constexpr (USE_OUTPUT_ACTIVATION)
								value = this->activation_function(value);

							activations_buffer[out_shared_mem_ping_pong_offset + neuron][sample]							 = static_cast<fp16>(value);
							train_activations_global[(sample_offset + sample) * NEURON_COUNT + layer_neuron_offset + neuron] = static_cast<fp16>(value);
						}
					}
				}
			}
			__syncthreads();
		}
#endif
	}

	HIPRT_DEVICE void forward_train(fp16 activations_buffer[ACTIVATION_WIDTH * 2][BLOCK_SIZE], fp16* train_activations_global, unsigned int sample_offset) const
	{
		static_assert(BlockSize_ % 32 == 0, "BlockSize must be a multiple of 32 for MLP");

		unsigned int lane_id = threadIdx.x & 31;
		unsigned int warp_id = threadIdx.x / 32;

		for (unsigned int layer_index = 1; layer_index < LAYER_COUNT; layer_index++)
		{
			unsigned int neurons_current_layer			 = get_layer_neuron_count(layer_index);
			unsigned int neurons_previous_layer			 = get_layer_neuron_count(layer_index - 1);
			unsigned int layer_neuron_offset			 = get_neuron_data_index(layer_index, 0);
			unsigned int layer_connection_offset		 = get_connection_data_index(layer_index, 0, 0);
			unsigned int in_shared_mem_ping_pong_offset	 = ((layer_index - 1) & 1) * ACTIVATION_WIDTH;
			unsigned int out_shared_mem_ping_pong_offset = (layer_index & 1) * ACTIVATION_WIDTH;

			constexpr unsigned int tiles_per_warp  = HiddenLayerSize_ / 16 / (BlockSize_ / 32);
			unsigned int current_neuron_tile_count = PAD_SIZE_WMMA(neurons_current_layer) / 16;
			unsigned int warp_first_tile		   = warp_id * tiles_per_warp;

			if (warp_first_tile < current_neuron_tile_count)
			{
				for (unsigned int tile = 0; tile < tiles_per_warp && warp_first_tile + tile < current_neuron_tile_count; tile++)
				{
					unsigned int neuron_base = (warp_first_tile + tile) * 16;

					for (unsigned int neuron_offset = 0; neuron_offset < 16 && neuron_base + neuron_offset < neurons_current_layer; neuron_offset++)
					{
						unsigned int neuron_index = neuron_base + neuron_offset;

						for (unsigned int sample_tile = 0; sample_tile < BlockSize_ / 32; sample_tile++)
						{
							unsigned int sample_index = lane_id + sample_tile * 32;
							float activation		  = 0.0f;

							for (unsigned int previous_neuron_index = 0; previous_neuron_index < neurons_previous_layer; previous_neuron_index++)
							{
								unsigned int connection_data_index = layer_connection_offset + neuron_index * neurons_previous_layer + previous_neuron_index;
								activation += connection_weights[connection_data_index] *
											  static_cast<float>(activations_buffer[in_shared_mem_ping_pong_offset + previous_neuron_index][sample_index]);
							}

							if constexpr (USE_BIASES)
								activation += neurons_biases[layer_neuron_offset + neuron_index];
							if (layer_index < LAYER_COUNT - 1)
								activation = this->activation_function(activation);
							else if constexpr (USE_OUTPUT_ACTIVATION)
								activation = this->activation_function(activation);

							activations_buffer[out_shared_mem_ping_pong_offset + neuron_index][sample_index] = static_cast<fp16>(activation);
							train_activations_global[(sample_offset + sample_index) * NEURON_COUNT + layer_neuron_offset + neuron_index] =
								static_cast<fp16>(activation);
						}
					}
				}
			}

			__syncthreads();
		}
	}

	HIPRT_DEVICE void backpropagation_wmma_from_output_gradient(fp16* train_activations_global,
																unsigned int sample_offset,
																fp16 activations_buffer[ACTIVATION_WIDTH * 2][BLOCK_SIZE],
																fp16 errors_buffer[ERROR_WIDTH * 2][BLOCK_SIZE],
																float* input_gradients,
																unsigned int input_gradient_count,
																const float* output_gradient,
																bool count_training_sample,
																float error_scale,
																bool output_errors_initialized
#if NISML_HAS_WMMA
																,
																NISMLTrainProfileRecord* profile_record
#endif
	) const
	{
		static_assert(INPUT_SIZE_PADDED_WMMA % 16 == 0, "INPUT_SIZE_PADDED_WMMA must be a multiple of 16 for WMMA");
		static_assert(HiddenLayerSize_ % 16 == 0, "HiddenLayerSize must be a multiple of 16 for WMMA");
		static_assert(OUTPUT_SIZE_PADDED_WMMA % 16 == 0, "OUTPUT_SIZE_PADDED_WMMA must be a multiple of 16 for WMMA");
		static_assert(BlockSize_ % 32 == 0, "BlockSize must be a multiple of 32 for WMMA");
		static_assert(OUTPUT_SIZE_PADDED_WMMA <= ERROR_WIDTH, "Error buffer must fit the padded output layer");
		static_assert(HIDDEN_LAYER_SIZE <= ERROR_WIDTH, "Error buffer must fit hidden layers");

#if NISML_HAS_WMMA
		unsigned int lane_id	  = threadIdx.x & 31;
		unsigned int warp_id	  = threadIdx.x / 32;
		unsigned int lane_id_wmma = lane_id & 15;
		unsigned int lane_high	  = lane_id / 16;
		unsigned int warp_count	  = BlockSize_ / 32;
		// NISML scales errors before storing them in FP16; restore the scale when accumulating FP32 parameter gradients.
		float inverse_error_scale			 = 1.0f / error_scale;
		unsigned long long int profile_start = 0;

		constexpr unsigned int errors_ping_pong_size = ERROR_WIDTH;
		unsigned int output_layer_index				 = LAYER_COUNT - 1;
		unsigned int output_layer_offset			 = get_neuron_data_index(output_layer_index, 0);
		unsigned int output_errors_offset			 = (output_layer_index & 1) * errors_ping_pong_size;

		NISML_TRAIN_PROFILE_START(profile_record, profile_start);
		if (!output_errors_initialized)
		{
			for (unsigned int linear_index = threadIdx.x; linear_index < OUTPUT_SIZE_PADDED_WMMA * BlockSize_; linear_index += BlockSize_)
			{
				unsigned int output_neuron = linear_index / BlockSize_;
				unsigned int sample_index  = linear_index % BlockSize_;
				fp16 error				   = static_cast<fp16>(0.0f);

				if (output_neuron < OutputSize_ && sample_index == threadIdx.x)
					error = static_cast<fp16>(output_gradient[output_neuron] * error_scale);

				errors_buffer[output_errors_offset + output_neuron][sample_index] = error;
			}
		}
		__syncthreads();
		NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_OUTPUT_ERROR_INITIALIZATION, profile_start);

		for (unsigned int layer_index = output_layer_index; layer_index > 0; layer_index--)
		{
			unsigned int current_neurons		 = get_layer_neuron_count(layer_index);
			unsigned int current_neurons_padded	 = PAD_SIZE_WMMA(current_neurons);
			unsigned int previous_neurons		 = get_layer_neuron_count(layer_index - 1);
			unsigned int previous_neurons_padded = PAD_SIZE_WMMA(previous_neurons);
			unsigned int current_layer_offset	 = get_neuron_data_index(layer_index, 0);
			unsigned int previous_layer_offset	 = get_neuron_data_index(layer_index - 1, 0);
			unsigned int connection_offset		 = get_connection_data_index(layer_index, 0, 0);
			unsigned int current_errors_offset	 = (layer_index & 1) * errors_ping_pong_size;
			unsigned int previous_errors_offset	 = ((layer_index - 1) & 1) * errors_ping_pong_size;
			unsigned int current_tile_count		 = current_neurons_padded / 16;
			unsigned int previous_tile_count	 = previous_neurons_padded / 16;
			unsigned int sample_tile_count		 = BlockSize_ / 16;

			NISML_TRAIN_PROFILE_START(profile_record, profile_start);
			for (unsigned int linear_index = threadIdx.x; linear_index < previous_neurons_padded * BlockSize_; linear_index += BlockSize_)
			{
				unsigned int previous_neuron = linear_index / BlockSize_;
				unsigned int sample_index	 = linear_index % BlockSize_;
				fp16 activation				 = static_cast<fp16>(0.0f);

				if (previous_neuron < previous_neurons)
				{
					unsigned int activation_index = (sample_offset + sample_index) * NEURON_COUNT + previous_layer_offset + previous_neuron;
					activation					  = train_activations_global[activation_index];
				}

				activations_buffer[previous_neuron][sample_index] = activation;
			}
			__syncthreads();
			NISML_TRAIN_PROFILE_STOP_ACCUMULATE(profile_record, NISML_TRAIN_PROFILE_ACTIVATION_RELOAD, profile_start);

			NISML_TRAIN_PROFILE_START(profile_record, profile_start);
			for (unsigned int current_tile = warp_id; current_tile < current_tile_count; current_tile += warp_count)
			{
				unsigned int current_neuron_base = current_tile * 16;

				for (unsigned int previous_tile = 0; previous_tile < previous_tile_count; previous_tile++)
				{
					unsigned int previous_neuron_base = previous_tile * 16;
					fp16x16 gradient_tile			  = {};

					for (unsigned int sample_tile = 0; sample_tile < sample_tile_count; sample_tile++)
					{
						fp16x16 errors_fragment;
						fp16x16 activations_fragment;
						unsigned int sample_base = sample_tile * 16;

						for (unsigned int sample = 0; sample < 16; sample++)
							errors_fragment[sample] = errors_buffer[current_errors_offset + current_neuron_base + lane_id_wmma][sample_base + sample];

						for (unsigned int sample = 0; sample < 16; sample++)
							activations_fragment[sample] = activations_buffer[previous_neuron_base + lane_id_wmma][sample_base + sample];

						gradient_tile = hippt::amdgcn_wmma_f16_16x16x16_f16_w32(errors_fragment, activations_fragment, gradient_tile);
					}

					for (unsigned int element = 0; element < 8; element++)
					{
						unsigned int current_neuron	 = current_neuron_base + element * 2 + lane_high;
						unsigned int previous_neuron = previous_neuron_base + lane_id_wmma;

						if (current_neuron < current_neurons && previous_neuron < previous_neurons)
						{
							unsigned int connection_index = connection_offset + current_neuron * previous_neurons + previous_neuron;
							hippt::atomic_fetch_add(&gradient_weights[connection_index], static_cast<float>(gradient_tile[element * 2]) * inverse_error_scale);
						}
					}
				}
			}
			NISML_TRAIN_PROFILE_STOP_ACCUMULATE(profile_record, NISML_TRAIN_PROFILE_WEIGHT_GRADIENTS, profile_start);

			NISML_TRAIN_PROFILE_START(profile_record, profile_start);
			if constexpr (USE_BIASES)
			{
				for (unsigned int neuron_index = threadIdx.x; neuron_index < current_neurons; neuron_index += BlockSize_)
				{
					float error_sum = 0.0f;
					for (unsigned int sample_index = 0; sample_index < BlockSize_; sample_index++)
						error_sum += static_cast<float>(errors_buffer[current_errors_offset + neuron_index][sample_index]);

					hippt::atomic_fetch_add(&gradient_biases[current_layer_offset + neuron_index], error_sum * inverse_error_scale);
				}
			}
			NISML_TRAIN_PROFILE_STOP_ACCUMULATE(profile_record, NISML_TRAIN_PROFILE_BIAS_GRADIENTS, profile_start);
			__syncthreads();

			if (layer_index > 1)
			{
				NISML_TRAIN_PROFILE_START(profile_record, profile_start);
				for (unsigned int previous_tile = warp_id; previous_tile < previous_tile_count; previous_tile += warp_count)
				{
					unsigned int previous_neuron_base		   = previous_tile * 16;
					fp16x16 propagated_errors[BlockSize_ / 16] = {};

					for (unsigned int current_tile = 0; current_tile < current_tile_count; current_tile++)
					{
						unsigned int current_neuron_base = current_tile * 16;
						fp16x16 weights_fragment;
						for (unsigned int current_neuron = 0; current_neuron < 16; current_neuron++)
						{
							unsigned int neuron = current_neuron_base + current_neuron;
							weights_fragment[current_neuron] =
								neuron < current_neurons
									? connection_weights_fp16[connection_offset + neuron * previous_neurons + previous_neuron_base + lane_id_wmma]
									: static_cast<fp16>(0.0f);
						}

						for (unsigned int sample_tile = 0; sample_tile < sample_tile_count; sample_tile++)
						{
							fp16x16 errors_fragment;
							unsigned int sample_base = sample_tile * 16;

							for (unsigned int current_neuron = 0; current_neuron < 16; current_neuron++)
								errors_fragment[current_neuron] =
									errors_buffer[current_errors_offset + current_neuron_base + current_neuron][sample_base + lane_id_wmma];

							propagated_errors[sample_tile] =
								hippt::amdgcn_wmma_f16_16x16x16_f16_w32(weights_fragment, errors_fragment, propagated_errors[sample_tile]);
						}
					}

					for (unsigned int sample_tile = 0; sample_tile < sample_tile_count; sample_tile++)
					{
						for (unsigned int element = 0; element < 8; element++)
						{
							unsigned int previous_neuron = previous_neuron_base + element * 2 + lane_high;
							unsigned int sample_index	 = sample_tile * 16 + lane_id_wmma;

							if (previous_neuron < previous_neurons)
								errors_buffer[previous_errors_offset + previous_neuron][sample_index] = propagated_errors[sample_tile][element * 2];
						}
					}
				}
				NISML_TRAIN_PROFILE_STOP_ACCUMULATE(profile_record, NISML_TRAIN_PROFILE_ERROR_PROPAGATION, profile_start);

				__syncthreads();

				for (unsigned int linear_index = threadIdx.x; linear_index < previous_neurons * BlockSize_; linear_index += BlockSize_)
				{
					unsigned int previous_neuron = linear_index / BlockSize_;
					unsigned int sample_index	 = linear_index % BlockSize_;
					float propagated_error		 = static_cast<float>(errors_buffer[previous_errors_offset + previous_neuron][sample_index]);
					float previous_activation	 = static_cast<float>(activations_buffer[previous_neuron][sample_index]);

					propagated_error *= this->activation_function_derivative(previous_activation);
					errors_buffer[previous_errors_offset + previous_neuron][sample_index] = static_cast<fp16>(propagated_error);
				}
			}

			__syncthreads();
		}

		if (input_gradients != nullptr)
		{
			NISML_TRAIN_PROFILE_START(profile_record, profile_start);
			unsigned int first_layer_connection_offset = get_connection_data_index(1, 0, 0);
			unsigned int first_hidden_error_offset	   = ERROR_WIDTH;
			for (unsigned int input_index = 0; input_index < input_gradient_count; input_index++)
			{
				float input_gradient = 0.0f;
				for (unsigned int hidden_index = 0; hidden_index < HIDDEN_LAYER_SIZE; hidden_index++)
				{
					unsigned int connection_index = first_layer_connection_offset + hidden_index * INPUT_SIZE_PADDED_WMMA + input_index;
					input_gradient += static_cast<float>(errors_buffer[first_hidden_error_offset + hidden_index][threadIdx.x]) *
									  static_cast<float>(connection_weights_fp16[connection_index]) * inverse_error_scale;
				}

				input_gradients[input_index] = input_gradient;
			}
			NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_INPUT_GRADIENTS, profile_start);
		}

		if (count_training_sample)
			hippt::atomic_fetch_add(last_training_sample_count, 1u);
#endif
	}

	HIPRT_DEVICE void backpropagation_wmma(fp16* train_activations_global,
										   unsigned int sample_offset,
										   fp16 activations_buffer[ACTIVATION_WIDTH * 2][BLOCK_SIZE],
										   fp16 errors_buffer[ERROR_WIDTH * 2][BLOCK_SIZE],
										   float* input_gradients,
										   unsigned int input_gradient_count,
										   const float* output_gradient,
										   float error_scale,
										   bool count_training_sample,
										   bool output_errors_initialized
#if NISML_HAS_WMMA
										   ,
										   NISMLTrainProfileRecord* profile_record
#endif
	) const
	{
		backpropagation_wmma_from_output_gradient(train_activations_global, sample_offset, activations_buffer, errors_buffer, input_gradients,
												  input_gradient_count, output_gradient, count_training_sample, error_scale, output_errors_initialized
#if NISML_HAS_WMMA
												  ,
												  profile_record
#endif
		);
	}
};

#endif
