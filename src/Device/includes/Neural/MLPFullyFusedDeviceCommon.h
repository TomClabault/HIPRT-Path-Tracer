/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_MLP_FULLY_FUSED_DEVICE_COMMON_H
#define DEVICE_INCLUDES_NEURAL_MLP_FULLY_FUSED_DEVICE_COMMON_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/AtomicType.h"
#include "HostDeviceCommon/Maths/Math.h"

#define PAD_SIZE_WMMA(size) ((size + 15) / 16 * 16)

enum class MLPActivationFunction
{
	LEAKY_RELU,
	RELU
};

template <unsigned int InputSizeEncoded_,
		  unsigned int HiddenLayerCount_,
		  unsigned int HiddenLayerSize_,
		  unsigned int OutputSize_,
		  unsigned int BlockSize_,
		  bool UseBiases_							= true,
		  MLPActivationFunction ActivationFunction_ = MLPActivationFunction::LEAKY_RELU,
		  bool UseOutputActivation_					= false>
struct MLPFullyFusedDeviceCommon
{
	static constexpr unsigned int INPUT_SIZE_ENCODED	  = InputSizeEncoded_;
	static constexpr unsigned int INPUT_SIZE_PADDED_WMMA  = PAD_SIZE_WMMA(INPUT_SIZE_ENCODED);
	static constexpr unsigned int OUTPUT_SIZE_PADDED_WMMA = (OutputSize_ + 15) / 16 * 16;
	static constexpr unsigned int LAYER_COUNT			  = HiddenLayerCount_ + 2;
	static constexpr unsigned int NEURON_COUNT			  = INPUT_SIZE_PADDED_WMMA + OUTPUT_SIZE_PADDED_WMMA + (HiddenLayerCount_ * HiddenLayerSize_);
	static constexpr unsigned int CONNECTIONS_COUNT		  = (INPUT_SIZE_PADDED_WMMA * HiddenLayerSize_) +
													  ((HiddenLayerCount_ - 1) * HiddenLayerSize_ * HiddenLayerSize_) +
													  (OUTPUT_SIZE_PADDED_WMMA * HiddenLayerSize_);
	static constexpr unsigned int SAMPLES_PER_BLOCK = BlockSize_;

	static constexpr unsigned int HIDDEN_LAYER_COUNT		   = HiddenLayerCount_;
	static constexpr unsigned int HIDDEN_LAYER_SIZE			   = HiddenLayerSize_;
	static constexpr unsigned int OUTPUT_SIZE				   = OutputSize_;
	static constexpr unsigned int BLOCK_SIZE				   = BlockSize_;
	static constexpr unsigned int ACTIVATION_WIDTH			   = hippt::max(INPUT_SIZE_PADDED_WMMA, hippt::max(HIDDEN_LAYER_SIZE, OUTPUT_SIZE_PADDED_WMMA));
	static constexpr unsigned int ERROR_WIDTH				   = hippt::max(HIDDEN_LAYER_SIZE, OUTPUT_SIZE_PADDED_WMMA);
	static constexpr bool USE_BIASES						   = UseBiases_;
	static constexpr MLPActivationFunction ACTIVATION_FUNCTION = ActivationFunction_;
	static constexpr bool USE_OUTPUT_ACTIVATION				   = UseOutputActivation_;

	struct InputLayer
	{
		float input[InputSizeEncoded_];
	};

	HIPRT_DEVICE static constexpr unsigned int get_layer_neuron_count(unsigned int layer)
	{
		return layer == 0 ? INPUT_SIZE_PADDED_WMMA : (layer == LAYER_COUNT - 1 ? OUTPUT_SIZE : HIDDEN_LAYER_SIZE);
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

	HIPRT_DEVICE void load_input_ref(const float* input, float* out_activations) const
	{
		for (unsigned int input_index = 0; input_index < INPUT_SIZE_ENCODED; input_index++)
			out_activations[input_index] = input[input_index];

		for (unsigned int input_index = INPUT_SIZE_ENCODED; input_index < INPUT_SIZE_PADDED_WMMA; input_index++)
			out_activations[input_index] = 0.0f;
	}

	HIPRT_DEVICE void inference_single_thread(const InputLayer& input, float* output) const
	{
		float activations_a[ACTIVATION_WIDTH];
		float activations_b[ACTIVATION_WIDTH];

		load_input_ref(input.input, activations_a);

		float* previous = activations_a;
		float* current	= activations_b;

		for (unsigned int layer = 1; layer < LAYER_COUNT; ++layer)
		{
			unsigned int previous_count = get_layer_neuron_count(layer - 1);
			unsigned int current_count	= get_layer_neuron_count(layer);

			for (unsigned int neuron = 0; neuron < current_count; ++neuron)
			{
				float value = 0.0f;

				if constexpr (USE_BIASES)
					value += neurons_biases[get_neuron_data_index(layer, neuron)];

				for (unsigned int previous_neuron = 0; previous_neuron < previous_count; ++previous_neuron)
				{
					unsigned int connection = get_connection_data_index(layer, previous_neuron, neuron);

					value += connection_weights[connection] * previous[previous_neuron];
				}

				if (layer != LAYER_COUNT - 1)
					value = activation_function(value);

				current[neuron] = value;
			}

			float* activation_swap = previous;
			previous			   = current;
			current				   = activation_swap;
		}

		for (unsigned int output_index = 0; output_index < OUTPUT_SIZE; ++output_index)
			output[output_index] = previous[output_index];
	}

	/**
	 * Runs a scalar single-thread forward pass and returns the post-activation values of a zero-based hidden layer.
	 * This is intended for debug
	 * inspection and does not use the WMMA execution path.
	 */
	HIPRT_DEVICE bool inference_single_thread_hidden_layer(const InputLayer& input, unsigned int hidden_layer_index, float* output) const
	{
		if (hidden_layer_index >= HIDDEN_LAYER_COUNT)
			return false;

		float activations_a[ACTIVATION_WIDTH];
		float activations_b[ACTIVATION_WIDTH];
		load_input_ref(input.input, activations_a);

		float* previous			  = activations_a;
		float* current			  = activations_b;
		unsigned int target_layer = hidden_layer_index + 1;

		for (unsigned int layer = 1; layer <= target_layer; ++layer)
		{
			unsigned int previous_count = get_layer_neuron_count(layer - 1);
			unsigned int current_count	= get_layer_neuron_count(layer);

			for (unsigned int neuron = 0; neuron < current_count; ++neuron)
			{
				float value = 0.0f;

				if constexpr (USE_BIASES)
					value += neurons_biases[get_neuron_data_index(layer, neuron)];

				for (unsigned int previous_neuron = 0; previous_neuron < previous_count; ++previous_neuron)
				{
					unsigned int connection = get_connection_data_index(layer, previous_neuron, neuron);
					value += connection_weights[connection] * previous[previous_neuron];
				}

				if (layer != LAYER_COUNT - 1)
					value = activation_function(value);

				current[neuron] = value;
			}

			if (layer == target_layer)
			{
				for (unsigned int neuron = 0; neuron < current_count; ++neuron)
					output[neuron] = current[neuron];

				return true;
			}

			float* activation_swap = previous;
			previous			   = current;
			current				   = activation_swap;
		}

		return false;
	}

	HIPRT_DEVICE constexpr float leaky_ReLU(float x) const
	{
		return x > 0.0f ? x : 0.01f * x;
	}

	HIPRT_DEVICE constexpr float leaky_ReLU_derivative(float x) const
	{
		return x > 0.0f ? 1.0f : 0.01f;
	}

	HIPRT_DEVICE constexpr float ReLU(float x) const
	{
		return x > 0.0f ? x : 0.0f;
	}

	HIPRT_DEVICE constexpr float ReLU_derivative(float x) const
	{
		return x > 0.0f ? 1.0f : 0.0f;
	}

	HIPRT_DEVICE constexpr float activation_function(float x) const
	{
		if constexpr (ACTIVATION_FUNCTION == MLPActivationFunction::RELU)
			return ReLU(x);
		else
			return leaky_ReLU(x);
	}

	HIPRT_DEVICE constexpr float activation_function_derivative(float x) const
	{
		if constexpr (ACTIVATION_FUNCTION == MLPActivationFunction::RELU)
			return ReLU_derivative(x);
		else
			return leaky_ReLU_derivative(x);
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

	// Adam optimizer parameters
	float adam_learning_rate = 0.001f;
};

#endif
