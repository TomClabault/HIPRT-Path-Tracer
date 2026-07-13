/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_MLP_FULLY_FUSED_TRAIN_H
#define KERNELS_MLP_FULLY_FUSED_TRAIN_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Neural/MLPFullyFusedDevice.h"
#include "HostDeviceCommon/KernelOptions/MLPTrainingTestOptions.h"
#include "HostDeviceCommon/Xorshift.h"

using TrainingTestMLP = MLPFullyFusedDevice<
	MLP_TRAINING_TEST_INPUT_SIZE_RAW,
	MLP_TRAINING_TEST_FREQUENCY_ENCODING_NUM_FREQUENCIES,
	MLP_TRAINING_TEST_HIDDEN_LAYER_COUNT,
	MLP_TRAINING_TEST_HIDDEN_LAYER_SIZE,
	MLP_TRAINING_TEST_OUTPUT_SIZE,
	MLP_TRAINING_TEST_THREAD_BLOCK_SIZE,
	MLP_TRAINING_TEST_USE_BIASES>;

GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(TrainingTestMLP::BLOCK_SIZE)
MLPFullyFusedTrain(TrainingTestMLP mlp, unsigned char* texture, unsigned int tex_w, unsigned int tex_h, unsigned int frame_number, fp16* train_activations)
{
	unsigned int sample_index = blockIdx.x * blockDim.x + threadIdx.x;

	Xorshift32Generator rng(sample_index * 19741 + frame_number * 31337);

	float uv[2] = { rng(), rng() };

	unsigned int tx = static_cast<unsigned int>(uv[0] * (tex_w - 1));
	unsigned int ty = static_cast<unsigned int>(uv[1] * (tex_h - 1));
	unsigned int pi = (ty * tex_w + tx) * 3;

	float target_color[3] = { texture[pi + 0] / 255.0f, texture[pi + 1] / 255.0f, texture[pi + 2] / 255.0f };

	__shared__ fp16 activations_buffer[TrainingTestMLP::HIDDEN_LAYER_SIZE * 2][TrainingTestMLP::BLOCK_SIZE];

	TrainingTestMLP::InputLayer input = { { uv[0], uv[1] } };
	mlp.encode_input(input.input, activations_buffer);

	// Save layer 0 (frequency-encoded input) activations for the backward pass
	fp16* sample_activations = train_activations + sample_index * TrainingTestMLP::NEURON_COUNT;
	for (unsigned int n = 0; n < TrainingTestMLP::INPUT_SIZE; n++)
		sample_activations[TrainingTestMLP::get_neuron_data_index(0, n)] = activations_buffer[n][threadIdx.x];

	// WMMA forward pass, saves layers 1..LAYER_COUNT-1 activations to global memory
	mlp.forward_train(activations_buffer, train_activations, blockIdx.x * blockDim.x);

	// Read activations for scalar backward
	float activations[TrainingTestMLP::NEURON_COUNT];
	for (unsigned int i = 0; i < TrainingTestMLP::NEURON_COUNT; i++)
		activations[i] = static_cast<float>(sample_activations[i]);

	mlp.backpropagation(activations, target_color);
}

#endif
