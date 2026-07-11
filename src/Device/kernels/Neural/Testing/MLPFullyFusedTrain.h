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

GLOBAL_KERNEL_SIGNATURE(void)
MLPFullyFusedTrain(TrainingTestMLP mlp, unsigned char* texture, unsigned int tex_w, unsigned int tex_h, unsigned int frame_number)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	Xorshift32Generator rng(x * 19741 + frame_number * 31337);

	float uv[2] = { rng(), rng() };

	unsigned int tx = static_cast<unsigned int>(uv[0] * (tex_w - 1));
	unsigned int ty = static_cast<unsigned int>(uv[1] * (tex_h - 1));
	unsigned int pi = (ty * tex_w + tx) * 3;

	float color[3] = { texture[pi + 0] / 255.0f, texture[pi + 1] / 255.0f, texture[pi + 2] / 255.0f };

	float activations[TrainingTestMLP::NEURON_COUNT];

	TrainingTestMLP::InputLayer input = { { uv[0], uv[1] } };
	mlp.forward_pass(input, activations);

	mlp.backpropagation(activations, color);
}

#endif
