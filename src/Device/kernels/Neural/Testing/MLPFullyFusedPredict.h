/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_MLP_FULLY_FUSED_PREDICT_H
#define KERNELS_MLP_FULLY_FUSED_PREDICT_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Neural/MLPFullyFusedDevice.h"
#include "HostDeviceCommon/KernelOptions/MLPTrainingTestOptions.h"

using TrainingTestMLP = MLPFullyFusedDevice<
	MLP_TRAINING_TEST_INPUT_SIZE_RAW,
	MLP_TRAINING_TEST_FREQUENCY_ENCODING_NUM_FREQUENCIES,
	MLP_TRAINING_TEST_HIDDEN_LAYER_COUNT,
	MLP_TRAINING_TEST_HIDDEN_LAYER_SIZE,
	MLP_TRAINING_TEST_OUTPUT_SIZE,
	MLP_TRAINING_TEST_THREAD_BLOCK_SIZE>;

GLOBAL_KERNEL_SIGNATURE(void) MLPFullyFusedPredict(TrainingTestMLP mlp, unsigned char* out_predicted_texture, unsigned int width, unsigned int height)
{
	unsigned int global_sample_index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int sample_in_chunk	 = threadIdx.x;

	unsigned int x	   = global_sample_index % width;
	unsigned int y	   = global_sample_index / width;
	bool active_thread = true;
	if (x >= width || y >= height)
		active_thread = false;

	TrainingTestMLP::InputLayer input = { { static_cast<float>(x) / static_cast<float>(width - 1),
											static_cast<float>(y) / static_cast<float>(height - 1) } };

	__shared__ fp16 activations[TrainingTestMLP::HIDDEN_LAYER_SIZE * 2][TrainingTestMLP::BLOCK_SIZE];

	mlp.inference(input, activations);

	TrainingTestMLP::OutputLayer output;
	if (active_thread)
		output = mlp.get_output_layer(activations);

	float r = hippt::clamp(0.0f, 1.0f, output.output[0]);
	float g = hippt::clamp(0.0f, 1.0f, output.output[1]);
	float b = hippt::clamp(0.0f, 1.0f, output.output[2]);

	unsigned int pixel_index = x + y * width;

	out_predicted_texture[pixel_index * 3 + 0] = static_cast<unsigned char>(r * 255.0f);
	out_predicted_texture[pixel_index * 3 + 1] = static_cast<unsigned char>(g * 255.0f);
	out_predicted_texture[pixel_index * 3 + 2] = static_cast<unsigned char>(b * 255.0f);
}

#endif
