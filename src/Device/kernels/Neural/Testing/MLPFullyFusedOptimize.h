/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_MLP_FULLY_FUSED_OPTIMIZE_H
#define KERNELS_MLP_FULLY_FUSED_OPTIMIZE_H

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

#define ADAM_BETA1	 0.9f
#define ADAM_BETA2	 0.999f
#define ADAM_EPSILON 1e-8f

GLOBAL_KERNEL_SIGNATURE(void) MLPFullyFusedOptimize(TrainingTestMLP mlp)
{
	unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int last_training_sample_count = hippt::atomic_load(mlp.last_training_sample_count);

	// Bias correction terms
	float t	  = static_cast<float>(mlp.training_step + 1);
	float b1t = powf(ADAM_BETA1, t);
	float b2t = powf(ADAM_BETA2, t);
	float b1c = 1.0f - b1t;
	float b2c = 1.0f - b2t;

	// Updating weights with Adam
	if (thread_index < TrainingTestMLP::CONNECTIONS_COUNT)
	{
		unsigned int connection_index = thread_index;

		float gradient_weight				   = hippt::atomic_load(&mlp.gradient_weights[connection_index]);
		mlp.gradient_weights[connection_index] = 0.0f;
		float grad							   = gradient_weight / static_cast<float>(last_training_sample_count);

		float mean	   = mlp.adam_weights_means[connection_index];
		float variance = mlp.adam_weights_variances[connection_index];

		mean	 = ADAM_BETA1 * mean + (1.0f - ADAM_BETA1) * grad;
		variance = ADAM_BETA2 * variance + (1.0f - ADAM_BETA2) * grad * grad;

		mlp.adam_weights_means[connection_index]	 = mean;
		mlp.adam_weights_variances[connection_index] = variance;

		float corrected_mean	 = mean / b1c;
		float corrected_variance = variance / b2c;

		float updated_weight = mlp.connection_weights[connection_index] - mlp.adam_learning_rate * corrected_mean / (sqrtf(corrected_variance) + ADAM_EPSILON);
		mlp.connection_weights[connection_index]	  = updated_weight;
		mlp.connection_weights_fp16[connection_index] = static_cast<fp16>(updated_weight);
	}

	// Updating biases with Adam
	if (thread_index < TrainingTestMLP::NEURON_COUNT)
	{
		unsigned int neuron_index = thread_index;

		float gradient_bias				  = hippt::atomic_load(&mlp.gradient_biases[neuron_index]);
		mlp.gradient_biases[neuron_index] = 0.0f;
		float grad						  = gradient_bias / static_cast<float>(last_training_sample_count);

		float mean	   = mlp.adam_biases_means[neuron_index];
		float variance = mlp.adam_biases_variances[neuron_index];

		mean	 = ADAM_BETA1 * mean + (1.0f - ADAM_BETA1) * grad;
		variance = ADAM_BETA2 * variance + (1.0f - ADAM_BETA2) * grad * grad;

		mlp.adam_biases_means[neuron_index]		= mean;
		mlp.adam_biases_variances[neuron_index] = variance;

		float corrected_mean	 = mean / b1c;
		float corrected_variance = variance / b2c;

		mlp.neurons_biases[neuron_index] -= mlp.adam_learning_rate * corrected_mean / (sqrtf(corrected_variance) + ADAM_EPSILON);
	}
}

#endif
