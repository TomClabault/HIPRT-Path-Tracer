/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_NIS_ML_OPTIMIZE_H
#define KERNELS_NIS_ML_OPTIMIZE_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingOptions.h"

#define NIS_ADAM_BETA1	 0.9f
#define NIS_ADAM_BETA2	 0.999f
#define NIS_ADAM_EPSILON 1e-8f

GLOBAL_KERNEL_SIGNATURE(void)
NISMLOptimize(NeuralImportanceSamplingMLP mlp, unsigned int adam_step)
{
	unsigned int thread_index		   = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int training_sample_count = hippt::atomic_load(mlp.last_training_sample_count);
	if (training_sample_count == 0u)
		return;

	float time_step		   = static_cast<float>(adam_step + 1u);
	float beta1_correction = 1.0f - powf(NIS_ADAM_BETA1, time_step);
	float beta2_correction = 1.0f - powf(NIS_ADAM_BETA2, time_step);

	if (thread_index < NeuralImportanceSamplingMLP::CONNECTIONS_COUNT)
	{
		float gradient					   = hippt::atomic_load(&mlp.gradient_weights[thread_index]) / static_cast<float>(training_sample_count);
		mlp.gradient_weights[thread_index] = 0.0f;

		float mean								 = NIS_ADAM_BETA1 * mlp.adam_weights_means[thread_index] + (1.0f - NIS_ADAM_BETA1) * gradient;
		float variance							 = NIS_ADAM_BETA2 * mlp.adam_weights_variances[thread_index] + (1.0f - NIS_ADAM_BETA2) * gradient * gradient;
		mlp.adam_weights_means[thread_index]	 = mean;
		mlp.adam_weights_variances[thread_index] = variance;

		float corrected_mean	 = mean / beta1_correction;
		float corrected_variance = variance / beta2_correction;
		float updated_weight = mlp.connection_weights[thread_index] - mlp.adam_learning_rate * corrected_mean / (sqrtf(corrected_variance) + NIS_ADAM_EPSILON);

		mlp.connection_weights[thread_index]	  = updated_weight;
		mlp.connection_weights_fp16[thread_index] = static_cast<fp16>(updated_weight);
	}

	if constexpr (NeuralImportanceSamplingMLP::USE_BIASES)
	{
		if (thread_index < NeuralImportanceSamplingMLP::NEURON_COUNT)
		{
			float gradient					  = hippt::atomic_load(&mlp.gradient_biases[thread_index]) / static_cast<float>(training_sample_count);
			mlp.gradient_biases[thread_index] = 0.0f;

			float mean								= NIS_ADAM_BETA1 * mlp.adam_biases_means[thread_index] + (1.0f - NIS_ADAM_BETA1) * gradient;
			float variance							= NIS_ADAM_BETA2 * mlp.adam_biases_variances[thread_index] + (1.0f - NIS_ADAM_BETA2) * gradient * gradient;
			mlp.adam_biases_means[thread_index]		= mean;
			mlp.adam_biases_variances[thread_index] = variance;

			float corrected_mean	 = mean / beta1_correction;
			float corrected_variance = variance / beta2_correction;
			mlp.neurons_biases[thread_index] -= mlp.adam_learning_rate * corrected_mean / (sqrtf(corrected_variance) + NIS_ADAM_EPSILON);
		}
	}
}

#endif
