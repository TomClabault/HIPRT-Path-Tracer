/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_NISML_GRID_OPTIMIZE_H
#define KERNELS_NISML_GRID_OPTIMIZE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Neural/NISML/NISMLPositionLearnableDenseGrid.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) NISMLGridOptimize(NISMLPositionLearnableDenseGridDevice grid, unsigned int training_sample_count, unsigned int adam_step)
#else // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline NISMLGridOptimize(NISMLPositionLearnableDenseGridDevice grid, unsigned int training_sample_count, unsigned int adam_step, unsigned int feature_index)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int feature_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif // #ifdef __KERNELCC__
	if (training_sample_count == 0u || feature_index >= NISML_POSITION_LEARNABLE_DENSE_GRID_TOTAL_PARAMETER_COUNT)
		return;

	float time_step		   = static_cast<float>(adam_step + 1u);
	float beta1_correction = 1.0f - powf(NISML_ADAM_BETA1, time_step);
	float beta2_correction = 1.0f - powf(NISML_ADAM_BETA2, time_step);

	float gradient						  = hippt::atomic_load(&grid.gradient_features[feature_index]) / static_cast<float>(training_sample_count);
	grid.gradient_features[feature_index] = 0.0f;

	float mean							   = NISML_ADAM_BETA1 * grid.adam_feature_means[feature_index] + (1.0f - NISML_ADAM_BETA1) * gradient;
	float variance						   = NISML_ADAM_BETA2 * grid.adam_feature_variances[feature_index] + (1.0f - NISML_ADAM_BETA2) * gradient * gradient;
	grid.adam_feature_means[feature_index] = mean;
	grid.adam_feature_variances[feature_index] = variance;

	float corrected_mean	 = mean / beta1_correction;
	float corrected_variance = variance / beta2_correction;
	float updated_feature	 = grid.features[feature_index] - grid.adam_learning_rate * corrected_mean / (sqrtf(corrected_variance) + NISML_ADAM_EPSILON);

	grid.features[feature_index]	  = updated_feature;
	grid.features_fp16[feature_index] = static_cast<fp16>(updated_feature);
}

#endif // #ifndef KERNELS_NISML_GRID_OPTIMIZE_H
