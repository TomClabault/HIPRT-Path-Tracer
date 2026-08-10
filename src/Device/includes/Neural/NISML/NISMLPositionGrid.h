/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_NIS_ML_POSITION_GRID_H
#define DEVICE_INCLUDES_NEURAL_NIS_ML_POSITION_GRID_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/AtomicType.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingOptions.h"
#include "HostDeviceCommon/Maths/VecTypes.h"

struct NISPositionGridInterpolation
{
	unsigned int corner_indices[8] = {};
	float corner_weights[8]		   = {};
};

struct NISPositionGridDevice
{
	HIPRT_DEVICE unsigned int get_level_resolution(unsigned int level) const
	{
		return level_resolutions[level];
	}

	HIPRT_DEVICE unsigned int get_level_offset(unsigned int level) const
	{
		return level_offsets[level];
	}

	float* features						 = nullptr;
	fp16* features_fp16					 = nullptr;
	AtomicType<float>* gradient_features = nullptr;
	float* adam_feature_means			 = nullptr;
	float* adam_feature_variances		 = nullptr;
	float adam_learning_rate			 = 0.001f;

	unsigned int level_resolutions[NIS_POSITION_GRID_LEVEL_COUNT] = {};
	unsigned int level_offsets[NIS_POSITION_GRID_LEVEL_COUNT]	  = {};
};

HIPRT_DEVICE inline float nis_position_grid_smoothstep(float value)
{
	return value * value * (3.0f - 2.0f * value);
}

HIPRT_DEVICE inline void compute_nis_position_grid_interpolation(const NISPositionGridDevice& grid,
																 unsigned int level,
																 float3_t normalized_position,
																 NISPositionGridInterpolation& interpolation)
{
	unsigned int resolution			   = grid.get_level_resolution(level);
	float coordinates[3]			   = { hippt::clamp(0.0f, 1.0f, normalized_position.x), hippt::clamp(0.0f, 1.0f, normalized_position.y),
										   hippt::clamp(0.0f, 1.0f, normalized_position.z) };
	unsigned int lower_coordinates[3]  = {};
	float interpolation_coordinates[3] = {};

	for (unsigned int dimension = 0; dimension < 3; dimension++)
	{
		float scaled_coordinate		 = coordinates[dimension] * static_cast<float>(resolution - 1u);
		lower_coordinates[dimension] = static_cast<unsigned int>(floorf(scaled_coordinate));
		if (lower_coordinates[dimension] >= resolution - 1u)
			lower_coordinates[dimension] = resolution - 2u;

		interpolation_coordinates[dimension] = nis_position_grid_smoothstep(scaled_coordinate - static_cast<float>(lower_coordinates[dimension]));
	}

	for (unsigned int corner = 0; corner < 8; corner++)
	{
		unsigned int x						 = lower_coordinates[0] + ((corner >> 0u) & 1u);
		unsigned int y						 = lower_coordinates[1] + ((corner >> 1u) & 1u);
		unsigned int z						 = lower_coordinates[2] + ((corner >> 2u) & 1u);
		interpolation.corner_indices[corner] = ((z * resolution + y) * resolution + x) * NIS_POSITION_GRID_FEATURE_COUNT + grid.get_level_offset(level);

		float x_weight						 = ((corner >> 0u) & 1u) != 0u ? interpolation_coordinates[0] : 1.0f - interpolation_coordinates[0];
		float y_weight						 = ((corner >> 1u) & 1u) != 0u ? interpolation_coordinates[1] : 1.0f - interpolation_coordinates[1];
		float z_weight						 = ((corner >> 2u) & 1u) != 0u ? interpolation_coordinates[2] : 1.0f - interpolation_coordinates[2];
		interpolation.corner_weights[corner] = x_weight * y_weight * z_weight;
	}
}

HIPRT_DEVICE inline void encode_nis_position_grid(const NISPositionGridDevice& grid, float3_t normalized_position, float* encoded_position)
{
	for (unsigned int level = 0; level < NIS_POSITION_GRID_LEVEL_COUNT; level++)
	{
		NISPositionGridInterpolation interpolation;
		compute_nis_position_grid_interpolation(grid, level, normalized_position, interpolation);

		for (unsigned int feature = 0; feature < NIS_POSITION_GRID_FEATURE_COUNT; feature++)
		{
			float feature_value = 0.0f;
			for (unsigned int corner = 0; corner < 8; corner++)
				feature_value += interpolation.corner_weights[corner] * static_cast<float>(grid.features_fp16[interpolation.corner_indices[corner] + feature]);

			encoded_position[level * NIS_POSITION_GRID_FEATURE_COUNT + feature] = feature_value;
		}
	}
}

HIPRT_DEVICE inline void accumulate_nis_position_grid_input_gradients(const NISPositionGridDevice& grid,
																	  float3_t normalized_position,
																	  const float* position_feature_gradients)
{
	for (unsigned int level = 0; level < NIS_POSITION_GRID_LEVEL_COUNT; level++)
	{
		NISPositionGridInterpolation interpolation;
		compute_nis_position_grid_interpolation(grid, level, normalized_position, interpolation);

		for (unsigned int feature = 0; feature < NIS_POSITION_GRID_FEATURE_COUNT; feature++)
		{
			float feature_gradient = position_feature_gradients[level * NIS_POSITION_GRID_FEATURE_COUNT + feature];
			for (unsigned int corner = 0; corner < 8; corner++)
				hippt::atomic_fetch_add(&grid.gradient_features[interpolation.corner_indices[corner] + feature],
										feature_gradient * interpolation.corner_weights[corner]);
		}
	}
}

#endif
