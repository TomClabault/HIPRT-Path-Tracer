/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_NEURAL_IMPORTANCE_SAMPLING_MANY_LIGHTS_OPTIONS_H
#define HOST_DEVICE_COMMON_NEURAL_IMPORTANCE_SAMPLING_MANY_LIGHTS_OPTIONS_H

// Temporary WMMA NISMLTrain profiling. Delete only after removing every profiling reference
#define PROFILING_ENABLED
// Uncomment to disable NISMLTrain profiling instrumentation while keeping the profiling declarations available.
#define NISML_TRAIN_PROFILING_DISABLED

#include "Device/includes/Neural/InputEncodings.h"
#include "Device/includes/Neural/MLPFullyFusedDevice.h"

#define NISML_MAX_CLUSTER_COUNT 64

#define NISML_NORMAL_ONE_BLOB_BIN_COUNT 32
#define NISML_NORMAL_ONE_BLOB_KERNEL	OneBlobKernel::QUARTIC

/**
 * Input order for Neural Importance Sampling Many Lights:
 * [position_learnable_dense_grid_features, outgoing_direction spherical harmonics degree 4, normal_x_one_blob,
 * normal_y_one_blob, normal_z_one_blob]
 *
 * where each normal component is mapped from [-1, 1] to [0, 1] before one-blob encoding.
 * The position grid features are ordered level-major, with all four features of one level contiguous.
 *
 * Position is normalized to the scene bounds before being passed to the grid encoder.
 */

#define NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT		8
#define NISML_POSITION_LEARNABLE_DENSE_GRID_FEATURE_COUNT	4
#define NISML_POSITION_LEARNABLE_DENSE_GRID_BASE_RESOLUTION 8
#define NISML_POSITION_LEARNABLE_DENSE_GRID_PER_LEVEL_SCALE 1.405f

template <unsigned int ElementCount>
struct NISMLPositionLearnableDenseGridConstexprValues
{
	unsigned int values[ElementCount] = {};

	constexpr unsigned int operator[](unsigned int index) const
	{
		return values[index];
	}
};

constexpr unsigned int nisml_position_learnable_dense_grid_level_resolution(unsigned int level)
{
	float resolution = static_cast<float>(NISML_POSITION_LEARNABLE_DENSE_GRID_BASE_RESOLUTION);
	for (unsigned int level_index = 0; level_index < level; level_index++)
		resolution *= NISML_POSITION_LEARNABLE_DENSE_GRID_PER_LEVEL_SCALE;

	unsigned int floored_resolution = static_cast<unsigned int>(resolution);
	return resolution > static_cast<float>(floored_resolution) ? floored_resolution + 1u : floored_resolution;
}

constexpr NISMLPositionLearnableDenseGridConstexprValues<NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT> nisml_position_learnable_dense_grid_level_resolutions()
{
	NISMLPositionLearnableDenseGridConstexprValues<NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT> resolutions;
	for (unsigned int level = 0; level < NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT; level++)
		resolutions.values[level] = nisml_position_learnable_dense_grid_level_resolution(level);

	return resolutions;
}

static constexpr NISMLPositionLearnableDenseGridConstexprValues<NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT>
	NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_RESOLUTIONS = nisml_position_learnable_dense_grid_level_resolutions();

constexpr NISMLPositionLearnableDenseGridConstexprValues<NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT> nisml_position_learnable_dense_grid_level_offsets()
{
	NISMLPositionLearnableDenseGridConstexprValues<NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT> offsets;
	unsigned int offset = 0;
	for (unsigned int level = 0; level < NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT; level++)
	{
		offsets.values[level]	= offset;
		unsigned int resolution = NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_RESOLUTIONS[level];
		offset += resolution * resolution * resolution * NISML_POSITION_LEARNABLE_DENSE_GRID_FEATURE_COUNT;
	}

	return offsets;
}

static constexpr NISMLPositionLearnableDenseGridConstexprValues<NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT>
	NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_OFFSETS = nisml_position_learnable_dense_grid_level_offsets();

constexpr unsigned int nisml_position_learnable_dense_grid_total_parameter_count()
{
	unsigned int total_parameter_count = 0;
	for (unsigned int level = 0; level < NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT; level++)
	{
		unsigned int resolution = NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_RESOLUTIONS[level];
		total_parameter_count += resolution * resolution * resolution * NISML_POSITION_LEARNABLE_DENSE_GRID_FEATURE_COUNT;
	}

	return total_parameter_count;
}

static constexpr unsigned int NISML_POSITION_LEARNABLE_DENSE_GRID_TOTAL_PARAMETER_COUNT = nisml_position_learnable_dense_grid_total_parameter_count();

// Learnable dense grid encoding for the normalized position (x, y, z)
#define NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE (NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT * NISML_POSITION_LEARNABLE_DENSE_GRID_FEATURE_COUNT)
// 16 for spherical harmonics degree 4 encoding of the view direction (tiny cuda nn convention: degree 4 = 16 features)
#define NISML_VIEW_DIRECTION_ENCODED_SIZE 16
// 3 * NISML_NORMAL_ONE_BLOB_BIN_COUNT for the one-blob encoding of the normal (x, y, z)
#define NISML_SURFACE_NORMAL_ENCODED_SIZE (3 * NISML_NORMAL_ONE_BLOB_BIN_COUNT)

#define NISML_INPUT_SIZE_ENCODED (NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE + NISML_VIEW_DIRECTION_ENCODED_SIZE + NISML_SURFACE_NORMAL_ENCODED_SIZE)
#define NISML_HIDDEN_LAYER_COUNT 3
#define NISML_HIDDEN_LAYER_SIZE	 64
#define NISML_THREAD_BLOCK_SIZE	 64
#define NISML_USE_BIASES		 1

#define NISML_ADAM_BETA1   0.9f
#define NISML_ADAM_BETA2   0.999f
#define NISML_ADAM_EPSILON 1e-8f

using NeuralImportanceSamplingMLP = MLPFullyFusedDevice<NISML_INPUT_SIZE_ENCODED,
														NISML_HIDDEN_LAYER_COUNT,
														NISML_HIDDEN_LAYER_SIZE,
														NISML_MAX_CLUSTER_COUNT,
														NISML_THREAD_BLOCK_SIZE,
														NISML_USE_BIASES,
														MLPActivationFunction::RELU,
														false>;

#endif
