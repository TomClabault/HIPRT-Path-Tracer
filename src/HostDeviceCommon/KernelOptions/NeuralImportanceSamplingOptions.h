/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_NEURAL_IMPORTANCE_SAMPLING_OPTIONS_H
#define HOST_DEVICE_COMMON_NEURAL_IMPORTANCE_SAMPLING_OPTIONS_H

#include "Device/includes/Neural/InputEncodings.h"
#include "Device/includes/Neural/MLPFullyFusedDevice.h"

#define NIS_MAX_CLUSTER_COUNT 64

#define NIS_NORMAL_ONE_BLOB_BIN_COUNT 32
#define NIS_NORMAL_ONE_BLOB_KERNEL	  OneBlobKernel::QUARTIC

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

#define NISML_POSITION_LEARNABLE_DENSE_GRID_TOTAL_PARAMETER_COUNT 4133144

static constexpr unsigned int NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_RESOLUTIONS[NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT] = { 8, 12, 16, 23, 32, 44, 62, 87 };
static constexpr unsigned int NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_OFFSETS[NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT] = { 0, 2048, 8960, 25344, 74012, 205084, 545820, 1499132 };

// Learnable dense grid encoding for the normalized position (x, y, z)
#define NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE (NISML_POSITION_LEARNABLE_DENSE_GRID_LEVEL_COUNT * NISML_POSITION_LEARNABLE_DENSE_GRID_FEATURE_COUNT)
// 16 for spherical harmonics degree 4 encoding of the view direction (tiny cuda nn convention: degree 4 = 16 features)
#define NIS_VIEW_DIRECTION_ENCODED_SIZE 16
// 3 * NIS_NORMAL_ONE_BLOB_BIN_COUNT for the one-blob encoding of the normal (x, y, z)
#define NIS_SURFACE_NORMAL_ENCODED_SIZE (3 * NIS_NORMAL_ONE_BLOB_BIN_COUNT)

#define NIS_INPUT_SIZE_ENCODED (NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE + NIS_VIEW_DIRECTION_ENCODED_SIZE + NIS_SURFACE_NORMAL_ENCODED_SIZE)
#define NIS_HIDDEN_LAYER_COUNT 3
#define NIS_HIDDEN_LAYER_SIZE  64
#define NIS_THREAD_BLOCK_SIZE  64
#define NIS_USE_BIASES		   1

#define NIS_ADAM_BETA1	 0.9f
#define NIS_ADAM_BETA2	 0.999f
#define NIS_ADAM_EPSILON 1e-8f

using NeuralImportanceSamplingMLP = MLPFullyFusedDevice<NIS_INPUT_SIZE_ENCODED,
														NIS_HIDDEN_LAYER_COUNT,
														NIS_HIDDEN_LAYER_SIZE,
														NIS_MAX_CLUSTER_COUNT,
														NIS_THREAD_BLOCK_SIZE,
														NIS_USE_BIASES,
														MLPActivationFunction::RELU,
														false>;

#endif
