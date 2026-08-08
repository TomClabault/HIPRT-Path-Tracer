/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_NEURAL_IMPORTANCE_SAMPLING_OPTIONS_H
#define HOST_DEVICE_COMMON_NEURAL_IMPORTANCE_SAMPLING_OPTIONS_H

#include "Device/includes/Neural/MLPFullyFusedDevice.h"

#define NIS_MAX_CLUSTER_COUNT 64

/**
 * Raw input order for the neural importance sampling MLP:
 * [position.xyz, wo.xyz, normal.xyz]
 *
 * Position is normalized to the scene bounds before being passed to the MLP.
 */
#define NIS_INPUT_SIZE_RAW					   9
#define NIS_FREQUENCY_ENCODING_NUM_FREQUENCIES 8
#define NIS_HIDDEN_LAYER_COUNT				   3
#define NIS_HIDDEN_LAYER_SIZE				   64
#define NIS_THREAD_BLOCK_SIZE				   64
#define NIS_USE_BIASES						   1

using NeuralImportanceSamplingMLP = MLPFullyFusedDevice<NIS_INPUT_SIZE_RAW,
														NIS_FREQUENCY_ENCODING_NUM_FREQUENCIES,
														NIS_HIDDEN_LAYER_COUNT,
														NIS_HIDDEN_LAYER_SIZE,
														NIS_MAX_CLUSTER_COUNT,
														NIS_THREAD_BLOCK_SIZE,
														NIS_USE_BIASES,
														MLPActivationFunction::RELU>;

#endif
