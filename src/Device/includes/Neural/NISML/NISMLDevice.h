/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_NIS_ML_DEVICE_H
#define DEVICE_INCLUDES_NEURAL_NIS_ML_DEVICE_H

#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingOptions.h"

struct NISMLDevice
{
	NeuralImportanceSamplingMLP mlp;

	unsigned int* cluster_node_indices	= nullptr;
	float* cluster_log_baseline_weights = nullptr;

	unsigned char* triangle_to_cluster = nullptr;
	unsigned char* cluster_node_depths = nullptr;
	unsigned int cluster_count		   = 0;
};

#endif
