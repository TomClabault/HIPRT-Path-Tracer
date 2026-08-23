/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DEVICE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeCoreDevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterDevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterTrainingSample.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNISMLDevice.h"

struct IlluminationAwareKDTreeDevice
{
	IlluminationAwareKDTreeCoreDevice core;
	IlluminationAwareKDTreeLearningToClusterDevice learning_to_cluster;
	IlluminationAwareKDTreeNISMLDevice nisml;

	unsigned char* any_cell_needs_split = nullptr;
};

#endif
