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
};

#endif
