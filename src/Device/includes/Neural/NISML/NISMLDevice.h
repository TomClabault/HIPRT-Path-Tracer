/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_NIS_ML_DEVICE_H
#define DEVICE_INCLUDES_NEURAL_NIS_ML_DEVICE_H

#include "Device/includes/Neural/NISML.h"
#include "Device/includes/Neural/NISML/NISMLPositionLearnableDenseGrid.h"
#include "HostDeviceCommon/AtomicType.h"
#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingOptions.h"
#include "HostDeviceCommon/Xorshift.h"

struct NISMLDevice
{
	HIPRT_DEVICE void append_training_record(const NISTrainingSample& record, Xorshift32Generator& random_number_generator)
	{
#if DirectLightNEEEstimator != LSS_NEURAL_MANY_LIGHTS
		return;
#endif

		if (!learning_enabled || training_record_probability <= 0.0f || random_number_generator() >= training_record_probability)
			return;

		unsigned int ticket = hippt::atomic_fetch_add(training_record_count, 1u);
		if (ticket < training_record_capacity)
		{
			// The buffer isn't full yet, just append
			training_records[ticket] = record;

			return;
		}

		// The buffer is full, replace randomly
		float replacement_probability = static_cast<float>(training_record_capacity) / static_cast<float>(ticket + 1u);
		if (random_number_generator() >= replacement_probability)
			return;

		unsigned int replacement_index = random_number_generator.random_index(training_record_capacity);

		training_records[replacement_index] = record;
	}

	NeuralImportanceSamplingMLP mlp;
	NISMLPositionLearnableDenseGridDevice position_learnable_dense_grid;

	unsigned int* cluster_node_indices	= nullptr;
	float* cluster_log_baseline_weights = nullptr;

	unsigned char* triangle_to_cluster = nullptr;
	unsigned char* cluster_node_depths = nullptr;
	unsigned int cluster_count		   = 0;

	NISTrainingSample* training_records				= nullptr;
	AtomicType<unsigned int>* training_record_count = nullptr;
	unsigned int training_record_capacity			= 0;

	bool learning_enabled			  = false;
	float training_record_probability = 0.0f;
};

#endif
