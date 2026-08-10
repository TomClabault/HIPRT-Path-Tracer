/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_NIS_ML_TRAIN_H
#define KERNELS_NIS_ML_TRAIN_H

#include "Device/includes/Compute/Common/WarpBlockReduce.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/NISML/NISML.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingOptions.h"

GLOBAL_KERNEL_SIGNATURE(void)
__launch_bounds__(NeuralImportanceSamplingMLP::BLOCK_SIZE) NISMLTrain(NeuralImportanceSamplingMLP mlp, HIPRTRenderData render_data, fp16* train_activations)
{
	unsigned int record_index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int record_count = hippt::min(hippt::atomic_load(render_data.nis_ml.training_record_count), render_data.nis_ml.training_record_capacity);

	bool valid_record		   = record_index < record_count;
	bool valid_training_sample = false;

	float neurons_activations[NeuralImportanceSamplingMLP::NEURON_COUNT];

	NeuralImportanceSamplingMLP::InputLayer input = {};
	NISTrainingSample record;

	if (valid_record)
	{
		record = render_data.nis_ml.training_records[record_index];
		build_nis_input(render_data.nis_ml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max,
						record.position, record.outgoing_direction, record.normal, input);
	}

	__shared__ fp16 activations_buffer[NeuralImportanceSamplingMLP::ACTIVATION_WIDTH * 2][NeuralImportanceSamplingMLP::BLOCK_SIZE];
	__shared__ fp16 errors_buffer[NeuralImportanceSamplingMLP::ERROR_WIDTH * 2][NeuralImportanceSamplingMLP::BLOCK_SIZE];

	mlp.load_input(input.input, activations_buffer);

	fp16* sample_activations = train_activations + record_index * NeuralImportanceSamplingMLP::NEURON_COUNT;
	for (unsigned int neuron_index = 0; neuron_index < NeuralImportanceSamplingMLP::INPUT_SIZE_PADDED_WMMA; neuron_index++)
		sample_activations[NeuralImportanceSamplingMLP::get_neuron_data_index(0, neuron_index)] = activations_buffer[neuron_index][threadIdx.x];

#if __gfx1100__ || __gfx1101__ || __gfx1102__ || __gfx1200__ || __gfx1201__
	mlp.forward_train_wmma(activations_buffer, train_activations, blockIdx.x * blockDim.x);
	mlp.forward_train(activations_buffer, train_activations, blockIdx.x * blockDim.x);

	float output_gradient[NIS_MAX_CLUSTER_COUNT]  = {};
	float input_gradients[NIS_INPUT_SIZE_ENCODED] = {};
	float weight								  = 0.0f;

	if (valid_record)
	{
		for (unsigned int neuron_index = 0; neuron_index < NeuralImportanceSamplingMLP::NEURON_COUNT; neuron_index++)
			neurons_activations[neuron_index] = static_cast<float>(sample_activations[neuron_index]);

		float residuals[NIS_MAX_CLUSTER_COUNT];
		for (unsigned int cluster_index = 0; cluster_index < NIS_MAX_CLUSTER_COUNT; cluster_index++)
			residuals[cluster_index] =
				neurons_activations[NeuralImportanceSamplingMLP::get_neuron_data_index(NeuralImportanceSamplingMLP::LAYER_COUNT - 1, cluster_index)];

		float log_baseline_weights[NIS_MAX_CLUSTER_COUNT];
		build_nis_log_baseline_weights(render_data, render_data.nis_ml, record.position, record.outgoing_direction, record.normal, record.sg_specular_weight,
									   record.alpha_x, record.alpha_y, log_baseline_weights);

		float probabilities[NIS_MAX_CLUSTER_COUNT];
		bool valid_softmax = evaluate_nis_softmax(log_baseline_weights, residuals, render_data.nis_ml.cluster_count, probabilities);

		bool valid_weight = valid_softmax && record.cluster_index < render_data.nis_ml.cluster_count && record.cluster_probability > 0.0f &&
							record.conditional_light_probability > 0.0f && record.point_on_light_pdf_solid_angle > 0.0f;
		if (valid_weight)
		{
			weight = record.cluster_probability > 0.0f ? record.contribution_luminance / (record.cluster_probability * record.conditional_light_probability *
																						  record.point_on_light_pdf_solid_angle)
													   : 0.0f;

			valid_training_sample = true;
			for (unsigned int output_index = 0; output_index < render_data.nis_ml.cluster_count; output_index++)
				output_gradient[output_index] = weight * (probabilities[output_index] - (output_index == record.cluster_index ? 1.0f : 0.0f));
		}
	}

#if __gfx1100__ || __gfx1101__ || __gfx1102__ || __gfx1200__ || __gfx1201__
	constexpr float TARGET_MAX_ERROR = 1024.0f;
	float error_scale				 = 1.0f;
	float maximum_weight			 = block_reduce<NeuralImportanceSamplingMLP::BLOCK_SIZE, float, OperatorMax<float>>(valid_training_sample ? weight : 0.0f);
	if (maximum_weight > TARGET_MAX_ERROR)
		error_scale = TARGET_MAX_ERROR / maximum_weight;

	mlp.backpropagation_wmma(train_activations, blockIdx.x * blockDim.x, activations_buffer, errors_buffer, output_gradient, error_scale, valid_training_sample,
							 input_gradients);
#else
	if (valid_training_sample)
		mlp.backpropagation_from_output_gradient(neurons_activations, output_gradient, input_gradients);
#endif

	if (valid_training_sample)
	{
		float3_t normalized_position = make_float3(
			(record.position.x - render_data.world_settings.scene_min.x) / (render_data.world_settings.scene_max.x - render_data.world_settings.scene_min.x),
			(record.position.y - render_data.world_settings.scene_min.y) / (render_data.world_settings.scene_max.y - render_data.world_settings.scene_min.y),
			(record.position.z - render_data.world_settings.scene_min.z) / (render_data.world_settings.scene_max.z - render_data.world_settings.scene_min.z));
		accumulate_nisml_position_grid_input_gradients(render_data.nis_ml.position_learnable_dense_grid, normalized_position, input_gradients);
	}
}

#endif
