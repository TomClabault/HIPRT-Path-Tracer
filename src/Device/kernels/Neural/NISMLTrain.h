/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_NISML_TRAIN_H
#define KERNELS_NISML_TRAIN_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/NISML/NISML.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingManyLightsOptions.h"

#ifndef __KERNELCC__

GLOBAL_KERNEL_SIGNATURE(void)
inline NISMLTrain(
	NeuralImportanceSamplingMLPCPU mlp, HIPRTRenderData render_data, fp16* train_activations, unsigned int training_record_count, unsigned int record_index)
{
	unsigned int record_count = hippt::min(training_record_count, render_data.nisml.training_record_capacity);
	if (record_index >= record_count)
		return;

	NISMLTrainingSample record = render_data.nisml.training_records[record_index];

	NeuralImportanceSamplingMLPCPU::InputLayer input = {};
	build_nisml_input(render_data.nisml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max,
					  record.position, record.outgoing_direction, record.normal, input);

	float neurons_activations[NeuralImportanceSamplingMLPCPU::NEURON_COUNT];
	mlp.forward_single_thread(input, neurons_activations);

	float output_gradient_or_probabilities[NISML_MAX_CLUSTER_COUNT]			= {};
	float* cluster_log_baseline_weights_or_probabilities					= output_gradient_or_probabilities;
	float residuals[NISML_MAX_CLUSTER_COUNT]								= {};
	float input_gradients[NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE] = {};

	for (unsigned int cluster_index = 0; cluster_index < NISML_MAX_CLUSTER_COUNT; cluster_index++)
		residuals[cluster_index] =
			neurons_activations[NeuralImportanceSamplingMLPCPU::get_neuron_data_index(NeuralImportanceSamplingMLPCPU::LAYER_COUNT - 1, cluster_index)];

	build_nisml_log_baseline_weights(render_data, render_data.nisml, record.position, record.outgoing_direction, record.normal, record.sg_specular_weight,
									 record.alpha_x, record.alpha_y, cluster_log_baseline_weights_or_probabilities);

	bool valid_softmax = evaluate_nisml_softmax(cluster_log_baseline_weights_or_probabilities, residuals, render_data.nisml.cluster_count);
	bool valid_weight  = valid_softmax && record.cluster_index < render_data.nisml.cluster_count && record.cluster_probability > 0.0f &&
						record.conditional_light_probability > 0.0f && record.point_on_light_pdf_solid_angle > 0.0f;
	float weight = 0.0f;
	if (valid_weight)
		weight = record.contribution_luminance / (record.cluster_probability * record.conditional_light_probability * record.point_on_light_pdf_solid_angle);

	if (valid_weight)
	{
		for (unsigned int output_index = 0; output_index < render_data.nisml.cluster_count; output_index++)
			output_gradient_or_probabilities[output_index] =
				weight * (cluster_log_baseline_weights_or_probabilities[output_index] - (output_index == record.cluster_index ? 1.0f : 0.0f));

		mlp.backpropagation_from_output_gradient(neurons_activations, output_gradient_or_probabilities, input_gradients,
												 NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE);

		float3_t normalized_position = make_float3(
			(record.position.x - render_data.world_settings.scene_min.x) / (render_data.world_settings.scene_max.x - render_data.world_settings.scene_min.x),
			(record.position.y - render_data.world_settings.scene_min.y) / (render_data.world_settings.scene_max.y - render_data.world_settings.scene_min.y),
			(record.position.z - render_data.world_settings.scene_min.z) / (render_data.world_settings.scene_max.z - render_data.world_settings.scene_min.z));
		accumulate_nisml_position_grid_input_gradients(render_data.nisml.position_learnable_dense_grid, normalized_position, input_gradients);
	}
}

#else
#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char NISML_RENDER_DATA[sizeof(HIPRTRenderData)];
}
#endif
#if NISML_HAS_WMMA

#include "Device/includes/Compute/Common/WarpBlockReduce.h"

GLOBAL_KERNEL_SIGNATURE(void)
__launch_bounds__(NeuralImportanceSamplingMLPGPU::BLOCK_SIZE)
	NISMLTrain(NeuralImportanceSamplingMLPGPU mlp, fp16* train_activations, unsigned int training_record_count)
{
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(NISML_RENDER_DATA);
	unsigned int record_index	 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int record_count	 = hippt::min(training_record_count, render_data.nisml.training_record_capacity);

	if (blockIdx.x * blockDim.x >= record_count)
		// Early-outing at the block level, not the thread level because we have some __synchthreads() in the kernel and it's UB to not have all threads in a
		// block reach the __synchthreads() call
		return;

	bool valid_record		   = record_index < record_count;
	bool valid_training_sample = false;

	// The second activation ping-pong region is unused after forward_train_wmma() and is reused for backpropagation errors.
	// The error buffer is shifted into the activation region that is no longer live during backpropagation.
	constexpr unsigned int ERROR_BUFFER_OFFSET =
		hippt::max(NeuralImportanceSamplingMLPGPU::ERROR_WIDTH, NeuralImportanceSamplingMLPGPU::ACTIVATION_WIDTH - NeuralImportanceSamplingMLPGPU::ERROR_WIDTH);
	__shared__ fp16 training_buffer[ERROR_BUFFER_OFFSET + NeuralImportanceSamplingMLPGPU::ERROR_WIDTH * 2][NeuralImportanceSamplingMLPGPU::BLOCK_SIZE];
	fp16(*activations_buffer)[NeuralImportanceSamplingMLPGPU::BLOCK_SIZE] = training_buffer;

	NISMLTrainProfileRecord* profile_record = &render_data.nisml.train_profile_records[blockIdx.x];
	unsigned long long int profile_start	= 0;
	if (threadIdx.x == 0)
		for (unsigned int phase = 0; phase < NISML_TRAIN_PROFILE_PHASE_COUNT; phase++)
			profile_record->phase_durations[phase] = 0;
	__syncthreads();

	NISMLTrainingSample record;
	if (valid_record)
		record = render_data.nisml.training_records[record_index];

	fp16(*errors_buffer)[NeuralImportanceSamplingMLPGPU::BLOCK_SIZE] = &training_buffer[ERROR_BUFFER_OFFSET];
	float* output_probabilities_buffer								 = reinterpret_cast<float*>(&training_buffer[0][0]);
	// The float probability matrix occupies the first 128 fp16 rows, so place output errors in the non-overlapping second error bank.
	unsigned int output_errors_offset = NeuralImportanceSamplingMLPGPU::ERROR_WIDTH;

	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	if (valid_record)
		load_nisml_input_wmma(render_data.nisml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max,
							  record.position, record.outgoing_direction, record.normal, &activations_buffer[0][0], threadIdx.x);
	else
		for (unsigned int input_index = 0; input_index < NeuralImportanceSamplingMLPGPU::INPUT_SIZE_PADDED_WMMA; input_index++)
			activations_buffer[input_index][threadIdx.x] = static_cast<fp16>(0.0f);

	__syncthreads();
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_INPUT_ENCODING, profile_start);

	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	unsigned int activation_block_base = blockIdx.x * NeuralImportanceSamplingMLPGPU::NEURON_COUNT * NeuralImportanceSamplingMLPGPU::BLOCK_SIZE;
	for (unsigned int neuron_index = 0; neuron_index < NeuralImportanceSamplingMLPGPU::INPUT_SIZE_PADDED_WMMA; neuron_index++)
		train_activations[activation_block_base + neuron_index * NeuralImportanceSamplingMLPGPU::BLOCK_SIZE + threadIdx.x] =
			activations_buffer[neuron_index][threadIdx.x];
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_INPUT_ACTIVATION_STORE, profile_start);

	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	mlp.forward_train_wmma(activations_buffer, train_activations, blockIdx.x * blockDim.x);
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_FORWARD, profile_start);

	float input_gradients[NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE] = {};
	bool valid_softmax														= false;
	bool valid_weight														= false;
	float weight															= 0.0f;

	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_OUTPUT_RESIDUALS, profile_start);

	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	if (valid_record)
		build_nisml_log_baseline_weights(render_data, render_data.nisml, record.position, record.outgoing_direction, record.normal, record.sg_specular_weight,
										 record.alpha_x, record.alpha_y, output_probabilities_buffer, NeuralImportanceSamplingMLPGPU::BLOCK_SIZE, threadIdx.x);
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_BASELINE, profile_start);

	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	if (valid_record)
	{
		constexpr unsigned int output_layer_offset = NeuralImportanceSamplingMLPGPU::get_neuron_data_index(NeuralImportanceSamplingMLPGPU::LAYER_COUNT - 1, 0);
		unsigned int output_activation_block_base  = blockIdx.x * NeuralImportanceSamplingMLPGPU::NEURON_COUNT * NeuralImportanceSamplingMLPGPU::BLOCK_SIZE;
		fp16* residuals = train_activations + output_activation_block_base + output_layer_offset * NeuralImportanceSamplingMLPGPU::BLOCK_SIZE;

		valid_softmax = evaluate_nisml_softmax(output_probabilities_buffer, residuals, render_data.nisml.cluster_count,
											   // Baseline/probability layout:
											   NeuralImportanceSamplingMLPGPU::BLOCK_SIZE, threadIdx.x,
											   // Residual layout:
											   NeuralImportanceSamplingMLPGPU::BLOCK_SIZE, threadIdx.x);
	}
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_SOFTMAX, profile_start);

	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	if (valid_record)
	{
		valid_weight = valid_softmax && record.cluster_index < render_data.nisml.cluster_count && record.cluster_probability > 0.0f &&
					   record.conditional_light_probability > 0.0f && record.point_on_light_pdf_solid_angle > 0.0f;
		if (valid_weight)
			weight = record.cluster_probability > 0.0f ? record.contribution_luminance / (record.cluster_probability * record.conditional_light_probability *
																						  record.point_on_light_pdf_solid_angle)
													   : 0.0f;
	}
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_SAMPLE_WEIGHT, profile_start);

	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	if (valid_record)
	{
		if (valid_weight)
			valid_training_sample = true;
	}

	// Scaling everything by a constant factor to avoid overflow in the FP16 representation of the errors. The maximum error is clamped to 1024.0f so that we
	// have some headroom when during backpropagation (the errors in the hidden layer may grow)
	constexpr float TARGET_MAX_ERROR = 1024.0f;

	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	float error_scale	 = 1.0f;
	float maximum_weight = block_reduce<NeuralImportanceSamplingMLPGPU::BLOCK_SIZE, float, OperatorMax<float>>(valid_training_sample ? weight : 0.0f);
	if (maximum_weight > TARGET_MAX_ERROR)
		error_scale = TARGET_MAX_ERROR / maximum_weight;
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_ERROR_SCALE, profile_start);

	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	for (unsigned int output_index = 0; output_index < NeuralImportanceSamplingMLPGPU::OUTPUT_SIZE_PADDED_WMMA; output_index++)
	{
		float output_gradient = 0.0f;
		if (valid_training_sample && output_index < render_data.nisml.cluster_count)
			output_gradient = weight * (output_probabilities_buffer[output_index * NeuralImportanceSamplingMLPGPU::BLOCK_SIZE + threadIdx.x] -
										(output_index == record.cluster_index ? 1.0f : 0.0f));

		errors_buffer[output_errors_offset + output_index][threadIdx.x] = static_cast<fp16>(output_gradient * error_scale);
	}
	__syncthreads();

	// The probability scratch overlaps the default error bank, so copy the finished output errors after the scratch is no longer needed.
	for (unsigned int output_index = 0; output_index < NeuralImportanceSamplingMLPGPU::OUTPUT_SIZE_PADDED_WMMA; output_index++)
		errors_buffer[output_index][threadIdx.x] = errors_buffer[output_errors_offset + output_index][threadIdx.x];
	__syncthreads();

	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_OUTPUT_GRADIENT, profile_start);

	mlp.backpropagation_wmma<NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE>(train_activations, blockIdx.x * blockDim.x, activations_buffer, errors_buffer,
																			   input_gradients, nullptr, error_scale, valid_training_sample, true,
																			   profile_record);

	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	if (valid_training_sample)
	{
		float3_t normalized_position = make_float3(
			(record.position.x - render_data.world_settings.scene_min.x) / (render_data.world_settings.scene_max.x - render_data.world_settings.scene_min.x),
			(record.position.y - render_data.world_settings.scene_min.y) / (render_data.world_settings.scene_max.y - render_data.world_settings.scene_min.y),
			(record.position.z - render_data.world_settings.scene_min.z) / (render_data.world_settings.scene_max.z - render_data.world_settings.scene_min.z));
		accumulate_nisml_position_grid_input_gradients(render_data.nisml.position_learnable_dense_grid, normalized_position, input_gradients);
	}
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_GRID_GRADIENTS, profile_start);
}

#else

GLOBAL_KERNEL_SIGNATURE(void)
__launch_bounds__(NeuralImportanceSamplingMLPGPU::BLOCK_SIZE)
	NISMLTrain(NeuralImportanceSamplingMLPGPU mlp, fp16* train_activations, unsigned int training_record_count)
{
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(NISML_RENDER_DATA);
	unsigned int record_index	 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int record_count	 = hippt::min(training_record_count, render_data.nisml.training_record_capacity);

	if (blockIdx.x * blockDim.x >= record_count)
		// Early-outing at the block level, not the thread level because we have some __synchthreads() in the kernel and it's UB to not have all threads in a
		// block reach the __synchthreads() call
		return;

	bool valid_record		   = record_index < record_count;
	bool valid_training_sample = false;

	__shared__ fp16 activations_buffer[NeuralImportanceSamplingMLPGPU::ACTIVATION_WIDTH * 2][NeuralImportanceSamplingMLPGPU::BLOCK_SIZE];
	NeuralImportanceSamplingMLPGPU::InputLayer input = {};
	float neurons_activations[NeuralImportanceSamplingMLPGPU::NEURON_COUNT];

	NISMLTrainingSample record;
	if (valid_record)
	{
		record = render_data.nisml.training_records[record_index];
		build_nisml_input(render_data.nisml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max,
						  record.position, record.outgoing_direction, record.normal, input);
	}

	mlp.load_input(input.input, activations_buffer);

	fp16* sample_activations = train_activations + record_index * NeuralImportanceSamplingMLPGPU::NEURON_COUNT;
	for (unsigned int neuron_index = 0; neuron_index < NeuralImportanceSamplingMLPGPU::INPUT_SIZE_PADDED_WMMA; neuron_index++)
		sample_activations[NeuralImportanceSamplingMLPGPU::get_neuron_data_index(0, neuron_index)] = activations_buffer[neuron_index][threadIdx.x];

	mlp.forward_train(activations_buffer, train_activations, blockIdx.x * blockDim.x);

	// Stores either the output gradients or the probabilities of the clusters, this avoids using two separate arrays when only one can do the job
	float output_gradient_or_probabilities[NISML_MAX_CLUSTER_COUNT] = {};
	float* cluster_log_baseline_weights_or_probabilities			= output_gradient_or_probabilities;
	// Not needed in the WMMA path, we can just read the residuals from sample_activations buffer
	float residuals[NISML_MAX_CLUSTER_COUNT] = {};

	float input_gradients[NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE] = {};
	bool valid_softmax														= false;
	bool valid_weight														= false;
	float weight															= 0.0f;

	if (valid_record)
	{
		for (unsigned int neuron_index = 0; neuron_index < NeuralImportanceSamplingMLPGPU::NEURON_COUNT; neuron_index++)
			neurons_activations[neuron_index] = static_cast<float>(sample_activations[neuron_index]);

		for (unsigned int cluster_index = 0; cluster_index < NISML_MAX_CLUSTER_COUNT; cluster_index++)
			residuals[cluster_index] =
				neurons_activations[NeuralImportanceSamplingMLPGPU::get_neuron_data_index(NeuralImportanceSamplingMLPGPU::LAYER_COUNT - 1, cluster_index)];
	}

	if (valid_record)
		build_nisml_log_baseline_weights(render_data, render_data.nisml, record.position, record.outgoing_direction, record.normal, record.sg_specular_weight,
										 record.alpha_x, record.alpha_y, cluster_log_baseline_weights_or_probabilities);

	if (valid_record)
		valid_softmax = evaluate_nisml_softmax(cluster_log_baseline_weights_or_probabilities, residuals, render_data.nisml.cluster_count);

	if (valid_record)
	{
		valid_weight = valid_softmax && record.cluster_index < render_data.nisml.cluster_count && record.cluster_probability > 0.0f &&
					   record.conditional_light_probability > 0.0f && record.point_on_light_pdf_solid_angle > 0.0f;
		if (valid_weight)
			weight = record.cluster_probability > 0.0f ? record.contribution_luminance / (record.cluster_probability * record.conditional_light_probability *
																						  record.point_on_light_pdf_solid_angle)
													   : 0.0f;
	}

	if (valid_record)
	{
		if (valid_weight)
		{
			valid_training_sample = true;
			for (unsigned int output_index = 0; output_index < render_data.nisml.cluster_count; output_index++)
				output_gradient_or_probabilities[output_index] =
					weight * (cluster_log_baseline_weights_or_probabilities[output_index] - (output_index == record.cluster_index ? 1.0f : 0.0f));
		}
		else
			for (unsigned int output_index = 0; output_index < NISML_MAX_CLUSTER_COUNT; output_index++)
				output_gradient_or_probabilities[output_index] = 0.0f;
	}

	if (valid_training_sample)
	{
		float3_t normalized_position = make_float3(
			(record.position.x - render_data.world_settings.scene_min.x) / (render_data.world_settings.scene_max.x - render_data.world_settings.scene_min.x),
			(record.position.y - render_data.world_settings.scene_min.y) / (render_data.world_settings.scene_max.y - render_data.world_settings.scene_min.y),
			(record.position.z - render_data.world_settings.scene_min.z) / (render_data.world_settings.scene_max.z - render_data.world_settings.scene_min.z));
		accumulate_nisml_position_grid_input_gradients(render_data.nisml.position_learnable_dense_grid, normalized_position, input_gradients);
	}
}

#endif

#endif

#endif
