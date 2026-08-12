/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_NISML_TRAIN_H
#define KERNELS_NISML_TRAIN_H

#include "Device/includes/Compute/Common/WarpBlockReduce.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/NISML/NISML.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingManyLightsOptions.h"

#if NISML_GPU
GLOBAL_KERNEL_SIGNATURE(void)
__launch_bounds__(NeuralImportanceSamplingMLP::BLOCK_SIZE)
	NISMLTrain(NeuralImportanceSamplingMLP mlp, HIPRTRenderData render_data, fp16* train_activations, unsigned int training_record_count)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline NISMLTrain(
	NeuralImportanceSamplingMLP mlp, HIPRTRenderData render_data, fp16* train_activations, unsigned int training_record_count, unsigned int record_index)
#endif
{
#if NISML_GPU
	unsigned int record_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif
	unsigned int record_count = hippt::min(training_record_count, render_data.nisml.training_record_capacity);

#if NISML_GPU
	if (blockIdx.x * blockDim.x >= record_count)
		// Early-outing at the block level, not the thread level because we have some __synchthreads() in the kernel and it's UB to not have all threads in a
		// block reach the __synchthreads() call
		return;
#endif

	bool valid_record		   = record_index < record_count;
	bool valid_training_sample = false;

#if !NISML_GPU || !NISML_HAS_WMMA
	NeuralImportanceSamplingMLP::InputLayer input = {};
#endif

#if NISML_GPU
#if NISML_HAS_WMMA
	// The second activation ping-pong region is unused after forward_train_wmma() and is reused for backpropagation errors.
	__shared__ fp16
		training_buffer[NeuralImportanceSamplingMLP::ACTIVATION_WIDTH + NeuralImportanceSamplingMLP::ERROR_WIDTH * 2][NeuralImportanceSamplingMLP::BLOCK_SIZE];
	fp16(*activations_buffer)[NeuralImportanceSamplingMLP::BLOCK_SIZE] = training_buffer;
#else
	__shared__ fp16 activations_buffer[NeuralImportanceSamplingMLP::ACTIVATION_WIDTH * 2][NeuralImportanceSamplingMLP::BLOCK_SIZE];
#endif
#endif

#if NISML_GPU && NISML_HAS_WMMA
	NISMLTrainProfileRecord* profile_record = &render_data.nisml.train_profile_records[blockIdx.x];
	unsigned long long int profile_start	= 0;
	if (threadIdx.x == 0)
		for (unsigned int phase = 0; phase < NISML_TRAIN_PROFILE_PHASE_COUNT; phase++)
			profile_record->phase_durations[phase] = 0;
	__syncthreads();
#endif

	NISMLTrainingSample record;

#if !NISML_GPU || !NISML_HAS_WMMA
	// That local array is only used in the CPU version of the kernel and in the GPU version for non-WMMA
	float neurons_activations[NeuralImportanceSamplingMLP::NEURON_COUNT];
#endif

	if (valid_record)
	{
		record = render_data.nisml.training_records[record_index];
#if !NISML_GPU || !NISML_HAS_WMMA
		build_nisml_input(render_data.nisml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max,
						  record.position, record.outgoing_direction, record.normal, input);
#endif
	}

#if NISML_GPU
#if NISML_HAS_WMMA
	fp16(*errors_buffer)[NeuralImportanceSamplingMLP::BLOCK_SIZE] = &training_buffer[NeuralImportanceSamplingMLP::ACTIVATION_WIDTH];

	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	if (valid_record)
		load_nisml_input_wmma(render_data.nisml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max,
							  record.position, record.outgoing_direction, record.normal, &activations_buffer[0][0], threadIdx.x);
	else
		for (unsigned int input_index = 0; input_index < NeuralImportanceSamplingMLP::INPUT_SIZE_PADDED_WMMA; input_index++)
			activations_buffer[input_index][threadIdx.x] = static_cast<fp16>(0.0f);

	__syncthreads();
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_INPUT_ENCODING, profile_start);
#else
	mlp.load_input(input.input, activations_buffer);
#endif

	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	fp16* sample_activations = train_activations + record_index * NeuralImportanceSamplingMLP::NEURON_COUNT;
	for (unsigned int neuron_index = 0; neuron_index < NeuralImportanceSamplingMLP::INPUT_SIZE_PADDED_WMMA; neuron_index++)
		sample_activations[NeuralImportanceSamplingMLP::get_neuron_data_index(0, neuron_index)] = activations_buffer[neuron_index][threadIdx.x];
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_INPUT_ACTIVATION_STORE, profile_start);

#if NISML_HAS_WMMA
	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
	mlp.forward_train_wmma(activations_buffer, train_activations, blockIdx.x * blockDim.x);
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_FORWARD, profile_start);
#else
	mlp.forward_train(activations_buffer, train_activations, blockIdx.x * blockDim.x);
#endif
#else
	if (!valid_record)
		return;

	mlp.forward_single_thread(input, neurons_activations);
#endif

	// Stores either the output gradients or the probabilities of the clusters, this avoids using two separate arrays when only one can do the job
	float output_gradient_or_probabilities[NISML_MAX_CLUSTER_COUNT] = {};
#if !NISML_GPU || !NISML_HAS_WMMA
	// Not needed in the WMMA path, we can just read the residuals from sample_activations buffer
	float residuals[NISML_MAX_CLUSTER_COUNT] = {};
#endif
	float input_gradients[NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE] = {};
	float* cluster_log_baseline_weights_or_probabilities					= output_gradient_or_probabilities;
	bool valid_softmax														= false;
	bool valid_weight														= false;
	float weight															= 0.0f;

#if NISML_GPU && NISML_HAS_WMMA
	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
#endif
	if (valid_record)
	{
#if NISML_GPU		// GPU
#if !NISML_HAS_WMMA // Non-WMMA
		for (unsigned int neuron_index = 0; neuron_index < NeuralImportanceSamplingMLP::NEURON_COUNT; neuron_index++)
			neurons_activations[neuron_index] = static_cast<float>(sample_activations[neuron_index]);

		for (unsigned int cluster_index = 0; cluster_index < NISML_MAX_CLUSTER_COUNT; cluster_index++)
			residuals[cluster_index] =
				neurons_activations[NeuralImportanceSamplingMLP::get_neuron_data_index(NeuralImportanceSamplingMLP::LAYER_COUNT - 1, cluster_index)];
#endif // !Non-WMMA
#else  // Non GPU
		for (unsigned int cluster_index = 0; cluster_index < NISML_MAX_CLUSTER_COUNT; cluster_index++)
			residuals[cluster_index] =
				neurons_activations[NeuralImportanceSamplingMLP::get_neuron_data_index(NeuralImportanceSamplingMLP::LAYER_COUNT - 1, cluster_index)];
#endif // !GPU
	}

#if NISML_GPU && NISML_HAS_WMMA
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_OUTPUT_RESIDUALS, profile_start);
	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
#endif

	if (valid_record)
	{
		build_nisml_log_baseline_weights(render_data, render_data.nisml, record.position, record.outgoing_direction, record.normal, record.sg_specular_weight,
										 record.alpha_x, record.alpha_y, cluster_log_baseline_weights_or_probabilities);
	}

#if NISML_GPU && NISML_HAS_WMMA
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_BASELINE, profile_start);
	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
#endif

	if (valid_record)
	{
#if NISML_HAS_WMMA
		// On WMMA we can just read the residuals from the sample_activations buffer, no need to copy them to a separate array
		valid_softmax =
			evaluate_nisml_softmax(cluster_log_baseline_weights_or_probabilities,
								   sample_activations + NeuralImportanceSamplingMLP::get_neuron_data_index(NeuralImportanceSamplingMLP::LAYER_COUNT - 1, 0),
								   render_data.nisml.cluster_count);
#else
		valid_softmax = evaluate_nisml_softmax(cluster_log_baseline_weights_or_probabilities, residuals, render_data.nisml.cluster_count);
#endif
	}

#if NISML_GPU && NISML_HAS_WMMA
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_SOFTMAX, profile_start);
	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
#endif

	if (valid_record)
	{
		valid_weight = valid_softmax && record.cluster_index < render_data.nisml.cluster_count && record.cluster_probability > 0.0f &&
					   record.conditional_light_probability > 0.0f && record.point_on_light_pdf_solid_angle > 0.0f;
		if (valid_weight)
		{
			weight = record.cluster_probability > 0.0f ? record.contribution_luminance / (record.cluster_probability * record.conditional_light_probability *
																						  record.point_on_light_pdf_solid_angle)
													   : 0.0f;
		}
	}

#if NISML_GPU && NISML_HAS_WMMA
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_SAMPLE_WEIGHT, profile_start);
	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
#endif

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
		{
			for (unsigned int output_index = 0; output_index < NISML_MAX_CLUSTER_COUNT; output_index++)
				output_gradient_or_probabilities[output_index] = 0.0f;
		}
	}

#if NISML_GPU && NISML_HAS_WMMA
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_OUTPUT_GRADIENT, profile_start);
	NISML_TRAIN_PROFILE_START(profile_record, profile_start);
#endif

#if NISML_GPU
#if NISML_HAS_WMMA
	// Scaling everything by a constant factor to avoid overflow in the FP16 representation of the errors. The maximum error is clamped to 1024.0f so that we
	// have some headroom when during backpropagation (the errors in the hidden layer may grow)
	constexpr float TARGET_MAX_ERROR = 1024.0f;

	float error_scale	 = 1.0f;
	float maximum_weight = block_reduce<NeuralImportanceSamplingMLP::BLOCK_SIZE, float, OperatorMax<float>>(valid_training_sample ? weight : 0.0f);
	if (maximum_weight > TARGET_MAX_ERROR)
		error_scale = TARGET_MAX_ERROR / maximum_weight;
	NISML_TRAIN_PROFILE_STOP(profile_record, NISML_TRAIN_PROFILE_ERROR_SCALE, profile_start);

	mlp.backpropagation_wmma(train_activations, blockIdx.x * blockDim.x, activations_buffer, errors_buffer, input_gradients,
							 NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE, output_gradient_or_probabilities, error_scale, valid_training_sample
#if NISML_HAS_WMMA
							 ,
							 profile_record
#endif
	);
#endif
#else
	if (valid_training_sample)
		mlp.backpropagation_from_output_gradient(neurons_activations, output_gradient_or_probabilities, input_gradients,
												 NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE);
#endif

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

#endif
