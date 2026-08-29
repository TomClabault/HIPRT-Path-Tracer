/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/NISMLRenderPass.h"

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeOptions.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
	const char* nisml_train_profile_phase_name(unsigned int phase)
	{
		switch (static_cast<NISMLTrainProfilePhase>(phase))
		{
		case NISML_TRAIN_PROFILE_INPUT_ENCODING:
			return "input encoding";
		case NISML_TRAIN_PROFILE_INPUT_ACTIVATION_STORE:
			return "input activation store";
		case NISML_TRAIN_PROFILE_FORWARD:
			return "forward";
		case NISML_TRAIN_PROFILE_OUTPUT_RESIDUALS:
			return "output residuals";
		case NISML_TRAIN_PROFILE_BASELINE:
			return "baseline construction";
		case NISML_TRAIN_PROFILE_SOFTMAX:
			return "softmax";
		case NISML_TRAIN_PROFILE_SAMPLE_WEIGHT:
			return "sample weight";
		case NISML_TRAIN_PROFILE_OUTPUT_GRADIENT:
			return "output gradient";
		case NISML_TRAIN_PROFILE_ERROR_SCALE:
			return "error scale";
		case NISML_TRAIN_PROFILE_OUTPUT_ERROR_INITIALIZATION:
			return "output error initialization";
		case NISML_TRAIN_PROFILE_ACTIVATION_RELOAD:
			return "activation reload";
		case NISML_TRAIN_PROFILE_WEIGHT_GRADIENTS:
			return "weight gradients";
		case NISML_TRAIN_PROFILE_BIAS_GRADIENTS:
			return "bias gradients";
		case NISML_TRAIN_PROFILE_ERROR_PROPAGATION:
			return "error propagation";
		case NISML_TRAIN_PROFILE_INPUT_GRADIENTS:
			return "input gradients";
		case NISML_TRAIN_PROFILE_GRID_GRADIENTS:
			return "grid gradients";
		default:
			return "unknown";
		}
	}

	const char* nisml_train_profile_timer_name()
	{
#if NISML_TRAIN_PROFILE_USE_CLOCK
		return "clock ticks";
#elif NISML_TRAIN_PROFILE_USE_CLOCK64
		return "clock64 ticks";
#else
		return "wall_clock64 ticks";
#endif // #if NISML_TRAIN_PROFILE_USE_CLOCK
	}

	struct NISMLTrainProfileSummary
	{
		const char* name;
		double average_ticks;
		unsigned long long int minimum_ticks;
		unsigned long long int maximum_ticks;
		unsigned long long int total_ticks;
	};

	bool compare_nisml_train_profile_summaries(const NISMLTrainProfileSummary& left, const NISMLTrainProfileSummary& right)
	{
		return left.average_ticks > right.average_ticks;
	}
} // namespace

const std::string NISMLRenderPass::NISML_RENDER_PASS_NAME = "Neural Importance Sampling Many Lights";
const std::string NISMLRenderPass::NISML_TRAIN			  = "NIS Many Lights Train";
const std::string NISMLRenderPass::NISML_OPTIMIZE		  = "NIS Many Lights Optimize";
const std::string NISMLRenderPass::NISML_GRID_OPTIMIZE	  = "NIS Many Lights Grid Optimize";

NISMLRenderPass::NISMLRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(NISMLRenderPass::NISML_RENDER_PASS_NAME, renderer, options)
{
	m_render_data_host_pinned.resize_host_pinned_mem(1);
	m_kernels[NISMLRenderPass::NISML_TRAIN] = std::make_shared<GPUKernel>(this->get_name() + "::" + NISMLRenderPass::NISML_TRAIN);
	m_kernels[NISMLRenderPass::NISML_TRAIN]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/NISMLTrain.h");
	m_kernels[NISMLRenderPass::NISML_TRAIN]->set_kernel_function_name("NISMLTrain");
	m_kernels[NISMLRenderPass::NISML_TRAIN]->synchronize_options_with(m_compiler_options, {});

	m_kernels[NISMLRenderPass::NISML_OPTIMIZE] = std::make_shared<GPUKernel>(this->get_name() + "::" + NISMLRenderPass::NISML_OPTIMIZE);
	m_kernels[NISMLRenderPass::NISML_OPTIMIZE]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/NISMLOptimize.h");
	m_kernels[NISMLRenderPass::NISML_OPTIMIZE]->set_kernel_function_name("NISMLOptimize");
	m_kernels[NISMLRenderPass::NISML_OPTIMIZE]->synchronize_options_with(m_compiler_options, {});

	m_kernels[NISMLRenderPass::NISML_GRID_OPTIMIZE] = std::make_shared<GPUKernel>(this->get_name() + "::" + NISMLRenderPass::NISML_GRID_OPTIMIZE);
	m_kernels[NISMLRenderPass::NISML_GRID_OPTIMIZE]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/NISMLGridOptimize.h");
	m_kernels[NISMLRenderPass::NISML_GRID_OPTIMIZE]->set_kernel_function_name("NISMLGridOptimize");
	m_kernels[NISMLRenderPass::NISML_GRID_OPTIMIZE]->synchronize_options_with(m_compiler_options, {});
}

bool NISMLRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
												   const std::vector<hiprtFuncNameSet>& func_name_sets,
												   bool silent,
												   bool use_cache)
{
	if (!is_render_pass_used(*m_compiler_options))
		return false;

	bool updated = false;
	for (auto& name_to_kernel : m_kernels)
	{
		if (name_to_kernel.second->has_been_compiled())
			continue;

		name_to_kernel.second->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
		updated = true;
	}

	return updated;
}

void NISMLRenderPass::resize(unsigned int new_width, unsigned int new_height) {}

bool NISMLRenderPass::pre_frame_render_update(float delta_time)
{
	if (!is_render_pass_used(*m_compiler_options))
		return false;

	bool render_data_needs_update = pre_render_update();

	m_nisml_data.reset();
	update_render_data();

	return render_data_needs_update;
}

bool NISMLRenderPass::pre_render_update()
{
	bool render_data_needs_update = false;
	if (m_mlp.maximum_size() == 0)
	{
		m_mlp.resize(NISMLDataHost<OrochiBuffer>::NISML_TRAINING_BATCH_SIZE);
		m_mlp.initialize(true);

		unsigned int profile_record_count =
			(NISMLDataHost<OrochiBuffer>::NISML_TRAINING_BATCH_SIZE + NeuralImportanceSamplingMLP::BLOCK_SIZE - 1) / NeuralImportanceSamplingMLP::BLOCK_SIZE;
		m_train_profile_records.resize(profile_record_count);

		m_position_learnable_dense_grid.resize();
		m_position_learnable_dense_grid.initialize();

		render_data_needs_update = true;
	}

	unsigned int training_record_buffer_capacity = static_cast<unsigned int>(std::max(m_training_record_buffer_capacity, 1));
	if (m_nisml_data.get_training_record_capacity() != training_record_buffer_capacity)
	{
		m_nisml_data.resize(training_record_buffer_capacity);
		render_data_needs_update = true;
	}

	return render_data_needs_update;
}

bool NISMLRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return false;
	if (!render_data.nisml.learning_enabled || m_training_record_percentage <= 0.0f)
		return true;

	NeuralImportanceSamplingMLP mlp_device = m_mlp.to_device(m_adam_learning_rate);

	unsigned int training_record_count = m_nisml_data.get_effective_training_record_count();
	if (training_record_count == 0)
		return true;

	fp16* train_activations = reinterpret_cast<fp16*>(m_mlp.m_mlp_data.template get_buffer_data_ptr<MLPDataHostBuffers::MLP_TRAIN_ACTIVATIONS>());
	upload_render_data(render_data);
	void* train_launch_args[] = { &mlp_device, &train_activations, &training_record_count };
	m_kernels[NISMLRenderPass::NISML_TRAIN]->launch_asynchronous(NeuralImportanceSamplingMLP::BLOCK_SIZE, 1, training_record_count, 1, train_launch_args,
																 m_renderer->get_main_stream());

	unsigned int training_sample_count = m_mlp.m_mlp_data.template download_buffer<MLPDataHostBuffers::MLP_LAST_TRAINING_SAMPLE_COUNT>()[0];

	print_train_profile(training_record_count);

	if (training_sample_count > 0)
	{
		unsigned int adam_step		 = m_adam_step;
		void* optimize_launch_args[] = { &mlp_device, &adam_step };
		m_kernels[NISMLRenderPass::NISML_OPTIMIZE]->launch_asynchronous(1024, 1, NeuralImportanceSamplingMLP::CONNECTIONS_COUNT, 1, optimize_launch_args,
																		m_renderer->get_main_stream());

		NISMLPositionLearnableDenseGridDevice position_learnable_dense_grid_device = m_position_learnable_dense_grid.to_device(m_adam_learning_rate);
		void* grid_optimize_launch_args[] = { &position_learnable_dense_grid_device, &training_sample_count, &adam_step };
		m_kernels[NISMLRenderPass::NISML_GRID_OPTIMIZE]->launch_asynchronous(1024, 1, NISML_POSITION_LEARNABLE_DENSE_GRID_TOTAL_PARAMETER_COUNT, 1,
																			 grid_optimize_launch_args, m_renderer->get_main_stream());
		m_adam_step++;
	}

	m_mlp.m_mlp_data.template memset_buffer<MLPDataHostBuffers::MLP_LAST_TRAINING_SAMPLE_COUNT>(0);

	return true;
}

void NISMLRenderPass::upload_render_data(HIPRTRenderData& render_data)
{
	HIPRTRenderData* host_pinned_render_data = m_render_data_host_pinned.get_host_pinned_pointer();
	*host_pinned_render_data				 = render_data;
	m_kernels[NISMLRenderPass::NISML_TRAIN]->upload_to_module_global("NISML_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData),
																	 m_renderer->get_main_stream());
}

void NISMLRenderPass::print_train_profile(unsigned int training_record_count)
{
#if defined(NISML_TRAIN_PROFILING_DISABLED)
	return;
#endif

	unsigned int profile_record_count	 = (training_record_count + NeuralImportanceSamplingMLP::BLOCK_SIZE - 1) / NeuralImportanceSamplingMLP::BLOCK_SIZE;
	unsigned int profile_record_capacity = static_cast<unsigned int>(m_train_profile_records.get_element_count());
	profile_record_count				 = std::min(profile_record_count, profile_record_capacity);
	if (profile_record_count == 0)
		return;

	std::vector<NISMLTrainProfileRecord> profile_records				= m_train_profile_records.download_data_partial(0, profile_record_count);
	unsigned long long int total_ticks[NISML_TRAIN_PROFILE_PHASE_COUNT] = {};
	unsigned long long int minimum_ticks[NISML_TRAIN_PROFILE_PHASE_COUNT];
	unsigned long long int maximum_ticks[NISML_TRAIN_PROFILE_PHASE_COUNT] = {};
	unsigned int measured_block_count[NISML_TRAIN_PROFILE_PHASE_COUNT]	  = {};

	for (unsigned int phase = 0; phase < NISML_TRAIN_PROFILE_PHASE_COUNT; phase++)
		minimum_ticks[phase] = std::numeric_limits<unsigned long long int>::max();

	for (const NISMLTrainProfileRecord& profile_record : profile_records)
	{
		for (unsigned int phase = 0; phase < NISML_TRAIN_PROFILE_PHASE_COUNT; phase++)
		{
			unsigned long long int phase_ticks = profile_record.phase_durations[phase];
			total_ticks[phase] += phase_ticks;
			if (phase_ticks == 0)
				continue;

			minimum_ticks[phase] = std::min(minimum_ticks[phase], phase_ticks);
			maximum_ticks[phase] = std::max(maximum_ticks[phase], phase_ticks);
			measured_block_count[phase]++;
		}
	}

	std::vector<NISMLTrainProfileSummary> summaries;
	summaries.reserve(NISML_TRAIN_PROFILE_PHASE_COUNT);

	double sum_minimum_ticks = 0.0;
	double sum_maximum_ticks = 0.0;
	double sum_average_ticks = 0.0;
	for (unsigned int phase = 0; phase < NISML_TRAIN_PROFILE_PHASE_COUNT; phase++)
	{
		unsigned long long int minimum_phase_ticks = measured_block_count[phase] > 0 ? minimum_ticks[phase] : 0;
		unsigned long long int maximum_phase_ticks = measured_block_count[phase] > 0 ? maximum_ticks[phase] : 0;
		double average_phase_ticks				   = static_cast<double>(total_ticks[phase]) / static_cast<double>(profile_record_count);

		sum_minimum_ticks += static_cast<double>(minimum_phase_ticks);
		sum_maximum_ticks += static_cast<double>(maximum_phase_ticks);
		sum_average_ticks += average_phase_ticks;

		summaries.push_back({ nisml_train_profile_phase_name(phase), average_phase_ticks, minimum_phase_ticks, maximum_phase_ticks, total_ticks[phase] });
	}

	summaries.push_back({ "total", sum_average_ticks, static_cast<unsigned long long int>(sum_minimum_ticks),
						  static_cast<unsigned long long int>(sum_maximum_ticks),
						  static_cast<unsigned long long int>(sum_average_ticks * static_cast<double>(profile_record_count)) });

	std::sort(summaries.begin(), summaries.end(), compare_nisml_train_profile_summaries);
	std::size_t phase_name_width = 0;
	for (const NISMLTrainProfileSummary& summary : summaries)
		phase_name_width = std::max(phase_name_width, std::strlen(summary.name));

	std::cout << "[NISMLTrain profile] records=" << training_record_count << ", blocks=" << profile_record_count
			  << ", timer=" << nisml_train_profile_timer_name() << "\n";
	for (const NISMLTrainProfileSummary& summary : summaries)
	{
		std::cout << "  " << std::left << std::setw(static_cast<int>(phase_name_width)) << summary.name << std::right << " | avg=" << std::setw(12)
				  << std::fixed << std::setprecision(2) << summary.average_ticks << " | min=" << std::setw(12) << summary.minimum_ticks
				  << " | max=" << std::setw(12) << summary.maximum_ticks << " | total=" << std::setw(14) << summary.total_ticks << "\n";
	}
}

void NISMLRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) {}

void NISMLRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used(*m_compiler_options))
	{
		render_data.nisml.cluster_node_indices			= nullptr;
		render_data.nisml.triangle_to_cluster			= nullptr;
		render_data.nisml.cluster_node_depths			= nullptr;
		render_data.nisml.cluster_log_baseline_weights	= nullptr;
		render_data.nisml.cluster_count					= 0;
		render_data.nisml.training_records				= nullptr;
		render_data.nisml.training_record_count			= nullptr;
		render_data.nisml.training_record_capacity		= 0;
		render_data.nisml.train_profile_records			= nullptr;
		render_data.nisml.position_learnable_dense_grid = {};
		render_data.nisml.learning_enabled				= false;
		render_data.nisml.training_record_probability	= 0.0f;

		return;
	}

	render_data.nisml.mlp							= m_mlp.to_device(m_adam_learning_rate);
	render_data.nisml.position_learnable_dense_grid = m_position_learnable_dense_grid.to_device(m_adam_learning_rate);
	render_data.nisml.cluster_log_baseline_weights	= nullptr;
	m_renderer->light_tree_sg_builder().get_nisml_data().to_device<OrochiBuffer>(render_data.nisml);

	NISMLDevice training_data				   = m_nisml_data.to_device();
	render_data.nisml.training_records		   = training_data.training_records;
	render_data.nisml.training_record_count	   = training_data.training_record_count;
	render_data.nisml.training_record_capacity = training_data.training_record_capacity;
	render_data.nisml.train_profile_records	   = m_train_profile_records.data();
	render_data.nisml.learning_enabled		   = m_training_spp <= 0 || render_data.render_settings.sample_number < static_cast<unsigned int>(m_training_spp);
	render_data.nisml.training_record_probability = std::clamp(m_training_record_percentage / 100.0f, 0.0f, 1.0f);
}

void NISMLRenderPass::reset(bool reset_by_camera_movement)
{
	m_nisml_data.reset();

	m_mlp.initialize(true);
	m_position_learnable_dense_grid.initialize();

	m_adam_step = 0;
}

bool NISMLRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return ILLUMINATION_AWARE_KD_TREE_IS_NISML(compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR),
											   compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY));
}

float& NISMLRenderPass::get_training_record_percentage()
{
	return m_training_record_percentage;
}

int& NISMLRenderPass::get_training_spp()
{
	return m_training_spp;
}

int& NISMLRenderPass::get_training_record_buffer_capacity()
{
	return m_training_record_buffer_capacity;
}

float& NISMLRenderPass::get_adam_learning_rate()
{
	return m_adam_learning_rate;
}

NISMLVRAMUsage NISMLRenderPass::get_vram_usage_breakdown() const
{
	NISMLVRAMUsage vram_usage;
	vram_usage.neurons_biases	  = GenericSoAHelpers::get_byte_size(m_mlp.m_mlp_data.template get_buffer<MLPDataHostBuffers::MLP_NEURONS_BIASES>());
	vram_usage.gradient_biases	  = GenericSoAHelpers::get_byte_size(m_mlp.m_mlp_data.template get_buffer<MLPDataHostBuffers::MLP_GRADIENT_BIASES>());
	vram_usage.connection_weights = GenericSoAHelpers::get_byte_size(m_mlp.m_mlp_data.template get_buffer<MLPDataHostBuffers::MLP_CONNECTION_WEIGHTS>());
	vram_usage.connection_weights_fp16 =
		GenericSoAHelpers::get_byte_size(m_mlp.m_mlp_data.template get_buffer<MLPDataHostBuffers::MLP_CONNECTION_WEIGHTS_FP16>());
	vram_usage.gradient_weights = GenericSoAHelpers::get_byte_size(m_mlp.m_mlp_data.template get_buffer<MLPDataHostBuffers::MLP_GRADIENT_WEIGHTS>());
	vram_usage.training_sample_count =
		GenericSoAHelpers::get_byte_size(m_mlp.m_mlp_data.template get_buffer<MLPDataHostBuffers::MLP_LAST_TRAINING_SAMPLE_COUNT>());
	vram_usage.adam_weights_means = GenericSoAHelpers::get_byte_size(m_mlp.m_mlp_data.template get_buffer<MLPDataHostBuffers::MLP_ADAM_WEIGHTS_MEANS>());
	vram_usage.adam_weights_variances =
		GenericSoAHelpers::get_byte_size(m_mlp.m_mlp_data.template get_buffer<MLPDataHostBuffers::MLP_ADAM_WEIGHTS_VARIANCES>());
	vram_usage.adam_biases_means	 = GenericSoAHelpers::get_byte_size(m_mlp.m_mlp_data.template get_buffer<MLPDataHostBuffers::MLP_ADAM_BIASES_MEANS>());
	vram_usage.adam_biases_variances = GenericSoAHelpers::get_byte_size(m_mlp.m_mlp_data.template get_buffer<MLPDataHostBuffers::MLP_ADAM_BIASES_VARIANCES>());
	vram_usage.train_activations	 = GenericSoAHelpers::get_byte_size(m_mlp.m_mlp_data.template get_buffer<MLPDataHostBuffers::MLP_TRAIN_ACTIVATIONS>());
	vram_usage.training_records		 = GenericSoAHelpers::get_byte_size(m_nisml_data.m_training_records);
	vram_usage.training_record_count = GenericSoAHelpers::get_byte_size(m_nisml_data.m_training_record_count);
	vram_usage.grid_features		 = GenericSoAHelpers::get_byte_size(
		m_position_learnable_dense_grid.m_grid_data
			.template get_buffer<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_FEATURES>());
	vram_usage.grid_features_fp16 = GenericSoAHelpers::get_byte_size(
		m_position_learnable_dense_grid.m_grid_data
			.template get_buffer<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_FEATURES_FP16>());
	vram_usage.grid_gradient_features = GenericSoAHelpers::get_byte_size(
		m_position_learnable_dense_grid.m_grid_data
			.template get_buffer<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_GRADIENT_FEATURES>());
	vram_usage.grid_adam_feature_means = GenericSoAHelpers::get_byte_size(
		m_position_learnable_dense_grid.m_grid_data
			.template get_buffer<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_ADAM_FEATURE_MEANS>());
	vram_usage.grid_adam_feature_variances = GenericSoAHelpers::get_byte_size(
		m_position_learnable_dense_grid.m_grid_data
			.template get_buffer<NISMLPositionLearnableDenseGridDataHostBuffers::NISML_POSITION_LEARNABLE_DENSE_GRID_ADAM_FEATURE_VARIANCES>());

	return vram_usage;
}

std::size_t NISMLRenderPass::get_vram_usage_bytes() const
{
	return get_vram_usage_breakdown().get_total_bytes();
}
