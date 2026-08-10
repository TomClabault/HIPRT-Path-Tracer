/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/NISMLRenderPass.h"

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

#include <algorithm>

const std::string NISMLRenderPass::NISML_RENDER_PASS_NAME = "Neural Importance Sampling Many Lights";
const std::string NISMLRenderPass::NISML_TRAIN			  = "NIS Many Lights Train";
const std::string NISMLRenderPass::NISML_OPTIMIZE		  = "NIS Many Lights Optimize";
const std::string NISMLRenderPass::NISML_GRID_OPTIMIZE	  = "NIS Many Lights Grid Optimize";

NISMLRenderPass::NISMLRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(NISMLRenderPass::NISML_RENDER_PASS_NAME, renderer, options)
{
	m_kernels[NISMLRenderPass::NISML_TRAIN] = std::make_shared<GPUKernel>(this->get_name() + "::" + NISMLRenderPass::NISML_TRAIN);
	m_kernels[NISMLRenderPass::NISML_TRAIN]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/NISMLTrain.h");
	m_kernels[NISMLRenderPass::NISML_TRAIN]->set_kernel_function_name("NISMLTrain");

	m_kernels[NISMLRenderPass::NISML_OPTIMIZE] = std::make_shared<GPUKernel>(this->get_name() + "::" + NISMLRenderPass::NISML_OPTIMIZE);
	m_kernels[NISMLRenderPass::NISML_OPTIMIZE]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/NISMLOptimize.h");
	m_kernels[NISMLRenderPass::NISML_OPTIMIZE]->set_kernel_function_name("NISMLOptimize");

	m_kernels[NISMLRenderPass::NISML_GRID_OPTIMIZE] = std::make_shared<GPUKernel>(this->get_name() + "::" + NISMLRenderPass::NISML_GRID_OPTIMIZE);
	m_kernels[NISMLRenderPass::NISML_GRID_OPTIMIZE]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/NISMLGridOptimize.h");
	m_kernels[NISMLRenderPass::NISML_GRID_OPTIMIZE]->set_kernel_function_name("NISMLGridOptimize");
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

bool NISMLRenderPass::pre_sample_update(float delta_time)
{
	if (!is_render_pass_used(*m_compiler_options))
		return false;

	bool render_data_needs_update = pre_render_update();

	m_nis_ml_data.reset();
	update_render_data();

	return render_data_needs_update;
}

bool NISMLRenderPass::pre_render_update()
{
	bool render_data_needs_update = false;
	if (m_mlp.maximum_size() == 0)
	{
		m_mlp.resize(NISMLDataHost<OrochiBuffer>::NIS_TRAINING_BATCH_SIZE);
		m_mlp.initialize(false);
		m_position_grid.resize();
		m_position_grid.initialize();
		render_data_needs_update = true;
	}

	unsigned int training_record_buffer_capacity = static_cast<unsigned int>(std::max(m_training_record_buffer_capacity, 1));
	if (m_nis_ml_data.get_training_record_capacity() != training_record_buffer_capacity)
	{
		m_nis_ml_data.resize(training_record_buffer_capacity);
		render_data_needs_update = true;
	}

	return render_data_needs_update;
}

bool NISMLRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return false;
	if (!render_data.nis_ml.learning_enabled || m_training_record_percentage <= 0.0f)
		return true;

	NeuralImportanceSamplingMLP mlp_device = m_mlp.to_device(m_adam_learning_rate);

	fp16* train_activations	  = reinterpret_cast<fp16*>(m_mlp.m_mlp_data.template get_buffer_data_ptr<MLPDataHostBuffers::MLP_TRAIN_ACTIVATIONS>());
	void* train_launch_args[] = { &mlp_device, &render_data, &train_activations };
	m_kernels[NISMLRenderPass::NISML_TRAIN]->launch_asynchronous(
		NeuralImportanceSamplingMLP::BLOCK_SIZE, 1, NISMLDataHost<OrochiBuffer>::NIS_TRAINING_BATCH_SIZE, 1, train_launch_args, m_renderer->get_main_stream());

	unsigned int training_sample_count = m_mlp.m_mlp_data.template download_buffer<MLPDataHostBuffers::MLP_LAST_TRAINING_SAMPLE_COUNT>()[0];
	if (training_sample_count > 0)
	{
		unsigned int adam_step		 = m_adam_step;
		void* optimize_launch_args[] = { &mlp_device, &adam_step };
		m_kernels[NISMLRenderPass::NISML_OPTIMIZE]->launch_asynchronous(1024, 1, NeuralImportanceSamplingMLP::CONNECTIONS_COUNT, 1, optimize_launch_args,
																		m_renderer->get_main_stream());

		NISPositionGridDevice position_grid_device = m_position_grid.to_device(m_adam_learning_rate);
		void* grid_optimize_launch_args[]		   = { &position_grid_device, &training_sample_count, &adam_step };
		m_kernels[NISMLRenderPass::NISML_GRID_OPTIMIZE]->launch_asynchronous(1024, 1, NIS_POSITION_GRID_TOTAL_PARAMETER_COUNT, 1, grid_optimize_launch_args,
																			 m_renderer->get_main_stream());
		m_adam_step++;
	}

	m_mlp.m_mlp_data.template memset_buffer<MLPDataHostBuffers::MLP_LAST_TRAINING_SAMPLE_COUNT>(0);

	return true;
}

void NISMLRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) {}

void NISMLRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used(*m_compiler_options))
	{
		render_data.nis_ml.cluster_node_indices			= nullptr;
		render_data.nis_ml.triangle_to_cluster			= nullptr;
		render_data.nis_ml.cluster_node_depths			= nullptr;
		render_data.nis_ml.cluster_log_baseline_weights = nullptr;
		render_data.nis_ml.cluster_count				= 0;
		render_data.nis_ml.training_records				= nullptr;
		render_data.nis_ml.training_record_count		= nullptr;
		render_data.nis_ml.training_record_capacity		= 0;
		render_data.nis_ml.position_grid				= {};
		render_data.nis_ml.learning_enabled				= false;
		render_data.nis_ml.training_record_probability	= 0.0f;

		return;
	}

	render_data.nis_ml.mlp							= m_mlp.to_device(m_adam_learning_rate);
	render_data.nis_ml.position_grid				= m_position_grid.to_device(m_adam_learning_rate);
	render_data.nis_ml.cluster_log_baseline_weights = nullptr;
	m_renderer->light_tree_sg_builder().get_nisml_data().to_device<OrochiBuffer>(render_data.nis_ml);

	NISMLDevice training_data					= m_nis_ml_data.to_device();
	render_data.nis_ml.training_records			= training_data.training_records;
	render_data.nis_ml.training_record_count	= training_data.training_record_count;
	render_data.nis_ml.training_record_capacity = training_data.training_record_capacity;
	render_data.nis_ml.learning_enabled			= m_training_spp <= 0 || render_data.render_settings.sample_number < static_cast<unsigned int>(m_training_spp);
	render_data.nis_ml.training_record_probability = std::clamp(m_training_record_percentage / 100.0f, 0.0f, 1.0f);
}

void NISMLRenderPass::reset(bool reset_by_camera_movement)
{
	m_nis_ml_data.reset();

	m_mlp.initialize(true);
	m_position_grid.initialize();

	m_adam_step = 0;
}

bool NISMLRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR) == LSS_NEURAL_MANY_LIGHTS;
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
	vram_usage.training_records		 = GenericSoAHelpers::get_byte_size(m_nis_ml_data.m_training_records);
	vram_usage.training_record_count = GenericSoAHelpers::get_byte_size(m_nis_ml_data.m_training_record_count);
	vram_usage.grid_features =
		GenericSoAHelpers::get_byte_size(m_position_grid.m_grid_data.template get_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_FEATURES>());
	vram_usage.grid_features_fp16 =
		GenericSoAHelpers::get_byte_size(m_position_grid.m_grid_data.template get_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_FEATURES_FP16>());
	vram_usage.grid_gradient_features = GenericSoAHelpers::get_byte_size(
		m_position_grid.m_grid_data.template get_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_GRADIENT_FEATURES>());
	vram_usage.grid_adam_feature_means = GenericSoAHelpers::get_byte_size(
		m_position_grid.m_grid_data.template get_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_ADAM_FEATURE_MEANS>());
	vram_usage.grid_adam_feature_variances = GenericSoAHelpers::get_byte_size(
		m_position_grid.m_grid_data.template get_buffer<NISPositionGridDataHostBuffers::NIS_POSITION_GRID_ADAM_FEATURE_VARIANCES>());

	return vram_usage;
}

std::size_t NISMLRenderPass::get_vram_usage_bytes() const
{
	return get_vram_usage_breakdown().get_total_bytes();
}
