/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/IlluminationAwareKDTreeRenderPass.h"

#include "HostDeviceCommon/RenderData.h"

const std::string IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME		= "Illumination-Aware KD-Tree Render Pass";
const std::string IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID						= "Initialize Root Node";
const std::string IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID		= "Accumulate Batch Training Samples";
const std::string IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID = "Accumulate Batch Statistics Into History";
const std::string IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID					= "Reset Batch Statistics";

IlluminationAwareKDTreeRenderPass::IlluminationAwareKDTreeRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME, renderer, options)
{
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY
																									   "/IlluminationAwareKDTree/InitializeRootNode.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID]->set_kernel_function_name("initialize_illumination_tree_root");
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/AccumulateBatchTrainingSamples.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTreeDevice_AccumulateBatchTrainingSamples");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/AccumulateBatchStatisticsIntoHistory.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTreeDevice_AccumulateBatchStatisticsIntoHistory");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY
																										 "/IlluminationAwareKDTree/ResetBatchStatistics.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID]->set_kernel_function_name("IlluminationAwareKDTreeDevice_ResetBatchStatistics");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});
}

void IlluminationAwareKDTreeRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	if (!is_render_pass_used(*m_compiler_options))
		m_illumination_aware_kd_tree.free();
}

bool IlluminationAwareKDTreeRenderPass::pre_render_update(float delta_time)
{
	if (!is_render_pass_used(*m_compiler_options))
		return m_illumination_aware_kd_tree.free();

	bool render_data_invalidated = false;
	if (m_illumination_aware_kd_tree.maximum_size() == 0)
	{
		m_illumination_aware_kd_tree.resize(IlluminationAwareKDTreeDataHost<OrochiBuffer>::MAXIMUM_NUMBER_OF_NODES);
		render_data_invalidated = true;
	}

	IlluminationAwareKDTreeDevice illumination_aware_kd_tree = m_illumination_aware_kd_tree.to_device();
	void* launch_args[]										 = { &illumination_aware_kd_tree };
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID]->launch_asynchronous(256, 1, illumination_aware_kd_tree.node_capacity, 1,
																										launch_args, m_renderer->get_main_stream());

	return render_data_invalidated;
}

bool IlluminationAwareKDTreeRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return false;

	return false;
}

void IlluminationAwareKDTreeRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return;

	IlluminationAwareKDTreeDevice illumination_aware_kd_tree = render_data.illumination_aware_kd_tree;
	void* launch_args[]										 = { &illumination_aware_kd_tree };

	// TODO maybe download the training_sample_count and launch the kernel with a single thread per sample instead of launching a fixed number of threads and
	// having threads beyond the training_sample_count do nothing? Maybe worth it in perf despite CPU overhead?
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->launch_asynchronous(
		256, 1, illumination_aware_kd_tree.training_sample_capacity, 1, launch_args, m_renderer->get_main_stream());
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->launch_asynchronous(
		256, 1, illumination_aware_kd_tree.node_capacity, 1, launch_args, m_renderer->get_main_stream());
}

void IlluminationAwareKDTreeRenderPass::update_render_data()
{
	if (!is_render_pass_used(*m_compiler_options))
	{
		m_renderer->get_render_data().illumination_aware_kd_tree = {};

		return;
	}

	HIPRTRenderData& render_data = m_renderer->get_render_data();

	render_data.illumination_aware_kd_tree					= m_illumination_aware_kd_tree.to_device();
	render_data.illumination_aware_kd_tree.subdivision_mode = m_subdivision_mode;
}

void IlluminationAwareKDTreeRenderPass::reset(bool reset_by_camera_movement)
{
	if (!is_render_pass_used(*m_compiler_options))
		return;

	if (m_illumination_aware_kd_tree.maximum_size() == 0)
		// Nothing to reset
		return;

	m_illumination_aware_kd_tree.reset();

	HIPRTRenderData& render_data						  = m_renderer->get_render_data();
	render_data.illumination_aware_kd_tree.debug_counters = {};

	IlluminationAwareKDTreeNode* nodes		  = m_illumination_aware_kd_tree.m_nodes_and_bounds.get_buffer<ILLUMINATION_AWARE_KD_TREE_NODES>().data();
	IlluminationAwareKDTreeNodeBounds* bounds = m_illumination_aware_kd_tree.m_nodes_and_bounds.get_buffer<ILLUMINATION_AWARE_KD_TREE_NODE_BOUNDS>().data();
	uint32_t* node_count					  = m_illumination_aware_kd_tree.m_node_count.get_device_pointer();
	uint32_t* active_guiding_nodes			  = m_illumination_aware_kd_tree.m_active_guiding_nodes.get_device_pointer();
	uint32_t* active_guiding_node_count		  = m_illumination_aware_kd_tree.m_active_guiding_node_count.get_device_pointer();
	float3_t scene_bounds_minimum			  = m_renderer->get_scene_metadata().scene_bounding_box.mini;
	float3_t scene_bounds_maximum			  = m_renderer->get_scene_metadata().scene_bounding_box.maxi;
	void* launch_args[] = { &nodes, &bounds, &node_count, &active_guiding_nodes, &active_guiding_node_count, &scene_bounds_minimum, &scene_bounds_maximum };

	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID]->launch_asynchronous(1, 1, 1, 1, launch_args, m_renderer->get_main_stream());
}

bool IlluminationAwareKDTreeRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	// TODO should SG tree + illum aware be a separate DIRECT_LIGHT_SAMPLING_STRATEGY or NEE Estimator? Maybe an estimator, this would avoid bloating all the
	// other estimators with the training stuff
	return compiler_options.get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_ILLUMINATION_AWARE_DISTRIBUTIONS) == KERNEL_OPTION_TRUE &&
		   compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_LIGHT_TREE_SG;
}
