/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/IlluminationAwareKDTreeRenderPass.h"

#include "HostDeviceCommon/RenderData.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

const std::string IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME		= "Illumination-Aware KD-Tree Render Pass";
const std::string IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID								= "Reset Tree";
const std::string IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID		= "Accumulate Batch Training Samples";
const std::string IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID = "Accumulate Batch Statistics Into History";
const std::string IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID					= "Reset Batch Statistics";
const std::string IlluminationAwareKDTreeRenderPass::EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID				= "Expand One Lookahead Level";
const std::string IlluminationAwareKDTreeRenderPass::REPLAY_TRAINING_SAMPLES_KERNEL_ID					= "Replay Training Samples";
const std::string IlluminationAwareKDTreeRenderPass::INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID			= "Initialize Created Node History";
const std::string IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID			= "Mark Guiding Cells For Splitting";
const std::string IlluminationAwareKDTreeRenderPass::PROMOTE_GUIDING_CELLS_KERNEL_ID					= "Promote Guiding Cells";

IlluminationAwareKDTreeRenderPass::IlluminationAwareKDTreeRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME, renderer, options)
{
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/ResetTree.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID]->set_kernel_function_name("IlluminationAwareKDTree_ResetTree");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/AccumulateBatchTrainingSamples.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_AccumulateBatchTrainingSamples");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/AccumulateBatchStatisticsIntoHistory.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_AccumulateBatchStatisticsIntoHistory");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY
																										 "/IlluminationAwareKDTree/ResetBatchStatistics.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID]->set_kernel_function_name("IlluminationAwareKDTree_ResetBatchStatistics");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/ExpandOneLookaheadLevel.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_ExpandOneLookaheadLevel");
	m_kernels[IlluminationAwareKDTreeRenderPass::EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_TRAINING_SAMPLES_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::REPLAY_TRAINING_SAMPLES_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_TRAINING_SAMPLES_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/ReplayTrainingSamplesKernel.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_TRAINING_SAMPLES_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_ReplayTrainingSamplesKernel");
	m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_TRAINING_SAMPLES_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/InitializeCreatedNodeHistoryKernel.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_InitializeCreatedNodeHistoryKernel");
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/MarkGuidingCellsForSplitting.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTreeDevice_MarkGuidingCellsForSplitting");
	m_kernels[IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::PROMOTE_GUIDING_CELLS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::PROMOTE_GUIDING_CELLS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::PROMOTE_GUIDING_CELLS_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY
																										"/IlluminationAwareKDTree/PromoteGuidingCells.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::PROMOTE_GUIDING_CELLS_KERNEL_ID]->set_kernel_function_name("IlluminationAwareKDTree_PromoteGuidingCells");
	m_kernels[IlluminationAwareKDTreeRenderPass::PROMOTE_GUIDING_CELLS_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});
}

void IlluminationAwareKDTreeRenderPass::resize(unsigned int new_width, unsigned int new_height) {}

bool IlluminationAwareKDTreeRenderPass::pre_render_update(float delta_time)
{
	if (!is_render_pass_used(*m_renderer->get_global_compiler_options()))
		return m_illumination_aware_kd_tree.free();

	bool render_data_invalidated = false;
	if (m_buffers_need_reallocation)
	{
		m_illumination_aware_kd_tree.resize(IlluminationAwareKDTreeDataHost<OrochiBuffer>::MAXIMUM_NUMBER_OF_NODES, m_training_sample_buffer_capacity);

		m_buffers_need_reallocation = false;
		render_data_invalidated		= true;

		reset(false);
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

	// Returning true because this pass is going to run in post sample update
	return true;
}

void IlluminationAwareKDTreeRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return;
	else if (m_frozen_tree)
		// Not doing any work if the tree is frozen
		return;

	IlluminationAwareKDTreeDevice illumination_aware_kd_tree = render_data.illumination_aware_kd_tree;
	void* launch_args[]										 = { &illumination_aware_kd_tree };

	// TODO maybe download the training_sample_count and launch the kernel with a single thread per sample instead of launching a fixed number of threads and
	// having threads beyond the training_sample_count do nothing? Maybe worth it in perf despite CPU overhead?
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->launch_asynchronous(
		256, 1, illumination_aware_kd_tree.training_sample_capacity, 1, launch_args, m_renderer->get_main_stream());
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->launch_asynchronous(
		256, 1, illumination_aware_kd_tree.node_capacity, 1, launch_args, m_renderer->get_main_stream());

	// TODO make this lower and lower as SPPs progress because we mostly need a lot of iterations at the beginning but then it naturally slows down anyways
	for (int split = 0; split < m_split_iterations_per_SPP; split++)
	{
		ensure_all_lookahead_cell_levels(render_data, compiler_options);

		void* mark_guiding_cell_launch_args[] = { &illumination_aware_kd_tree };
		m_kernels[IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID]->launch_asynchronous(
			256, 1, illumination_aware_kd_tree.node_capacity, 1, mark_guiding_cell_launch_args, m_renderer->get_main_stream());
		// TODO if no cell was marked for splitting, no need to continue this whole loop, we can break

		// TODO this download data could be done with a DtoD async copy of the current guiding count into another 1*unsigned int buffer
		m_cached_current_guiding_node_count = m_illumination_aware_kd_tree.m_active_guiding_node_count.download_data()[0];
		// TODO same here download async
		m_cached_current_node_count	  = m_illumination_aware_kd_tree.m_node_count.download_data()[0];
		void* promotion_launch_args[] = { &illumination_aware_kd_tree, &m_cached_current_guiding_node_count };
		m_kernels[IlluminationAwareKDTreeRenderPass::PROMOTE_GUIDING_CELLS_KERNEL_ID]->launch_asynchronous(
			256, 1, illumination_aware_kd_tree.node_capacity, 1, promotion_launch_args, m_renderer->get_main_stream());
	}
}

void IlluminationAwareKDTreeRenderPass::ensure_all_lookahead_cell_levels(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	IlluminationAwareKDTreeDevice illumination_aware_kd_tree = render_data.illumination_aware_kd_tree;

	unsigned int* current_frontier					 = illumination_aware_kd_tree.active_guiding_nodes;
	unsigned int* next_frontier						 = m_illumination_aware_kd_tree.m_current_frontier.data();
	AtomicType<unsigned int>* current_frontier_count = illumination_aware_kd_tree.active_guiding_node_count;
	AtomicType<unsigned int>* next_frontier_count	 = m_illumination_aware_kd_tree.m_current_frontier_count.get_atomic_device_pointer();

	bool next_frontier_uses_first_buffer = true;

	int max_lookahead_levels = compiler_options.get_macro_value(GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_MAXIMUM_LOOKAHEAD_LEVEL_COUNT);
	for (unsigned int level = 0; level < max_lookahead_levels; level++)
	{
		illumination_aware_kd_tree.current_frontier		  = current_frontier;
		illumination_aware_kd_tree.current_frontier_count = current_frontier_count;
		illumination_aware_kd_tree.next_frontier		  = next_frontier;
		illumination_aware_kd_tree.next_frontier_count	  = next_frontier_count;
		if (next_frontier_uses_first_buffer)
			m_illumination_aware_kd_tree.m_current_frontier_count.memset_whole_buffer(0u);
		else
			m_illumination_aware_kd_tree.m_next_frontier_count.memset_whole_buffer(0u);

		unsigned int creation_tag	  = m_next_creation_tag++;
		void* expansion_launch_args[] = { &illumination_aware_kd_tree, &creation_tag };
		m_kernels[IlluminationAwareKDTreeRenderPass::EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID]->launch_asynchronous(
			256, 1, illumination_aware_kd_tree.node_capacity, 1, expansion_launch_args, m_renderer->get_main_stream());
		m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_TRAINING_SAMPLES_KERNEL_ID]->launch_asynchronous(
			256, 1, illumination_aware_kd_tree.training_sample_capacity, 1, expansion_launch_args, m_renderer->get_main_stream());
		m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID]->launch_asynchronous(
			256, 1, illumination_aware_kd_tree.node_capacity, 1, expansion_launch_args, m_renderer->get_main_stream());

		current_frontier	   = next_frontier;
		current_frontier_count = next_frontier_count;

		if (next_frontier_uses_first_buffer)
		{
			next_frontier		= m_illumination_aware_kd_tree.m_next_frontier.data();
			next_frontier_count = m_illumination_aware_kd_tree.m_next_frontier_count.get_atomic_device_pointer();
		}
		else
		{
			next_frontier		= m_illumination_aware_kd_tree.m_current_frontier.data();
			next_frontier_count = m_illumination_aware_kd_tree.m_current_frontier_count.get_atomic_device_pointer();
		}

		next_frontier_uses_first_buffer = !next_frontier_uses_first_buffer;
	}
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

	if (m_frozen_tree)
		// If the tree is frozen, we don't want to reset it even if the camera moves. Useful for debugging to see how the tree is subdivided over the scene by
		// moving around
		return;

	m_illumination_aware_kd_tree.reset();
	m_lookahead_frontier_initialized		 = false;
	m_next_creation_tag						 = 0;
	m_mark_guiding_cells_debug_check_done	 = false;
	m_promote_guiding_cells_debug_check_done = false;

	IlluminationAwareKDTreeDevice kd_tree_device = m_illumination_aware_kd_tree.to_device();
	float3_t scene_bounds_minimum				 = m_renderer->get_scene_metadata().scene_bounding_box.mini;
	float3_t scene_bounds_maximum				 = m_renderer->get_scene_metadata().scene_bounding_box.maxi;
	void* launch_args[]							 = { &kd_tree_device, &scene_bounds_minimum, &scene_bounds_maximum };

	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID]->launch_asynchronous(
		256, 1, m_illumination_aware_kd_tree.m_nodes_and_bounds.maximum_size(), 1, launch_args, m_renderer->get_main_stream());
}

bool IlluminationAwareKDTreeRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	// TODO should SG tree + illum aware be a separate DIRECT_LIGHT_SAMPLING_STRATEGY or NEE Estimator? Maybe an estimator, this would avoid bloating all the
	// other estimators with the training stuff
	return compiler_options.get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_ILLUMINATION_AWARE_DISTRIBUTIONS) == KERNEL_OPTION_TRUE &&
		   compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_LIGHT_TREE_SG;
}

int& IlluminationAwareKDTreeRenderPass::get_split_iterations_per_SPP()
{
	return m_split_iterations_per_SPP;
}

int& IlluminationAwareKDTreeRenderPass::get_training_sample_buffer_capacity()
{
	return m_training_sample_buffer_capacity;
}

std::size_t IlluminationAwareKDTreeRenderPass::get_current_node_buffer_capacity() const
{
	return m_illumination_aware_kd_tree.m_nodes_and_bounds.maximum_size();
}

std::size_t IlluminationAwareKDTreeRenderPass::get_current_node_count() const
{
	return m_cached_current_node_count;
}

std::size_t IlluminationAwareKDTreeRenderPass::get_current_guiding_node_count() const
{
	return m_cached_current_guiding_node_count;
}

void IlluminationAwareKDTreeRenderPass::mark_buffers_need_reallocation()
{
	m_buffers_need_reallocation = true;
}

bool& IlluminationAwareKDTreeRenderPass::get_frozen_tree()
{
	return m_frozen_tree;
}

std::size_t IlluminationAwareKDTreeRenderPass::get_vram_usage_bytes() const
{
	return m_illumination_aware_kd_tree.get_byte_size();
}
