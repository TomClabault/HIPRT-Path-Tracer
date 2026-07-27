/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/IlluminationAwareKDTreeRenderPass.h"

#include "HostDeviceCommon/RenderData.h"

#include <algorithm>
#include <vector>

IlluminationAwareKDTreeNode make_debug_node(const uint32_t left_child_index, const uint8_t flags, const uint8_t split_axis, const float split_position)
{
	IlluminationAwareKDTreeNode node;
	node.left_child_index			= left_child_index;
	node.guiding_distribution_index = IlluminationAwareKDTreeNode::INVALID_GUIDING_SLOT;
	node.creation_tag				= IlluminationAwareKDTreeNode::INVALID_CREATION_TAG;
	node.split_position				= split_position;
	node.split_axis					= split_axis;
	node.flags						= flags;
	node.padding					= 0;

	return node;
}

bool debug_nodes_match(const IlluminationAwareKDTreeNode& first, const IlluminationAwareKDTreeNode& second)
{
	return first.left_child_index == second.left_child_index && first.guiding_distribution_index == second.guiding_distribution_index &&
		   first.creation_tag == second.creation_tag && first.split_position == second.split_position && first.split_axis == second.split_axis &&
		   first.flags == second.flags && first.padding == second.padding;
}

float get_bound_component(const float3_t& value, const uint8_t axis)
{
	if (axis == 0)
		return value.x;
	if (axis == 1)
		return value.y;
	return value.z;
}

bool bounds_contain(const IlluminationAwareKDTreeNodeBounds& parent_bounds, const IlluminationAwareKDTreeNodeBounds& child_bounds)
{
	return child_bounds.minimum.x >= parent_bounds.minimum.x && child_bounds.minimum.y >= parent_bounds.minimum.y &&
		   child_bounds.minimum.z >= parent_bounds.minimum.z && child_bounds.maximum.x <= parent_bounds.maximum.x &&
		   child_bounds.maximum.y <= parent_bounds.maximum.y && child_bounds.maximum.z <= parent_bounds.maximum.z;
}

bool position_is_inside_bounds(const IlluminationAwareKDTreeNodeBounds& bounds, const float3_t& position)
{
	return position.x >= bounds.minimum.x && position.y >= bounds.minimum.y && position.z >= bounds.minimum.z && position.x <= bounds.maximum.x &&
		   position.y <= bounds.maximum.y && position.z <= bounds.maximum.z;
}

void report_verification_failure(bool& reported, const char* message)
{
	if (reported)
		return;

	reported = true;
	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Illumination-aware KD-tree verification failed: %s", message);
	Debug::debugbreak();
}

void verify_illumination_aware_kd_tree(const std::vector<IlluminationAwareKDTreeNode>& nodes,
									   const std::vector<IlluminationAwareKDTreeNodeBounds>& node_bounds,
									   const uint32_t node_count,
									   const std::vector<uint32_t>& active_guiding_nodes,
									   const uint32_t active_guiding_node_count,
									   const std::vector<IlluminationAwareKDTreeDirectIlluminationTrainingSample>& training_samples)
{
	bool invalid_child_indices_reported	   = false;
	bool invalid_child_bounds_reported	   = false;
	bool invalid_sibling_bounds_reported   = false;
	bool invalid_depth_reported			   = false;
	bool expanded_level_six_reported	   = false;
	bool excessive_sample_updates_reported = false;
	bool invalid_level_counts_reported	   = false;

	for (uint32_t node_index = 0; node_index < node_count; node_index++)
	{
		const IlluminationAwareKDTreeNode& node = nodes[node_index];
		if (!(node.flags & IlluminationAwareKDTreeNodeFlag_HasChildren))
			continue;

		const uint32_t left_child_index = node.left_child_index;
		if (left_child_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX || left_child_index >= node_count || left_child_index + 1u >= node_count)
		{
			report_verification_failure(invalid_child_indices_reported, "a physical child index is not below node_count");
			continue;
		}

		const IlluminationAwareKDTreeNodeBounds& parent_bounds = node_bounds[node_index];
		const IlluminationAwareKDTreeNodeBounds& left_bounds   = node_bounds[left_child_index];
		const IlluminationAwareKDTreeNodeBounds& right_bounds  = node_bounds[left_child_index + 1u];
		if (!bounds_contain(parent_bounds, left_bounds) || !bounds_contain(parent_bounds, right_bounds))
			report_verification_failure(invalid_child_bounds_reported, "a child bound is not contained in its parent");

		if (node.split_axis > 2 || get_bound_component(left_bounds.maximum, node.split_axis) != node.split_position ||
			get_bound_component(right_bounds.minimum, node.split_axis) != node.split_position)
			report_verification_failure(invalid_sibling_bounds_reported, "sibling bounds do not meet exactly at the parent split");
	}

	std::vector<int> node_levels(node_count, -1);
	std::vector<uint32_t> nodes_to_visit;
	uint32_t level_counts[IlluminationAwareKDTreeMaximumLookaheadDepth + 1u] = {};

	const uint32_t root_count = std::min(active_guiding_node_count, static_cast<uint32_t>(active_guiding_nodes.size()));
	for (uint32_t root_index = 0; root_index < root_count; root_index++)
	{
		const uint32_t node_index = active_guiding_nodes[root_index];
		if (node_index >= node_count)
		{
			report_verification_failure(invalid_child_indices_reported, "an active guiding node index is not below node_count");
			continue;
		}

		node_levels[node_index] = 0;
		nodes_to_visit.push_back(node_index);
	}

	for (std::size_t visit_index = 0; visit_index < nodes_to_visit.size(); visit_index++)
	{
		const uint32_t node_index = nodes_to_visit[visit_index];
		const int node_level	  = node_levels[node_index];
		if (node_level < 0 || node_level > IlluminationAwareKDTreeMaximumLookaheadDepth)
		{
			report_verification_failure(invalid_depth_reported, "a node exists beyond the six-level lookahead depth");
			continue;
		}

		level_counts[node_level]++;
		const IlluminationAwareKDTreeNode& node = nodes[node_index];
		if (!(node.flags & IlluminationAwareKDTreeNodeFlag_HasChildren))
			continue;

		if (node_level == IlluminationAwareKDTreeMaximumLookaheadDepth)
			report_verification_failure(expanded_level_six_reported, "a level-six frontier node has children");

		const uint32_t left_child_index = node.left_child_index;
		if (left_child_index >= node_count || left_child_index + 1u >= node_count)
			continue;

		const uint32_t child_level = static_cast<uint32_t>(node_level + 1);
		for (uint32_t child_offset = 0; child_offset < 2; child_offset++)
		{
			const uint32_t child_index = left_child_index + child_offset;
			if (node_levels[child_index] != -1)
			{
				report_verification_failure(invalid_depth_reported, "the physical tree contains a cycle or shared child");
				continue;
			}

			node_levels[child_index] = static_cast<int>(child_level);
			nodes_to_visit.push_back(child_index);
		}
	}

	for (const IlluminationAwareKDTreeDirectIlluminationTrainingSample& sample : training_samples)
	{
		if (!sample.valid)
			continue;

		uint32_t node_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
		for (uint32_t root_index = 0; root_index < root_count; root_index++)
		{
			const uint32_t root_node_index = active_guiding_nodes[root_index];
			if (root_node_index < node_count && position_is_inside_bounds(node_bounds[root_node_index], sample.position))
			{
				node_index = root_node_index;
				break;
			}
		}

		if (node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			continue;

		uint32_t sample_cell_count = 0;
		while (node_index < node_count)
		{
			const IlluminationAwareKDTreeNode& node = nodes[node_index];
			sample_cell_count++;
			if (!(node.flags & IlluminationAwareKDTreeNodeFlag_HasChildren))
				break;

			if (sample_cell_count >= IlluminationAwareKDTreeMaximumLookaheadDepth + 1u)
			{
				report_verification_failure(excessive_sample_updates_reported, "a sample traverses more than seven cells");
				break;
			}
			if (node.split_axis > 2)
			{
				report_verification_failure(invalid_depth_reported, "a node with children has an invalid split axis");
				break;
			}

			const float* position_components = &sample.position.x;
			const uint32_t left_child_index	 = node.left_child_index;
			node_index						 = position_components[node.split_axis] < node.split_position ? left_child_index : left_child_index + 1u;
		}
	}

	if (active_guiding_node_count == 1 && node_count == 127)
	{
		const uint32_t expected_level_counts[IlluminationAwareKDTreeMaximumLookaheadDepth + 1u] = { 1, 2, 4, 8, 16, 32, 64 };
		for (uint32_t level = 0; level <= IlluminationAwareKDTreeMaximumLookaheadDepth; level++)
			if (level_counts[level] != expected_level_counts[level])
				report_verification_failure(invalid_level_counts_reported,
											"a complete tree does not contain the expected 1, 2, 4, 8, 16, 32, 64 nodes per level");
	}
}

const std::string IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME		= "Illumination-Aware KD-Tree Render Pass";
const std::string IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID						= "Initialize Root Node";
const std::string IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID		= "Accumulate Batch Training Samples";
const std::string IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID = "Accumulate Batch Statistics Into History";
const std::string IlluminationAwareKDTreeRenderPass::RESET_BATCH_STATISTICS_KERNEL_ID					= "Reset Batch Statistics";
const std::string IlluminationAwareKDTreeRenderPass::EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID				= "Expand One Lookahead Level";
const std::string IlluminationAwareKDTreeRenderPass::REPLAY_TRAINING_SAMPLES_KERNEL_ID					= "Replay Training Samples";
const std::string IlluminationAwareKDTreeRenderPass::INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID			= "Initialize Created Node History";
const std::string IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID			= "Mark Guiding Cells For Splitting";

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
}

void IlluminationAwareKDTreeRenderPass::resize(unsigned int new_width, unsigned int new_height) {}

bool IlluminationAwareKDTreeRenderPass::pre_render_update(float delta_time)
{
	if (!is_render_pass_used(*m_renderer->get_global_compiler_options()))
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

	// Returning true because this pass is going to run in post sample update
	return true;
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

	unsigned int* current_frontier					 = illumination_aware_kd_tree.active_guiding_nodes;
	unsigned int* next_frontier						 = m_illumination_aware_kd_tree.m_current_frontier.data();
	AtomicType<unsigned int>* current_frontier_count = illumination_aware_kd_tree.active_guiding_node_count;
	AtomicType<unsigned int>* next_frontier_count	 = m_illumination_aware_kd_tree.m_current_frontier_count.get_atomic_device_pointer();

	bool next_frontier_uses_first_buffer = true;

	for (uint32_t depth = 0; depth < IlluminationAwareKDTreeMaximumLookaheadDepth; depth++)
	{
		illumination_aware_kd_tree.current_frontier		  = current_frontier;
		illumination_aware_kd_tree.current_frontier_count = current_frontier_count;
		illumination_aware_kd_tree.next_frontier		  = next_frontier;
		illumination_aware_kd_tree.next_frontier_count	  = next_frontier_count;
		if (next_frontier_uses_first_buffer)
			m_illumination_aware_kd_tree.m_current_frontier_count.memset_whole_buffer(0u);
		else
			m_illumination_aware_kd_tree.m_next_frontier_count.memset_whole_buffer(0u);

		uint32_t creation_tag		  = m_next_creation_tag++;
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

	void* evaluation_launch_args[] = { &illumination_aware_kd_tree };
	m_kernels[IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID]->launch_asynchronous(
		256, 1, illumination_aware_kd_tree.node_capacity, 1, evaluation_launch_args, m_renderer->get_main_stream());

	// DEBUG block
	{
		run_mark_guiding_cells_for_splitting_debug_check();

		// The verification intentionally synchronizes the stream and downloads the tree because it is only meant for CPU-side development validation.
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

		const std::vector<IlluminationAwareKDTreeNode> nodes =
			m_illumination_aware_kd_tree.m_nodes_and_bounds.get_buffer<ILLUMINATION_AWARE_KD_TREE_NODES>().download_data();
		const std::vector<IlluminationAwareKDTreeNodeBounds> node_bounds =
			m_illumination_aware_kd_tree.m_nodes_and_bounds.get_buffer<ILLUMINATION_AWARE_KD_TREE_NODE_BOUNDS>().download_data();
		const std::vector<unsigned int> node_count_data				   = m_illumination_aware_kd_tree.m_node_count.download_data();
		const std::vector<unsigned int> active_guiding_node_count_data = m_illumination_aware_kd_tree.m_active_guiding_node_count.download_data();

		if (!node_count_data.empty() && !active_guiding_node_count_data.empty())
		{
			const uint32_t node_count						 = std::min(node_count_data[0], static_cast<uint32_t>(nodes.size()));
			const uint32_t active_guiding_node_count		 = active_guiding_node_count_data[0];
			const std::vector<uint32_t> active_guiding_nodes = m_illumination_aware_kd_tree.m_active_guiding_nodes.download_data_partial(
				0, std::min(active_guiding_node_count, static_cast<uint32_t>(m_illumination_aware_kd_tree.m_active_guiding_nodes.size())));

			const std::vector<unsigned int> training_sample_count_data = m_illumination_aware_kd_tree.m_training_sample_count.download_data();
			std::vector<IlluminationAwareKDTreeDirectIlluminationTrainingSample> training_samples;
			if (!training_sample_count_data.empty())
			{
				const uint32_t training_sample_count =
					std::min(training_sample_count_data[0], static_cast<uint32_t>(m_illumination_aware_kd_tree.m_training_samples.size()));
				if (training_sample_count > 0)
					training_samples = m_illumination_aware_kd_tree.m_training_samples.download_data_partial(0, training_sample_count);
			}

			verify_illumination_aware_kd_tree(nodes, node_bounds, node_count, active_guiding_nodes, active_guiding_node_count, training_samples);
		}
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

void IlluminationAwareKDTreeRenderPass::run_mark_guiding_cells_for_splitting_debug_check()
{
	if (m_mark_guiding_cells_debug_check_done)
		return;

	m_mark_guiding_cells_debug_check_done = true;

	constexpr uint32_t SYNTHETIC_NODE_COUNT			   = 7;
	constexpr uint32_t SYNTHETIC_TRIGGERING_NODE_INDEX = 6;

	IlluminationAwareKDTreeDataHost<OrochiBuffer> synthetic_tree;
	synthetic_tree.resize(SYNTHETIC_NODE_COUNT);

	std::vector<IlluminationAwareKDTreeNode> synthetic_nodes(SYNTHETIC_NODE_COUNT);
	synthetic_nodes[0] = make_debug_node(1, IlluminationAwareKDTreeNodeFlag_Guiding | IlluminationAwareKDTreeNodeFlag_HasChildren, 0, 0.5f);
	synthetic_nodes[1] = make_debug_node(3, IlluminationAwareKDTreeNodeFlag_HasChildren, 1, 0.5f);
	synthetic_nodes[2] = make_debug_node(5, IlluminationAwareKDTreeNodeFlag_HasChildren, 1, 0.5f);
	synthetic_nodes[3] = make_debug_node(IlluminationAwareKDTreeNode::INVALID_NODE_INDEX, IlluminationAwareKDTreeNodeFlag_None,
										 IlluminationAwareKDTreeNode::INVALID_SPLIT_AXIS, 0.0f);
	synthetic_nodes[4] = synthetic_nodes[3];
	synthetic_nodes[5] = synthetic_nodes[3];
	synthetic_nodes[6] = synthetic_nodes[3];

	std::vector<IlluminationAwareKDTreeNodeBounds> synthetic_bounds(SYNTHETIC_NODE_COUNT);
	synthetic_bounds[0] = { make_float3(0.0f), make_float3(1.0f) };
	synthetic_bounds[1] = { make_float3(0.0f, 0.0f, 0.0f), make_float3(0.5f, 1.0f, 1.0f) };
	synthetic_bounds[2] = { make_float3(0.5f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f) };
	synthetic_bounds[3] = { make_float3(0.0f, 0.0f, 0.0f), make_float3(0.5f, 0.5f, 1.0f) };
	synthetic_bounds[4] = { make_float3(0.0f, 0.5f, 0.0f), make_float3(0.5f, 1.0f, 1.0f) };
	synthetic_bounds[5] = { make_float3(0.5f, 0.0f, 0.0f), make_float3(1.0f, 0.5f, 1.0f) };
	synthetic_bounds[6] = { make_float3(0.5f, 0.5f, 0.0f), make_float3(1.0f, 1.0f, 1.0f) };

	synthetic_tree.m_nodes_and_bounds.upload_to_buffer_partial<ILLUMINATION_AWARE_KD_TREE_NODES>(0, synthetic_nodes, SYNTHETIC_NODE_COUNT);
	synthetic_tree.m_nodes_and_bounds.upload_to_buffer_partial<ILLUMINATION_AWARE_KD_TREE_NODE_BOUNDS>(0, synthetic_bounds, SYNTHETIC_NODE_COUNT);
	synthetic_tree.m_node_count.upload_data(std::vector<unsigned int>{ SYNTHETIC_NODE_COUNT });
	const std::vector<uint32_t> active_guiding_nodes = { 0 };
	synthetic_tree.m_active_guiding_nodes.upload_data_partial(0, active_guiding_nodes.data(), 1);
	synthetic_tree.m_active_guiding_node_count.upload_data(std::vector<unsigned int>{ 1 });

	std::vector<IlluminationAwareKDTreeIlluminationSignature> synthetic_history_signatures(SYNTHETIC_NODE_COUNT);
	for (IlluminationAwareKDTreeIlluminationSignature& signature : synthetic_history_signatures)
	{
		signature.valid_observation_count	  = 1000;
		signature.scalar_radiance_sum		  = 1000.0f;
		signature.squared_scalar_radiance_sum = 1000.0f;
	}

	// This descendant has a mean radiance of 1.5 while its guiding ancestor has a mean of 1.0.
	synthetic_history_signatures[0].valid_observation_count									  = 2000;
	synthetic_history_signatures[0].scalar_radiance_sum										  = 2000.0f;
	synthetic_history_signatures[0].squared_scalar_radiance_sum								  = 2000.0f;
	synthetic_history_signatures[SYNTHETIC_TRIGGERING_NODE_INDEX].scalar_radiance_sum		  = 1500.0f;
	synthetic_history_signatures[SYNTHETIC_TRIGGERING_NODE_INDEX].squared_scalar_radiance_sum = 2250.0f;
	synthetic_tree.m_history_signatures.upload_data(synthetic_history_signatures);

	IlluminationAwareKDTreeDevice synthetic_device = synthetic_tree.to_device();
	void* launch_arguments[]					   = { &synthetic_device };
	m_kernels[IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID]->launch_asynchronous(
		256, 1, SYNTHETIC_NODE_COUNT, 1, launch_arguments, m_renderer->get_main_stream());

	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

	const std::vector<uint8_t> needs_split		 = synthetic_tree.m_needs_split.download_data_partial(0, 1);
	const std::vector<uint32_t> triggering_nodes = synthetic_tree.m_triggering_lookahead_nodes.download_data_partial(0, 1);
	const std::vector<IlluminationAwareKDTreeNode> nodes_after =
		synthetic_tree.m_nodes_and_bounds.get_buffer<ILLUMINATION_AWARE_KD_TREE_NODES>().download_data_partial(0, SYNTHETIC_NODE_COUNT);

	bool topology_unchanged = nodes_after.size() == synthetic_nodes.size();
	if (topology_unchanged)
	{
		for (uint32_t node_index = 0; node_index < SYNTHETIC_NODE_COUNT; ++node_index)
		{
			if (!debug_nodes_match(nodes_after[node_index], synthetic_nodes[node_index]))
			{
				topology_unchanged = false;
				break;
			}
		}
	}

	const bool root_was_marked			 = needs_split.size() == 1 && needs_split[0] == 1;
	const bool triggering_node_was_found = triggering_nodes.size() == 1 && triggering_nodes[0] == SYNTHETIC_TRIGGERING_NODE_INDEX;
	if (!root_was_marked || !triggering_node_was_found || !topology_unchanged)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Illumination-aware KD-tree MarkGuidingCellsForSplitting debug check failed.");
		Debug::debugbreak();
	}
	else
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Illumination-aware KD-tree MarkGuidingCellsForSplitting debug check passed.");
}

void IlluminationAwareKDTreeRenderPass::reset(bool reset_by_camera_movement)
{
	if (!is_render_pass_used(*m_compiler_options))
		return;

	if (m_illumination_aware_kd_tree.maximum_size() == 0)
		// Nothing to reset
		return;

	m_illumination_aware_kd_tree.reset();
	m_lookahead_frontier_initialized	  = false;
	m_next_creation_tag					  = 0;
	m_mark_guiding_cells_debug_check_done = false;

	IlluminationAwareKDTreeNode* nodes		  = m_illumination_aware_kd_tree.m_nodes_and_bounds.get_buffer<ILLUMINATION_AWARE_KD_TREE_NODES>().data();
	IlluminationAwareKDTreeNodeBounds* bounds = m_illumination_aware_kd_tree.m_nodes_and_bounds.get_buffer<ILLUMINATION_AWARE_KD_TREE_NODE_BOUNDS>().data();
	AtomicType<uint32_t>* node_count		  = m_illumination_aware_kd_tree.m_node_count.get_atomic_device_pointer();
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
