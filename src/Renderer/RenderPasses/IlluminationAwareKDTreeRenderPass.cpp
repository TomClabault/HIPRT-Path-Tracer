/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/IlluminationAwareKDTreeRenderPass.h"

#include "HostDeviceCommon/RenderData.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

const std::string IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME = "Illumination-Aware KD-Tree Render Pass";

const std::string IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID							 = "Reset Tree";
const std::string IlluminationAwareKDTreeRenderPass::RESET_TREE_CUT_SAMPLING_DISTRIBUTIONS_KERNEL_ID = "Reset Tree Cut Sampling Distributions";
const std::string IlluminationAwareKDTreeRenderPass::INITIALIZE_GLOBAL_TREE_CUT_PRIOR_SAMPLING_DISTRIBUTION_KERNEL_ID =
	"Initialize Global Tree Cut Prior Sampling Distribution";
const std::string IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_TREE_CUT_SAMPLING_DISTRIBUTION_KERNEL_ID =
	"Initialize Root Tree Cut Sampling Distribution";
const std::string IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID		= "Accumulate Batch Training Samples";
const std::string IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID = "Accumulate Batch Statistics Into History";
const std::string IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_NEE_DISTRIBUTIONS_STATISTICS_KERNEL_ID =
	"Reset Batch KD-Tree And NEE Distributions Statistics";
const std::string IlluminationAwareKDTreeRenderPass::EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID					= "Expand One Lookahead Level";
const std::string IlluminationAwareKDTreeRenderPass::REPLAY_TRAINING_SAMPLES_KERNEL_ID						= "Replay Training Samples";
const std::string IlluminationAwareKDTreeRenderPass::ACCUMULATE_NEE_DISTRIBUTION_TRAINING_RECORDS_KERNEL_ID = "Accumulate NEE Distribution Training Records";
const std::string IlluminationAwareKDTreeRenderPass::REBUILD_ACTIVE_NEE_DISTRIBUTIONS_KERNEL_ID				= "Rebuild Active NEE Distributions";
const std::string IlluminationAwareKDTreeRenderPass::INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID				= "Initialize Created Node History";
const std::string IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID				= "Mark Guiding Cells For Splitting";
const std::string IlluminationAwareKDTreeRenderPass::PROMOTE_GUIDING_CELLS_KERNEL_ID						= "Promote Guiding Cells";

namespace
{
	// BEGIN TEMPORARY SPP 0 NEE DISTRIBUTION DIAGNOSTICS
	void log_spp_0_final_nee_distribution_statistics(const IlluminationAwareKDTreeDataHost<OrochiBuffer>& illumination_aware_kd_tree,
													 unsigned int active_guiding_count,
													 unsigned int tree_cut_size)
	{
		if (active_guiding_count == 0 || tree_cut_size == 0)
			return;

		std::vector<unsigned int> active_guiding_nodes = illumination_aware_kd_tree.m_active_guiding_nodes.download_data_partial(0, active_guiding_count);
		std::vector<IlluminationAwareKDTreeNode> nodes = illumination_aware_kd_tree.m_nodes.download_data();

		unsigned int maximum_guiding_distribution_index = 0;
		for (unsigned int active_guiding_index = 0; active_guiding_index < active_guiding_count; active_guiding_index++)
		{
			unsigned int guiding_node_index = active_guiding_nodes[active_guiding_index];
			if (guiding_node_index >= nodes.size())
			{
				Debug::debugbreak();
				continue;
			}

			unsigned int guiding_distribution_index = nodes[guiding_node_index].guiding_distribution_index;
			if (guiding_distribution_index != IlluminationAwareKDTreeNode::INVALID_GUIDING_DISTRIBUTION_INDEX)
				maximum_guiding_distribution_index = std::max(maximum_guiding_distribution_index, guiding_distribution_index);
		}

		unsigned int distribution_buffer_element_count = (maximum_guiding_distribution_index + 1) * tree_cut_size;
		std::vector<unsigned int> cell_sample_counts =
			illumination_aware_kd_tree.m_history_per_cell_sample_count.download_data_partial(0, maximum_guiding_distribution_index + 1);
		std::vector<unsigned int> batch_selected_slot_counts =
			illumination_aware_kd_tree.m_batch_per_cut_node_sample_count.download_data_partial(0, distribution_buffer_element_count);
		std::vector<unsigned short int> final_probabilities =
			illumination_aware_kd_tree.m_tree_cut_sampling_probabilities.download_data_partial(0, distribution_buffer_element_count);

		std::printf("\n[IlluminationAwareKDTree SPP 0 final NEE distribution diagnostics]\n");
		for (unsigned int active_guiding_index = 0; active_guiding_index < active_guiding_count; active_guiding_index++)
		{
			unsigned int guiding_node_index = active_guiding_nodes[active_guiding_index];
			if (guiding_node_index >= nodes.size())
			{
				Debug::debugbreak();
				continue;
			}

			unsigned int guiding_distribution_index = nodes[guiding_node_index].guiding_distribution_index;
			if (guiding_distribution_index == IlluminationAwareKDTreeNode::INVALID_GUIDING_DISTRIBUTION_INDEX)
			{
				Debug::debugbreak();
				continue;
			}

			unsigned int distribution_offset				   = guiding_distribution_index * tree_cut_size;
			unsigned int number_of_unique_selected_slots	   = 0;
			unsigned int number_of_nonzero_final_probabilities = 0;
			float maximum_probability						   = 0.0f;
			float probability_squared_sum					   = 0.0f;
			float entropy									   = 0.0f;
			bool incoherent_value_detected					   = false;

			float probability_sum = 0.0f;
			for (unsigned int slot = 0; slot < tree_cut_size; slot++)
			{
				unsigned int distribution_slot = distribution_offset + slot;
				if (batch_selected_slot_counts[distribution_slot] > 0)
					number_of_unique_selected_slots++;

				float probability = static_cast<float>(final_probabilities[distribution_slot]) / 65535.0f;
				if (!std::isfinite(probability) || probability < 0.0f || probability > 1.0f)
					incoherent_value_detected = true;

				probability_sum += probability;
				if (probability > 0.0f)
				{
					number_of_nonzero_final_probabilities++;
					maximum_probability = std::max(maximum_probability, probability);
					probability_squared_sum += probability * probability;
					entropy -= probability * std::log(probability);
				}
			}

			float effective_support = probability_squared_sum > 0.0f ? 1.0f / probability_squared_sum : 0.0f;
			if (!std::isfinite(probability_sum) || probability_sum > 1.01f || !std::isfinite(entropy) || !std::isfinite(effective_support))
				incoherent_value_detected = true;

			std::printf(
				"cell=%u node=%u cell_sample_count=%u number_of_unique_selected_slots=%u number_of_nonzero_final_probabilities=%u maximum_probability=%.9g "
				"entropy=%.9g effective_support=%.9g\n",
				active_guiding_index, guiding_node_index, cell_sample_counts[guiding_distribution_index], number_of_unique_selected_slots,
				number_of_nonzero_final_probabilities, maximum_probability, entropy, effective_support);

			if (incoherent_value_detected)
				Debug::debugbreak();
		}
	}
	// END TEMPORARY SPP 0 NEE DISTRIBUTION DIAGNOSTICS

	// BEGIN TEMPORARY SPP 0 TREE-CUT PRIOR DIAGNOSTICS
	void log_spp_0_tree_cut_prior_statistics(const LightTreeSGDevice& light_tree_sg,
											 const IlluminationAwareKDTreeDataHost<OrochiBuffer>& illumination_aware_kd_tree,
											 const OrochiBuffer<unsigned int>& debug_tree_cut_node_index,
											 unsigned int tree_cut_size)
	{
		if (tree_cut_size == 0)
			return;

		std::vector<unsigned int> tree_cut_node_indices			= OrochiBuffer<unsigned int>::download_data(light_tree_sg.tree_cut_node_indices, tree_cut_size);
		std::vector<unsigned int> initial_tree_cut_node_indices = debug_tree_cut_node_index.download_data_partial(0, tree_cut_size);
		std::vector<unsigned short int> prior_probabilities = illumination_aware_kd_tree.m_tree_cut_sampling_prior_pdfs.download_data_partial(0, tree_cut_size);
		if (tree_cut_node_indices.size() != tree_cut_size || initial_tree_cut_node_indices.size() != tree_cut_size ||
			prior_probabilities.size() != tree_cut_size)
			Debug::debugbreak();

		unsigned int maximum_tree_cut_node_index = 0;
		bool has_valid_tree_cut_node			 = false;
		for (unsigned int slot = 0; slot < tree_cut_size; slot++)
		{
			unsigned int node_index = tree_cut_node_indices[slot];
			if (initial_tree_cut_node_indices[slot] == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX ||
				node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
				Debug::debugbreak();

			if (node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
				continue;

			maximum_tree_cut_node_index = std::max(maximum_tree_cut_node_index, node_index);
			has_valid_tree_cut_node		= true;
		}

		std::vector<LightTreeSGNodeDevice> light_tree_nodes;
		if (has_valid_tree_cut_node)
			light_tree_nodes = OrochiBuffer<LightTreeSGNodeDevice>::download_data(light_tree_sg.nodes, maximum_tree_cut_node_index + 1);

		std::printf(
			"\n[IlluminationAwareKDTree SPP 0 tree-cut prior diagnostics] tree_cut_node_indices_pointer=%p nodes_pointer=%p effective_tree_cut_size=%u\n",
			static_cast<void*>(light_tree_sg.tree_cut_node_indices), static_cast<void*>(light_tree_sg.nodes), tree_cut_size);
		for (unsigned int slot = 0; slot < tree_cut_size; slot++)
		{
			unsigned int node_index		   = tree_cut_node_indices[slot];
			bool valid					   = node_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX && node_index < light_tree_nodes.size();
			float total_power			   = valid ? light_tree_nodes[node_index].get_total_power() : 0.0f;
			float prior_probability		   = static_cast<float>(prior_probabilities[slot]) / 65535.0f;
			bool incoherent_value_detected = initial_tree_cut_node_indices[slot] != node_index || !std::isfinite(total_power) || total_power < 0.0f ||
											 !std::isfinite(prior_probability) || prior_probability < 0.0f || prior_probability > 1.0f;

			std::printf("slot=%u initial_node=%u current_node=%u valid=%s total_power=%.9g prior_probability=%.9g\n", slot, initial_tree_cut_node_indices[slot],
						node_index, valid ? "true" : "false", total_power, prior_probability);

			if (incoherent_value_detected)
				Debug::debugbreak();
		}
	}
	// END TEMPORARY SPP 0 TREE-CUT PRIOR DIAGNOSTICS

	// BEGIN TEMPORARY SPP 0 GLOBAL PRIOR INITIALIZATION DIAGNOSTICS
	void log_spp_0_global_prior_initialization_diagnostics(const OrochiBuffer<float>& debug_power,
														   const OrochiBuffer<float>& debug_total_power,
														   const OrochiBuffer<float>& debug_probability,
														   const OrochiBuffer<unsigned short int>& debug_probability_u16,
														   const OrochiBuffer<unsigned short int>& stored_prior_probabilities,
														   unsigned int tree_cut_size)
	{
		std::vector<float> power							   = debug_power.download_data_partial(0, tree_cut_size);
		std::vector<float> total_power						   = debug_total_power.download_data_partial(0, tree_cut_size);
		std::vector<float> probability						   = debug_probability.download_data_partial(0, tree_cut_size);
		std::vector<unsigned short int> probability_u16		   = debug_probability_u16.download_data_partial(0, tree_cut_size);
		std::vector<unsigned short int> stored_probability_u16 = stored_prior_probabilities.download_data_partial(0, tree_cut_size);

		float sum_debug_probability			 = 0.0f;
		float sum_decoded_stored_probability = 0.0f;
		std::printf("\n[IlluminationAwareKDTree SPP 0 global prior initialization diagnostics]\n");
		for (unsigned int slot = 0; slot < tree_cut_size; slot++)
		{
			bool expected_is_defined	   = total_power[slot] > 0.0f;
			bool incoherent_value_detected = !std::isfinite(power[slot]) || !std::isfinite(total_power[slot]) || power[slot] < 0.0f ||
											 total_power[slot] < 0.0f || power[slot] > total_power[slot];

			float expected								= expected_is_defined ? power[slot] / total_power[slot] : 0.0f;
			unsigned short int expected_probability_u16 = static_cast<unsigned short int>(expected * 65535.0f);
			bool probability_matches_expected			= expected_is_defined && std::fabs(probability[slot] - expected) <= 1.0e-6f;
			bool probability_u16_matches_expected		= expected_is_defined && probability_u16[slot] == expected_probability_u16;
			bool stored_probability_matches_debug		= stored_probability_u16[slot] == probability_u16[slot];
			float decoded_stored_probability			= static_cast<float>(stored_probability_u16[slot]) / 65535.0f;
			if (!expected_is_defined || !std::isfinite(expected) || expected < 0.0f || expected > 1.0f || !std::isfinite(probability[slot]) ||
				probability[slot] < 0.0f || probability[slot] > 1.0f || !probability_matches_expected || !probability_u16_matches_expected ||
				!stored_probability_matches_debug)
				incoherent_value_detected = true;

			sum_debug_probability += probability[slot];
			sum_decoded_stored_probability += decoded_stored_probability;

			std::printf("slot=%u power=%.9g total_power=%.9g debug_probability=%.9g expected=%.9g debug_probability_u16=%u expected_probability_u16=%u "
						"stored_probability_u16=%u expected_is_defined=%s debug_probability_matches_expected=%s debug_probability_u16_matches_expected=%s "
						"stored_probability_matches_debug=%s decoded_stored_probability=%.9g\n",
						slot, power[slot], total_power[slot], probability[slot], expected, static_cast<unsigned int>(probability_u16[slot]),
						static_cast<unsigned int>(expected_probability_u16), static_cast<unsigned int>(stored_probability_u16[slot]),
						expected_is_defined ? "true" : "false", probability_matches_expected ? "true" : "false",
						probability_u16_matches_expected ? "true" : "false", stored_probability_matches_debug ? "true" : "false", decoded_stored_probability);

			if (incoherent_value_detected)
				Debug::debugbreak();
		}

		std::printf("sum(debug_probability)=%.9g sum(decoded_stored_probabilities)=%.9g\n", sum_debug_probability, sum_decoded_stored_probability);
		if (!std::isfinite(sum_debug_probability) || !std::isfinite(sum_decoded_stored_probability) || std::fabs(sum_debug_probability - 1.0f) > 1.0e-3f ||
			std::fabs(sum_decoded_stored_probability - 1.0f) > 1.0e-2f)
			Debug::debugbreak();
	}
	// END TEMPORARY SPP 0 GLOBAL PRIOR INITIALIZATION DIAGNOSTICS
} // namespace

IlluminationAwareKDTreeRenderPass::IlluminationAwareKDTreeRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME, renderer, options)
{
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/ResetTree.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID]->set_kernel_function_name("IlluminationAwareKDTree_ResetTree");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_CUT_SAMPLING_DISTRIBUTIONS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::RESET_TREE_CUT_SAMPLING_DISTRIBUTIONS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_CUT_SAMPLING_DISTRIBUTIONS_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/ResetTreeCutSamplingDistributions.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_CUT_SAMPLING_DISTRIBUTIONS_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_ResetTreeCutSamplingDistributions");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_CUT_SAMPLING_DISTRIBUTIONS_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_GLOBAL_TREE_CUT_PRIOR_SAMPLING_DISTRIBUTION_KERNEL_ID] = std::make_shared<GPUKernel>(
		this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::INITIALIZE_GLOBAL_TREE_CUT_PRIOR_SAMPLING_DISTRIBUTION_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_GLOBAL_TREE_CUT_PRIOR_SAMPLING_DISTRIBUTION_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/InitializeGlobalTreeCutPriorSamplingDistribution.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_GLOBAL_TREE_CUT_PRIOR_SAMPLING_DISTRIBUTION_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_InitializeGlobalTreeCutPriorSamplingDistribution");
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_GLOBAL_TREE_CUT_PRIOR_SAMPLING_DISTRIBUTION_KERNEL_ID]->synchronize_options_with(m_compiler_options,
																																			 {});

	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_TREE_CUT_SAMPLING_DISTRIBUTION_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_TREE_CUT_SAMPLING_DISTRIBUTION_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_TREE_CUT_SAMPLING_DISTRIBUTION_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/InitializeRootTreeCutSamplingDistribution.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_TREE_CUT_SAMPLING_DISTRIBUTION_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_InitializeRootTreeCutSamplingDistribution");
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_TREE_CUT_SAMPLING_DISTRIBUTION_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

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

	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_NEE_DISTRIBUTIONS_STATISTICS_KERNEL_ID] = std::make_shared<GPUKernel>(
		this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_NEE_DISTRIBUTIONS_STATISTICS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_NEE_DISTRIBUTIONS_STATISTICS_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/ResetBatchKDTreeAndNEEDistributionsStatistics.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_NEE_DISTRIBUTIONS_STATISTICS_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_ResetBatchKDTreeAndNEEDistributionsStatistics");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_NEE_DISTRIBUTIONS_STATISTICS_KERNEL_ID]->synchronize_options_with(m_compiler_options,
																																		   {});

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

	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_NEE_DISTRIBUTION_TRAINING_RECORDS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::ACCUMULATE_NEE_DISTRIBUTION_TRAINING_RECORDS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_NEE_DISTRIBUTION_TRAINING_RECORDS_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/AccumulateNEEDistributionTrainingRecords.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_NEE_DISTRIBUTION_TRAINING_RECORDS_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_AccumulateNEEDistributionTrainingRecords");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_NEE_DISTRIBUTION_TRAINING_RECORDS_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::REBUILD_ACTIVE_NEE_DISTRIBUTIONS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::REBUILD_ACTIVE_NEE_DISTRIBUTIONS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::REBUILD_ACTIVE_NEE_DISTRIBUTIONS_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/RebuildActiveNEEDistributions.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::REBUILD_ACTIVE_NEE_DISTRIBUTIONS_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_RebuildActiveNEEDistributions");
	m_kernels[IlluminationAwareKDTreeRenderPass::REBUILD_ACTIVE_NEE_DISTRIBUTIONS_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

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

bool IlluminationAwareKDTreeRenderPass::pre_sample_update(float delta_time)
{
	if (!is_render_pass_used(*m_renderer->get_global_compiler_options()))
	{
		m_buffers_need_reallocation = true;

		// BEGIN TEMPORARY SPP 0 GLOBAL PRIOR INITIALIZATION DIAGNOSTICS
		m_debug_power.free();
		m_debug_total_power.free();
		m_debug_probability.free();
		m_debug_probability_u16.free();
		m_debug_tree_cut_node_index.free();
		// END TEMPORARY SPP 0 GLOBAL PRIOR INITIALIZATION DIAGNOSTICS

		return m_illumination_aware_kd_tree.free();
	}

	bool render_data_invalidated = false;
	if (m_buffers_need_reallocation)
	{
		int sg_tree_cut_size = m_renderer->get_light_tree_sg_sampling_data_structure().get_tree_cut_size();
		m_illumination_aware_kd_tree.resize(IlluminationAwareKDTreeDataHost<OrochiBuffer>::MAXIMUM_NUMBER_OF_NODES, m_training_sample_buffer_capacity,
											sg_tree_cut_size);

		// BEGIN TEMPORARY SPP 0 GLOBAL PRIOR INITIALIZATION DIAGNOSTICS
		m_debug_power.resize(sg_tree_cut_size);
		m_debug_total_power.resize(sg_tree_cut_size);
		m_debug_probability.resize(sg_tree_cut_size);
		m_debug_probability_u16.resize(sg_tree_cut_size);
		m_debug_tree_cut_node_index.resize(sg_tree_cut_size);
		// END TEMPORARY SPP 0 GLOBAL PRIOR INITIALIZATION DIAGNOSTICS
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

		m_buffers_need_reallocation = false;
		render_data_invalidated		= true;

		reset(false);
	}

	IlluminationAwareKDTreeDevice illumination_aware_kd_tree = m_illumination_aware_kd_tree.to_device(m_renderer->get_render_data());
	LightTreeSGDevice light_tree_sg							 = m_renderer->get_render_data().light_tree_sg;
	unsigned int tree_cut_size								 = light_tree_sg.settings.effective_tree_cut_size;
	unsigned int active_node_count							 = m_illumination_aware_kd_tree.m_node_count.download_data()[0];
	unsigned int distribution_slot_count					 = active_node_count * tree_cut_size;
	// Total number of nodes * tree cut node, to reset everything, not just active nodes as 'distribution_slot_count' represents
	unsigned int all_distribution_slot_count = m_illumination_aware_kd_tree.m_nodes.size() * tree_cut_size;

	if (m_renderer->get_render_data().render_settings.sample_number == 0)
	{
		void* reset_distribution_launch_args[] = { &illumination_aware_kd_tree, &tree_cut_size };
		m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_CUT_SAMPLING_DISTRIBUTIONS_KERNEL_ID]->launch_asynchronous(
			256, 1, all_distribution_slot_count, 1, reset_distribution_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

		// BEGIN TEMPORARY SPP 0 GLOBAL PRIOR INITIALIZATION DIAGNOSTICS
		float* debug_power						  = m_debug_power.data();
		float* debug_total_power				  = m_debug_total_power.data();
		float* debug_probability				  = m_debug_probability.data();
		unsigned short int* debug_probability_u16 = m_debug_probability_u16.data();
		unsigned int* debug_tree_cut_node_index	  = m_debug_tree_cut_node_index.data();
		void* global_prior_launch_args[]		  = { &illumination_aware_kd_tree, &light_tree_sg,	   &debug_power,
													  &debug_total_power,		   &debug_probability, &debug_probability_u16,
													  &debug_tree_cut_node_index };
		// END TEMPORARY SPP 0 GLOBAL PRIOR INITIALIZATION DIAGNOSTICS
		m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_GLOBAL_TREE_CUT_PRIOR_SAMPLING_DISTRIBUTION_KERNEL_ID]->launch_asynchronous(
			IlluminationAwareKDTreeTreeCutInitializationBlockSize, 1, tree_cut_size, 1, global_prior_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

		// BEGIN TEMPORARY SPP 0 GLOBAL PRIOR INITIALIZATION DIAGNOSTICS
		log_spp_0_global_prior_initialization_diagnostics(m_debug_power, m_debug_total_power, m_debug_probability, m_debug_probability_u16,
														  m_illumination_aware_kd_tree.m_tree_cut_sampling_prior_pdfs, tree_cut_size);
		// END TEMPORARY SPP 0 GLOBAL PRIOR INITIALIZATION DIAGNOSTICS

		// BEGIN TEMPORARY SPP 0 TREE-CUT PRIOR DIAGNOSTICS
		log_spp_0_tree_cut_prior_statistics(light_tree_sg, m_illumination_aware_kd_tree, m_debug_tree_cut_node_index, tree_cut_size);
		// END TEMPORARY SPP 0 TREE-CUT PRIOR DIAGNOSTICS

		void* root_distribution_launch_args[] = { &illumination_aware_kd_tree, &tree_cut_size };
		m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_TREE_CUT_SAMPLING_DISTRIBUTION_KERNEL_ID]->launch_asynchronous(
			256, 1, tree_cut_size, 1, root_distribution_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
	}

	unsigned int reset_thread_count = std::max(active_node_count, distribution_slot_count);
	void* launch_args[]				= { &illumination_aware_kd_tree, &tree_cut_size };
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_NEE_DISTRIBUTIONS_STATISTICS_KERNEL_ID]->launch_asynchronous(
		1024, 1, reset_thread_count, 1, launch_args, m_renderer->get_main_stream());
	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

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

	LightTreeSGDevice light_tree_sg							 = render_data.light_tree_sg;
	IlluminationAwareKDTreeDevice illumination_aware_kd_tree = render_data.illumination_aware_kd_tree;

	// TODO maybe download the training_sample_count and launch the kernel with a single thread per sample instead of launching a fixed number of threads and
	// having threads beyond the training_sample_count do nothing? Maybe worth it in perf despite CPU overhead?
	void* launch_args[] = { &illumination_aware_kd_tree };
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->launch_asynchronous(
		256, 1, illumination_aware_kd_tree.training_sample_capacity, 1, launch_args, m_renderer->get_main_stream());
	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->launch_asynchronous(
		256, 1, illumination_aware_kd_tree.node_capacity, 1, launch_args, m_renderer->get_main_stream());
	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

	int split_iterations = m_split_iterations_per_SPP;
	if (m_auto_split_iterations_per_SPP)
	{
		if (render_data.render_settings.sample_number == 0)
			// Lots of splits at the very first SPP to quickly get a good tree structure, then we can slow down the splits
			split_iterations = 16;
		else
			// Maximum 3 because we don't really need more after the first few SPPs
			split_iterations = std::min(split_iterations, 3);
	}

	for (int split = 0; split < split_iterations; split++)
	{
		ensure_all_lookahead_cell_levels(render_data, compiler_options);

		void* mark_guiding_cell_launch_args[] = { &illumination_aware_kd_tree };
		m_kernels[IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID]->launch_asynchronous(
			256, 1, illumination_aware_kd_tree.node_capacity, 1, mark_guiding_cell_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
		// TODO if no cell was marked for splitting, no need to continue this whole loop, we can break

		// TODO this download data could be done with a DtoD async copy of the current guiding count into another 1*unsigned int buffer
		// Number of guiding nodes before the splitting
		m_cached_current_guiding_node_count = m_illumination_aware_kd_tree.m_active_guiding_node_count.download_data()[0];
		if (m_cached_current_guiding_node_count == 0)
			// Should never happen we should at least have the root node
			Debug::debugbreak();
		// TODO same here download async
		m_cached_current_node_count	  = m_illumination_aware_kd_tree.m_node_count.download_data()[0];
		void* promotion_launch_args[] = { &illumination_aware_kd_tree, &light_tree_sg.settings.effective_tree_cut_size, &m_cached_current_guiding_node_count };
		// We launch blocks of 1024 threads here, and as many blocks as needed to cover all the guiding nodes that need to be promoted. This is because each
		// thread block will be in charge of one cell to copy NEE guiding distributions from the parent to the 2 new children
		//
		// TODO we could be launching only number of blocks = nodes that have been marked for splitting instead of all the guiding nodes
		m_kernels[IlluminationAwareKDTreeRenderPass::PROMOTE_GUIDING_CELLS_KERNEL_ID]->launch_asynchronous(
			1024, 1, m_cached_current_guiding_node_count * 1024, 1, promotion_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
	}

	unsigned int tree_cut_size				 = light_tree_sg.settings.effective_tree_cut_size;
	void* nee_training_records_launch_args[] = { &illumination_aware_kd_tree, &tree_cut_size };
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_NEE_DISTRIBUTION_TRAINING_RECORDS_KERNEL_ID]->launch_asynchronous(
		256, 1, illumination_aware_kd_tree.nee_learnt_distributions.nee_training_record_capacity, 1, nee_training_records_launch_args,
		m_renderer->get_main_stream());
	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

	unsigned int active_guiding_count			  = m_illumination_aware_kd_tree.m_active_guiding_node_count.download_data()[0];
	m_cached_current_guiding_node_count			  = active_guiding_count;
	void* rebuild_nee_distributions_launch_args[] = { &illumination_aware_kd_tree, &tree_cut_size, &active_guiding_count };
	m_kernels[IlluminationAwareKDTreeRenderPass::REBUILD_ACTIVE_NEE_DISTRIBUTIONS_KERNEL_ID]->launch_asynchronous(
		1024, 1, active_guiding_count * 1024, 1, rebuild_nee_distributions_launch_args, m_renderer->get_main_stream());
	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

	// BEGIN TEMPORARY SPP 0 NEE DISTRIBUTION DIAGNOSTICS
	if (render_data.render_settings.sample_number == 0)
		log_spp_0_final_nee_distribution_statistics(m_illumination_aware_kd_tree, active_guiding_count, tree_cut_size);
	// END TEMPORARY SPP 0 NEE DISTRIBUTION DIAGNOSTICS

	// BEGIN TEMPORARY SPP 0 TREE-CUT PRIOR DIAGNOSTICS
	if (render_data.render_settings.sample_number == 0)
		log_spp_0_tree_cut_prior_statistics(light_tree_sg, m_illumination_aware_kd_tree, m_debug_tree_cut_node_index, tree_cut_size);
	// END TEMPORARY SPP 0 TREE-CUT PRIOR DIAGNOSTICS
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
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
		m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_TRAINING_SAMPLES_KERNEL_ID]->launch_asynchronous(
			256, 1, illumination_aware_kd_tree.training_sample_capacity, 1, expansion_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
		m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID]->launch_asynchronous(
			256, 1, illumination_aware_kd_tree.node_capacity, 1, expansion_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

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

	m_renderer->get_render_data().illumination_aware_kd_tree = m_illumination_aware_kd_tree.to_device(m_renderer->get_render_data());
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

	IlluminationAwareKDTreeDevice kd_tree_device = m_illumination_aware_kd_tree.to_device(m_renderer->get_render_data());
	float3_t scene_bounds_minimum				 = m_renderer->get_scene_metadata().scene_bounding_box.mini;
	float3_t scene_bounds_maximum				 = m_renderer->get_scene_metadata().scene_bounding_box.maxi;
	void* launch_args[]							 = { &kd_tree_device, &scene_bounds_minimum, &scene_bounds_maximum };

	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID]->launch_asynchronous(256, 1, m_illumination_aware_kd_tree.m_nodes.size(), 1, launch_args,
																							m_renderer->get_main_stream());
	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
}

bool IlluminationAwareKDTreeRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR) == LSS_SG_TREE_LEARNT_DISTRIBUTIONS &&
		   compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_LIGHT_TREE_SG;
}

int& IlluminationAwareKDTreeRenderPass::get_split_iterations_per_SPP()
{
	return m_split_iterations_per_SPP;
}

bool& IlluminationAwareKDTreeRenderPass::get_auto_split_iterations_per_SPP()
{
	return m_auto_split_iterations_per_SPP;
}

int& IlluminationAwareKDTreeRenderPass::get_training_sample_buffer_capacity()
{
	return m_training_sample_buffer_capacity;
}

std::size_t IlluminationAwareKDTreeRenderPass::get_current_node_buffer_capacity() const
{
	return m_illumination_aware_kd_tree.m_nodes.size();
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
	return get_vram_usage_breakdown().get_total_bytes();
}

IlluminationAwareKDTreeVRAMUsage IlluminationAwareKDTreeRenderPass::get_vram_usage_breakdown() const
{
	IlluminationAwareKDTreeVRAMUsage vram_usage;

	vram_usage.nodes	   = m_illumination_aware_kd_tree.m_nodes.get_byte_size();
	vram_usage.node_bounds = m_illumination_aware_kd_tree.m_node_bounds.get_byte_size();
	vram_usage.node_count  = m_illumination_aware_kd_tree.m_node_count.get_byte_size();

	vram_usage.active_guiding_nodes		  = m_illumination_aware_kd_tree.m_active_guiding_nodes.get_byte_size();
	vram_usage.active_guiding_node_count  = m_illumination_aware_kd_tree.m_active_guiding_node_count.get_byte_size();
	vram_usage.needs_split				  = m_illumination_aware_kd_tree.m_needs_split.get_byte_size();
	vram_usage.guiding_distribution_count = m_illumination_aware_kd_tree.m_guiding_distribution_count.get_byte_size();

	vram_usage.current_frontier		  = m_illumination_aware_kd_tree.m_current_frontier.get_byte_size();
	vram_usage.current_frontier_count = m_illumination_aware_kd_tree.m_current_frontier_count.get_byte_size();
	vram_usage.next_frontier		  = m_illumination_aware_kd_tree.m_next_frontier.get_byte_size();
	vram_usage.next_frontier_count	  = m_illumination_aware_kd_tree.m_next_frontier_count.get_byte_size();

	vram_usage.training_samples			 = m_illumination_aware_kd_tree.m_training_samples.get_byte_size();
	vram_usage.training_sample_count	 = m_illumination_aware_kd_tree.m_training_sample_count.get_byte_size();
	vram_usage.nee_training_records		 = m_illumination_aware_kd_tree.m_nee_training_records.get_byte_size();
	vram_usage.nee_training_record_count = m_illumination_aware_kd_tree.m_nee_training_record_count.get_byte_size();

	vram_usage.batch_signatures		   = m_illumination_aware_kd_tree.m_batch_signatures.get_byte_size();
	vram_usage.history_signatures	   = m_illumination_aware_kd_tree.m_history_signatures.get_byte_size();
	vram_usage.batch_spatial_moments   = m_illumination_aware_kd_tree.m_batch_spatial_moments.get_byte_size();
	vram_usage.history_spatial_moments = m_illumination_aware_kd_tree.m_history_spatial_moments.get_byte_size();

	vram_usage.tree_cut_sampling_probabilities				= m_illumination_aware_kd_tree.m_tree_cut_sampling_probabilities.get_byte_size();
	vram_usage.tree_cut_sampling_cdfs						= m_illumination_aware_kd_tree.m_tree_cut_sampling_cdfs.get_byte_size();
	vram_usage.history_per_cell_sample_count				= m_illumination_aware_kd_tree.m_history_per_cell_sample_count.get_byte_size();
	vram_usage.history_per_cell_normal_sum_x				= m_illumination_aware_kd_tree.m_history_per_cell_normal_sum_x.get_byte_size();
	vram_usage.history_per_cell_normal_sum_y				= m_illumination_aware_kd_tree.m_history_per_cell_normal_sum_y.get_byte_size();
	vram_usage.history_per_cell_normal_sum_z				= m_illumination_aware_kd_tree.m_history_per_cell_normal_sum_z.get_byte_size();
	vram_usage.history_per_cell_normal_count				= m_illumination_aware_kd_tree.m_history_per_cell_normal_count.get_byte_size();
	vram_usage.history_per_cut_node_estimated_second_moment = m_illumination_aware_kd_tree.m_history_per_cut_node_estimated_second_moment.get_byte_size();
	vram_usage.history_per_cut_node_sample_count			= m_illumination_aware_kd_tree.m_history_per_cut_node_sample_count.get_byte_size();
	vram_usage.batch_per_cut_node_second_moment_sum			= m_illumination_aware_kd_tree.m_batch_per_cut_node_second_moment_sum.get_byte_size();
	vram_usage.batch_per_cut_node_sample_count				= m_illumination_aware_kd_tree.m_batch_per_cut_node_sample_count.get_byte_size();

	vram_usage.tree_cut_sampling_prior_pdfs = m_illumination_aware_kd_tree.m_tree_cut_sampling_prior_pdfs.get_byte_size();
	vram_usage.tree_cut_sampling_prior_cdfs = m_illumination_aware_kd_tree.m_tree_cut_sampling_prior_cdfs.get_byte_size();

	return vram_usage;
}
