/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/IlluminationAwareKDTreeRenderPass.h"

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeOptions.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingManyLightsOptions.h"
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
const std::string IlluminationAwareKDTreeRenderPass::REPLAY_NISML_TRAINING_SAMPLES_KERNEL_ID				= "Replay NISML Training Samples";
const std::string IlluminationAwareKDTreeRenderPass::BUILD_NISML_CACHES_KERNEL_ID							= "Build NISML Caches";

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

	m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_NISML_TRAINING_SAMPLES_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::REPLAY_NISML_TRAINING_SAMPLES_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_NISML_TRAINING_SAMPLES_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/ReplayNISMLTrainingSamplesKernel.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_NISML_TRAINING_SAMPLES_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_ReplayNISMLTrainingSamplesKernel");
	m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_NISML_TRAINING_SAMPLES_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::BUILD_NISML_CACHES_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::BUILD_NISML_CACHES_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::BUILD_NISML_CACHES_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY
																									 "/IlluminationAwareKDTree/BuildNISMLCaches.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::BUILD_NISML_CACHES_KERNEL_ID]->set_kernel_function_name("IlluminationAwareKDTree_BuildNISMLCaches");
	m_kernels[IlluminationAwareKDTreeRenderPass::BUILD_NISML_CACHES_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});
}

bool IlluminationAwareKDTreeRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
																	 const std::vector<hiprtFuncNameSet>& func_name_sets,
																	 bool silent,
																	 bool use_cache)
{
	if (!is_render_pass_used(*m_compiler_options))
		return false;

	bool updated													 = false;
	std::map<std::string, std::shared_ptr<GPUKernel>> active_kernels = get_all_kernels();
	for (std::map<std::string, std::shared_ptr<GPUKernel>>::value_type& name_to_kernel : active_kernels)
	{
		if (name_to_kernel.second->has_been_compiled())
			continue;

		name_to_kernel.second->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
		updated = true;
	}

	return updated;
}

std::map<std::string, std::shared_ptr<GPUKernel>> IlluminationAwareKDTreeRenderPass::get_all_kernels()
{
	if (!is_render_pass_used(*m_compiler_options))
		return {};

	std::map<std::string, std::shared_ptr<GPUKernel>> active_kernels = m_kernels;
	bool using_nisml												 = is_using_nisml(*m_compiler_options);
	bool using_nee_learnt_distributions =
		ILLUMINATION_AWARE_KD_TREE_IS_NEE_LEARNT_DISTRIBUTIONS(m_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR),
															   m_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY));

	if (!using_nee_learnt_distributions)
	{
		active_kernels.erase(RESET_TREE_CUT_SAMPLING_DISTRIBUTIONS_KERNEL_ID);
		active_kernels.erase(INITIALIZE_GLOBAL_TREE_CUT_PRIOR_SAMPLING_DISTRIBUTION_KERNEL_ID);
		active_kernels.erase(INITIALIZE_ROOT_TREE_CUT_SAMPLING_DISTRIBUTION_KERNEL_ID);
		active_kernels.erase(ACCUMULATE_NEE_DISTRIBUTION_TRAINING_RECORDS_KERNEL_ID);
		active_kernels.erase(REBUILD_ACTIVE_NEE_DISTRIBUTIONS_KERNEL_ID);
	}

	if (!using_nisml)
	{
		active_kernels.erase(REPLAY_NISML_TRAINING_SAMPLES_KERNEL_ID);
		active_kernels.erase(BUILD_NISML_CACHES_KERNEL_ID);
	}

	return active_kernels;
}

void IlluminationAwareKDTreeRenderPass::resize(unsigned int new_width, unsigned int new_height) {}

bool IlluminationAwareKDTreeRenderPass::pre_sample_update(float delta_time)
{
	m_nisml_representative_capacity = std::max(m_nisml_representative_capacity, 1);

	if (!is_render_pass_used(*m_renderer->get_global_compiler_options()))
	{
		m_buffers_need_reallocation = true;

		return m_illumination_aware_kd_tree.free();
	}

	unsigned int nisml_representative_capacity = static_cast<unsigned int>(m_nisml_representative_capacity);
	bool nisml_representative_capacity_changed = m_illumination_aware_kd_tree.m_nisml_data.m_representative_capacity != nisml_representative_capacity;
	if (nisml_representative_capacity_changed)
		m_buffers_need_reallocation = true;

	bool render_data_invalidated = false;
	if (m_buffers_need_reallocation)
	{
		int sg_tree_cut_size = m_renderer->get_light_tree_sg_sampling_data_structure().get_tree_cut_size();
		m_illumination_aware_kd_tree.resize(IlluminationAwareKDTreeDataHost<OrochiBuffer>::MAXIMUM_NUMBER_OF_NODES, m_training_sample_buffer_capacity,
											sg_tree_cut_size, nisml_representative_capacity);

		if (nisml_representative_capacity_changed)
			m_illumination_aware_kd_tree.m_nisml_data.clear_representative_metadata();

		m_buffers_need_reallocation = false;
		render_data_invalidated		= true;

		reset(false);
	}

	IlluminationAwareKDTreeDevice kd_tree_device = m_illumination_aware_kd_tree.to_device(m_renderer->get_render_data());
	LightTreeSGDevice light_tree_sg				 = m_renderer->get_render_data().light_tree_sg;
	unsigned int tree_cut_size					 = light_tree_sg.settings.effective_tree_cut_size;
	unsigned int active_node_count				 = m_illumination_aware_kd_tree.download_counter(m_illumination_aware_kd_tree.m_kd_tree_data.m_node_count);
	unsigned int distribution_slot_count		 = active_node_count * tree_cut_size;
	// Total number of nodes * tree cut node, to reset everything, not just active nodes as 'distribution_slot_count' represents
	unsigned int all_distribution_slot_count = m_illumination_aware_kd_tree.m_kd_tree_data.m_nodes.size() * tree_cut_size;
	distribution_slot_count *= static_cast<unsigned int>(SurfaceNormalFace_Count);
	all_distribution_slot_count *= static_cast<unsigned int>(SurfaceNormalFace_Count);

	if (!is_using_nisml(*m_compiler_options) && m_renderer->get_render_data().render_settings.sample_number == 0)
	{
		void* reset_distribution_launch_args[] = { &kd_tree_device, &tree_cut_size };
		m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_CUT_SAMPLING_DISTRIBUTIONS_KERNEL_ID]->launch_asynchronous(
			256, 1, all_distribution_slot_count, 1, reset_distribution_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

		void* global_prior_launch_args[] = { &kd_tree_device, &light_tree_sg };
		m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_GLOBAL_TREE_CUT_PRIOR_SAMPLING_DISTRIBUTION_KERNEL_ID]->launch_asynchronous(
			IlluminationAwareKDTreeTreeCutInitializationBlockSize, 1, tree_cut_size, 1, global_prior_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

		void* root_distribution_launch_args[] = { &kd_tree_device, &tree_cut_size };
		m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_TREE_CUT_SAMPLING_DISTRIBUTION_KERNEL_ID]->launch_asynchronous(
			256, 1, tree_cut_size, 1, root_distribution_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
	}

	unsigned int reset_thread_count = is_using_nisml(*m_compiler_options) ? active_node_count : std::max(active_node_count, distribution_slot_count);
	void* launch_args[]				= { &kd_tree_device, &tree_cut_size };
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_NEE_DISTRIBUTIONS_STATISTICS_KERNEL_ID]->launch_asynchronous(
		1024, 1, reset_thread_count, 1, launch_args, m_renderer->get_main_stream());
	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

	return render_data_invalidated;
}

bool IlluminationAwareKDTreeRenderPass::is_using_nisml(const GPUKernelCompilerOptions& compiler_options) const
{
	return ILLUMINATION_AWARE_KD_TREE_IS_NISML(compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR),
											   compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY));
}

void IlluminationAwareKDTreeRenderPass::build_nisml(HIPRTRenderData& render_data)
{
	if (render_data.nisml.cluster_node_indices == nullptr || render_data.nisml.cluster_count == 0 || render_data.nisml.cluster_count > NISML_MAX_CLUSTER_COUNT)
		return;

	IlluminationAwareKDTreeDevice kd_tree_device = m_illumination_aware_kd_tree.to_device(render_data);
	unsigned int node_count						 = m_illumination_aware_kd_tree.download_counter(m_illumination_aware_kd_tree.m_kd_tree_data.m_node_count);
	unsigned int pending_cell_count = m_illumination_aware_kd_tree.download_counter(m_illumination_aware_kd_tree.m_nisml_data.m_pending_cell_count);
	if (node_count == 0 || pending_cell_count == 0)
		return;

	HIPRTRenderData cache_render_data = render_data;
	cache_render_data.kd_tree_device  = kd_tree_device;
	void* launch_args[]				  = { &kd_tree_device, &cache_render_data };
	unsigned int cache_entry_count	  = node_count * ILLUMINATION_AWARE_KD_TREE_NISML_NORMAL_FACE_COUNT;
	m_kernels[IlluminationAwareKDTreeRenderPass::BUILD_NISML_CACHES_KERNEL_ID]->launch_asynchronous(256, 1, cache_entry_count, 1, launch_args,
																									m_renderer->get_main_stream());
	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
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

	LightTreeSGDevice light_tree_sg				 = render_data.light_tree_sg;
	IlluminationAwareKDTreeDevice kd_tree_device = render_data.kd_tree_device;

	// Using -1 because this counter is 0-based but the SPPs displayed at the top of the UI are 1-based. This is just to match the user's HUD
	if (!m_frozen_tree && render_data.render_settings.sample_number <= kd_tree_device.core.user_settings.stop_refining_after_SPP - 1)
	{
		// TODO maybe download the training_sample_count and launch the kernel with a single thread per sample instead of launching a fixed number of threads
		// and
		// having threads beyond the training_sample_count do nothing? Maybe worth it in perf despite CPU overhead?
		void* launch_args[] = { &kd_tree_device };
		m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->launch_asynchronous(
			256, 1, kd_tree_device.core.training_sample_capacity, 1, launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
		m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->launch_asynchronous(
			256, 1, kd_tree_device.core.node_capacity, 1, launch_args, m_renderer->get_main_stream());
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

			void* mark_guiding_cell_launch_args[] = { &kd_tree_device };
			m_kernels[IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID]->launch_asynchronous(
				256, 1, kd_tree_device.core.node_capacity, 1, mark_guiding_cell_launch_args, m_renderer->get_main_stream());
			OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
			// TODO if no cell was marked for splitting, no need to continue this whole loop, we can break

			// TODO this download data could be done with a DtoD async copy of the current guiding count into another 1*unsigned int buffer
			// Number of guiding nodes before the splitting
			m_cached_current_guiding_node_count =
				m_illumination_aware_kd_tree.download_counter(m_illumination_aware_kd_tree.m_kd_tree_data.m_active_guiding_node_count);
			if (m_cached_current_guiding_node_count == 0)
				// Should never happen we should at least have the root node
				Debug::debugbreak();
			// TODO same here download async
			m_cached_current_node_count	  = m_illumination_aware_kd_tree.download_counter(m_illumination_aware_kd_tree.m_kd_tree_data.m_node_count);
			void* promotion_launch_args[] = { &kd_tree_device, &light_tree_sg.settings.effective_tree_cut_size, &m_cached_current_guiding_node_count };
			// We launch blocks of 1024 threads here, and as many blocks as needed to cover all the guiding nodes that need to be promoted. This is because each
			// thread block will be in charge of one cell to copy NEE guiding distributions from the parent to the 2 new children
			//
			// TODO we could be launching only number of blocks = nodes that have been marked for splitting instead of all the guiding nodes
			m_kernels[IlluminationAwareKDTreeRenderPass::PROMOTE_GUIDING_CELLS_KERNEL_ID]->launch_asynchronous(
				1024, 1, m_cached_current_guiding_node_count * 1024, 1, promotion_launch_args, m_renderer->get_main_stream());
			OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
		}
	}

	if (!m_frozen_tree && !is_using_nisml(compiler_options))
	{
		unsigned int tree_cut_size = light_tree_sg.settings.effective_tree_cut_size;

		void* nee_training_records_launch_args[] = { &kd_tree_device, &tree_cut_size };
		m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_NEE_DISTRIBUTION_TRAINING_RECORDS_KERNEL_ID]->launch_asynchronous(
			256, 1, kd_tree_device.nee_distributions.nee_training_record_capacity, 1, nee_training_records_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

		unsigned int active_guiding_count =
			m_illumination_aware_kd_tree.download_counter(m_illumination_aware_kd_tree.m_kd_tree_data.m_active_guiding_node_count);
		m_cached_current_guiding_node_count = active_guiding_count;
		active_guiding_count *= static_cast<unsigned int>(SurfaceNormalFace_Count);
		void* rebuild_nee_distributions_launch_args[] = { &kd_tree_device, &tree_cut_size, &active_guiding_count };
		m_kernels[IlluminationAwareKDTreeRenderPass::REBUILD_ACTIVE_NEE_DISTRIBUTIONS_KERNEL_ID]->launch_asynchronous(
			1024, 1, active_guiding_count * 1024, 1, rebuild_nee_distributions_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
	}

	if (is_using_nisml(compiler_options))
	{
		unsigned int training_record_capacity = render_data.nisml.training_record_capacity;
		if (training_record_capacity > 0)
		{
			void* replay_nisml_training_samples_launch_args[] = { &render_data };
			m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_NISML_TRAINING_SAMPLES_KERNEL_ID]->launch_asynchronous(
				256, 1, training_record_capacity, 1, replay_nisml_training_samples_launch_args, m_renderer->get_main_stream());
			OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
		}

		build_nisml(render_data);
	}
}

void IlluminationAwareKDTreeRenderPass::ensure_all_lookahead_cell_levels(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	IlluminationAwareKDTreeDevice kd_tree_device = render_data.kd_tree_device;

	unsigned int* current_frontier					 = kd_tree_device.core.active_guiding_nodes;
	unsigned int* next_frontier						 = m_illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier.data();
	AtomicType<unsigned int>* current_frontier_count = kd_tree_device.core.active_guiding_node_count;
	AtomicType<unsigned int>* next_frontier_count	 = m_illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier_count.get_atomic_device_pointer();

	bool next_frontier_uses_first_buffer = true;

	int max_lookahead_levels = compiler_options.get_macro_value(GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_MAXIMUM_LOOKAHEAD_LEVEL_COUNT);
	for (unsigned int level = 0; level < max_lookahead_levels; level++)
	{
		kd_tree_device.core.current_frontier	   = current_frontier;
		kd_tree_device.core.current_frontier_count = current_frontier_count;
		kd_tree_device.core.next_frontier		   = next_frontier;
		kd_tree_device.core.next_frontier_count	   = next_frontier_count;
		if (next_frontier_uses_first_buffer)
			m_illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier_count.memset_whole_buffer(0u);
		else
			m_illumination_aware_kd_tree.m_kd_tree_data.m_next_frontier_count.memset_whole_buffer(0u);

		unsigned int creation_tag	  = m_next_creation_tag++;
		void* expansion_launch_args[] = { &kd_tree_device, &creation_tag };
		m_kernels[IlluminationAwareKDTreeRenderPass::EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID]->launch_asynchronous(
			256, 1, kd_tree_device.core.node_capacity, 1, expansion_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
		m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_TRAINING_SAMPLES_KERNEL_ID]->launch_asynchronous(
			256, 1, kd_tree_device.core.training_sample_capacity, 1, expansion_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
		m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID]->launch_asynchronous(
			256, 1, kd_tree_device.core.node_capacity, 1, expansion_launch_args, m_renderer->get_main_stream());
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

		current_frontier	   = next_frontier;
		current_frontier_count = next_frontier_count;

		if (next_frontier_uses_first_buffer)
		{
			next_frontier		= m_illumination_aware_kd_tree.m_kd_tree_data.m_next_frontier.data();
			next_frontier_count = m_illumination_aware_kd_tree.m_kd_tree_data.m_next_frontier_count.get_atomic_device_pointer();
		}
		else
		{
			next_frontier		= m_illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier.data();
			next_frontier_count = m_illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier_count.get_atomic_device_pointer();
		}

		next_frontier_uses_first_buffer = !next_frontier_uses_first_buffer;
	}
}

void IlluminationAwareKDTreeRenderPass::update_render_data()
{
	if (!is_render_pass_used(*m_compiler_options))
	{
		m_renderer->get_render_data().kd_tree_device = {};

		return;
	}

	m_renderer->get_render_data().kd_tree_device = m_illumination_aware_kd_tree.to_device(m_renderer->get_render_data());
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

	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID]->launch_asynchronous(256, 1, m_illumination_aware_kd_tree.m_kd_tree_data.m_nodes.size(),
																							1, launch_args, m_renderer->get_main_stream());
	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));
}

bool IlluminationAwareKDTreeRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return ILLUMINATION_AWARE_KD_TREE_IS_ENABLED(compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR),
												 compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY));
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

int& IlluminationAwareKDTreeRenderPass::get_nisml_representative_capacity()
{
	return m_nisml_representative_capacity;
}

std::size_t IlluminationAwareKDTreeRenderPass::get_current_node_buffer_capacity() const
{
	return m_illumination_aware_kd_tree.m_kd_tree_data.m_nodes.size();
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

	vram_usage.nodes	   = m_illumination_aware_kd_tree.m_kd_tree_data.m_nodes.get_byte_size();
	vram_usage.node_bounds = m_illumination_aware_kd_tree.m_kd_tree_data.m_node_bounds.get_byte_size();
	vram_usage.node_count  = m_illumination_aware_kd_tree.m_kd_tree_data.m_node_count.get_byte_size();

	vram_usage.active_guiding_nodes		  = m_illumination_aware_kd_tree.m_kd_tree_data.m_active_guiding_nodes.get_byte_size();
	vram_usage.active_guiding_node_count  = m_illumination_aware_kd_tree.m_kd_tree_data.m_active_guiding_node_count.get_byte_size();
	vram_usage.needs_split				  = m_illumination_aware_kd_tree.m_kd_tree_data.m_needs_split.get_byte_size();
	vram_usage.guiding_distribution_count = m_illumination_aware_kd_tree.m_kd_tree_data.m_guiding_distribution_count.get_byte_size();

	vram_usage.current_frontier		  = m_illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier.get_byte_size();
	vram_usage.current_frontier_count = m_illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier_count.get_byte_size();
	vram_usage.next_frontier		  = m_illumination_aware_kd_tree.m_kd_tree_data.m_next_frontier.get_byte_size();
	vram_usage.next_frontier_count	  = m_illumination_aware_kd_tree.m_kd_tree_data.m_next_frontier_count.get_byte_size();

	vram_usage.training_samples			 = m_illumination_aware_kd_tree.m_kd_tree_data.m_training_samples.get_byte_size();
	vram_usage.training_sample_count	 = m_illumination_aware_kd_tree.m_kd_tree_data.m_training_sample_count.get_byte_size();
	vram_usage.nee_training_records		 = m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_training_records.get_byte_size();
	vram_usage.nee_training_record_count = m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_training_record_count.get_byte_size();

	vram_usage.batch_signatures		   = m_illumination_aware_kd_tree.m_kd_tree_data.m_batch_signatures.get_byte_size();
	vram_usage.history_signatures	   = m_illumination_aware_kd_tree.m_kd_tree_data.m_history_signatures.get_byte_size();
	vram_usage.batch_spatial_moments   = m_illumination_aware_kd_tree.m_kd_tree_data.m_batch_spatial_moments.get_byte_size();
	vram_usage.history_spatial_moments = m_illumination_aware_kd_tree.m_kd_tree_data.m_history_spatial_moments.get_byte_size();

	vram_usage.nisml_cache							= m_illumination_aware_kd_tree.m_nisml_data.m_cache.get_byte_size();
	vram_usage.nisml_representative_sample_counts	= m_illumination_aware_kd_tree.m_nisml_data.m_representative_sample_counts.get_byte_size();
	vram_usage.nisml_representative_occupied_counts = m_illumination_aware_kd_tree.m_nisml_data.m_representative_occupied_counts.get_byte_size();
	vram_usage.nisml_representative_write_locks		= m_illumination_aware_kd_tree.m_nisml_data.m_representative_write_locks.get_byte_size();
	vram_usage.nisml_representative_dirty			= m_illumination_aware_kd_tree.m_nisml_data.m_representative_dirty.get_byte_size();
	vram_usage.nisml_cache_ready					= m_illumination_aware_kd_tree.m_nisml_data.m_cache_ready.get_byte_size();
	vram_usage.nisml_pending_cell_count				= m_illumination_aware_kd_tree.m_nisml_data.m_pending_cell_count.get_byte_size();

	vram_usage.tree_cut_sampling_probabilities = m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_tree_cut_sampling_probabilities.get_byte_size();
	vram_usage.tree_cut_sampling_cdfs		   = m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_tree_cut_sampling_cdfs.get_byte_size();
	vram_usage.history_per_cell_sample_count   = m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_history_per_cell_sample_count.get_byte_size();
	vram_usage.history_per_cell_normal_sum_x   = m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_history_per_cell_normal_sum_x.get_byte_size();
	vram_usage.history_per_cell_normal_sum_y   = m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_history_per_cell_normal_sum_y.get_byte_size();
	vram_usage.history_per_cell_normal_sum_z   = m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_history_per_cell_normal_sum_z.get_byte_size();
	vram_usage.history_per_cell_normal_count   = m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_history_per_cell_normal_count.get_byte_size();
	vram_usage.history_per_cut_node_estimated_second_moment =
		m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_history_per_cut_node_estimated_second_moment.get_byte_size();
	vram_usage.history_per_cut_node_sample_count =
		m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_history_per_cut_node_sample_count.get_byte_size();
	vram_usage.batch_per_cut_node_second_moment_sum =
		m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_batch_per_cut_node_second_moment_sum.get_byte_size();
	vram_usage.batch_per_cut_node_sample_count = m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_batch_per_cut_node_sample_count.get_byte_size();

	vram_usage.tree_cut_sampling_prior_pdfs = m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_tree_cut_sampling_prior_pdfs.get_byte_size();
	vram_usage.tree_cut_sampling_prior_cdfs = m_illumination_aware_kd_tree.m_nee_learnt_distributions_data.m_tree_cut_sampling_prior_cdfs.get_byte_size();

	return vram_usage;
}
