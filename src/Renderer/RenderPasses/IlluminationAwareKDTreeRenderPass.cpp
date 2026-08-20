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

const std::string IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID									= "Reset Tree";
const std::string IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_LIGHT_CLUSTERING_KERNEL_ID				= "Initialize Root Light Clustering";
const std::string IlluminationAwareKDTreeRenderPass::ACCUMULATE_NORMAL_FACE_OBSERVATIONS_KERNEL_ID			= "Accumulate Normal Face Observations";
const std::string IlluminationAwareKDTreeRenderPass::ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_KERNEL_ID		= "Allocate Normal Face Light Clusterings";
const std::string IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID			= "Accumulate Batch Training Samples";
const std::string IlluminationAwareKDTreeRenderPass::ACCUMULATE_LIGHT_CLUSTERING_TRAINING_SAMPLES_KERNEL_ID = "Accumulate Light Clustering Training Samples";
const std::string IlluminationAwareKDTreeRenderPass::COMMIT_LIGHT_CLUSTER_RESERVOIR_PROPOSALS_KERNEL_ID		= "Commit Light Cluster Reservoir Proposals";
const std::string IlluminationAwareKDTreeRenderPass::UPDATE_LIGHT_CLUSTER_STATISTICS_KERNEL_ID				= "Update Light Cluster Statistics";
const std::string IlluminationAwareKDTreeRenderPass::REFINE_LIGHT_CLUSTERINGS_KERNEL_ID						= "Refine Light Clusterings";
const std::string IlluminationAwareKDTreeRenderPass::APPLY_PENDING_LIGHT_CLUSTER_Q_UPDATES_KERNEL_ID		= "Apply Pending Light Cluster Q Updates";
const std::string IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID		= "Accumulate Batch Statistics Into History";
const std::string IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_STATISTICS_KERNEL_ID				= "Reset Batch KD-Tree Statistics";
const std::string IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_LIGHT_CLUSTERING_STATISTICS_KERNEL_ID =
	"Reset Batch KD-Tree And Light Clustering Statistics";
const std::string IlluminationAwareKDTreeRenderPass::EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID		= "Expand One Lookahead Level";
const std::string IlluminationAwareKDTreeRenderPass::REPLAY_TRAINING_SAMPLES_KERNEL_ID			= "Replay Training Samples";
const std::string IlluminationAwareKDTreeRenderPass::INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID	= "Initialize Created Node History";
const std::string IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID = "Mark Guiding Cells For Splitting";
const std::string IlluminationAwareKDTreeRenderPass::PROMOTE_GUIDING_CELLS_KERNEL_ID			= "Promote Guiding Cells";
const std::string IlluminationAwareKDTreeRenderPass::REPLAY_NISML_TRAINING_SAMPLES_KERNEL_ID	= "Replay NISML Training Samples";
const std::string IlluminationAwareKDTreeRenderPass::BUILD_NISML_CACHES_KERNEL_ID				= "Build NISML Caches";

IlluminationAwareKDTreeRenderPass::IlluminationAwareKDTreeRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME, renderer, options)
{
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/ResetTree.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID]->set_kernel_function_name("IlluminationAwareKDTree_ResetTree");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_TREE_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_LIGHT_CLUSTERING_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_LIGHT_CLUSTERING_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_LIGHT_CLUSTERING_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/InitializeRootLightClustering.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_LIGHT_CLUSTERING_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_InitializeRootLightClustering");
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_LIGHT_CLUSTERING_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_NORMAL_FACE_OBSERVATIONS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::ACCUMULATE_NORMAL_FACE_OBSERVATIONS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_NORMAL_FACE_OBSERVATIONS_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/AccumulateNormalFaceObservations.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_NORMAL_FACE_OBSERVATIONS_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_AccumulateNormalFaceObservations");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_NORMAL_FACE_OBSERVATIONS_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/AllocateNormalFaceLightClusterings.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_AllocateNormalFaceLightClusterings");
	m_kernels[IlluminationAwareKDTreeRenderPass::ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/AccumulateBatchTrainingSamples.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_AccumulateBatchTrainingSamples");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_LIGHT_CLUSTERING_TRAINING_SAMPLES_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::ACCUMULATE_LIGHT_CLUSTERING_TRAINING_SAMPLES_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_LIGHT_CLUSTERING_TRAINING_SAMPLES_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/AccumulateLightClusteringTrainingSamples.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_LIGHT_CLUSTERING_TRAINING_SAMPLES_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_AccumulateLightClusteringTrainingSamples");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_LIGHT_CLUSTERING_TRAINING_SAMPLES_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::COMMIT_LIGHT_CLUSTER_RESERVOIR_PROPOSALS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::COMMIT_LIGHT_CLUSTER_RESERVOIR_PROPOSALS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::COMMIT_LIGHT_CLUSTER_RESERVOIR_PROPOSALS_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/CommitLightClusterReservoirProposals.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::COMMIT_LIGHT_CLUSTER_RESERVOIR_PROPOSALS_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_CommitLightClusterReservoirProposals");
	m_kernels[IlluminationAwareKDTreeRenderPass::COMMIT_LIGHT_CLUSTER_RESERVOIR_PROPOSALS_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::UPDATE_LIGHT_CLUSTER_STATISTICS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::UPDATE_LIGHT_CLUSTER_STATISTICS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::UPDATE_LIGHT_CLUSTER_STATISTICS_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/UpdateLightClusterStatistics.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::UPDATE_LIGHT_CLUSTER_STATISTICS_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_UpdateLightClusterStatistics");
	m_kernels[IlluminationAwareKDTreeRenderPass::UPDATE_LIGHT_CLUSTER_STATISTICS_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::REFINE_LIGHT_CLUSTERINGS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::REFINE_LIGHT_CLUSTERINGS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::REFINE_LIGHT_CLUSTERINGS_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY
																										   "/IlluminationAwareKDTree/RefineLightClusterings.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::REFINE_LIGHT_CLUSTERINGS_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_RefineLightClusterings");
	m_kernels[IlluminationAwareKDTreeRenderPass::REFINE_LIGHT_CLUSTERINGS_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::APPLY_PENDING_LIGHT_CLUSTER_Q_UPDATES_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::APPLY_PENDING_LIGHT_CLUSTER_Q_UPDATES_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::APPLY_PENDING_LIGHT_CLUSTER_Q_UPDATES_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/ApplyPendingLightClusterQUpdates.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::APPLY_PENDING_LIGHT_CLUSTER_Q_UPDATES_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_ApplyPendingLightClusterQUpdates");
	m_kernels[IlluminationAwareKDTreeRenderPass::APPLY_PENDING_LIGHT_CLUSTER_Q_UPDATES_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/AccumulateBatchStatisticsIntoHistory.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_AccumulateBatchStatisticsIntoHistory");
	m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_STATISTICS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_STATISTICS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_STATISTICS_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/ResetBatchKDTreeStatistics.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_STATISTICS_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_ResetBatchKDTreeStatistics");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_STATISTICS_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});

	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_LIGHT_CLUSTERING_STATISTICS_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_LIGHT_CLUSTERING_STATISTICS_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_LIGHT_CLUSTERING_STATISTICS_KERNEL_ID]->set_kernel_file_path(
		DEVICE_KERNELS_DIRECTORY "/IlluminationAwareKDTree/ResetBatchKDTreeAndLightClusteringStatistics.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_LIGHT_CLUSTERING_STATISTICS_KERNEL_ID]->set_kernel_function_name(
		"IlluminationAwareKDTree_ResetBatchKDTreeAndLightClusteringStatistics");
	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_LIGHT_CLUSTERING_STATISTICS_KERNEL_ID]->synchronize_options_with(m_compiler_options,
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
	bool using_learning_to_cluster									 = is_using_learning_to_cluster(*m_compiler_options);
	if (!using_nisml)
	{
		active_kernels.erase(REPLAY_NISML_TRAINING_SAMPLES_KERNEL_ID);
		active_kernels.erase(BUILD_NISML_CACHES_KERNEL_ID);
	}
	if (!using_learning_to_cluster)
	{
		active_kernels.erase(INITIALIZE_ROOT_LIGHT_CLUSTERING_KERNEL_ID);
		active_kernels.erase(ACCUMULATE_NORMAL_FACE_OBSERVATIONS_KERNEL_ID);
		active_kernels.erase(ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_KERNEL_ID);
		active_kernels.erase(ACCUMULATE_LIGHT_CLUSTERING_TRAINING_SAMPLES_KERNEL_ID);
		active_kernels.erase(COMMIT_LIGHT_CLUSTER_RESERVOIR_PROPOSALS_KERNEL_ID);
		active_kernels.erase(UPDATE_LIGHT_CLUSTER_STATISTICS_KERNEL_ID);
		active_kernels.erase(REFINE_LIGHT_CLUSTERINGS_KERNEL_ID);
		active_kernels.erase(APPLY_PENDING_LIGHT_CLUSTER_Q_UPDATES_KERNEL_ID);
		active_kernels.erase(RESET_BATCH_KD_TREE_AND_LIGHT_CLUSTERING_STATISTICS_KERNEL_ID);
	}

	return active_kernels;
}

void IlluminationAwareKDTreeRenderPass::resize(unsigned int new_width, unsigned int new_height) {}

bool IlluminationAwareKDTreeRenderPass::pre_sample_update(float delta_time)
{
	m_nisml_representative_capacity = std::max(m_nisml_representative_capacity, 1);

	if (!is_render_pass_used(*m_renderer->get_global_compiler_options()))
	{
		m_buffers_need_reallocation		  = true;
		m_nisml_hash_occupied_entry_count = 0;

		return m_illumination_aware_kd_tree.free();
	}

	unsigned int nisml_representative_capacity = static_cast<unsigned int>(m_nisml_representative_capacity);
	bool nisml_representative_capacity_changed = m_illumination_aware_kd_tree.m_nisml_data.m_representative_capacity != nisml_representative_capacity;
	if (nisml_representative_capacity_changed)
		m_buffers_need_reallocation = true;

	unsigned int nisml_hash_table_reserved_bytes = static_cast<unsigned int>(m_nisml_hash_table_size_mb) * 1000000u;
	unsigned int nisml_hash_normal_precision	 = static_cast<unsigned int>(m_nisml_hash_normal_precision);
	bool nisml_hash_settings_changed			 = m_illumination_aware_kd_tree.m_nisml_data.m_hash_table_reserved_bytes != nisml_hash_table_reserved_bytes ||
												   m_illumination_aware_kd_tree.m_nisml_data.m_hash_normal_precision != nisml_hash_normal_precision;
	if (nisml_hash_settings_changed)
		m_buffers_need_reallocation = true;

	bool render_data_invalidated = false;
	if (m_buffers_need_reallocation)
	{
		m_illumination_aware_kd_tree.resize(m_nodes_buffer_capacity, m_training_sample_buffer_capacity, nisml_representative_capacity,
											nisml_hash_table_reserved_bytes, nisml_hash_normal_precision);

		m_buffers_need_reallocation = false;
		render_data_invalidated		= true;

		// Reallocation creates fresh cache buffers whose metadata must be initialized even when the tree is frozen.
		m_illumination_aware_kd_tree.m_nisml_data.clear_representative_metadata();
		reset(false);
	}

	IlluminationAwareKDTreeDevice kd_tree_device = m_illumination_aware_kd_tree.to_device(m_renderer->get_render_data());
	unsigned int node_reset_thread_count		 = kd_tree_device.core.node_capacity;

	void* launch_args[] = { &kd_tree_device };
	if (is_using_learning_to_cluster(*m_renderer->get_global_compiler_options()))
	{
		unsigned int reservoir_proposal_reset_thread_count =
			kd_tree_device.learning_to_cluster.light_clustering_capacity * kd_tree_device.learning_to_cluster.pending_record_stride;
		unsigned int reset_thread_count			= std::max(node_reset_thread_count, reservoir_proposal_reset_thread_count);
		void* learning_to_cluster_launch_args[] = { &kd_tree_device };
		m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_AND_LIGHT_CLUSTERING_STATISTICS_KERNEL_ID]->launch_asynchronous(
			1024, 1, reset_thread_count, 1, learning_to_cluster_launch_args, m_renderer->get_main_stream());

		if (m_renderer->get_render_data().render_settings.sample_number == 0)
		{
			const LightTreeSGBuildResult<OrochiBuffer>& light_tree_sg_build_result = m_renderer->get_light_tree_sg_sampling_data_structure().get_build_result();
			const std::vector<unsigned int>& second_tree_cut_node_indices		   = light_tree_sg_build_result.second_tree_cut_node_indices;
			unsigned int effective_second_tree_cut_size							   = light_tree_sg_build_result.effective_second_tree_cut_size;
			kd_tree_device.learning_to_cluster.effective_initial_light_cut_size	   = effective_second_tree_cut_size;
			m_renderer->get_render_data().kd_tree_device.learning_to_cluster.effective_initial_light_cut_size = effective_second_tree_cut_size;
			if (!second_tree_cut_node_indices.empty() && effective_second_tree_cut_size > 0)
			{
				m_illumination_aware_kd_tree.m_initial_light_cut_node_indices.upload_data_partial(0, second_tree_cut_node_indices.data(),
																								  effective_second_tree_cut_size);

				LightTreeSGDevice light_tree_sg						 = m_renderer->get_render_data().light_tree_sg;
				void* initialize_root_light_clustering_launch_args[] = { &kd_tree_device, &light_tree_sg };
				m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_LIGHT_CLUSTERING_KERNEL_ID]->launch_asynchronous(
					IlluminationAwareKDTreeLightClusteringBlockSize, 1, IlluminationAwareKDTreeLightClusteringBlockSize, 1,
					initialize_root_light_clustering_launch_args, m_renderer->get_main_stream());
			}
		}

		return render_data_invalidated;
	}

	m_kernels[IlluminationAwareKDTreeRenderPass::RESET_BATCH_KD_TREE_STATISTICS_KERNEL_ID]->launch_asynchronous(1024, 1, node_reset_thread_count, 1,
																												launch_args, m_renderer->get_main_stream());

	return render_data_invalidated;
}

bool IlluminationAwareKDTreeRenderPass::is_using_nisml(const GPUKernelCompilerOptions& compiler_options) const
{
	return ILLUMINATION_AWARE_KD_TREE_IS_NISML(compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR),
											   compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY));
}

bool IlluminationAwareKDTreeRenderPass::is_using_learning_to_cluster(const GPUKernelCompilerOptions& compiler_options) const
{
	return ILLUMINATION_AWARE_KD_TREE_IS_LEARNING_TO_CLUSTER(compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR),
															 compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY));
}

void IlluminationAwareKDTreeRenderPass::build_nisml(HIPRTRenderData& render_data)
{
	if (render_data.nisml.cluster_node_indices == nullptr || render_data.nisml.cluster_count == 0 || render_data.nisml.cluster_count > NISML_MAX_CLUSTER_COUNT)
		return;

	IlluminationAwareKDTreeDevice kd_tree_device = m_illumination_aware_kd_tree.to_device(render_data);
	if (kd_tree_device.nisml.nisml_hash_table_capacity == 0u)
		return;

	HIPRTRenderData cache_render_data = render_data;
	cache_render_data.kd_tree_device  = kd_tree_device;
	void* launch_args[]				  = { &kd_tree_device, &cache_render_data };
	unsigned int hash_table_capacity  = kd_tree_device.nisml.nisml_hash_table_capacity;
	m_kernels[IlluminationAwareKDTreeRenderPass::BUILD_NISML_CACHES_KERNEL_ID]->launch_asynchronous(256, 1, hash_table_capacity, 1, launch_args,
																									m_renderer->get_main_stream());
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

	IlluminationAwareKDTreeDevice kd_tree_device = render_data.kd_tree_device;

	// Using -1 because this counter is 0-based but the SPPs displayed at the top of the UI are 1-based. This is just to match the user's HUD
	bool refinement_spp_budget_reached = render_data.render_settings.sample_number > kd_tree_device.core.user_settings.stop_refining_after_SPP - 1;
	if (!m_frozen_tree && !refinement_spp_budget_reached)
	{
		// TODO maybe download the training_sample_count and launch the kernel with a single thread per sample instead of launching a fixed number of threads
		// and
		// having threads beyond the training_sample_count do nothing? Maybe worth it in perf despite CPU overhead?
		void* launch_args[] = { &kd_tree_device };
		m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID]->launch_asynchronous(
			256, 1, kd_tree_device.core.training_sample_capacity, 1, launch_args, m_renderer->get_main_stream());

		m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID]->launch_asynchronous(
			256, 1, kd_tree_device.core.node_capacity, 1, launch_args, m_renderer->get_main_stream());

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

			unsigned char* any_cell_needs_split_host_pinned = m_illumination_aware_kd_tree.m_any_cell_needs_split_host_pinned.get_host_pinned_pointer();
			*any_cell_needs_split_host_pinned				= 0;
			m_illumination_aware_kd_tree.m_any_cell_needs_split.memset_whole_buffer_async(any_cell_needs_split_host_pinned, 1, m_renderer->get_main_stream());

			void* mark_guiding_cell_launch_args[] = { &kd_tree_device };
			m_kernels[IlluminationAwareKDTreeRenderPass::MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID]->launch_asynchronous(
				256, 1, kd_tree_device.core.node_capacity, 1, mark_guiding_cell_launch_args, m_renderer->get_main_stream());

			// The host-pinned flag lets us break out when no cell was marked for splitting.
			m_illumination_aware_kd_tree.m_any_cell_needs_split.download_data_into(any_cell_needs_split_host_pinned);
			if (*any_cell_needs_split_host_pinned == 0)
				break;

			// The number of guiding nodes before the splitting is read directly by the GPU promotion kernel.
			// The GPU kernel reads the active guiding count directly and returns for threads beyond the current count.
			// Launching at node capacity avoids synchronizing these counters back to the host.
			void* promotion_launch_args[] = { &kd_tree_device };
			m_kernels[IlluminationAwareKDTreeRenderPass::PROMOTE_GUIDING_CELLS_KERNEL_ID]->launch_asynchronous(
				1024, 1, kd_tree_device.core.node_capacity * 1024, 1, promotion_launch_args, m_renderer->get_main_stream());
		}

		m_illumination_aware_kd_tree.m_kd_tree_data.m_active_guiding_node_count.download_data_async(&m_cached_current_guiding_node_count,
																									m_renderer->get_main_stream());
		m_illumination_aware_kd_tree.m_kd_tree_data.m_node_count.download_data_async(&m_cached_current_node_count, m_renderer->get_main_stream());
	}

	if (is_using_learning_to_cluster(compiler_options))
	{
		bool learning_to_cluster_learning_budget_reached =
			render_data.render_settings.sample_number >= m_learning_to_cluster_learning_spp && m_learning_to_cluster_learning_spp > 0;
		if (learning_to_cluster_learning_budget_reached)
			return;

		void* learning_to_cluster_launch_args[] = { &kd_tree_device };
		m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_NORMAL_FACE_OBSERVATIONS_KERNEL_ID]->launch_asynchronous(
			256, 1, kd_tree_device.learning_to_cluster_training_sample_capacity, 1, learning_to_cluster_launch_args, m_renderer->get_main_stream());

		unsigned int maximum_light_clustering_work_count =
			kd_tree_device.core.node_capacity * SurfaceNormalFace_Count * IlluminationAwareKDTreeLightClusteringBlockSize;
		m_kernels[IlluminationAwareKDTreeRenderPass::ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_KERNEL_ID]->launch_asynchronous(
			IlluminationAwareKDTreeLightClusteringBlockSize, 1, maximum_light_clustering_work_count, 1, learning_to_cluster_launch_args,
			m_renderer->get_main_stream());

		m_kernels[IlluminationAwareKDTreeRenderPass::ACCUMULATE_LIGHT_CLUSTERING_TRAINING_SAMPLES_KERNEL_ID]->launch_asynchronous(
			256, 1, kd_tree_device.learning_to_cluster_training_sample_capacity, 1, learning_to_cluster_launch_args, m_renderer->get_main_stream());

		unsigned int maximum_light_clustering_reservoir_proposal_count =
			kd_tree_device.learning_to_cluster.light_clustering_capacity * kd_tree_device.learning_to_cluster.pending_record_stride;
		m_kernels[IlluminationAwareKDTreeRenderPass::COMMIT_LIGHT_CLUSTER_RESERVOIR_PROPOSALS_KERNEL_ID]->launch_asynchronous(
			IlluminationAwareKDTreeLightClusteringBlockSize, 1, maximum_light_clustering_reservoir_proposal_count, 1, learning_to_cluster_launch_args,
			m_renderer->get_main_stream());

		LightTreeSGDevice light_tree_sg		 = render_data.light_tree_sg;
		void* light_clustering_launch_args[] = { &kd_tree_device, &light_tree_sg };
		m_kernels[IlluminationAwareKDTreeRenderPass::UPDATE_LIGHT_CLUSTER_STATISTICS_KERNEL_ID]->launch_asynchronous(
			IlluminationAwareKDTreeLightClusteringBlockSize, 1, maximum_light_clustering_work_count, 1, light_clustering_launch_args,
			m_renderer->get_main_stream());

		m_kernels[IlluminationAwareKDTreeRenderPass::REFINE_LIGHT_CLUSTERINGS_KERNEL_ID]->launch_asynchronous(
			IlluminationAwareKDTreeLightClusteringBlockSize, 1, maximum_light_clustering_work_count, 1, light_clustering_launch_args,
			m_renderer->get_main_stream());

		m_kernels[IlluminationAwareKDTreeRenderPass::APPLY_PENDING_LIGHT_CLUSTER_Q_UPDATES_KERNEL_ID]->launch_asynchronous(
			IlluminationAwareKDTreeLightClusteringBlockSize, 1, maximum_light_clustering_work_count, 1, light_clustering_launch_args,
			m_renderer->get_main_stream());
	}

	if (is_using_nisml(compiler_options))
	{
		if (!render_data.nisml.learning_enabled)
			return;

		unsigned int training_record_capacity = render_data.nisml.training_record_capacity;
		if (training_record_capacity > 0)
		{
			void* replay_nisml_training_samples_launch_args[] = { &render_data };
			m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_NISML_TRAINING_SAMPLES_KERNEL_ID]->launch_asynchronous(
				256, 1, training_record_capacity, 1, replay_nisml_training_samples_launch_args, m_renderer->get_main_stream());
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

		m_kernels[IlluminationAwareKDTreeRenderPass::REPLAY_TRAINING_SAMPLES_KERNEL_ID]->launch_asynchronous(
			256, 1, kd_tree_device.core.training_sample_capacity, 1, expansion_launch_args, m_renderer->get_main_stream());

		m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID]->launch_asynchronous(
			256, 1, kd_tree_device.core.node_capacity, 1, expansion_launch_args, m_renderer->get_main_stream());

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

	// The tree may reuse node indices after a full reset, so clear the separate hash-backed cache before rebuilding the root.
	m_illumination_aware_kd_tree.m_nisml_data.clear_representative_metadata();
	m_nisml_hash_occupied_entry_count = 0;
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

int& IlluminationAwareKDTreeRenderPass::get_learning_to_cluster_learning_spp()
{
	return m_learning_to_cluster_learning_spp;
}

int& IlluminationAwareKDTreeRenderPass::get_nisml_representative_capacity()
{
	return m_nisml_representative_capacity;
}

int& IlluminationAwareKDTreeRenderPass::get_nisml_hash_table_size_mb()
{
	return m_nisml_hash_table_size_mb;
}

int& IlluminationAwareKDTreeRenderPass::get_nisml_hash_normal_precision()
{
	return m_nisml_hash_normal_precision;
}

unsigned int IlluminationAwareKDTreeRenderPass::get_nisml_hash_occupied_entry_count() const
{
	return m_nisml_hash_occupied_entry_count;
}

unsigned int IlluminationAwareKDTreeRenderPass::get_nisml_hash_table_capacity() const
{
	return static_cast<unsigned int>(m_illumination_aware_kd_tree.m_nisml_data.m_hash_table_capacity);
}

int& IlluminationAwareKDTreeRenderPass::get_current_node_buffer_capacity()
{
	return m_nodes_buffer_capacity;
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

	vram_usage.active_guiding_nodes		 = m_illumination_aware_kd_tree.m_kd_tree_data.m_active_guiding_nodes.get_byte_size();
	vram_usage.active_guiding_node_count = m_illumination_aware_kd_tree.m_kd_tree_data.m_active_guiding_node_count.get_byte_size();
	vram_usage.needs_split				 = m_illumination_aware_kd_tree.m_kd_tree_data.m_needs_split.get_byte_size();

	vram_usage.current_frontier		  = m_illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier.get_byte_size();
	vram_usage.current_frontier_count = m_illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier_count.get_byte_size();
	vram_usage.next_frontier		  = m_illumination_aware_kd_tree.m_kd_tree_data.m_next_frontier.get_byte_size();
	vram_usage.next_frontier_count	  = m_illumination_aware_kd_tree.m_kd_tree_data.m_next_frontier_count.get_byte_size();

	vram_usage.training_samples		 = m_illumination_aware_kd_tree.m_kd_tree_data.m_training_samples.get_byte_size();
	vram_usage.training_sample_count = m_illumination_aware_kd_tree.m_kd_tree_data.m_training_sample_count.get_byte_size();

	vram_usage.batch_signatures		   = m_illumination_aware_kd_tree.m_kd_tree_data.m_batch_signatures.get_byte_size();
	vram_usage.history_signatures	   = m_illumination_aware_kd_tree.m_kd_tree_data.m_history_signatures.get_byte_size();
	vram_usage.batch_spatial_moments   = m_illumination_aware_kd_tree.m_kd_tree_data.m_batch_spatial_moments.get_byte_size();
	vram_usage.history_spatial_moments = m_illumination_aware_kd_tree.m_kd_tree_data.m_history_spatial_moments.get_byte_size();

	vram_usage.nisml_cache							= m_illumination_aware_kd_tree.m_nisml_data.m_cache.get_byte_size();
	vram_usage.nisml_hash_keys						= m_illumination_aware_kd_tree.m_nisml_data.m_hash_keys.get_byte_size();
	vram_usage.nisml_hash_entry_states				= m_illumination_aware_kd_tree.m_nisml_data.m_hash_entry_states.get_byte_size();
	vram_usage.nisml_hash_occupied_entry_count		= m_illumination_aware_kd_tree.m_nisml_data.m_hash_occupied_entry_count.get_byte_size();
	vram_usage.nisml_representative_sample_counts	= m_illumination_aware_kd_tree.m_nisml_data.m_representative_sample_counts.get_byte_size();
	vram_usage.nisml_representative_occupied_counts = m_illumination_aware_kd_tree.m_nisml_data.m_representative_occupied_counts.get_byte_size();
	vram_usage.nisml_representative_valid			= m_illumination_aware_kd_tree.m_nisml_data.m_representative_valid.get_byte_size();
	vram_usage.nisml_representative_write_locks		= m_illumination_aware_kd_tree.m_nisml_data.m_representative_write_locks.get_byte_size();
	vram_usage.nisml_representative_dirty			= m_illumination_aware_kd_tree.m_nisml_data.m_representative_dirty.get_byte_size();
	vram_usage.nisml_cache_ready					= m_illumination_aware_kd_tree.m_nisml_data.m_cache_ready.get_byte_size();
	vram_usage.nisml_pending_cell_count				= m_illumination_aware_kd_tree.m_nisml_data.m_pending_cell_count.get_byte_size();

	return vram_usage;
}
