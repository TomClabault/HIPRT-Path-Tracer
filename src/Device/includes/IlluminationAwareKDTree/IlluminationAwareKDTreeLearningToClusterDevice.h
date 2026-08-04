/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_DEVICE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterUserSettings.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLightClusterBatchStatistics.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLightClusterBatchStatisticsSoADevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeSurfaceNormalFace.h"
#include "HostDeviceCommon/AtomicType.h"
#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeOptions.h"
#include "HostDeviceCommon/Maths/VecTypes.h"

struct IlluminationAwareKDTreeLightClusterStatistics
{
	// Q_x(c): estimated contribution of this light cluster, light clusters of the cut are sampled proportionally to this value
	float estimated_importance_Q = 0.0f;

	// Estimate of E[Y_c^2].
	// Used to reconstruct Var[Y_c] = E[Y_c^2] - E[Y_c]^2
	float estimated_second_moment = 0.0f;

	// Variance used by the light-clustering refinement probability, Equation 7
	float variance = 0.0f;

	// n_c in Equations 7 and 8: how many times this cluster has actually been selected
	unsigned int visit_count = 0;
};

static_assert(sizeof(IlluminationAwareKDTreeLightClusterStatistics) == 16);

struct IlluminationAwareKDTreeLightClusteringData
{
	// Current number of active SG nodes in the cut
	unsigned int cut_size = 0;

	unsigned int iteration = 0;

	// t' in the paper's refinement stopping rule
	unsigned int last_refinement_iteration = 0;

	// Samples accumulated since the previous light-cut refinement attempt
	unsigned int refinement_sample_count = 0;

	// Q has been initialized using Equation 5
	unsigned int Q0_initialized = false;

	// Permanently set when the paper's Gamma stopping condition is reached
	unsigned int refinement_stopped = false;
};

struct IlluminationAwareKDTreeSGShadingContext
{
	float3_t position;
	float3_t shading_normal;
	float3_t view_direction;

	// Values already derived from the material by the SG sampler, to sample the SG lobe with the specular SG approximation
	float sg_specular_weight;
	float alpha_x;
	float alpha_y;
};

struct IlluminationAwareKDTreeNormalClusteringSet
{
	unsigned int clustering_indices[SurfaceNormalFace_Count];
};

struct IlluminationAwareKDTreeLearningToClusterDevice
{
	static constexpr unsigned int REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT = 0u;
	static constexpr unsigned int REPRESENTATIVE_SHADING_CONTEXT_STATE_WRITING	  = 1u;
	static constexpr unsigned int REPRESENTATIVE_SHADING_CONTEXT_STATE_READY	  = 2u;

	IlluminationAwareKDTreeLearningToClusterUserSettings user_settings;

	HIPRT_DEVICE unsigned int get_light_cluster_offset(unsigned int light_clustering_index, unsigned int slot) const
	{
		return light_clustering_index * IlluminationAwareKDTreeMaximumLightCutSize + slot;
	}

	HIPRT_DEVICE unsigned int get_normal_face_observation_offset(unsigned int set_index, unsigned int normal_face) const
	{
		return set_index * SurfaceNormalFace_Count + normal_face;
	}

	AtomicType<unsigned int>* light_clustering_count = nullptr;
	unsigned int light_clustering_capacity			 = 0;

	IlluminationAwareKDTreeNormalClusteringSet* normal_clustering_sets = nullptr;
	AtomicType<unsigned int>* normal_clustering_set_count			   = nullptr;
	unsigned int normal_clustering_set_capacity						   = 0;
	AtomicType<unsigned int>* normal_face_observation_counts		   = nullptr;

	unsigned int* initial_light_cut_node_indices  = nullptr;
	unsigned int effective_initial_light_cut_size = 0;

	unsigned int* light_cluster_node_indices								= nullptr;
	IlluminationAwareKDTreeLightClusterStatistics* light_cluster_statistics = nullptr;
	IlluminationAwareKDTreeLightClusterBatchStatisticsSoADevice light_cluster_batch_statistics;

	IlluminationAwareKDTreeLightClusteringData* light_clustering_data = nullptr;
	AtomicType<unsigned int>* light_clustering_batch_sample_counts	  = nullptr;

	IlluminationAwareKDTreeSGShadingContext* representative_shading_contexts = nullptr;
	AtomicType<unsigned int>* representative_shading_context_states			 = nullptr;
};

#endif
