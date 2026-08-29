/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_DEVICE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterUserSettings.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeSurfaceNormalFace.h"
#include "HostDeviceCommon/AtomicType.h"
#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeLearningToClusterOptions.h"
#include "HostDeviceCommon/Maths/VecTypes.h"

struct IlluminationAwareKDTreeLearningToClusterTrainingSample;

struct IlluminationAwareKDTreeLightClusterStatistics
{
	// Q_x(c): estimated contribution of this light cluster, light clusters of the cut are sampled proportionally to this value
	float estimated_importance_Q = 0.0f;

	// Running mean of the selected-sample observations used by refinement
	float mean = 0.0f;

	// Running sum of squared deviations used by refinement
	float M2 = 0.0f;

	// n_c in Equations 7 and 8: how many times this cluster has actually been selected
	unsigned int visit_count = 0;

	HIPRT_DEVICE float get_refinement_variance() const
	{
		return M2 / static_cast<float>(visit_count + 1u);
	}
};

struct IlluminationAwareKDTreeLightClusteringData
{
	// Current number of active SG nodes in the cut
	unsigned int lightcut_size = 0;

	// Current iteration of this lightcut
	unsigned int iteration = 0;

	// t' in the paper's refinement stopping rule
	unsigned int last_refinement_iteration = 0;

	// Q has been initialized using Equation 5
	unsigned int Q0_initialized = false;

	// The persistent CDF used to sample the light cut needs to be rebuilt after Q or cut changes
	unsigned int lightcut_cdf_dirty = true;

	// Permanently set when the paper's Gamma stopping condition is reached
	unsigned int refinement_stopped = false;
};

struct IlluminationAwareKDTreeSGShadingContext
{
	// TODO those fields are duplicated with LearningToClusterTrainingSampleSoA
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
	unsigned int lightcut_indices[SurfaceNormalFace_Count];
};

struct IlluminationAwareKDTreeLearningToClusterTrainingSampleSoADevice
{
	float3_t* positions				 = nullptr;
	float3_t* shading_normals		 = nullptr;
	unsigned int* valid_for_lightcut = nullptr;
};

struct IlluminationAwareKDTreeLearningToClusterDevice
{
	static constexpr unsigned int REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT = 0u;
	static constexpr unsigned int REPRESENTATIVE_SHADING_CONTEXT_STATE_WRITING	  = 1u;
	static constexpr unsigned int REPRESENTATIVE_SHADING_CONTEXT_STATE_READY	  = 2u;

	HIPRT_DEVICE void append_learning_to_cluster_training_sample(const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample);

	HIPRT_DEVICE unsigned int get_light_cluster_offset(unsigned int lightcut_index, unsigned int slot) const
	{
		return lightcut_index * LearningToClusterMaximumLightCutSize + slot;
	}

	HIPRT_DEVICE unsigned int get_normal_face_observation_offset(unsigned int set_index, unsigned int normal_face) const
	{
		return set_index * SurfaceNormalFace_Count + normal_face;
	}

	IlluminationAwareKDTreeLearningToClusterUserSettings user_settings;

	IlluminationAwareKDTreeLearningToClusterTrainingSample* training_samples = nullptr;
	IlluminationAwareKDTreeLearningToClusterTrainingSampleSoADevice training_samples_soa;
	AtomicType<unsigned int>* training_sample_count = nullptr;
	unsigned int training_sample_capacity			= 0;

	AtomicType<unsigned int>* lightcut_count = nullptr;
	unsigned int lightcut_capacity			 = 0;

	IlluminationAwareKDTreeNormalClusteringSet* normal_lightcut_sets = nullptr;
	AtomicType<unsigned int>* normal_lightcut_set_count				 = nullptr;
	unsigned int normal_lightcut_set_capacity						 = 0;
	AtomicType<unsigned int>* normal_face_observation_counts		 = nullptr;

	unsigned int* initial_lightcut_node_indices	 = nullptr;
	unsigned int effective_initial_lightcut_size = 0;

	unsigned int* lightcut_node_indices								   = nullptr;
	IlluminationAwareKDTreeLightClusterStatistics* lightcut_statistics = nullptr;
	unsigned short int* lightcut_cdfs								   = nullptr;

	IlluminationAwareKDTreeLightClusteringData* lightcut_data = nullptr;
	// Incremented once per sample matched to any slot.
	// Used by refinement eligibility.
	// Reset after Q updates, so it is a per-update count, not cumulative.
	AtomicType<unsigned int>* lightcut_sample_counts = nullptr;

	IlluminationAwareKDTreeSGShadingContext* lightcut_representative_shading_contexts = nullptr;
	AtomicType<unsigned int>* lightcut_representative_shading_context_states		  = nullptr;
};

#endif
