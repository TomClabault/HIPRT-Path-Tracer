/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_DEVICE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterUserSettings.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLightClusterBatchStatisticsSoADevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeSurfaceNormalFace.h"
#include "HostDeviceCommon/AtomicType.h"
#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeLearningToClusterOptions.h"
#include "HostDeviceCommon/Maths/VecTypes.h"

struct IlluminationAwareKDTreeLearningToClusterTrainingSample;

struct IlluminationAwareKDTreeLightClusterStatistics
{
	// Q_x(c): estimated contribution of this light cluster, light clusters of the cut are sampled proportionally to this value
	float estimated_importance_Q = 0.0f;
	// Fast EMA estimate before regularization by the inherited prior
	float learned_importance_Q = 0.0f;
	// Q estimate inherited at initialization, spatial promotion, or lightcut refinement
	float prior_importance_Q = 0.0f;
	// Number of real observations incorporated into learned_importance_Q
	unsigned int Q_observation_count = 0u;

	// Running mean of the selected-sample observations used by refinement
	float mean = 0.0f;

	// Running sum of squared deviations used by refinement
	float M2 = 0.0f;

	// n_c in Equations 7 and 8: how many times this cluster has actually been selected
	unsigned int visit_count = 0;

	HIPRT_DEVICE void initialize_importance_prior(float importance)
	{
		estimated_importance_Q = importance;
		learned_importance_Q   = importance;
		prior_importance_Q	   = importance;
		Q_observation_count	   = 0u;
	}

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

struct IlluminationAwareKDTreeLearningToClusterLightcutSet
{
	static constexpr unsigned int PER_MESH_ID_LIGHTCUT_COUNT	 = 2u;
	static constexpr unsigned int PER_FACE_NORMAL_LIGHTCUT_COUNT = PER_MESH_ID_LIGHTCUT_COUNT + 1u;
	static constexpr unsigned int INVALID_MESH_ID				 = 0xffffffffu;

	struct PerSurfaceLightcut
	{
		unsigned int mesh_id;
		unsigned int lightcut_index;
	};

	struct PerNormalFaceLightcuts
	{
		unsigned int shared_lightcut_index;
		PerSurfaceLightcut per_mesh_id_lightcuts[PER_MESH_ID_LIGHTCUT_COUNT];
	};

	PerNormalFaceLightcuts face_lightcuts[SurfaceNormalFace_Count];

	HIPRT_DEVICE void initialize_invalid()
	{
		for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
		{
			face_lightcuts[normal_face].shared_lightcut_index = IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
			for (unsigned int per_mesh_id_lightcut_slot = 0; per_mesh_id_lightcut_slot < PER_MESH_ID_LIGHTCUT_COUNT; per_mesh_id_lightcut_slot++)
			{
				face_lightcuts[normal_face].per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].mesh_id = INVALID_MESH_ID;
				face_lightcuts[normal_face].per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].lightcut_index =
					IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
			}
		}
	}
};

struct IlluminationAwareKDTreeLearningToClusterTrainingSampleSoADevice
{
	float3_t* positions						= nullptr;
	float3_t* shading_normals				= nullptr;
	unsigned int* mesh_ids					= nullptr;
	unsigned int* valid_for_lightcut		= nullptr;
	unsigned int* replayed_lightcut_indices = nullptr;
	unsigned int* replayed_lightcut_slots	= nullptr;
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

	HIPRT_DEVICE unsigned int resolve_lightcut_for_normal_face(unsigned int set_index, unsigned int normal_face, unsigned int mesh_id) const
	{
		const IlluminationAwareKDTreeLearningToClusterLightcutSet::PerNormalFaceLightcuts& face = normal_lightcut_sets[set_index].face_lightcuts[normal_face];

		for (unsigned int per_mesh_id_lightcut_slot = 0;
			 per_mesh_id_lightcut_slot < IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_MESH_ID_LIGHTCUT_COUNT; per_mesh_id_lightcut_slot++)
		{
			const IlluminationAwareKDTreeLearningToClusterLightcutSet::PerSurfaceLightcut& per_mesh_id_lightcut =
				face.per_mesh_id_lightcuts[per_mesh_id_lightcut_slot];
			if (per_mesh_id_lightcut.mesh_id == mesh_id && per_mesh_id_lightcut.lightcut_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
				return per_mesh_id_lightcut.lightcut_index;
		}

		return face.shared_lightcut_index;
	}

	HIPRT_DEVICE unsigned int claim_per_mesh_id_lightcut(unsigned int set_index, unsigned int normal_face, unsigned int mesh_id) const
	{
		if (mesh_id == IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID)
			return IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

		IlluminationAwareKDTreeLearningToClusterLightcutSet::PerNormalFaceLightcuts& face = normal_lightcut_sets[set_index].face_lightcuts[normal_face];
		for (unsigned int per_mesh_id_lightcut_slot = 0;
			 per_mesh_id_lightcut_slot < IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_MESH_ID_LIGHTCUT_COUNT; per_mesh_id_lightcut_slot++)
		{
#ifdef __KERNELCC__
			unsigned int previous_mesh_id = hippt::atomic_compare_exchange(&face.per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].mesh_id,
																		   IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID, mesh_id);
#else
			unsigned int previous_mesh_id = face.per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].mesh_id;
			if (previous_mesh_id == IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID)
				face.per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].mesh_id = mesh_id;
#endif
			if (previous_mesh_id == IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID || previous_mesh_id == mesh_id)
				return per_mesh_id_lightcut_slot;
		}

		return IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	}

	HIPRT_DEVICE void clone_lightcut_as_fresh_child(unsigned int shared_lightcut_index, unsigned int child_lightcut_index)
	{
		if (shared_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX ||
			child_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX || child_lightcut_index >= lightcut_capacity)
			return;

		IlluminationAwareKDTreeLightClusteringData& shared_lightcut_data = lightcut_data[shared_lightcut_index];
		IlluminationAwareKDTreeLightClusteringData& child_lightcut_data	 = lightcut_data[child_lightcut_index];
		child_lightcut_data												 = shared_lightcut_data;
		child_lightcut_data.iteration									 = 0u;
		child_lightcut_data.last_refinement_iteration					 = 0u;
		child_lightcut_data.refinement_stopped							 = false;

		lightcut_sample_counts[child_lightcut_index]				   = 0u;
		lightcut_representative_shading_contexts[child_lightcut_index] = lightcut_representative_shading_contexts[shared_lightcut_index];
		lightcut_representative_shading_context_states[child_lightcut_index] =
			hippt::atomic_load(&lightcut_representative_shading_context_states[shared_lightcut_index]);

		for (unsigned int slot = 0; slot < LearningToClusterMaximumLightCutSize; slot++)
		{
			unsigned int shared_offset = get_light_cluster_offset(shared_lightcut_index, slot);
			unsigned int child_offset  = get_light_cluster_offset(child_lightcut_index, slot);

			lightcut_node_indices[child_offset] = lightcut_node_indices[shared_offset];
			lightcut_cdfs[child_offset]			= lightcut_cdfs[shared_offset];

			float shared_estimated_importance_Q = lightcut_statistics[shared_offset].estimated_importance_Q;
			lightcut_statistics[child_offset]	= {};
			lightcut_statistics[child_offset].initialize_importance_prior(shared_estimated_importance_Q);
			lightcut_batch_statistics.reset(child_offset);
		}
	}

	IlluminationAwareKDTreeLearningToClusterUserSettings user_settings;

	IlluminationAwareKDTreeLearningToClusterTrainingSample* training_samples = nullptr;
	IlluminationAwareKDTreeLearningToClusterTrainingSampleSoADevice training_samples_soa;
	AtomicType<unsigned int>* training_sample_count = nullptr;
	unsigned int training_sample_capacity			= 0;

	AtomicType<unsigned int>* lightcut_count = nullptr;
	unsigned int lightcut_capacity			 = 0;

	IlluminationAwareKDTreeLearningToClusterLightcutSet* normal_lightcut_sets = nullptr;
	AtomicType<unsigned int>* normal_lightcut_set_count						  = nullptr;
	unsigned int normal_lightcut_set_capacity								  = 0;
	AtomicType<unsigned int>* normal_face_observation_counts				  = nullptr;

	unsigned int* initial_lightcut_node_indices	 = nullptr;
	unsigned int effective_initial_lightcut_size = 0;

	unsigned int* lightcut_node_indices								   = nullptr;
	IlluminationAwareKDTreeLightClusterStatistics* lightcut_statistics = nullptr;
	IlluminationAwareKDTreeLightClusterBatchStatisticsSoADevice lightcut_batch_statistics;
	unsigned short int* lightcut_cdfs = nullptr;

	IlluminationAwareKDTreeLightClusteringData* lightcut_data = nullptr;
	// Incremented once per sample matched to any slot.
	// Used by refinement eligibility.
	// Reset after Q updates, so it is a per-update count, not cumulative.
	AtomicType<unsigned int>* lightcut_sample_counts = nullptr;

	IlluminationAwareKDTreeSGShadingContext* lightcut_representative_shading_contexts = nullptr;
	AtomicType<unsigned int>* lightcut_representative_shading_context_states		  = nullptr;
};

#endif // #ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_DEVICE_H
