/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_LIGHT_TREE_LIGHT_TREE_SG_SAMPLING_LEARNING_TO_CLUSTER_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_LIGHT_TREE_LIGHT_TREE_SG_SAMPLING_LEARNING_TO_CLUSTER_H

#include "Device/includes/CDF.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterCutTriangleSample.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterDevice.h"
#include "Device/includes/IlluminationAwareKDTree/LearningToClusterClusterImportance.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSampling.h"

#include "HostDeviceCommon/Material/MaterialTraits.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

template <typename MaterialType>
HIPRT_DEVICE IlluminationAwareKDTreeSGShadingContext build_light_clustering_shading_context(const float3_t& shading_point,
																							const float3_t& view_direction,
																							const float3_t& shading_normal,
																							const MaterialType& material)
{
	IlluminationAwareKDTreeSGShadingContext context{};

	context.position	   = shading_point;
	context.view_direction = view_direction;
	context.shading_normal = shading_normal;

	float sg_roughness	= MaterialConstants::ROUGHNESS_CLAMP;
	float sg_anisotropy = 0.0f;

	if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationAll && MaterialTraits<MaterialType>::is_principled)
	{
		float material_specular_weight =
			(1.0f - material.metallic) * (1.0f - material.specular_transmission * (1.0f - material.diffuse_transmission)) * material.specular;
		float specular_lobes_sum = material.coat + material.metallic + material_specular_weight;

		context.sg_specular_weight = hippt::max(material.coat, hippt::max(material.metallic, material_specular_weight));

		if (specular_lobes_sum > 0.0f)
		{
			sg_roughness = material.coat * material.coat_roughness + material.metallic * material.roughness + material_specular_weight * material.roughness;
			sg_roughness /= specular_lobes_sum;

			sg_anisotropy = material.coat * material.coat_anisotropy + material.metallic * material.anisotropy + material_specular_weight * material.anisotropy;
			sg_anisotropy /= specular_lobes_sum;
		}
	}
	else if constexpr (MaterialTraits<MaterialType>::has_metallic)
	{
		context.sg_specular_weight = 1.0f;
		sg_roughness			   = material.roughness;
		sg_anisotropy			   = material.anisotropy;
	}
	else if constexpr (MaterialTraits<MaterialType>::has_specular)
	{
		context.sg_specular_weight = material.specular;
		sg_roughness			   = material.roughness;
		sg_anisotropy			   = material.anisotropy;
	}

	sg_roughness = hippt::max(MaterialConstants::ROUGHNESS_CLAMP, sg_roughness);
	MaterialUtils::get_alphas(sg_roughness, sg_anisotropy, context.alpha_x, context.alpha_y);

	return context;
}

HIPRT_DEVICE IlluminationAwareKDTreeSGShadingContext build_light_clustering_shading_context(const float3_t& shading_point,
																							const float3_t& view_direction,
																							const float3_t& shading_normal,
																							const LightTreeSGProposalState& proposal_state)
{
	IlluminationAwareKDTreeSGShadingContext context{};
	context.position		   = shading_point;
	context.view_direction	   = view_direction;
	context.shading_normal	   = shading_normal;
	context.sg_specular_weight = proposal_state.sg_specular_weight;
	context.alpha_x			   = proposal_state.alpha_x;
	context.alpha_y			   = proposal_state.alpha_y;
	return context;
}

/**
 * Whether the lightcuts of learning to cluster hold at least one cluster: emissive geometry or the envmap. Without emissive geometry, the envmap
 * can be the only cluster of the lightcuts
 */
HIPRT_DEVICE bool learning_to_cluster_has_light_to_sample(const HIPRTRenderData& render_data)
{
	return render_data.buffers.emissive_triangles_count > 0 || render_data.kd_tree_device.learning_to_cluster.envmap_cluster_enabled;
}

HIPRT_DEVICE float light_clustering_cluster_probability(const IlluminationAwareKDTreeDevice& kd_tree,
														unsigned int lightcut_index,
														unsigned int slot,
														unsigned int lightcut_size)
{
	unsigned int offset	   = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);
	unsigned int cdf_start = slot == 0u ? 0u : kd_tree.learning_to_cluster.lightcut_cdfs[offset];
	unsigned int cdf_end   = slot + 1u < lightcut_size ? kd_tree.learning_to_cluster.lightcut_cdfs[offset + 1u] : 65535u;

	return static_cast<float>(cdf_end - cdf_start) / 65535.0f;
}

/**
 * 'lightcut_index' and 'guiding_node_index' are the results of IlluminationAwareKDTreeDevice::resolve_lightcut() at the shading point of 'context'
 */
HIPRT_DEVICE IlluminationAwareKDTreeLearningToClusterCutTriangleSample sample_cluster_from_light_cut(const HIPRTRenderData& render_data,
																									 const IlluminationAwareKDTreeSGShadingContext& context,
																									 unsigned int lightcut_index,
																									 unsigned int guiding_node_index,
																									 Xorshift32Generator& random_number_generator)
{
	IlluminationAwareKDTreeLearningToClusterCutTriangleSample result{};
	const IlluminationAwareKDTreeDevice& kd_tree = render_data.kd_tree_device;

	result.guiding_node_index = guiding_node_index;

	if (lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
	{
		unsigned int initial_lightcut_size = kd_tree.learning_to_cluster.effective_initial_lightcut_size;
		if (initial_lightcut_size == 0 || initial_lightcut_size > LearningToClusterMaximumLightCutSize)
			return result;

		float total_weight		   = 0.0f;
		float selected_weight	   = 0.0f;
		unsigned int selected_slot = initial_lightcut_size - 1;

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDF_MODEL != BSDF_LAMBERTIAN && BSDF_MODEL != BSDF_OREN_NAYAR
		SGSpecularImportanceData specular_data(context.view_direction, context.shading_normal, context.alpha_x, context.alpha_y);
#else
		SGSpecularImportanceData specular_data;
#endif

		for (unsigned int slot = 0; slot < initial_lightcut_size; slot++)
		{
			unsigned int node_index = kd_tree.learning_to_cluster.initial_lightcut_node_indices[slot];

			float weight = light_clustering_cluster_importance(kd_tree.learning_to_cluster, render_data.light_tree_sg, node_index, specular_data, context);
			weight		 = hippt::max(weight, 0.0f);
			total_weight += weight;
			if (weight > 0.0f && random_number_generator() < weight / total_weight)
			{
				selected_slot	= slot;
				selected_weight = weight;
			}
		}

		if (total_weight <= 0.0f)
		{
			float initial_lightcut_size_as_float = kd_tree.learning_to_cluster.effective_initial_lightcut_size_as_float;

			selected_slot	= random_number_generator.random_index(initial_lightcut_size, initial_lightcut_size_as_float);
			selected_weight = 1.0f;
			total_weight	= initial_lightcut_size_as_float;
		}

		result.cluster_node_index  = kd_tree.learning_to_cluster.initial_lightcut_node_indices[selected_slot];
		result.cluster_probability = selected_weight / total_weight;

		return result;
	}

	const IlluminationAwareKDTreeLightClusteringData& lightcut_data = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];

	unsigned int lightcut_size = lightcut_data.lightcut_size;
	if (lightcut_size == 0 || lightcut_size > LearningToClusterMaximumLightCutSize)
		return result;

	unsigned int cdf_offset = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, 0);

	CDFDeviceU16 lightcut_cdf;
	lightcut_cdf.cdf_u16 = kd_tree.learning_to_cluster.lightcut_cdfs + cdf_offset;
	lightcut_cdf.size	 = lightcut_size;

	unsigned int selected_slot	 = lightcut_cdf.sample(random_number_generator);
	unsigned int selected_offset = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, selected_slot);

	float selected_probability = light_clustering_cluster_probability(kd_tree, lightcut_index, selected_slot, lightcut_size);

	result.lightcut_index			 = lightcut_index;
	result.lightcut_slot			 = selected_slot;
	result.cluster_node_index		 = kd_tree.learning_to_cluster.lightcut_node_indices[selected_offset];
	result.lightcut_size_at_sampling = lightcut_size;
	result.cluster_probability		 = selected_probability;

	return result;
}

HIPRT_DEVICE bool sample_light_inside_cluster(const HIPRTRenderData& render_data,
											  const IlluminationAwareKDTreeSGShadingContext& context,
											  Xorshift32Generator& random_number_generator,
											  IlluminationAwareKDTreeLearningToClusterCutTriangleSample& sample)
{
	const LightTreeSGNodeDevice* nodes		 = render_data.light_tree_sg.nodes;
	const SpatialSGLobeDevice* spatial_lobes = render_data.light_tree_sg.spatial_lobes;
	unsigned int spatial_lobe_count			 = render_data.light_tree_sg.settings.spatial_lobe_count;
	float cumulative_probability			 = 1.0f;

	// The traversal fields of the chosen child are carried to the next level: loading them again from the node at the next level would be one
	// more dependent memory round trip per level
	unsigned int current_triangle_count				  = nodes[sample.cluster_node_index].triangle_count;
	unsigned int current_left_child_or_first_triangle = nodes[sample.cluster_node_index].left_child_index_or_first_triangle_index;

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDF_MODEL != BSDF_LAMBERTIAN && BSDF_MODEL != BSDF_OREN_NAYAR
	SGSpecularImportanceData specular_data(context.view_direction, context.shading_normal, context.alpha_x, context.alpha_y);
#else
	SGSpecularImportanceData specular_data;
#endif

	while (current_triangle_count == 0)
	{
		unsigned int left_index					= current_left_child_or_first_triangle;
		unsigned int right_index				= left_index + 1;
		const LightTreeSGNodeDevice& left_node	= nodes[left_index];
		const LightTreeSGNodeDevice& right_node = nodes[right_index];

		// The lobes are addressed from the node index, not through the lobe pointer of the node, so that they load together with the node
		float left_importance =
			light_tree_sg_node_importance(left_node, spatial_lobes + left_index * spatial_lobe_count, spatial_lobe_count, specular_data, context.position,
										  context.view_direction, context.shading_normal, context.sg_specular_weight, context.alpha_x, context.alpha_y);
		float right_importance =
			light_tree_sg_node_importance(right_node, spatial_lobes + right_index * spatial_lobe_count, spatial_lobe_count, specular_data, context.position,
										  context.view_direction, context.shading_normal, context.sg_specular_weight, context.alpha_x, context.alpha_y);

		float importance_sum = left_importance + right_importance;
		if (importance_sum <= 0.0f)
			return false;

		float left_probability = left_importance / importance_sum;
		bool go_left		   = random_number_generator() < left_probability;

		cumulative_probability *= go_left ? left_probability : 1.0f - left_probability;
		current_triangle_count = go_left ? left_node.triangle_count : right_node.triangle_count;
		current_left_child_or_first_triangle =
			go_left ? left_node.left_child_index_or_first_triangle_index : right_node.left_child_index_or_first_triangle_index;
	}

	unsigned int local_triangle_index	   = random_number_generator.random_index(current_triangle_count);
	unsigned int light_tree_triangle_index = current_left_child_or_first_triangle + local_triangle_index;
	int triangle_index					   = render_data.light_tree_sg.indices_array[light_tree_triangle_index];
	sample.emissive_triangle_global_index  = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

	cumulative_probability /= static_cast<float>(current_triangle_count);
	sample.conditional_triangle_probability = cumulative_probability;

	return sample.conditional_triangle_probability > 0.0f;
}

/**
 * Overload for a caller that already resolved the lightcut of the shading point of 'context': 'lightcut_index' and 'guiding_node_index' are the
 * results of IlluminationAwareKDTreeDevice::resolve_lightcut() there
 */
HIPRT_DEVICE IlluminationAwareKDTreeLearningToClusterCutTriangleSample
sample_one_emissive_triangle_learning_to_cluster(const HIPRTRenderData& render_data,
												 const IlluminationAwareKDTreeSGShadingContext& context,
												 unsigned int lightcut_index,
												 unsigned int guiding_node_index,
												 Xorshift32Generator& random_number_generator)
{
	IlluminationAwareKDTreeLearningToClusterCutTriangleSample sample =
		sample_cluster_from_light_cut(render_data, context, lightcut_index, guiding_node_index, random_number_generator);
	if (sample.cluster_probability <= 0.0f)
		return {};

	if (sample.is_envmap())
	{
		// The caller samples the direction from the envmap distribution
		sample.conditional_triangle_probability = 1.0f;

		return sample;
	}

	if (!sample_light_inside_cluster(render_data, context, random_number_generator, sample))
		return sample;

	return sample;
}

HIPRT_DEVICE IlluminationAwareKDTreeLearningToClusterCutTriangleSample
sample_one_emissive_triangle_learning_to_cluster(const HIPRTRenderData& render_data,
												 const IlluminationAwareKDTreeSGShadingContext& context,
												 unsigned int mesh_id,
												 Xorshift32Generator& random_number_generator)
{
	unsigned int guiding_node_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	unsigned int lightcut_index		= render_data.kd_tree_device.resolve_lightcut(context, mesh_id, &guiding_node_index);

	return sample_one_emissive_triangle_learning_to_cluster(render_data, context, lightcut_index, guiding_node_index, random_number_generator);
}

/**
 * 'lightcut_index' is the result of IlluminationAwareKDTreeDevice::resolve_lightcut() at the shading point of 'context': the caller resolves it,
 * once for every PDF and sample of that shading point
 */
HIPRT_DEVICE float pdf_of_emissive_triangle_learning_to_cluster(const HIPRTRenderData& render_data,
																const IlluminationAwareKDTreeSGShadingContext& context,
																unsigned int lightcut_index,
																int emissive_triangle_global_index)
{
	if (emissive_triangle_global_index < 0 || render_data.buffers.emissive_triangles_count == 0)
		return 0.0f;

	const IlluminationAwareKDTreeDevice& kd_tree = render_data.kd_tree_device;
	bool initial_lightcut						 = lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;

	unsigned int lightcut_size = initial_lightcut ? kd_tree.learning_to_cluster.effective_initial_lightcut_size
												  : kd_tree.learning_to_cluster.lightcut_data[lightcut_index].lightcut_size;
	if (lightcut_size == 0 || lightcut_size > LearningToClusterMaximumLightCutSize)
		return 0.0f;

	const unsigned int* cluster_node_indices =
		initial_lightcut ? kd_tree.learning_to_cluster.initial_lightcut_node_indices
						 : kd_tree.learning_to_cluster.lightcut_node_indices + kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, 0u);
	const LightTreeSGNodeDevice* nodes = render_data.light_tree_sg.nodes;
	const uint2_t* node_path_codes	   = render_data.light_tree_sg.node_path_codes;
	unsigned int bit_trail			   = render_data.light_tree_sg.bit_trails[emissive_triangle_global_index];
	unsigned int selected_slot		   = lightcut_size;
	unsigned int selected_depth		   = LightTreeSGDevice::INVALID_NODE_PATH_DEPTH;

	// The cluster of the hit triangle is the node of the cut whose path from the root is a prefix of the trail of the triangle; ancestors above
	// the cut are not sampled.
	//
	// Every slot is tested on its own, without an early exit, so that the loads of many slots can be in flight together (following the trail
	// from the root and scanning the cut at each depth would be a chain of dependent loads). The shallowest match is kept: this is the node that the
	// root-down search finds first.
	//
	// The slots are tested in unrolled groups of 8 so that the loads of a whole group are in flight together: in a plain loop over the slots,
	// every slot would wait for its own loads before the loads of the next slot are issued
	constexpr unsigned int slot_group_size = 8;
	// The last group reads the slots up to the next multiple of the group size, which must stay inside the storage of the cut
	static_assert(LearningToClusterMaximumLightCutSize % 8 == 0, "The cut search of the PDF reads the slots of a cut by groups of 8");

	for (unsigned int group_start = 0; group_start < lightcut_size; group_start += slot_group_size)
	{
		// The node indices of the group are read with two 16-byte loads: with one load per slot, the compiler issued them one after the other with
		// a wait after each. A cut has LearningToClusterMaximumLightCutSize slots of storage, a multiple of the group size, so the slots of the
		// last group past the end of the cut are still in bounds
		const uint4_t* group_node_indices_vectors		 = reinterpret_cast<const uint4_t*>(cluster_node_indices + group_start);
		uint4_t first_node_indices						 = group_node_indices_vectors[0];
		uint4_t second_node_indices						 = group_node_indices_vectors[1];
		unsigned int group_node_indices[slot_group_size] = { first_node_indices.x,	first_node_indices.y,  first_node_indices.z,  first_node_indices.w,
															 second_node_indices.x, second_node_indices.y, second_node_indices.z, second_node_indices.w };

		uint2_t group_path_codes[slot_group_size];

		// Two loops, all the loads of the group then all the tests, only to make that order visible: the unrolling is what lets the loads be in
		// flight together and a single unrolled loop would most likely compile to the same code
		UNROLL_LOOP
		for (unsigned int group_slot = 0; group_slot < slot_group_size; group_slot++)
		{
			// The slots past the end of the cut do not hold a valid node index: they read the root instead and they are never selected.
			//
			// The envmap slot has no node in the light tree either: it also reads the root and its depth is replaced by the invalid depth,
			// which never matches
			unsigned int group_node_index = group_node_indices[group_slot];
			bool is_envmap_slot			  = group_node_index == IlluminationAwareKDTreeNode::ENVMAP_CLUSTER_NODE_INDEX;
			bool is_light_tree_node_slot  = (group_start + group_slot < lightcut_size) & !is_envmap_slot;
			unsigned int node_index		  = is_light_tree_node_slot ? group_node_index : 0u;

			group_path_codes[group_slot]   = node_path_codes[node_index];
			group_path_codes[group_slot].y = is_envmap_slot ? LightTreeSGDevice::INVALID_NODE_PATH_DEPTH : group_path_codes[group_slot].y;
		}

		UNROLL_LOOP
		for (unsigned int group_slot = 0; group_slot < slot_group_size; group_slot++)
		{
			unsigned int slot		= group_start + group_slot;
			unsigned int path_depth = group_path_codes[group_slot].y;
			unsigned int path_mask	= path_depth >= 32u ? 0xFFFFFFFFu : (1u << path_depth) - 1u;

			// Bitwise & instead of &&: the short-circuit operator would compile to a branch between the loads of the depth and of the path
			bool matches =
				(slot < lightcut_size) & (path_depth <= 32u) & ((bit_trail & path_mask) == group_path_codes[group_slot].x) & (path_depth < selected_depth);
			selected_slot  = matches ? slot : selected_slot;
			selected_depth = matches ? path_depth : selected_depth;
		}
	}

	if (selected_slot == lightcut_size)
		return 0.0f;

	unsigned int current_node_index = cluster_node_indices[selected_slot];
	unsigned int current_depth		= selected_depth;

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDF_MODEL != BSDF_LAMBERTIAN && BSDF_MODEL != BSDF_OREN_NAYAR
	SGSpecularImportanceData specular_data(context.view_direction, context.shading_normal, context.alpha_x, context.alpha_y);
#else
	SGSpecularImportanceData specular_data;
#endif

	float cluster_probability;

	if (initial_lightcut)
	{
		// Before a learned cut exists, sampling uses normalized SG importance, with uniform fallback for an all-zero cut.
		float total_weight	  = 0.0f;
		float selected_weight = 0.0f;

		for (unsigned int slot = 0; slot < lightcut_size; slot++)
		{
			float weight =
				light_clustering_cluster_importance(kd_tree.learning_to_cluster, render_data.light_tree_sg, cluster_node_indices[slot], specular_data, context);
			weight = hippt::max(weight, 0.0f);
			total_weight += weight;

			if (slot == selected_slot)
				selected_weight = weight;
		}

		cluster_probability = total_weight > 0.0f ? selected_weight / total_weight : 1.0f / static_cast<float>(lightcut_size);
	}
	else
		cluster_probability = light_clustering_cluster_probability(kd_tree, lightcut_index, selected_slot, lightcut_size);

	if (cluster_probability <= 0.0f)
		return 0.0f;

	// Replay the unsplit SG traversal inside the selected cluster, including uniform triangle selection at the leaf.
	// Same load pattern as sample_light_inside_cluster(): lobes addressed from the node index, traversal fields of the chosen child carried to
	// the next level
	const SpatialSGLobeDevice* spatial_lobes		  = render_data.light_tree_sg.spatial_lobes;
	unsigned int spatial_lobe_count					  = render_data.light_tree_sg.settings.spatial_lobe_count;
	unsigned int current_triangle_count				  = nodes[current_node_index].triangle_count;
	unsigned int current_left_child_or_first_triangle = nodes[current_node_index].left_child_index_or_first_triangle_index;

	float conditional_probability = 1.0f;

	while (current_triangle_count == 0u)
	{
		if (current_depth >= sizeof(unsigned int) * 8u)
			return 0.0f;

		unsigned int left_index					= current_left_child_or_first_triangle;
		unsigned int right_index				= left_index + 1u;
		const LightTreeSGNodeDevice& left_node	= nodes[left_index];
		const LightTreeSGNodeDevice& right_node = nodes[right_index];

		float left_importance =
			light_tree_sg_node_importance(left_node, spatial_lobes + left_index * spatial_lobe_count, spatial_lobe_count, specular_data, context.position,
										  context.view_direction, context.shading_normal, context.sg_specular_weight, context.alpha_x, context.alpha_y);
		float right_importance =
			light_tree_sg_node_importance(right_node, spatial_lobes + right_index * spatial_lobe_count, spatial_lobe_count, specular_data, context.position,
										  context.view_direction, context.shading_normal, context.sg_specular_weight, context.alpha_x, context.alpha_y);

		float importance_sum = left_importance + right_importance;
		if (importance_sum <= 0.0f)
			return 0.0f;

		float left_probability = left_importance / importance_sum;
		bool go_left		   = (bit_trail & (1u << current_depth)) == 0u;

		conditional_probability *= go_left ? left_probability : 1.0f - left_probability;
		current_triangle_count = go_left ? left_node.triangle_count : right_node.triangle_count;
		current_left_child_or_first_triangle =
			go_left ? left_node.left_child_index_or_first_triangle_index : right_node.left_child_index_or_first_triangle_index;
		current_depth++;
	}

	conditional_probability /= static_cast<float>(current_triangle_count);
	return cluster_probability * conditional_probability;
}

/**
 * Probability that the lightcut of the shading point of 'context' selects the envmap cluster. The PDF of an envmap direction is that probability
 * times the PDF of the direction in the envmap distribution.
 *
 * 'lightcut_index' is the result of IlluminationAwareKDTreeDevice::resolve_lightcut() at the shading point of 'context'.
 *
 * Returns 0 when the envmap is not sampled by learning to cluster
 */
HIPRT_DEVICE float pdf_of_envmap_cluster_learning_to_cluster(const HIPRTRenderData& render_data,
															 const IlluminationAwareKDTreeSGShadingContext& context,
															 unsigned int lightcut_index)
{
	const IlluminationAwareKDTreeDevice& kd_tree = render_data.kd_tree_device;
	if (!kd_tree.learning_to_cluster.envmap_cluster_enabled)
		return 0.0f;

	bool initial_lightcut = lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;

	unsigned int lightcut_size = initial_lightcut ? kd_tree.learning_to_cluster.effective_initial_lightcut_size
												  : kd_tree.learning_to_cluster.lightcut_data[lightcut_index].lightcut_size;
	if (lightcut_size == 0 || lightcut_size > LearningToClusterMaximumLightCutSize)
		return 0.0f;

	// The envmap is always the slot 0 of the cut
	if (!initial_lightcut)
		return light_clustering_cluster_probability(kd_tree, lightcut_index, 0u, lightcut_size);

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDF_MODEL != BSDF_LAMBERTIAN && BSDF_MODEL != BSDF_OREN_NAYAR
	SGSpecularImportanceData specular_data(context.view_direction, context.shading_normal, context.alpha_x, context.alpha_y);
#else
	SGSpecularImportanceData specular_data;
#endif

	// Same normalized importance as sample_cluster_from_light_cut() for a shading point without a learned cut, with the same uniform fallback
	const unsigned int* cluster_node_indices = kd_tree.learning_to_cluster.initial_lightcut_node_indices;

	float total_weight = 0.0f;

	for (unsigned int slot = 0; slot < lightcut_size; slot++)
	{
		float weight =
			light_clustering_cluster_importance(kd_tree.learning_to_cluster, render_data.light_tree_sg, cluster_node_indices[slot], specular_data, context);
		total_weight += hippt::max(weight, 0.0f);
	}

	float envmap_weight = hippt::max(kd_tree.learning_to_cluster.envmap_cluster_importance_prior, 0.0f);

	return total_weight > 0.0f ? envmap_weight / total_weight : 1.0f / static_cast<float>(lightcut_size);
}

#endif // #ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_LIGHT_TREE_LIGHT_TREE_SG_SAMPLING_LEARNING_TO_CLUSTER_H
