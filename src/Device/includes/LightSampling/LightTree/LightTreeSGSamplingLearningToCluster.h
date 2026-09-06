/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_LIGHT_TREE_LIGHT_TREE_SG_SAMPLING_LEARNING_TO_CLUSTER_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_LIGHT_TREE_LIGHT_TREE_SG_SAMPLING_LEARNING_TO_CLUSTER_H

#include "Device/includes/CDF.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterCutTriangleSample.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterDevice.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSampling.h"

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_DEVICE IlluminationAwareKDTreeSGShadingContext build_light_clustering_shading_context(const float3_t& shading_point,
																							const float3_t& view_direction,
																							const float3_t& shading_normal,
																							const DeviceUnpackedEffectiveMaterial& material)
{
	IlluminationAwareKDTreeSGShadingContext context{};

	context.position	   = shading_point;
	context.view_direction = view_direction;
	context.shading_normal = shading_normal;

	float material_specular_weight =
		(1.0f - material.metallic) * (1.0f - material.specular_transmission * (1.0f - material.diffuse_transmission)) * material.specular;
	float specular_lobes_sum = material.coat + material.metallic + material_specular_weight;

	context.sg_specular_weight = hippt::max(material.coat, hippt::max(material.metallic, material_specular_weight));

	float sg_roughness	= MaterialConstants::ROUGHNESS_CLAMP;
	float sg_anisotropy = 0.0f;
	if (specular_lobes_sum > 0.0f)
	{
		sg_roughness = material.coat * material.coat_roughness + material.metallic * material.roughness + material_specular_weight * material.roughness;
		sg_roughness /= specular_lobes_sum;

		sg_anisotropy = material.coat * material.coat_anisotropy + material.metallic * material.anisotropy + material_specular_weight * material.anisotropy;
		sg_anisotropy /= specular_lobes_sum;
	}

	sg_roughness = hippt::max(MaterialConstants::ROUGHNESS_CLAMP, sg_roughness);
	MaterialUtils::get_alphas(sg_roughness, sg_anisotropy, context.alpha_x, context.alpha_y);

	return context;
}

HIPRT_DEVICE IlluminationAwareKDTreeLearningToClusterCutTriangleSample sample_cluster_from_light_cut(const HIPRTRenderData& render_data,
																									 const IlluminationAwareKDTreeSGShadingContext& context,
																									 unsigned int mesh_id,
																									 Xorshift32Generator& random_number_generator)
{
	IlluminationAwareKDTreeLearningToClusterCutTriangleSample result{};
	const IlluminationAwareKDTreeDevice& kd_tree = render_data.kd_tree_device;

	unsigned int guiding_node_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	unsigned int lightcut_index		= kd_tree.resolve_lightcut(context, mesh_id, &guiding_node_index);
	result.guiding_node_index		= guiding_node_index;
	if (lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
	{
		unsigned int initial_lightcut_size = kd_tree.learning_to_cluster.effective_initial_lightcut_size;
		if (initial_lightcut_size == 0 || initial_lightcut_size > LearningToClusterMaximumLightCutSize)
			return result;

		float total_weight		   = 0.0f;
		float selected_weight	   = 0.0f;
		unsigned int selected_slot = initial_lightcut_size - 1;

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
		SGSpecularImportanceData specular_data(context.view_direction, context.shading_normal, context.alpha_x, context.alpha_y);
#else
		SGSpecularImportanceData specular_data;
#endif

		for (unsigned int slot = 0; slot < initial_lightcut_size; slot++)
		{
			unsigned int node_index = kd_tree.learning_to_cluster.initial_lightcut_node_indices[slot];
			float weight = light_tree_sg_node_importance(render_data.light_tree_sg.nodes[node_index], specular_data, context.position, context.view_direction,
														 context.shading_normal, context.sg_specular_weight, context.alpha_x, context.alpha_y);
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
			selected_slot	= random_number_generator.random_index(initial_lightcut_size);
			selected_weight = 1.0f;
			total_weight	= static_cast<float>(initial_lightcut_size);
		}

		result.cluster_node_index  = kd_tree.learning_to_cluster.initial_lightcut_node_indices[selected_slot];
		result.cluster_probability = selected_weight / total_weight;

		return result;
	}

	const IlluminationAwareKDTreeLightClusteringData& lightcut_data = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
	unsigned int lightcut_size										= lightcut_data.lightcut_size;
	if (lightcut_size == 0 || lightcut_size > LearningToClusterMaximumLightCutSize)
		return result;

	unsigned int cdf_offset = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, 0);

	CDFDeviceU16 lightcut_cdf;
	lightcut_cdf.cdf_u16 = kd_tree.learning_to_cluster.lightcut_cdfs + cdf_offset;
	lightcut_cdf.size	 = lightcut_size;

	unsigned int selected_slot	 = lightcut_cdf.sample(random_number_generator);
	unsigned int selected_offset = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, selected_slot);

	unsigned short int selected_cdf_start = selected_slot == 0u ? 0u : kd_tree.learning_to_cluster.lightcut_cdfs[selected_offset];
	unsigned short int selected_cdf_end	  = selected_slot + 1u < lightcut_size ? kd_tree.learning_to_cluster.lightcut_cdfs[selected_offset + 1u] : 65535u;
	unsigned int selected_cdf_range		  = static_cast<unsigned int>(selected_cdf_end) - static_cast<unsigned int>(selected_cdf_start);
	float selected_probability			  = static_cast<float>(selected_cdf_range) / 65535.0f;

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
	const LightTreeSGNodeDevice* nodes = render_data.light_tree_sg.nodes;
	unsigned int current_node_index	   = sample.cluster_node_index;
	float cumulative_probability	   = 1.0f;

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	SGSpecularImportanceData specular_data(context.view_direction, context.shading_normal, context.alpha_x, context.alpha_y);
#else
	SGSpecularImportanceData specular_data;
#endif

	while (nodes[current_node_index].triangle_count == 0)
	{
		const LightTreeSGNodeDevice& current_node = nodes[current_node_index];
		unsigned int left_index					  = current_node.left_child_index_or_first_triangle_index;
		unsigned int right_index				  = left_index + 1;
		const LightTreeSGNodeDevice& left_node	  = nodes[left_index];
		const LightTreeSGNodeDevice& right_node	  = nodes[right_index];

		float left_importance  = light_tree_sg_node_importance(left_node, specular_data, context.position, context.view_direction, context.shading_normal,
															   context.sg_specular_weight, context.alpha_x, context.alpha_y);
		float right_importance = light_tree_sg_node_importance(right_node, specular_data, context.position, context.view_direction, context.shading_normal,
															   context.sg_specular_weight, context.alpha_x, context.alpha_y);
		float importance_sum   = left_importance + right_importance;
		if (importance_sum <= 0.0f)
			return false;

		float left_probability = left_importance / importance_sum;
		if (random_number_generator() < left_probability)
		{
			current_node_index = left_index;
			cumulative_probability *= left_probability;
		}
		else
		{
			current_node_index = right_index;
			cumulative_probability *= 1.0f - left_probability;
		}
	}

	const LightTreeSGNodeDevice& leaf_node = nodes[current_node_index];
	if (leaf_node.triangle_count == 0)
		return false;

	unsigned int local_triangle_index	   = random_number_generator.random_index(leaf_node.triangle_count);
	unsigned int light_tree_triangle_index = leaf_node.left_child_index_or_first_triangle_index + local_triangle_index;
	int triangle_index					   = render_data.light_tree_sg.indices_array[light_tree_triangle_index];
	sample.emissive_triangle_global_index  = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

	cumulative_probability /= static_cast<float>(leaf_node.triangle_count);
	sample.conditional_triangle_probability = cumulative_probability;

	return sample.conditional_triangle_probability > 0.0f;
}

HIPRT_DEVICE IlluminationAwareKDTreeLearningToClusterCutTriangleSample
sample_one_emissive_triangle_learning_to_cluster(const HIPRTRenderData& render_data,
												 const IlluminationAwareKDTreeSGShadingContext& context,
												 unsigned int mesh_id,
												 Xorshift32Generator& random_number_generator)
{
	IlluminationAwareKDTreeLearningToClusterCutTriangleSample sample = sample_cluster_from_light_cut(render_data, context, mesh_id, random_number_generator);
	if (sample.cluster_probability <= 0.0f)
		return {};

	if (!sample_light_inside_cluster(render_data, context, random_number_generator, sample))
		return sample;

	return sample;
}

#endif // #ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_LIGHT_TREE_LIGHT_TREE_SG_SAMPLING_LEARNING_TO_CLUSTER_H
