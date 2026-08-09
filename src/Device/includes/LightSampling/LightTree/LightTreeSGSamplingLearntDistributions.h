/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_LEARNT_DISTRIBUTIONS_H
#define DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_LEARNT_DISTRIBUTIONS_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/includes/LightSampling/LightSampleInformation.h"
#include "Device/includes/LightSampling/LightTree/LightTreeATSSampling.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSamplingCommon.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSampling.h"
#include "Device/includes/LightSampling/TriangleSampling.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"

HIPRT_DEVICE LightSampleArray<1> sample_one_emissive_triangle_light_tree_sg_learnt_distributions(const HIPRTRenderData& render_data,
																								 float3_t shading_point,
																								 float3_t view_direction,
																								 float3_t shading_normal,
																								 float3_t geometric_normal,
																								 const DeviceUnpackedEffectiveMaterial& material,
																								 int last_hit_primitive_index,
																								 Xorshift32Generator& random_number_generator,
																								 IlluminationAwareKDTreeSampledCutNode& out_sampled_cut_node)
{
	out_sampled_cut_node = IlluminationAwareKDTreeSampledCutNode{};

	float material_specular_weight =
		(1.0f - material.metallic) * (1.0f - material.specular_transmission * (1.0f - material.diffuse_transmission)) * material.specular;

	float specular_lobes_sum = material.coat + material.metallic + material_specular_weight;
	float specular			 = hippt::max(material.coat, hippt::max(material.metallic, material_specular_weight));
	float roughness	 = hippt::max(MaterialConstants::ROUGHNESS_CLAMP, (material.coat * material.coat_roughness + material.metallic * material.roughness +
																	   material_specular_weight * material.roughness) /
																		  specular_lobes_sum);
	float anisotropy = (material.coat * material.coat_anisotropy + material.metallic * material.anisotropy + material_specular_weight * material.anisotropy) /
					   specular_lobes_sum;

	float alpha_x;
	float alpha_y;
	MaterialUtils::get_alphas(roughness, anisotropy, alpha_x, alpha_y);

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	SGSpecularImportanceData spec_data(view_direction, shading_normal, alpha_x, alpha_y);
#else
	SGSpecularImportanceData spec_data;
#endif

	unsigned int guiding_node_index = render_data.illumination_aware_kd_tree.find_guiding_cell(shading_point);
	if (guiding_node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
		return LightSampleArray<1>{ LightSampleInformation{ -1, IlluminationAwareKDTreeSampledCutNode::INVALID_PROBABILITY } };

	const IlluminationAwareKDTreeNode& guiding_node = render_data.illumination_aware_kd_tree.nodes[guiding_node_index];
	unsigned int guiding_distribution_index			= guiding_node.guiding_distribution_index;

	IlluminationAwareKDTreeSampledCutNode sampled_cut_node = render_data.illumination_aware_kd_tree.nee_learnt_distributions.sample_global_cut_node(
		render_data.light_tree_sg, guiding_distribution_index, shading_normal, random_number_generator);

	if (sampled_cut_node.probability == IlluminationAwareKDTreeSampledCutNode::INVALID_PROBABILITY)
		// Returning a sample with invalid probability so we can detect that in the NEE estimator and fallback to normal light sampling
		return LightSampleArray<1>{ LightSampleInformation{ -1, IlluminationAwareKDTreeSampledCutNode::INVALID_PROBABILITY } };

	out_sampled_cut_node = sampled_cut_node;

	IlluminationAwareKDTreeConditionalLightTreeSample sampled_subtree =
		sample_light_tree_subtree(render_data.light_tree_sg.nodes, sampled_cut_node.light_tree_node_index, shading_point, view_direction, shading_normal,
								  spec_data, specular, alpha_x, alpha_y, random_number_generator);

	if (!(sampled_subtree.conditional_leaf_probability > 0.0f))
		return LightSampleArray<1>{ LightSampleInformation{ -1, 0.0f } };

	const LightTreeSGNodeDevice& sampled_leaf = render_data.light_tree_sg.nodes[sampled_subtree.light_leaf_index];
	if (sampled_leaf.triangle_count == 0)
		return LightSampleArray<1>{ LightSampleInformation{ -1, 0.0f } };

	int index					= sampled_leaf.left_child_index_or_first_triangle_index + random_number_generator.random_index(sampled_leaf.triangle_count);
	int triangle_index			= render_data.light_tree_sg.indices_array[index];
	int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

	LightSampleInformation light_sample;
	light_sample.emissive_triangle_global_index = emissive_triangle_index;
	light_sample.pdf							= sampled_cut_node.probability * sampled_subtree.conditional_leaf_probability;
	// Uniformly sample one triangle in the leaf node
	light_sample.pdf /= sampled_leaf.triangle_count;

	return LightSampleArray<1>{ light_sample };
}

#endif
