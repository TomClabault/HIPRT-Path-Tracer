/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_BUILD_LIGHT_CLUSTER_SAMPLING_CDFS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_BUILD_LIGHT_CLUSTER_SAMPLING_CDFS_H

#include "Device/includes/Compute/Common/WarpBlockScan.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"

HIPRT_DEVICE float get_light_cluster_sampling_weight(const IlluminationAwareKDTreeDevice& kd_tree,
													 const LightTreeSGDevice& light_tree_sg,
													 const IlluminationAwareKDTreeLightClusteringData& lightcut_data,
													 unsigned int cluster_offset,
													 unsigned int cluster_node_index)
{
	float weight = 0.0f;
	if (lightcut_data.Q0_initialized)
		weight = kd_tree.learning_to_cluster.lightcut_statistics[cluster_offset].estimated_importance_Q;
	else
		weight = light_tree_sg.nodes[cluster_node_index].get_total_power();

	return hippt::max(weight, 0.0f);
}

#ifndef __KERNELCC__
HIPRT_DEVICE void build_light_cluster_sampling_cdf_cpu(IlluminationAwareKDTreeDevice kd_tree,
													   const LightTreeSGDevice& light_tree_sg,
													   unsigned int lightcut_index)
{
	IlluminationAwareKDTreeLightClusteringData& lightcut_data = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
	if (!lightcut_data.lightcut_cdf_dirty)
		return;

	unsigned int lightcut_size = lightcut_data.lightcut_size;
	unsigned int cdf_offset	   = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, 0);
	if (lightcut_size == 0 || lightcut_size > LearningToClusterMaximumLightCutSize)
	{
		kd_tree.learning_to_cluster.lightcut_cdfs[cdf_offset] = 0u;

		lightcut_data.lightcut_cdf_dirty = false;

		return;
	}

	float total_weight = 0.0f;
	for (unsigned int slot = 0; slot < lightcut_size; slot++)
	{
		unsigned int cluster_offset		= kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);
		unsigned int cluster_node_index = kd_tree.learning_to_cluster.lightcut_node_indices[cluster_offset];

		total_weight += get_light_cluster_sampling_weight(kd_tree, light_tree_sg, lightcut_data, cluster_offset, cluster_node_index);
	}

	// CDFDeviceU16 ignores cdf[0], so it is used as a marker for whether the cut has any positive sampling weight.
	kd_tree.learning_to_cluster.lightcut_cdfs[cdf_offset] = total_weight > 0.0f ? 65535u : 0u;

	float cumulative_weight = 0.0f;
	for (unsigned int slot = 1; slot < lightcut_size; slot++)
	{
		unsigned int cluster_offset				 = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);
		unsigned int previous_cluster_node_index = kd_tree.learning_to_cluster.lightcut_node_indices[cluster_offset - 1];

		float previous_weight = get_light_cluster_sampling_weight(kd_tree, light_tree_sg, lightcut_data, cluster_offset - 1, previous_cluster_node_index);

		cumulative_weight += previous_weight;

		float normalized_prefix									  = total_weight > 0.0f ? cumulative_weight / total_weight : 0.0f;
		kd_tree.learning_to_cluster.lightcut_cdfs[cluster_offset] = static_cast<unsigned short int>(hippt::min(normalized_prefix, 1.0f) * 65535.0f);
	}

	lightcut_data.lightcut_cdf_dirty = false;
}

GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_LearningToClusterBuildLightClusterSamplingCDFs(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg, int x)
{
	unsigned int lightcut_index = static_cast<unsigned int>(x);
	if (lightcut_index >= kd_tree.learning_to_cluster.lightcut_capacity)
		return;

	build_light_cluster_sampling_cdf_cpu(kd_tree, light_tree_sg, lightcut_index);
}
#else  // #ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_LearningToClusterBuildLightClusterSamplingCDFs(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg)
{
	unsigned int lightcut_index = blockIdx.x;
	unsigned int slot			= threadIdx.x;
	unsigned int lightcut_count = *kd_tree.learning_to_cluster.lightcut_count;
	if (lightcut_index >= lightcut_count || lightcut_index >= kd_tree.learning_to_cluster.lightcut_capacity)
		return;

	IlluminationAwareKDTreeLightClusteringData& lightcut_data = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
	if (!lightcut_data.lightcut_cdf_dirty)
		return;

	unsigned int lightcut_size = lightcut_data.lightcut_size;
	unsigned int cdf_offset	   = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, 0);
	if (lightcut_size == 0 || lightcut_size > LearningToClusterMaximumLightCutSize)
	{
		if (slot == 0u)
		{
			kd_tree.learning_to_cluster.lightcut_cdfs[cdf_offset] = 0u;
			lightcut_data.lightcut_cdf_dirty					  = false;
		}

		return;
	}

	float weight = 0.0f;
	if (slot < lightcut_size)
	{
		unsigned int cluster_offset		= kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);
		unsigned int cluster_node_index = kd_tree.learning_to_cluster.lightcut_node_indices[cluster_offset];

		weight = get_light_cluster_sampling_weight(kd_tree, light_tree_sg, lightcut_data, cluster_offset, cluster_node_index);
	}

	__shared__ float total_weight;
	float exclusive_prefix_weight = block_prefix_scan_exclusive<LearningToClusterMaximumLightCutSize>(weight);
	if (slot == lightcut_size - 1u)
		total_weight = exclusive_prefix_weight + weight;

	__syncthreads();

	if (slot == 0u)
		// CDFDeviceU16 ignores cdf[0], so it is used as a marker for whether the cut has any positive sampling weight.
		kd_tree.learning_to_cluster.lightcut_cdfs[cdf_offset] = total_weight > 0.0f ? 65535u : 0u;
	else if (slot < lightcut_size)
	{
		unsigned int cluster_offset = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);

		float normalized_prefix									  = total_weight > 0.0f ? exclusive_prefix_weight / total_weight : 0.0f;
		kd_tree.learning_to_cluster.lightcut_cdfs[cluster_offset] = static_cast<unsigned short int>(hippt::min(normalized_prefix, 1.0f) * 65535.0f);
	}

	__syncthreads();

	if (slot == 0u)
		lightcut_data.lightcut_cdf_dirty = false;
}
#endif // #ifndef __KERNELCC__

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_BUILD_LIGHT_CLUSTER_SAMPLING_CDFS_H
