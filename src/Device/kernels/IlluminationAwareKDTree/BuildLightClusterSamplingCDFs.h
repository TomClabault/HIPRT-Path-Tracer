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
													 const IlluminationAwareKDTreeLightClusteringData& cluster_data,
													 unsigned int cluster_offset,
													 unsigned int cluster_node_index)
{
	float weight = 0.0f;
	if (cluster_data.Q0_initialized)
		weight = kd_tree.learning_to_cluster.light_cluster_statistics[cluster_offset].estimated_importance_Q;
	else
		weight = light_tree_sg.nodes[cluster_node_index].get_total_power();

	return hippt::max(weight, 0.0f);
}

#ifndef __KERNELCC__
HIPRT_DEVICE void build_light_cluster_sampling_cdf_cpu(IlluminationAwareKDTreeDevice kd_tree,
													   const LightTreeSGDevice& light_tree_sg,
													   unsigned int clustering_index)
{
	IlluminationAwareKDTreeLightClusteringData& cluster_data = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	if (!cluster_data.light_cluster_cdf_dirty)
		return;

	unsigned int cut_size	= cluster_data.cut_size;
	unsigned int cdf_offset = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, 0);
	if (cut_size == 0 || cut_size > LearningToClusterMaximumLightCutSize)
	{
		kd_tree.learning_to_cluster.light_cluster_cdfs[cdf_offset] = 0u;

		cluster_data.light_cluster_cdf_dirty = false;

		return;
	}

	float total_weight = 0.0f;
	for (unsigned int slot = 0; slot < cut_size; slot++)
	{
		unsigned int cluster_offset		= kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
		unsigned int cluster_node_index = kd_tree.learning_to_cluster.light_cluster_node_indices[cluster_offset];

		total_weight += get_light_cluster_sampling_weight(kd_tree, light_tree_sg, cluster_data, cluster_offset, cluster_node_index);
	}

	// CDFDeviceU16 ignores cdf[0], so it is used as a marker for whether the cut has any positive sampling weight.
	kd_tree.learning_to_cluster.light_cluster_cdfs[cdf_offset] = total_weight > 0.0f ? 65535u : 0u;

	float cumulative_weight = 0.0f;
	for (unsigned int slot = 1; slot < cut_size; slot++)
	{
		unsigned int cluster_offset				 = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
		unsigned int previous_cluster_node_index = kd_tree.learning_to_cluster.light_cluster_node_indices[cluster_offset - 1];

		float previous_weight = get_light_cluster_sampling_weight(kd_tree, light_tree_sg, cluster_data, cluster_offset - 1, previous_cluster_node_index);

		cumulative_weight += previous_weight;

		float normalized_prefix										   = total_weight > 0.0f ? cumulative_weight / total_weight : 0.0f;
		kd_tree.learning_to_cluster.light_cluster_cdfs[cluster_offset] = static_cast<unsigned short int>(hippt::min(normalized_prefix, 1.0f) * 65535.0f);
	}

	cluster_data.light_cluster_cdf_dirty = false;
}

GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_BuildLightClusterSamplingCDFs(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg, int x)
{
	unsigned int clustering_index = static_cast<unsigned int>(x);
	if (clustering_index >= kd_tree.learning_to_cluster.light_clustering_capacity)
		return;

	build_light_cluster_sampling_cdf_cpu(kd_tree, light_tree_sg, clustering_index);
}
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_BuildLightClusterSamplingCDFs(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg)
{
	unsigned int clustering_index		= blockIdx.x;
	unsigned int slot					= threadIdx.x;
	unsigned int light_clustering_count = *kd_tree.learning_to_cluster.light_clustering_count;
	if (clustering_index >= light_clustering_count || clustering_index >= kd_tree.learning_to_cluster.light_clustering_capacity)
		return;

	IlluminationAwareKDTreeLightClusteringData& cluster_data = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	if (!cluster_data.light_cluster_cdf_dirty)
		return;

	unsigned int cut_size	= cluster_data.cut_size;
	unsigned int cdf_offset = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, 0);
	if (cut_size == 0 || cut_size > LearningToClusterMaximumLightCutSize)
	{
		if (slot == 0u)
		{
			kd_tree.learning_to_cluster.light_cluster_cdfs[cdf_offset] = 0u;
			cluster_data.light_cluster_cdf_dirty					   = false;
		}

		return;
	}

	float weight = 0.0f;
	if (slot < cut_size)
	{
		unsigned int cluster_offset		= kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
		unsigned int cluster_node_index = kd_tree.learning_to_cluster.light_cluster_node_indices[cluster_offset];

		weight = get_light_cluster_sampling_weight(kd_tree, light_tree_sg, cluster_data, cluster_offset, cluster_node_index);
	}

	__shared__ float total_weight;
	float exclusive_prefix_weight = block_prefix_scan_exclusive<LearningToClusterMaximumLightCutSize>(weight);
	if (slot == cut_size - 1u)
		total_weight = exclusive_prefix_weight + weight;

	__syncthreads();

	if (slot == 0u)
		// CDFDeviceU16 ignores cdf[0], so it is used as a marker for whether the cut has any positive sampling weight.
		kd_tree.learning_to_cluster.light_cluster_cdfs[cdf_offset] = total_weight > 0.0f ? 65535u : 0u;
	else if (slot < cut_size)
	{
		unsigned int cluster_offset = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);

		float normalized_prefix										   = total_weight > 0.0f ? exclusive_prefix_weight / total_weight : 0.0f;
		kd_tree.learning_to_cluster.light_cluster_cdfs[cluster_offset] = static_cast<unsigned short int>(hippt::min(normalized_prefix, 1.0f) * 65535.0f);
	}

	__syncthreads();

	if (slot == 0u)
		cluster_data.light_cluster_cdf_dirty = false;
}
#endif

#endif
