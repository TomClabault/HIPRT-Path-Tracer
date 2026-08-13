/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_BUILD_NISML_CACHES_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_BUILD_NISML_CACHES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/NISML/NISML.h"

static_assert(ILLUMINATION_AWARE_KD_TREE_NISML_CLUSTER_COUNT == NISML_MAX_CLUSTER_COUNT);

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_BuildNISMLCaches(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, HIPRTRenderData render_data, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_BuildNISMLCaches(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, HIPRTRenderData render_data)
#endif
{
#ifdef __KERNELCC__
	unsigned int cache_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int cache_index = x;
#endif

	unsigned int node_count		   = *illumination_aware_kd_tree.node_count;
	unsigned int cache_entry_count = node_count * ILLUMINATION_AWARE_KD_TREE_NISML_NORMAL_FACE_COUNT;
	if (cache_index >= cache_entry_count || illumination_aware_kd_tree.nisml_cache == nullptr ||
		illumination_aware_kd_tree.nisml_representative_ready == nullptr || illumination_aware_kd_tree.nisml_cache_ready == nullptr ||
		illumination_aware_kd_tree.nisml_pending_cell_count == nullptr)
		return;

	if (illumination_aware_kd_tree.nisml_cache_ready[cache_index] != 0 || illumination_aware_kd_tree.nisml_representative_ready[cache_index] == 0)
		return;

	IlluminationAwareKDTreeNISMLCache& cache						  = illumination_aware_kd_tree.nisml_cache[cache_index];
	HIPRTRenderData baseline_render_data							  = render_data;
	baseline_render_data.illumination_aware_kd_tree.nisml_cache_ready = nullptr;
	float log_importances[NISML_MAX_CLUSTER_COUNT];
	build_nisml_log_baseline_weights(baseline_render_data, baseline_render_data.nisml, cache.representative_position, cache.representative_view_direction,
									 cache.representative_normal, cache.representative_sg_specular_weight, cache.representative_alpha_x,
									 cache.representative_alpha_y, log_importances);

	for (unsigned int cluster_index = 0; cluster_index < NISML_MAX_CLUSTER_COUNT; cluster_index++)
		cache.log_importances[cluster_index] = log_importances[cluster_index];

	// The cache is published only after all logits have been written. The host synchronizes this kernel before the next frame is uploaded.
	illumination_aware_kd_tree.nisml_cache_ready[cache_index] = 1;
	hippt::atomic_fetch_add(illumination_aware_kd_tree.nisml_pending_cell_count, static_cast<unsigned int>(-1));
}

#endif
