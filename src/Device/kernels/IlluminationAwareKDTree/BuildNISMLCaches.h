/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_BUILD_NISML_CACHES_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_BUILD_NISML_CACHES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/NISML/NISML.h"

static_assert(ILLUMINATION_AWARE_KD_TREE_NISML_CLUSTER_COUNT == NISML_MAX_CLUSTER_COUNT);

HIPRT_DEVICE bool normalize_nisml_baseline_distribution(float* log_importances, unsigned int cluster_count)
{
	float maximum_log_importance = -INFINITY;
	bool has_valid_importance	 = false;
	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		float log_importance = log_importances[cluster_index];
		if (log_importance != log_importance || log_importance == INFINITY)
			return false;

		if (log_importance == -INFINITY)
			continue;

		maximum_log_importance = hippt::max(maximum_log_importance, log_importance);
		has_valid_importance   = true;
	}

	if (!has_valid_importance)
		return false;

	float exponential_denominator = 0.0f;
	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		if (log_importances[cluster_index] == -INFINITY)
			continue;

		float probability = hippt::intrin_expf(log_importances[cluster_index] - maximum_log_importance);
		if (probability != probability || probability == INFINITY)
			return false;

		log_importances[cluster_index] = probability;
		exponential_denominator += probability;
	}

	if (!(exponential_denominator > 0.0f) || exponential_denominator == INFINITY)
		return false;

	float reciprocal_denominator = 1.0f / exponential_denominator;
	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		if (log_importances[cluster_index] == -INFINITY)
			continue;

		log_importances[cluster_index] *= reciprocal_denominator;
	}

	return true;
}

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_BuildNISMLCaches(IlluminationAwareKDTreeDevice kd_tree_device, HIPRTRenderData render_data, int x)
#else
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char ILLUMINATION_AWARE_KD_TREE_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_BuildNISMLCaches(IlluminationAwareKDTreeDevice kd_tree_device)
#endif
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(ILLUMINATION_AWARE_KD_TREE_RENDER_DATA);
	unsigned int cache_index	 = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int cache_index = x;
#endif

	if (cache_index >= kd_tree_device.nisml.nisml_hash_table_capacity || kd_tree_device.nisml.nisml_hash_keys == nullptr ||
		kd_tree_device.nisml.nisml_hash_entry_states == nullptr || kd_tree_device.nisml.nisml_cache == nullptr ||
		kd_tree_device.nisml.nisml_representative_occupied_counts == nullptr || kd_tree_device.nisml.nisml_representative_dirty == nullptr ||
		kd_tree_device.nisml.nisml_representative_valid == nullptr || kd_tree_device.nisml.nisml_cache_ready == nullptr ||
		kd_tree_device.nisml.nisml_pending_cell_count == nullptr || kd_tree_device.nisml.nisml_representative_capacity == 0u)
		return;
	if (hippt::atomic_load(kd_tree_device.nisml.nisml_pending_cell_count) == 0u)
		return;

	if (hippt::atomic_load(&kd_tree_device.nisml.nisml_hash_keys[cache_index]) == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX ||
		hippt::atomic_load(&kd_tree_device.nisml.nisml_hash_entry_states[cache_index]) != ILLUMINATION_AWARE_KD_TREE_NISML_HASH_ENTRY_READY)
		return;

	if (hippt::atomic_load(&kd_tree_device.nisml.nisml_representative_dirty[cache_index]) == 0)
		return;

	unsigned int occupied_representative_count = hippt::atomic_load(&kd_tree_device.nisml.nisml_representative_occupied_counts[cache_index]);
	if (occupied_representative_count == 0u)
		return;

	unsigned int cluster_count = hippt::min(render_data.nisml.cluster_count, static_cast<unsigned int>(NISML_MAX_CLUSTER_COUNT));
	if (cluster_count == 0u)
		return;

	float average_probabilities[NISML_MAX_CLUSTER_COUNT] = {};
	unsigned int valid_representative_count				 = 0;
	for (unsigned int representative_index = 0; representative_index < kd_tree_device.nisml.nisml_representative_capacity; representative_index++)
	{
		unsigned int flat_representative_index = kd_tree_device.nisml.get_nisml_representative_index(cache_index, representative_index);
		if (kd_tree_device.nisml.nisml_representative_valid[flat_representative_index] == 0)
			continue;

		IlluminationAwareKDTreeNISMLCache& representative			= kd_tree_device.nisml.nisml_cache[flat_representative_index];
		HIPRTRenderData baseline_render_data						= render_data;
		baseline_render_data.kd_tree_device.nisml.nisml_cache_ready = nullptr;

		float normalized_probabilities[NISML_MAX_CLUSTER_COUNT];
		build_nisml_log_baseline_weights(baseline_render_data, baseline_render_data.nisml, representative.representative_position,
										 representative.representative_view_direction, representative.representative_normal,
										 representative.representative_sg_specular_weight, representative.representative_alpha_x,
										 representative.representative_alpha_y, normalized_probabilities);
		if (!normalize_nisml_baseline_distribution(normalized_probabilities, cluster_count))
			continue;

		for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
		{
			if (normalized_probabilities[cluster_index] != -INFINITY)
				average_probabilities[cluster_index] += normalized_probabilities[cluster_index];
		}
		valid_representative_count++;
	}

	if (valid_representative_count == 0u)
		return;

	float reciprocal_representative_count	 = 1.0f / static_cast<float>(valid_representative_count);
	IlluminationAwareKDTreeNISMLCache& cache = kd_tree_device.nisml.nisml_cache[kd_tree_device.nisml.get_nisml_representative_index(cache_index, 0u)];
	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		float average_probability			 = average_probabilities[cluster_index] * reciprocal_representative_count;
		cache.log_importances[cluster_index] = average_probability > 0.0f ? hippt::intrin_logf(average_probability) : -INFINITY;
	}

	for (unsigned int cluster_index = cluster_count; cluster_index < NISML_MAX_CLUSTER_COUNT; cluster_index++)
		cache.log_importances[cluster_index] = -INFINITY;

	// The cache is published only after all logits have been written. The host synchronizes this kernel before the next frame is uploaded.
	kd_tree_device.nisml.nisml_cache_ready[cache_index] = 1;
	hippt::atomic_compare_exchange(&kd_tree_device.nisml.nisml_representative_dirty[cache_index], static_cast<unsigned char>(1), static_cast<unsigned char>(0));
	hippt::atomic_fetch_add(kd_tree_device.nisml.nisml_pending_cell_count, static_cast<unsigned int>(-1));
}

#endif
