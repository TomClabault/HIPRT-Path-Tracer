/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_LEARNING_TO_CLUSTER_ENVMAP_SELECTION_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_LEARNING_TO_CLUSTER_ENVMAP_SELECTION_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * How the NEE of learning to cluster chooses between the envmap and the emissive lights at a shading point, when the envmap is a cluster
 * of the lightcuts
 */
enum LearningToClusterEnvmapNEEMode : unsigned int
{
	// The envmap is the slot 0 of the lightcut and is sampled with the probability of that slot
	LEARNING_TO_CLUSTER_ENVMAP_NEE_LIGHTCUT_SLOT,
	// The envmap and the emissive lights each get their own NEE sample, the emissive lights being sampled from the lightcut without its envmap
	// slot. Used where nothing is learned yet about the visibility of the envmap
	LEARNING_TO_CLUSTER_ENVMAP_NEE_SEPARATE,
	// One NEE sample, the envmap with the probability learned at the pixel, the emissive lights otherwise, from the lightcut without its
	// envmap slot. Primary hits only
	LEARNING_TO_CLUSTER_ENVMAP_NEE_PIXEL_PROBABILITY,
};

/**
 * Whether the NEE at the primary hit of 'pixel_index' reads and updates the per-pixel envmap statistics
 */
HIPRT_DEVICE bool learning_to_cluster_uses_pixel_envmap_statistics(const HIPRTRenderData& render_data, int bounce)
{
#if PathSamplingStrategyIsReSTIRPT
	return false;
#else
	const IlluminationAwareKDTreeLearningToClusterDevice& learning_to_cluster = render_data.kd_tree_device.learning_to_cluster;

	// Without emissive geometry, the envmap is the only light and is always sampled. At low resolution, the pixels of the kernel do not map
	// one to one to the pixels of the statistics
	return bounce == 0 && learning_to_cluster.envmap_cluster_enabled && learning_to_cluster.user_settings.envmap_per_pixel_probability &&
		   render_data.aux_buffers.learning_to_cluster_pixel_envmap_statistics != nullptr && render_data.buffers.emissive_triangles_count > 0 &&
		   !render_data.render_settings.do_render_low_resolution();
#endif
}

HIPRT_DEVICE LearningToClusterEnvmapNEEMode choose_learning_to_cluster_envmap_nee_mode(const HIPRTRenderData& render_data,
																					   unsigned int lightcut_index,
																					   int bounce,
																					   unsigned int pixel_index)
{
#if PathSamplingStrategyIsReSTIRPT
	return LEARNING_TO_CLUSTER_ENVMAP_NEE_LIGHTCUT_SLOT;
#else
	const IlluminationAwareKDTreeLearningToClusterDevice& learning_to_cluster = render_data.kd_tree_device.learning_to_cluster;
	if (!learning_to_cluster.envmap_cluster_enabled)
		return LEARNING_TO_CLUSTER_ENVMAP_NEE_LIGHTCUT_SLOT;

	if (learning_to_cluster_uses_pixel_envmap_statistics(render_data, bounce))
	{
		if (render_data.aux_buffers.learning_to_cluster_pixel_envmap_statistics[pixel_index].observation_count > 0)
			return LEARNING_TO_CLUSTER_ENVMAP_NEE_PIXEL_PROBABILITY;
		else
			// The first sample of the pixel observes both techniques
			return LEARNING_TO_CLUSTER_ENVMAP_NEE_SEPARATE;
	}

	bool learned_lightcut = lightcut_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
	if (!learned_lightcut && learning_to_cluster.user_settings.envmap_separate_sampling_without_learned_lightcut)
		return LEARNING_TO_CLUSTER_ENVMAP_NEE_SEPARATE;

	return LEARNING_TO_CLUSTER_ENVMAP_NEE_LIGHTCUT_SLOT;
#endif
}

/**
 * Probability of sampling the envmap rather than the emissive lights at the primary hit of 'pixel_index', from the statistics of the previous
 * samples of that pixel
 */
HIPRT_DEVICE float learning_to_cluster_pixel_envmap_probability(const HIPRTRenderData& render_data, unsigned int pixel_index)
{
	const IlluminationAwareKDTreeLearningToClusterUserSettings& user_settings = render_data.kd_tree_device.learning_to_cluster.user_settings;
	const LearningToClusterPixelEnvmapStatistics& statistics = render_data.aux_buffers.learning_to_cluster_pixel_envmap_statistics[pixel_index];

	float envmap_weight	  = hippt::max(statistics.envmap_sum, 0.0f);
	float emissive_weight = hippt::max(statistics.emissive_sum, 0.0f);

	float total_weight		 = envmap_weight + emissive_weight;
	float envmap_probability = total_weight > 0.0f ? envmap_weight / total_weight : 0.5f;

	// A probability of 0 would never sample the light that only one of the techniques reaches
	float probability_floor = hippt::clamp(1.0e-3f, 0.5f, user_settings.envmap_per_pixel_probability_floor);

	return hippt::clamp(probability_floor, 1.0f - probability_floor, envmap_probability);
}

/**
 * Adds 'observation' to the statistic of the envmap or of the emissive lights of 'pixel_index'.
 *
 * 'observation' is the luminance of the MIS-weighted NEE estimator of the technique, already divided by the probability that the technique was
 * sampled: its expectation over the selection is the contribution of the technique.
 *
 * Only the thread that shades the primary hit of the pixel, or that traces its shadow rays, writes the statistics of the pixel: no atomic is
 * needed
 */
HIPRT_DEVICE void add_learning_to_cluster_pixel_envmap_statistic_observation(const HIPRTRenderData& render_data,
																			 unsigned int pixel_index,
																			 bool is_envmap,
																			 float observation)
{
	LearningToClusterPixelEnvmapStatistics& statistics = render_data.aux_buffers.learning_to_cluster_pixel_envmap_statistics[pixel_index];

	if (is_envmap)
		statistics.envmap_sum += observation;
	else
		statistics.emissive_sum += observation;
}

/**
 * Where a NEE sample of the primary hit adds its observation to the per-pixel statistics: 'record' is false for the NEE samples of the other
 * vertices
 */
struct LearningToClusterPixelObservationTarget
{
	unsigned int pixel_index = 0;
	bool record				 = false;
};

#endif // #ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_LEARNING_TO_CLUSTER_ENVMAP_SELECTION_H
