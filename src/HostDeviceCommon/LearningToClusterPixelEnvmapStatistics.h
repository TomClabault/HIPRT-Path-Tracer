/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_LEARNING_TO_CLUSTER_PIXEL_ENVMAP_STATISTICS_H
#define HOST_DEVICE_COMMON_LEARNING_TO_CLUSTER_PIXEL_ENVMAP_STATISTICS_H

/**
 * Statistics of the NEE samples taken at the primary hit of one pixel by learning to cluster, used to choose between the envmap and the
 * emissive lights at that primary hit.
 *
 * Each sum accumulates, over the NEE samples of the pixel, an unbiased estimate of the contribution of the NEE estimator of its technique,
 * visibility included. A sample that does not sample a technique adds 0 to the sum of that technique. Both sums
 * have the same number of terms, 'observation_count', so their ratio is the ratio of the moments
 */
struct LearningToClusterPixelEnvmapStatistics
{
	float envmap_sum			   = 0.0f;
	float emissive_sum			   = 0.0f;
	unsigned int observation_count = 0;
};

#endif // #ifndef HOST_DEVICE_COMMON_LEARNING_TO_CLUSTER_PIXEL_ENVMAP_STATISTICS_H
