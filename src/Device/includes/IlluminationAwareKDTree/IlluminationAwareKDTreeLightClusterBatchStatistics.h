/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_LIGHT_CLUSTER_BATCH_STATISTICS_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_LIGHT_CLUSTER_BATCH_STATISTICS_H

struct IlluminationAwareKDTreeLightClusterBatchStatistics
{
	// Sum of Y_c and Y_c^2 observations for that cluster
	float contribution_sum		   = 0.0f;
	float squared_contribution_sum = 0.0f;

	// Number of times that cluster was selected
	unsigned int selected_count = 0;
};

#endif // #ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_LIGHT_CLUSTER_BATCH_STATISTICS_H
