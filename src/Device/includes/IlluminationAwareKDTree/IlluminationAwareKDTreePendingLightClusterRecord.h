/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_PENDING_LIGHT_CLUSTER_RECORD_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_PENDING_LIGHT_CLUSTER_RECORD_H

struct IlluminationAwareKDTreePendingLightClusterRecord
{
	unsigned int cluster_node_index = 0;
	float q_reward					= 0.0f;
	float variance_observation		= 0.0f;
	unsigned int stream_index			= 0;
};

#endif
