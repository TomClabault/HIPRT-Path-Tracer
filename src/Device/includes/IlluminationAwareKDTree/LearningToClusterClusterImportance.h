/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_CLUSTER_IMPORTANCE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_CLUSTER_IMPORTANCE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterDevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSampling.h"

/**
 * Importance of the cluster of a lightcut slot at the shading point of 'context', for the clusters whose sampling weight is not learned yet.
 *
 * 'cluster_node_index' is either a node of the SG light tree, whose importance is its SG importance, or
 * IlluminationAwareKDTreeNode::ENVMAP_CLUSTER_NODE_INDEX, whose importance is the envmap prior computed by the host
 */
HIPRT_DEVICE float light_clustering_cluster_importance(const IlluminationAwareKDTreeLearningToClusterDevice& learning_to_cluster,
													   const LightTreeSGDevice& light_tree_sg,
													   unsigned int cluster_node_index,
													   const SGSpecularImportanceData& specular_data,
													   const IlluminationAwareKDTreeSGShadingContext& context)
{
	if (cluster_node_index == IlluminationAwareKDTreeNode::ENVMAP_CLUSTER_NODE_INDEX)
		return learning_to_cluster.envmap_cluster_importance_prior;

	return light_tree_sg_node_importance(light_tree_sg.nodes[cluster_node_index], specular_data, context.position, context.view_direction,
										 context.shading_normal, context.sg_specular_weight, context.alpha_x, context.alpha_y);
}

/**
 * Same as light_clustering_cluster_importance() but for the weights that do not depend on a shading point: the total power of the SG node, or
 * the envmap power prior computed by the host
 */
HIPRT_DEVICE float light_clustering_cluster_total_power(const IlluminationAwareKDTreeLearningToClusterDevice& learning_to_cluster,
														const LightTreeSGDevice& light_tree_sg,
														unsigned int cluster_node_index)
{
	if (cluster_node_index == IlluminationAwareKDTreeNode::ENVMAP_CLUSTER_NODE_INDEX)
		return learning_to_cluster.envmap_cluster_power_prior;

	return light_tree_sg.nodes[cluster_node_index].get_total_power();
}

#endif // #ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_CLUSTER_IMPORTANCE_H
