/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ACCUMULATE_NEE_DISTRIBUTION_TRAINING_RECORDS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ACCUMULATE_NEE_DISTRIBUTION_TRAINING_RECORDS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_AccumulateNEEDistributionTrainingRecords(IlluminationAwareKDTreeDevice kd_tree_device, unsigned int tree_cut_size, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_AccumulateNEEDistributionTrainingRecords(IlluminationAwareKDTreeDevice kd_tree_device, unsigned int tree_cut_size)
#endif
{
#ifdef __KERNELCC__
	unsigned int record_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int record_index = x;
#endif

	unsigned int record_count = *kd_tree_device.nee_distributions.nee_training_record_count;

	if (record_count > kd_tree_device.nee_distributions.nee_training_record_capacity)
		record_count = kd_tree_device.nee_distributions.nee_training_record_capacity;

	if (record_index >= record_count)
		return;

	const IlluminationAwareKDTreeNEEDistributionTrainingRecord& record = kd_tree_device.nee_distributions.nee_training_records[record_index];

	unsigned int guiding_node_index = kd_tree_device.core.find_guiding_cell(record.shading_position);
	if (guiding_node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
		return;

	unsigned int distribution_index = kd_tree_device.core.nodes[guiding_node_index].guiding_distribution_index;
	if (distribution_index == IlluminationAwareKDTreeNode::INVALID_GUIDING_DISTRIBUTION_INDEX)
		return;

	unsigned int normal_face = illumination_aware_kd_tree_classify_surface_normal_face(record.shading_normal);
	distribution_index		 = kd_tree_device.nee_distributions.get_normal_face_distribution_index(distribution_index, normal_face);

	if (record.selected_cut_slot >= tree_cut_size)
		return;

	unsigned int distribution_slot = kd_tree_device.nee_distributions.get_tree_cut_offset(distribution_index, tree_cut_size) + record.selected_cut_slot;

	hippt::atomic_fetch_add(&kd_tree_device.nee_distributions.batch_per_cut_node_second_moment_sum[distribution_slot],
							record.conditional_second_moment_contribution);
	hippt::atomic_fetch_add(&kd_tree_device.nee_distributions.batch_per_cut_node_sample_count[distribution_slot], 1u);
	hippt::atomic_fetch_add(&kd_tree_device.nee_distributions.history_per_cell_sample_count[distribution_index], 1u);

	hippt::atomic_fetch_add(&kd_tree_device.nee_distributions.history_per_cell_normal_sum_x[distribution_index], record.shading_normal.x);
	hippt::atomic_fetch_add(&kd_tree_device.nee_distributions.history_per_cell_normal_sum_y[distribution_index], record.shading_normal.y);
	hippt::atomic_fetch_add(&kd_tree_device.nee_distributions.history_per_cell_normal_sum_z[distribution_index], record.shading_normal.z);

	hippt::atomic_fetch_add(&kd_tree_device.nee_distributions.history_per_cell_normal_count[distribution_index], 1u);
}

#endif
