/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_NISML_TRAINING_SAMPLES_KERNEL_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_NISML_TRAINING_SAMPLES_KERNEL_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/includes/LightSampling/NISML/NISML.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_ReplayNISMLTrainingSamplesKernel(HIPRTRenderData render_data, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_ReplayNISMLTrainingSamplesKernel(HIPRTRenderData render_data)
#endif
{
#ifdef __KERNELCC__
	unsigned int training_record_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int training_record_index = x;
#endif

	if (render_data.nisml.training_records == nullptr || render_data.nisml.training_record_count == nullptr)
		return;

	unsigned int training_record_count = *render_data.nisml.training_record_count;
	if (training_record_index >= training_record_count)
		return;

	const NISMLTrainingSample& training_record				  = render_data.nisml.training_records[training_record_index];
	IlluminationAwareKDTreeDevice& illumination_aware_kd_tree = render_data.illumination_aware_kd_tree;
	unsigned int node_index									  = illumination_aware_kd_tree.core.find_guiding_cell(training_record.position);
	if (node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX || illumination_aware_kd_tree.nisml.nisml_representative_ready == nullptr)
		return;

	unsigned int normal_face = illumination_aware_kd_tree_classify_surface_normal_face(training_record.normal);
	unsigned int cache_index = illumination_aware_kd_tree.nisml.get_nisml_cache_index(node_index, normal_face);
	if (illumination_aware_kd_tree.nisml.nisml_representative_ready[cache_index] != 0)
		return;

	Xorshift32Generator random_number_generator(training_record_index + 1u);
	illumination_aware_kd_tree.nisml.append_nisml_representative(node_index, illumination_aware_kd_tree.core.node_capacity, training_record.position,
																 training_record.outgoing_direction, training_record.normal, training_record.sg_specular_weight,
																 training_record.alpha_x, training_record.alpha_y, random_number_generator);
}

#endif
