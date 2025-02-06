/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_RESTIR_DI_DEFAULT_SETTINGS_H
#define HOST_DEVICE_RESTIR_DI_DEFAULT_SETTINGS_H

#include "HostDeviceCommon/ReSTIR/ReSTIRCommonSettings.h"

static ReSTIRCommonTemporalPassSettings RESTIR_DI_DEFAULT_COMMON_TEMPORAL_PASS_SETTINGS = {
	.do_temporal_reuse_pass = true,

	.use_permutation_sampling = false,
	.permutation_sampling_random_bits = 42,

	.max_neighbor_search_count = 8,
	.neighbor_search_radius = 4,

	.temporal_buffer_clear_requested = false
};

static ReSTIRCommonSpatialPassSettings RESTIR_DI_DEFAULT_COMMON_SPATIAl_PASS_SETTINGS = {
	.do_spatial_reuse_pass = true,

	.spatial_pass_index = 0,
	.number_of_passes = 2,
	.reuse_radius = 16,
	.reuse_neighbor_count = 3,

	.do_disocclusion_reuse_boost = true,
	.disocclusion_reuse_count = 5,

	.debug_neighbor_location = false,
	.debug_neighbor_location_direction = 0,

	.do_neighbor_rotation = true,

	.allow_converged_neighbors_reuse = false,
	.converged_neighbor_reuse_probability = 0.5f,

	.do_visibility_only_last_pass = true,
	.neighbor_visibility_count = 5
};

static ReSTIRCommonNeighborSimiliaritySettings RESTIR_DI_DEFAULT_COMMON_NEIGHBOR_SIMILARITY_SETTINGS = {
	.use_normal_similarity_heuristic = true,
	.normal_similarity_angle_degrees = 25.0f,
	.normal_similarity_angle_precomp = 0.906307787f,

	.use_plane_distance_heuristic = true,
	.plane_distance_threshold = 0.1f,

	.use_roughness_similarity_heuristic = false,
	.roughness_similarity_threshold = 0.25f
};

static ReSTIRCommonSettings RESTIR_DI_DEFAULT_COMMON_SETTINGS = {
	.common_temporal_pass = RESTIR_DI_DEFAULT_COMMON_TEMPORAL_PASS_SETTINGS,
	.common_spatial_pass = RESTIR_DI_DEFAULT_COMMON_SPATIAl_PASS_SETTINGS,
	.neighbor_similarity_settings = RESTIR_DI_DEFAULT_COMMON_NEIGHBOR_SIMILARITY_SETTINGS,

	.m_cap = 3,
	.use_confidence_weights = true
};

#endif
