/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_RESTIR_GI_SETTINGS_H
#define HOST_DEVICE_RESTIR_GI_SETTINGS_H

#include "HostDeviceCommon/ReSTIR/ReSTIRGIDefaultSettings.h"

struct ReSTIRGIReservoir;

struct ReSTIRGIInitialCandidatesPassSettings
{
	// Buffer that contains the reservoirs that will hold the reservoir
	// for the initial candidates generated
	ReSTIRGIReservoir* initial_candidates_buffer = nullptr;
};

struct ReSTIRGITemporalPassSettings
{
	// Buffer that contains the input reservoirs for the temporal reuse pass
	ReSTIRGIReservoir* input_reservoirs = nullptr;
	// Buffer that contains the output reservoir of the temporal reuse pass
	ReSTIRGIReservoir* output_reservoirs = nullptr;
};

struct ReSTIRGISpatialPassSettings
{
	// Buffer that contains the input reservoirs for the spatial reuse pass
	ReSTIRGIReservoir* input_reservoirs = nullptr;
	// Buffer that contains the output reservoir of the spatial reuse pass
	ReSTIRGIReservoir* output_reservoirs = nullptr;
};

enum ReSTIRGIDebugView
{
	NO_DEBUG = 0,
	FINAL_RESERVOIR_UCW = 1,
	TARGET_FUNCTION = 2,
	WEIGHT_SUM = 3,
	M_COUNT = 4,
};

struct ReSTIRGISettings : public ReSTIRCommonSettings
{
	HIPRT_HOST_DEVICE ReSTIRGISettings() 
	{
		common_temporal_pass.do_temporal_reuse_pass = true;
		
		common_temporal_pass.use_permutation_sampling = false;
		common_temporal_pass.permutation_sampling_random_bits = 42;

		common_temporal_pass.max_neighbor_search_count = 8;
		common_temporal_pass.neighbor_search_radius = 4;

		common_temporal_pass.temporal_buffer_clear_requested = false;





		common_spatial_pass.do_spatial_reuse_pass = true;

		common_spatial_pass.spatial_pass_index = 0;
		common_spatial_pass.number_of_passes = 1;
		common_spatial_pass.reuse_radius = 8;
		common_spatial_pass.reuse_neighbor_count = 1;

		common_spatial_pass.do_disocclusion_reuse_boost = false;
		common_spatial_pass.disocclusion_reuse_count = 5;

		common_spatial_pass.debug_neighbor_location = true;
		common_spatial_pass.debug_neighbor_location_direction = 0;

		common_spatial_pass.do_neighbor_rotation = true;

		common_spatial_pass.allow_converged_neighbors_reuse = false;
		common_spatial_pass.converged_neighbor_reuse_probability = 0.5f;

		common_spatial_pass.do_visibility_only_last_pass = true;
		common_spatial_pass.neighbor_visibility_count = common_spatial_pass.do_disocclusion_reuse_boost ? common_spatial_pass.disocclusion_reuse_count : common_spatial_pass.reuse_neighbor_count;




		neighbor_similarity_settings.use_normal_similarity_heuristic = false;

		neighbor_similarity_settings.normal_similarity_angle_degrees = 25.0f;
		neighbor_similarity_settings.normal_similarity_angle_precomp = 0.906307787f;

		neighbor_similarity_settings.use_plane_distance_heuristic = false;
		neighbor_similarity_settings.plane_distance_threshold = 0.1f;

		neighbor_similarity_settings.use_roughness_similarity_heuristic = false;
		neighbor_similarity_settings.roughness_similarity_threshold = 0.25f;

		m_cap = 3;
		use_confidence_weights = true;
	}

	int debug_seed = 32;

	ReSTIRGIInitialCandidatesPassSettings initial_candidates;
	ReSTIRGITemporalPassSettings temporal_pass;
	ReSTIRGISpatialPassSettings spatial_pass;
	
	ReSTIRGIReservoir* restir_output_reservoirs = nullptr;

	ReSTIRGIDebugView debug_view = ReSTIRGIDebugView::NO_DEBUG;
	float debug_view_scale_factor = 0.04f;
};

#endif
