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
	PER_PIXEL_REUSE_RADIUS = 5,
	PER_PIXEL_VALID_DIRECTIONS_PERCENTAGE = 6,
};

struct ReSTIRGISettings : public ReSTIRCommonSettings
{
	HIPRT_HOST_DEVICE ReSTIRGISettings() 
	{
		common_temporal_pass.do_temporal_reuse_pass = false;
		
		common_temporal_pass.use_permutation_sampling = false;
		common_temporal_pass.permutation_sampling_random_bits = 42;

		common_temporal_pass.max_neighbor_search_count = 8;
		common_temporal_pass.neighbor_search_radius = 4;

		common_temporal_pass.temporal_buffer_clear_requested = false;





		common_spatial_pass.do_spatial_reuse_pass = false;

		common_spatial_pass.spatial_pass_index = 0;
		common_spatial_pass.number_of_passes = 2;
		common_spatial_pass.reuse_radius = 20;
		common_spatial_pass.reuse_neighbor_count = 10;

		common_spatial_pass.do_disocclusion_reuse_boost = false;
		common_spatial_pass.disocclusion_reuse_count = 5;

		common_spatial_pass.debug_neighbor_location = false;
		common_spatial_pass.debug_neighbor_location_direction = 0;

		common_spatial_pass.do_neighbor_rotation = false;
		common_spatial_pass.use_hammersley = false;
		common_spatial_pass.spatial_neighbors_rng_seed = 42;
		common_spatial_pass.use_adaptive_directional_spatial_reuse = false;

		common_spatial_pass.allow_converged_neighbors_reuse = false;
		common_spatial_pass.converged_neighbor_reuse_probability = 0.5f;

		common_spatial_pass.do_visibility_only_last_pass = true;
		common_spatial_pass.neighbor_visibility_count = common_spatial_pass.do_disocclusion_reuse_boost ? common_spatial_pass.disocclusion_reuse_count : common_spatial_pass.reuse_neighbor_count;

		common_spatial_pass.compute_spatial_reuse_hit_rate = false;




		neighbor_similarity_settings.use_normal_similarity_heuristic = true;
		neighbor_similarity_settings.normal_similarity_angle_degrees = 37.5f;
		neighbor_similarity_settings.normal_similarity_angle_precomp = 0.906307787f;
		neighbor_similarity_settings.reject_using_geometric_normals = true;

		neighbor_similarity_settings.use_plane_distance_heuristic = true;
		neighbor_similarity_settings.plane_distance_threshold = 0.1f;

		neighbor_similarity_settings.use_roughness_similarity_heuristic = false;
		neighbor_similarity_settings.roughness_similarity_threshold = 0.25f;

		use_jacobian_rejection_heuristic = true;
		jacobian_rejection_threshold = 15.0f;

		use_neighbor_sample_point_roughness_heuristic = false;
		neighbor_sample_point_roughness_threshold = 0.1f;

		m_cap = 3;
		use_confidence_weights = true;

		debug_view = ReSTIRGIDebugView::NO_DEBUG;
		debug_view_scale_factor = 1.0f;
	}

	ReSTIRGIInitialCandidatesPassSettings initial_candidates;
	ReSTIRGITemporalPassSettings temporal_pass;
	ReSTIRGISpatialPassSettings spatial_pass;
	
	ReSTIRGIReservoir* restir_output_reservoirs = nullptr;

	ReSTIRGIDebugView debug_view;
	float debug_view_scale_factor;

	// If a neighbor has its sample point on a glossy surface, we don't want to reuse
	// that sample with the reconnection shift if it is below a given roughness threshold because
	// the BSDF at the neighbor's glossy sample point is going to evaluate to 0 anyways if we change
	// its view direction
	bool use_neighbor_sample_point_roughness_heuristic;
	float neighbor_sample_point_roughness_threshold;

	bool use_jacobian_rejection_heuristic;

	HIPRT_HOST_DEVICE float get_jacobian_heuristic_threshold() const
	{
		if (use_jacobian_rejection_heuristic)
			return jacobian_rejection_threshold;
		else
			// Returning a super high threshold so that neighbors are basically
			// never rejected based on their jacobian
			return 1.0e20f;
	}

	/**
	 * This function is used by ImGui to get a pointer to the private member
	 */
	HIPRT_HOST float* get_jacobian_heuristic_threshold_pointer() { return &jacobian_rejection_threshold;}

	HIPRT_HOST void set_jacobian_heuristic_threshold(float new_threshold) { jacobian_rejection_threshold = new_threshold;}

private:
	float jacobian_rejection_threshold;
};

#endif
