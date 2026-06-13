/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_RESTIR_PT_SETTINGS_H
#define HOST_DEVICE_RESTIR_PT_SETTINGS_H

#include "HostDeviceCommon/ReSTIR/ReSTIRPTDefaultSettings.h"

struct ReSTIRPTReservoir;

struct ReSTIRPTInitialCandidatesPassSettings
{
	// How many path trees are resampled per pixel during the initial candidates generation pass.
	int initial_path_trees_count = 1;

	// ReSTIR PT uses RIS at each path vertex to sampel lights and BSDFs for the initial candidates generation pass. These parameters control how many
	// candidates are generated for each type of sampling. Both candidate types are mixed with MIS
	int nee_ris_number_of_light_candidates	= 0;
	int nee_ris_number_of_envmap_candidates = 0;
	int nee_ris_number_of_bsdf_candidates	= 1;

	// Buffer that contains the reservoirs that will hold the reservoir
	// for the initial candidates generated
	ReSTIRPTReservoir* initial_candidates_buffer = nullptr;
};

struct ReSTIRPTTemporalPassSettings
{
	// Buffer that contains the input reservoirs for the temporal reuse pass
	ReSTIRPTReservoir* input_reservoirs = nullptr;
	// Buffer that contains the output reservoir of the temporal reuse pass
	ReSTIRPTReservoir* output_reservoirs = nullptr;
};

struct ReSTIRPTSpatialPassSettings
{
	// Buffer that contains the input reservoirs for the spatial reuse pass
	ReSTIRPTReservoir* input_reservoirs = nullptr;
	// Buffer that contains the output reservoir of the spatial reuse pass
	ReSTIRPTReservoir* output_reservoirs = nullptr;
};

enum ReSTIRPTDebugView
{
	PT_NO_DEBUG								 = 0,
	PT_SHADE_ONLY_INITIAL_CANDIDATES		 = 1,
	PT_FINAL_RESERVOIR_UCW					 = 2,
	PT_TARGET_FUNCTION						 = 3,
	PT_WEIGHT_SUM							 = 4,
	PT_M_COUNT								 = 5,
	PT_PER_PIXEL_REUSE_RADIUS				 = 6,
	PT_PER_PIXEL_VALID_DIRECTIONS_PERCENTAGE = 7,
};

struct ReSTIRPTSettings : public ReSTIRCommonSettings
{
	HIPRT_HOST_DEVICE ReSTIRPTSettings()
	{
		common_temporal_pass.do_temporal_reuse_pass = false;

		common_temporal_pass.max_neighbor_search_count = 8;
		common_temporal_pass.neighbor_search_radius	   = 4;

		common_temporal_pass.temporal_buffer_clear_requested = false;

		common_spatial_pass.do_spatial_reuse_pass = true;

		common_spatial_pass.spatial_pass_index	 = 0;
		common_spatial_pass.number_of_passes	 = 2;
		common_spatial_pass.reuse_radius		 = 20;
		common_spatial_pass.reuse_neighbor_count = 1;

		common_spatial_pass.debug_neighbor_location			  = false;
		common_spatial_pass.debug_neighbor_location_direction = 0;

		common_spatial_pass.use_adaptive_directional_spatial_reuse = true;
		common_spatial_pass.spatial_neighbors_rng_seed			   = 42;

		common_spatial_pass.allow_converged_neighbors_reuse		 = false;
		common_spatial_pass.converged_neighbor_reuse_probability = 0.5f;

		common_spatial_pass.compute_spatial_reuse_hit_rate = false;

		neighbor_similarity_settings.use_normal_similarity_heuristic = true;
		neighbor_similarity_settings.normal_similarity_angle_degrees = 37.5f;
		neighbor_similarity_settings.normal_similarity_angle_precomp = 0.906307787f;
		neighbor_similarity_settings.reject_using_geometric_normals	 = true;

		neighbor_similarity_settings.use_plane_distance_heuristic = true;
		neighbor_similarity_settings.plane_distance_threshold	  = 0.1f;

		neighbor_similarity_settings.use_roughness_similarity_heuristic = true;
		neighbor_similarity_settings.roughness_similarity_threshold		= 0.25f;

		use_jacobian_rejection_heuristic = false;
		jacobian_rejection_threshold	 = 15.0f;

		use_neighbor_sample_point_roughness_heuristic = false;
		neighbor_sample_point_roughness_threshold	  = 0.1f;

		// Very very small m-cap to avoid correlations
		m_cap				   = 1;
		use_confidence_weights = true;

		debug_view				= ReSTIRPTDebugView::PT_NO_DEBUG;
		debug_view_scale_factor = 1.0f;
	}

	ReSTIRPTInitialCandidatesPassSettings initial_candidates;
	ReSTIRPTTemporalPassSettings temporal_pass;
	ReSTIRPTSpatialPassSettings spatial_pass;

	ReSTIRPTReservoir* restir_output_reservoirs = nullptr;

	ReSTIRPTDebugView debug_view;
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
	HIPRT_HOST_DEVICE float* get_jacobian_heuristic_threshold_pointer()
	{
		return &jacobian_rejection_threshold;
	}

	HIPRT_HOST_DEVICE void set_jacobian_heuristic_threshold(float new_threshold)
	{
		jacobian_rejection_threshold = new_threshold;
	}

private:
	float jacobian_rejection_threshold;
};

#endif
