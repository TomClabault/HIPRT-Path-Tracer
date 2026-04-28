/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_RESTIR_DI_SETTINGS_H
#define HOST_DEVICE_RESTIR_DI_SETTINGS_H

#include "HostDeviceCommon/ReSTIR/ReSTIRDIDefaultSettings.h"

struct ReSTIRDIReservoir;

struct ReSTIRDIInitialCandidatesSettings
{
	// How many light candidates to resamples during the initial candidates sampling pass
	int number_of_initial_light_candidates = 1;
	// How many BSDF candidates to resamples during the initial candidates sampling pass
	int number_of_initial_bsdf_candidates = 0;
	// For each 'number_of_initial_light_candidates', the probability that this light sample
	// will sample the envmap instead of a light in the scene
	float envmap_candidate_probability = 0.5f;

	// Buffer that contains the reservoirs that will hold the reservoir
	// for the initial candidates generated
	ReSTIRDIReservoir* output_reservoirs = nullptr;
};

struct ReSTIRDITemporalPassSettings
{
	// The temporal reuse pass resamples the initial candidates as well as the last frame reservoirs which
	// are accessed through this pointer
	ReSTIRDIReservoir* input_reservoirs = nullptr;
	// Buffer that holds the output of the temporal reuse pass
	ReSTIRDIReservoir* output_reservoirs = nullptr;
};

struct ReSTIRDISpatialPassSettings
{
	// Buffer that contains the input reservoirs for the spatial reuse pass
	ReSTIRDIReservoir* input_reservoirs = nullptr;
	// Buffer that contains the output reservoir of the spatial reuse pass
	ReSTIRDIReservoir* output_reservoirs = nullptr;
};

struct ReSTIRDISettings : public ReSTIRCommonSettings
{
	HIPRT_HOST_DEVICE ReSTIRDISettings()
	{
		common_temporal_pass.do_temporal_reuse_pass = true;

		common_temporal_pass.max_neighbor_search_count = 8;
		common_temporal_pass.neighbor_search_radius	   = 4;

		common_temporal_pass.temporal_buffer_clear_requested = false;

		common_spatial_pass.do_spatial_reuse_pass = true;

		common_spatial_pass.spatial_pass_index	 = 0;
		common_spatial_pass.number_of_passes	 = 1;
		common_spatial_pass.reuse_radius		 = 16;
		common_spatial_pass.reuse_neighbor_count = 5;

		common_spatial_pass.do_disocclusion_reuse_boost = false;
		common_spatial_pass.disocclusion_reuse_count	= 5;

		common_spatial_pass.debug_neighbor_location			  = false;
		common_spatial_pass.debug_neighbor_location_direction = 0;

		common_spatial_pass.spatial_neighbors_rng_seed			   = 42;
		common_spatial_pass.use_adaptive_directional_spatial_reuse = true;

		common_spatial_pass.allow_converged_neighbors_reuse		 = false;
		common_spatial_pass.converged_neighbor_reuse_probability = 0.5f;

		common_spatial_pass.compute_spatial_reuse_hit_rate = false;

		neighbor_similarity_settings.use_normal_similarity_heuristic = true;
		neighbor_similarity_settings.normal_similarity_angle_degrees = 37.5f;
		neighbor_similarity_settings.normal_similarity_angle_precomp = 0.906307787f;
		neighbor_similarity_settings.reject_using_geometric_normals	 = true;

		neighbor_similarity_settings.use_plane_distance_heuristic = true;
		neighbor_similarity_settings.plane_distance_threshold	  = 0.1f;

		neighbor_similarity_settings.use_roughness_similarity_heuristic = false;
		neighbor_similarity_settings.roughness_similarity_threshold		= 0.25f;

		m_cap				   = 3;
		use_confidence_weights = true;
	}

	// Settings for the initial candidates generation pass
	ReSTIRDIInitialCandidatesSettings initial_candidates;
	// Settings for the temporal reuse pass
	ReSTIRDITemporalPassSettings temporal_pass;
	// Settings for the spatial reuse pass
	ReSTIRDISpatialPassSettings spatial_pass;

	// Whether or not to trace a visibility ray when evaluating the final light sample produced by ReSTIR.
	// This is strongly biased but allows good performance.
	bool do_final_shading_visibility = true;

	// Pointer to the buffer that contains the output of all the passes of ReSTIR DI
	// This the buffer that should be used when evaluating direct lighting in the path tracer
	//
	// This buffer isn't allocated but is actually just a pointer
	// to the buffer that was last used as the output of the resampling
	// passes last frame.
	// For example if there was spatial reuse in last frame, this buffer
	// is going to be a pointer to the output of the spatial reuse pass
	// If there was only temporal reuse pass last frame, this buffer is going
	// to be a pointer to the output of the temporal reuse pass
	//
	// This is handy to know which buffer the temporal reuse pass is going to use
	// as input on the next frame
	ReSTIRDIReservoir* restir_output_reservoirs = nullptr;
};

#endif
