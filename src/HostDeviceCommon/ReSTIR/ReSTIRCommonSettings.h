/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTIR_COMMON_SETTINGS_H
#define HOST_DEVICE_COMMON_RESTIR_COMMON_SETTINGS_H

#include "HostDeviceCommon/ReSTIR/ReSTIRPTSPMISSettings.h"

/**
 * Note that no default values are set here because they are all set in
 * the ReSTIR_XXX_DefaultSettings structure/header files
 */

struct ReSTIRCommonTemporalPassSettings
{
	bool do_temporal_reuse_pass;

	// How many neighbors at most to check around the temporal back-projected pixel location
	// to find a valid neighbor
	int max_neighbor_search_count;
	// Radius around the temporal reprojected location of a pixel in which to look for an
	// acceptable temporal neighbor
	int neighbor_search_radius;

	// If set to true, the temporal buffers will be cleared by the camera
	// rays kernel
	bool temporal_buffer_clear_requested;
};

struct ReSTIRCommonSpatialPassSettings
{
	bool do_spatial_reuse_pass;

	// What spatial pass are we currently performing?
	// Takes values in [0, number_of_passes - 1]
	int spatial_pass_index;
	// How many spatial reuse pass to perform
	int number_of_passes;
	// The radius within which neighbor are going to be reused spatially
	int reuse_radius;
	// if true, the reuse radius will automatically be adjusted based on the render resolution
	bool auto_reuse_radius = true;
	// How many neighbors to reuse during the spatial pass
	int reuse_neighbor_count;

	// If true, reused neighbors will be hardcoded to always be 'reuse_radius' pixels to the right,
	// not in a circle around the center pixel.
	bool debug_neighbor_location;
	// If this is 0, the debug location will be horizontal
	// If this is 1, the debug location will be vertical
	// If this is 2, the debug location will be in diagonal
	int debug_neighbor_location_direction;

	// This seed is used to generate the spatial neighbors positions if not using Hammersley
	unsigned int spatial_neighbors_rng_seed;

	// If true, the best per-pixel spatial reuse radius to use as
	// well as the sectors in the spatial reuse disk (split in 32 sectors) that should be used for reuse
	// will be precomputed in a prepass
	//
	// This increases the spatial reuse "hit rate" (i.e. the number of neighbors that are not rejected by G-Buffer heuristics)
	// and thus increases convergence speed.
	bool use_adaptive_directional_spatial_reuse;

	/**
	 * If you want to check whether you should use the features of the adaptive directional spatial
	 * reuse, prefer using this function rather than directly checking the 'use_adaptive_directional_spatial_reuse'
	 * member
	 *
	 * This is because the directional spatial reuse feature cannot be used in realtime mode so if you use the
	 * 'use_adaptive_directional_spatial_reuse' member directly, you would also have to check for 'render_data.render_settings.accumulate'
	 * everytime.
	 *
	 * This function does it all
	 */
	HIPRT_HOST_DEVICE bool do_adaptive_directional_spatial_reuse(bool render_data_render_settings_accumulate) const
	{
		return use_adaptive_directional_spatial_reuse && render_data_render_settings_accumulate;
	}

	unsigned long long int* per_pixel_spatial_reuse_directions_mask_ull = nullptr;
	// Framebuffer that contains per-pixel spatial radius for use in the spatial reuse passes of ReSTIR.
	// This framebuffer is filled by the
	unsigned char* per_pixel_spatial_reuse_radius = nullptr;
	// The minimum radius that will be used per pixel when the optimal per - pixel spatial reuse
	// radius is computed by adaptive-directional spatial reuse
	int minimum_per_pixel_reuse_radius = 3;

	// This variable here is spatial because it is written to at the beginning of the spatial reuse pass.
	// The only goal of this variable is to be able to carry around the function the direction reuse mask
	// (i.e. which directions are allowed for reuse)of the pixel.
	//
	// This is purely to avoid passing yet another arguments to every function in the code...
	unsigned long long int current_pixel_directions_reuse_mask = 0;
};

struct ReSTIRCommonNeighborSimiliaritySettings
{
	bool use_normal_similarity_heuristic;
	// User-friendly (for ImGui) normal angle. When resampling a neighbor (temporal or spatial),
	// the normal of the neighbor being re-sampled must be similar to our normal. This angle gives the
	// "similarity threshold". Normals must be within 25 degrees of each other by default
	float normal_similarity_angle_degrees;
	// Precomputed cosine of the angle for use in the shader
	float normal_similarity_angle_precomp; // Normals must be within 25 degrees by default
	// If true, the geometric normals will be compared for the normal rejection heuristic.
	// If false, smooth vertex normals (or normal map normals) will be compared
	//
	// Geometric normals are prefered as they are not disturbed by high details normal maps
	bool reject_using_geometric_normals;

	bool use_plane_distance_heuristic;
	// Threshold used when determining whether a temporal neighbor is acceptable
	// for temporal reuse regarding the spatial proximity of the neighbor and the current
	// point.
	// This is a world space distance.
	float plane_distance_threshold;

	bool use_roughness_similarity_heuristic;
	// How close the roughness of the neighbor's surface must be to ours to resample that neighbor
	// If this value is 0.25f for example, then the roughnesses must be within 0.25f of each other. Simple.
	float roughness_similarity_threshold;
};

struct ReSTIRCommonSettings
{
	// Settings for the initial candidates generation pass
	ReSTIRCommonTemporalPassSettings common_temporal_pass;
	// Settings for the spatial reuse pass
	ReSTIRCommonSpatialPassSettings common_spatial_pass;

	ReSTIRCommonNeighborSimiliaritySettings neighbor_similarity_settings;

	// When finalizing the reservoir in the spatial reuse pass, what value
	// to cap the reservoirs's M value to.
	//
	// The point of this parameter is to avoid too much correlation between frames if using
	// MIS weights that uses confidence weights. Without M-capping, the M value of a reservoir
	// will keep growing exponentially through temporal and spatial reuse and when that exponentially
	// grown M value is used in confidence weights, it results in new samples being very unlikely
	// to be chosen which in turn results in non-convergence since always the same sample is evaluated
	// for a given pixel.
	//
	// A M-cap value between 5 - 30 is usually good
	//
	// 0 for infinite M-cap (don't...)
	int m_cap;

	// Beta exponent to the difference function for symmetric and asymmetric ratio MIS weights
	float symmetric_ratio_mis_weights_beta_exponent = 2.0f;
};

#endif // #ifndef HOST_DEVICE_COMMON_RESTIR_COMMON_SETTINGS_H
