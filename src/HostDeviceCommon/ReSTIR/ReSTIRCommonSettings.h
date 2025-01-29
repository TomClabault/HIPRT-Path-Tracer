/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTIR_COMMON_SETTINGS_H
#define HOST_DEVICE_COMMON_RESTIR_COMMON_SETTINGS_H

struct ReSTIRCommonTemporalPassSettings
{
	bool do_temporal_reuse_pass = true;

	// If true, the position of the canonical temporal neighbor will be shuffled to increase
	// variation between frames and make the render more denoiser friendly
	bool use_permutation_sampling = false;
	// Random bits used for all the pixels in the image for the permutation sampling
	int permutation_sampling_random_bits = 42;

	// How many neighbors at most to check around the temporal back-projected pixel location 
	// to find a valid neighbor
	int max_neighbor_search_count = 8;
	// Radius around the temporal reprojected location of a pixel in which to look for an
	// acceptable temporal neighbor
	int neighbor_search_radius = 4;

	// If set to true, the temporal buffers will be cleared by the camera
	// rays kernel
	bool temporal_buffer_clear_requested = false;
};

struct ReSTIRCommonSpatialPassSettings
{
	bool do_spatial_reuse_pass = true;

	// What spatial pass are we currently performing?
	// Takes values in [0, number_of_passes - 1]
	int spatial_pass_index = 0;
	// How many spatial reuse pass to perform
	int number_of_passes = 2;
	// The radius within which neighbor are going to be reused spatially
	int reuse_radius = 16;
	// How many neighbors to reuse during the spatial pass
	int reuse_neighbor_count = 3;

	// constexpr here just to be able to auto-initialize the 'neighbor_visibility_count'
	// property at compile time
	static constexpr bool DO_DISOCCLUSION_BOOST = false;
	// Whether or not to increase the number of spatially resampled neighbor
	// for disoccluded pixels (that have no temporal history)
	bool do_disocclusion_reuse_boost = DO_DISOCCLUSION_BOOST;
	// How many neighbors to spatially reuse when a disocclusion is detected.
	// This reduces the increased variance of disoccluded regions
	int disocclusion_reuse_count = 5;

	// If true, reused neighbors will be hardcoded to always be 15 pixels to the right,
	// not in a circle around the center pixel.
	bool debug_neighbor_location = false;

	// Whether or not to rotate the spatial neighbor locations generated.
	// Pretty much mandatory when using Hammersley points otherwise the neighbors
	// will always be the exact same
	bool do_neighbor_rotation = true;

	// If true, neighboring pixels that have converged (if adaptive sampling is enabled)
	// won't be reused to reduce bias.
	// If false, even neighboring pixels that have converged can be reused by the spatial pass
	bool allow_converged_neighbors_reuse = false;
	// If we're allowing the spatial reuse of converged neighbors, we're doing so we're a given
	// probability instead of always/never. This helps trade performance for bias.
	float converged_neighbor_reuse_probability = 0.5f;

	// If true, the visibility in the target function will only be used on the last spatial reuse
	// pass (and also if visibility is wanted)
	bool do_visibility_only_last_pass = true;
	// Visibility term in the target function will only be used for the first
	// 'neighbor_visibility_count' neighbors, not all.
	int neighbor_visibility_count = DO_DISOCCLUSION_BOOST ? disocclusion_reuse_count : reuse_neighbor_count;
};

struct ReSTIRCommonNeighborSimiliaritySettings
{
	bool use_normal_similarity_heuristic = true;
	// User-friendly (for ImGui) normal angle. When resampling a neighbor (temporal or spatial),
	// the normal of the neighbor being re-sampled must be similar to our normal. This angle gives the
	// "similarity threshold". Normals must be within 25 degrees of each other by default
	float normal_similarity_angle_degrees = 25.0f;
	// Precomputed cosine of the angle for use in the shader
	float normal_similarity_angle_precomp = 0.906307787f; // Normals must be within 25 degrees by default

	bool use_plane_distance_heuristic = true;
	// Threshold used when determining whether a temporal neighbor is acceptable
	// for temporal reuse regarding the spatial proximity of the neighbor and the current
	// point. 
	// This is a world space distance.
	float plane_distance_threshold = 0.1f;

	bool use_roughness_similarity_heuristic = false;
	// How close the roughness of the neighbor's surface must be to ours to resample that neighbor
	// If this value is 0.25f for example, then the roughnesses must be within 0.25f of each other. Simple.
	float roughness_similarity_threshold = 0.25f;
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
	// a bias correction that uses confidence weights. Without M-capping, the M value of a reservoir
	// will keep growing exponentially through temporal and spatial reuse and when that exponentially
	// grown M value is used in confidence weights, it results in new samples being very unlikely 
	// to be chosen which in turn results in non-convergence since always the same sample is evaluated
	// for a given pixel.
	//
	// A M-cap value between 5 - 30 is usually good
	//
	// 0 for infinite M-cap (don't...)
	int m_cap = 3;

	// Whether or not to use confidence weights when resampling neighbors.
	bool use_confidence_weights = true;
};

#endif
