/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTIR_COMMON_SETTINGS_H
#define HOST_DEVICE_COMMON_RESTIR_COMMON_SETTINGS_H

/**
 * Note that no default values are set here because they are all set in
 * the ReSTIR_XXX_DefaultSettings structure/header files
 */

struct ReSTIRCommonTemporalPassSettings
{
	bool do_temporal_reuse_pass;

	// If true, the position of the canonical temporal neighbor will be shuffled to increase
	// variation between frames and make the render more denoiser friendly
	bool use_permutation_sampling;
	// Random bits used for all the pixels in the image for the permutation sampling
	int permutation_sampling_random_bits;

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

	// Whether or not to increase the number of spatially resampled neighbor
	// for disoccluded pixels (that have no temporal history)
	bool do_disocclusion_reuse_boost;
	// How many neighbors to spatially reuse when a disocclusion is detected.
	// This reduces the increased variance of disoccluded regions
	int disocclusion_reuse_count;

	// If true, reused neighbors will be hardcoded to always be 'reuse_radius' pixels to the right,
	// not in a circle around the center pixel.
	bool debug_neighbor_location;
	// If this is 0, the debug location will be horizontal
	// If this is 1, the debug location will be vertical
	// If this is 2, the debug location will be in diagonal
	int debug_neighbor_location_direction;

	// Whether or not to rotate the spatial neighbor locations generated.
	// Pretty much mandatory when using Hammersley points otherwise the neighbors
	// will always be the exact same
	bool do_neighbor_rotation;
	// Whether or not to use a Hammersley point set for generating the position of the
	// spatial neighbors
	//
	// If not using Hammersely, uncorrelated random numbers will be used
	bool use_hammersley;
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

	// If true, neighboring pixels that have converged (if adaptive sampling is enabled)
	// won't be reused to reduce bias.
	// If false, even neighboring pixels that have converged can be reused by the spatial pass
	bool allow_converged_neighbors_reuse;
	// If we're allowing the spatial reuse of converged neighbors, we're doing so we're a given
	// probability instead of always/never. This helps trade performance for bias.
	float converged_neighbor_reuse_probability;

	// If true, the visibility in the target function will only be used on the last spatial reuse
	// pass (and also if visibility is wanted)
	bool do_visibility_only_last_pass;
	// Visibility term in the target function will only be used for the first
	// 'neighbor_visibility_count' neighbors, not all.
	int neighbor_visibility_count;

	unsigned int* per_pixel_spatial_reuse_directions_mask_u = nullptr;
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

	// Whether or not to gather statistics on the hit rate of the spatial reuse pass (i.e. how many
	// neighbors are rejected because of the G-Buffer heuristics vs. the maximum number of neighbors that can be reused)
	bool compute_spatial_reuse_hit_rate;
	// Counters for gathering the statistics on the spatial reuse hit rate
	AtomicType<unsigned long long int>* spatial_reuse_hit_rate_hits = nullptr;
	AtomicType<unsigned long long int>* spatial_reuse_hit_rate_total = nullptr;

	// If the decoupled shading reuse is enabled, the reservoirs will be shaded during the spatial reuse
	// and the shading result will be stored in this buffer and will then be looked up during path tracing
	// when we want our ReSTIR DI direct lighting estimation
	ColorRGB32F* decoupled_shading_reuse_buffer = nullptr;
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
	// a bias correction that uses confidence weights. Without M-capping, the M value of a reservoir
	// will keep growing exponentially through temporal and spatial reuse and when that exponentially
	// grown M value is used in confidence weights, it results in new samples being very unlikely 
	// to be chosen which in turn results in non-convergence since always the same sample is evaluated
	// for a given pixel.
	//
	// A M-cap value between 5 - 30 is usually good
	//
	// 0 for infinite M-cap (don't...)
	int m_cap;

	// Whether or not to use confidence weights when resampling neighbors.
	bool use_confidence_weights;

	// Beta exponent to the difference function for symmetric and asymmetric ratio MIS weights
	float symmetric_ratio_mis_weights_beta_exponent = 2.0f;
};

#endif
