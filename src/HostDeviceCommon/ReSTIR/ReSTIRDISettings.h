/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_RESTIR_DI_SETTINGS_H
#define HOST_DEVICE_RESTIR_DI_SETTINGS_H

struct ReSTIRDIReservoir;
struct ReSTIRDIPresampledLight;

struct ReSTIRDIInitialCandidatesSettings
{
	// How many light candidates to resamples during the initial candidates sampling pass
	int number_of_initial_light_candidates = 4;
	// How many BSDF candidates to resamples during the initial candidates sampling pass
	int number_of_initial_bsdf_candidates = 1;
	// For each 'number_of_initial_light_candidates', the probability that this light sample
	// will sample the envmap instead of a light in the scene
	float envmap_candidate_probability = 0.25f;

	// Buffer that contains the reservoirs that will hold the reservoir
	// for the initial candidates generated
	ReSTIRDIReservoir* output_reservoirs = nullptr;
};

struct ReSTIRDITemporalPassSettings
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

	// The temporal reuse pass resamples the initial candidates as well as the last frame reservoirs which
	// are accessed through this pointer
	ReSTIRDIReservoir* input_reservoirs = nullptr;
	// Buffer that holds the output of the temporal reuse pass
	ReSTIRDIReservoir* output_reservoirs = nullptr;
};

struct ReSTIRDISpatialPassSettings
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

	// Buffer that contains the input reservoirs for the spatial reuse pass
	ReSTIRDIReservoir* input_reservoirs = nullptr;
	// Buffer that contains the output reservoir of the spatial reuse pass
	ReSTIRDIReservoir* output_reservoirs = nullptr;
};

struct ReSTIRDILightPresamplingSettings
{
	// From all the lights of the scene, how many subsets to presample
	int number_of_subsets = 128;
	// How many lights to presample in each subset
	int subset_size = 1024;
	// All threads in a tile_size * tile_size block of pixels will sample from the same subset of light samples
	int tile_size = 8;

	// Buffer for the presampled light samples
	ReSTIRDIPresampledLight* light_samples;
};

struct ReSTIRDISettings
{
	// Settings for the initial candidates generation pass
	ReSTIRDIInitialCandidatesSettings initial_candidates;
	// Settings for the temporal reuse pass
	ReSTIRDITemporalPassSettings temporal_pass;
	// Settings for the spatial reuse pass
	ReSTIRDISpatialPassSettings spatial_pass;
	// Settings for the light presampling pass
	ReSTIRDILightPresamplingSettings light_presampling;

	// If true, the spatial and temporal pass will be fused into a single kernel call.
	// This avois a synchronization barrier between the temporal pass and the spatial pass
	// and increases performance.
	// Because the spatial must then resample without the output of the temporal pass, the spatial
	// pass only resamples on the temporal reservoir buffer, not the temporal + initial candidates reservoir
	// (which is the output of the temporal pass). This is usually imperceptible.
	bool do_fused_spatiotemporal = false;

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
	// This is handy to remember which buffer the temporal reuse pass is going to use
	// as input on the next frame
	ReSTIRDIReservoir* restir_output_reservoirs;
};

#endif
