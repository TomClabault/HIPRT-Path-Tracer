/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_RESTIR_DI_SETTINGS_H
#define HOST_DEVICE_RESTIR_DI_SETTINGS_H

class ReSTIRDIReservoir;

struct InitialCandidatesSettings
{
	// How many light candidates to resamples during the initial candidates sampling pass
	int number_of_initial_light_candidates = 4;
	// How many BSDF candidates to resamples during the initial candidates sampling pass
	int number_of_initial_bsdf_candidates = 1;
	// For each 'number_of_initial_light_candidates', the probability that this light sample
	// will sample the envmap instead of a light in the scene
	float envmap_candidate_probability = 0.20f;

	// Buffer that contains the reservoirs that will hold the reservoir
	// for the initial candidates generated
	ReSTIRDIReservoir* output_reservoirs = nullptr;
};

struct TemporalPassSettings
{
	bool do_temporal_reuse_pass = true;

	// Whether or not to use the G-buffer of last frame when resampling the temporal neighbor.
	// This is required to avoid bias with camera movements but this comes at a VRAM cost
	// which we may not want to pay (if we're accumulating frames with a still camera for example, ...
	// we don't need that)
	bool use_last_frame_g_buffer = true;

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
	int neighbor_search_radius = 8;

	// If set to true, the temporal buffers will be cleared by the camera
	// rays kernel
	bool temporal_buffer_clear_requested = false;

	// The temporal reuse pass resamples the initial candidates as well as the last frame reservoirs which
	// are accessed through this pointer
	ReSTIRDIReservoir* input_reservoirs = nullptr;
	// Buffer that holds the output of the temporal reuse pass
	ReSTIRDIReservoir* output_reservoirs = nullptr;
};

struct SpatialPassSettings
{
	bool do_spatial_reuse_pass = true;

	// What spatial pass are we currently performing?
	// Takes values in [0, number_of_passes - 1]
	int spatial_pass_index = 0;
	// How many spatial reuse pass to perform
	int number_of_passes = 2;
	// The radius within which neighbor are going to be reused spatially
	int spatial_reuse_radius = 20;
	// How many neighbors to reuse during the spatial pass
	int spatial_reuse_neighbor_count = 2;

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

	// Buffer that contains the input reservoirs for the spatial reuse pass
	ReSTIRDIReservoir* input_reservoirs = nullptr;
	// Buffer that contains the output reservoir of the spatial reuse pass
	ReSTIRDIReservoir* output_reservoirs = nullptr;
};

struct ReSTIRDISettings
{
	// Settings for the initial candidates generation pass
	InitialCandidatesSettings initial_candidates;
	// Settings for the temporal reuse pass
	TemporalPassSettings temporal_pass;
	// Settings for the spatial reuse pass
	SpatialPassSettings spatial_pass;

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
	int m_cap = 10;
	// M-cap for glossy surfaces: helps to reduce darkening with cameray ray jittering + temporal reuse 
	// + accumulation
	int glossy_m_cap = 3;
	// Below what roughness is a surface considering glossy and will use the glossy m-cap
	float glossy_threshold = 0.1f;

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
