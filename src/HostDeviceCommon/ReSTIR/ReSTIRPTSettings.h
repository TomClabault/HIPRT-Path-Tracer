/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_RESTIR_PT_SETTINGS_H
#define HOST_DEVICE_RESTIR_PT_SETTINGS_H

#include "HostDeviceCommon/KernelOptions/ReSTIRPTOptions.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRPTSPMISSettings.h"

struct ReSTIRPTReservoir;
struct ReSTIRPTReservoirSample;

/**
 * Note that no default values are set in the temporal pass, spatial pass and neighbor similarity
 * settings structures because they are all set in the ReSTIRPTSettings constructor
 */

struct ReSTIRPTInitialCandidatesPassSettings
{
	// How many path trees are resampled per pixel during the initial candidates generation pass is the ReSTIR_PT_InitialPathTreesCount kernel option

	// ReSTIR PT only supports the LSS_LEARNING_TO_CLUSTER_MIS NEE estimator. Each path vertex resamples one learning-to-cluster light candidate,
	// which also samples the envmap when it is a cluster of the lightcuts, and the BSDF sample that continues the path, MIS weighted together

	// Buffer that contains the reservoirs that will hold the reservoir
	// for the initial candidates generated
	ReSTIRPTReservoir* initial_candidates_buffer = nullptr;
	// One sample per pixel: the path sample that the initial candidates kernel builds while tracing the path of the pixel.
	//
	// The fields of the reconnection vertex are written once (bounces 0 and 1) but read by every later candidate: kept in registers, they stay live
	// across every NEE and every traced ray of the path. In memory, they are only read back when a candidate is selected
	ReSTIRPTReservoirSample* initial_candidates_working_samples_buffer = nullptr;
};

struct ReSTIRPTTemporalPassSettings
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

	// Buffer that contains the input reservoirs for the temporal reuse pass
	ReSTIRPTReservoir* input_reservoirs = nullptr;
	// Buffer that contains the output reservoir of the temporal reuse pass
	ReSTIRPTReservoir* output_reservoirs = nullptr;
};

struct ReSTIRPTSpatialPassSettings
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

	// Buffer that contains the input reservoirs for the spatial reuse pass
	ReSTIRPTReservoir* input_reservoirs = nullptr;
	// Buffer that contains the output reservoir of the spatial reuse pass
	ReSTIRPTReservoir* output_reservoirs = nullptr;
};

struct ReSTIRPTNeighborSimiliaritySettings
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
	PT_SPMIS_CELLS							 = 8,
	PT_CELL_VARIANCE						 = 9,
};

struct ReSTIRPTSettings
{
	HIPRT_HOST_DEVICE ReSTIRPTSettings()
	{
		temporal_pass.do_temporal_reuse_pass = false;

		temporal_pass.max_neighbor_search_count = 8;
		temporal_pass.neighbor_search_radius	= 4;

		temporal_pass.temporal_buffer_clear_requested = false;

		spatial_pass.do_spatial_reuse_pass = true;

		spatial_pass.spatial_pass_index	  = 0;
		spatial_pass.number_of_passes	  = 1;
		spatial_pass.reuse_radius		  = 20;
		spatial_pass.reuse_neighbor_count = 5;

		spatial_pass.debug_neighbor_location		   = false;
		spatial_pass.debug_neighbor_location_direction = 0;

		spatial_pass.use_adaptive_directional_spatial_reuse = false;
		spatial_pass.spatial_neighbors_rng_seed				= 42;

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
		m_cap = 1;

		debug_view				= ReSTIRPTDebugView::PT_NO_DEBUG;
		debug_view_scale_factor = 2.0f;
	}

	ReSTIRPTInitialCandidatesPassSettings initial_candidates;
	// Settings for the temporal reuse pass
	ReSTIRPTTemporalPassSettings temporal_pass;
	// Settings for the spatial reuse pass
	ReSTIRPTSpatialPassSettings spatial_pass;
	ReSTIRPTSPMISSettings spmis_settings;

	ReSTIRPTNeighborSimiliaritySettings neighbor_similarity_settings;

	ReSTIRPTReservoir* restir_pt_output_reservoirs = nullptr;

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

	/**
	 * Whether the SPMIS spatial reuse uses the union domain reuse (ReSTIRPTSPMISUnionDomainSettings) or the legacy single reuse cell path.
	 *
	 * 'restir_pt_mis_weights_type' is the value of the ReSTIR_PT_MISWeightsType kernel option: the macro itself on the device, the value of
	 * GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE on the host.
	 *
	 * The union domain reuse needs:
	 *	- its setting enabled
	 *	- the non-defensive stochastic pairwise MIS weights type (the group-wise MIS weights replace them, there is no defensive version)
	 *	- no temporal reuse and a single spatial reuse pass: the group-wise MIS weights are an exact partition of unity only when all the input
	 *	  reservoirs have the same confidence, which is the case for the initial candidates reservoirs (confidence 1)
	 *
	 * Prefer using this function rather than checking 'spmis_settings.union_domain.do_union_domain_reuse' directly
	 */
	HIPRT_HOST_DEVICE bool do_spmis_union_domain_reuse(int restir_pt_mis_weights_type) const
	{
		return spmis_settings.union_domain.do_union_domain_reuse && restir_pt_mis_weights_type == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS &&
			   !temporal_pass.do_temporal_reuse_pass && spatial_pass.number_of_passes == 1;
	}

private:
	float jacobian_rejection_threshold;
};

#endif // #ifndef HOST_DEVICE_RESTIR_PT_SETTINGS_H
