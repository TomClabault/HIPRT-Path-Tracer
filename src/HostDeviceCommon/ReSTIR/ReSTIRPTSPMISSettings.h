/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_PT_COMMON_SPMIS_SETTINGS_H
#define DEVICE_INCLUDES_RESTIR_PT_COMMON_SPMIS_SETTINGS_H

#include "HostDeviceCommon/AtomicType.h"

static constexpr unsigned int RESTIR_PT_SPMIS_REUSE_CELL_CACHE_SIZE = 4;

// Maximum number of distinct SPMIS cells in the reuse domain of the union domain reuse (ReSTIRPTSPMISUnionDomainSettings)
static constexpr int RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS = 4;
// Maximum number of SPMIS cell searches done to find the cells of a union domain (a search may not find a new cell)
static constexpr int RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELL_SEARCHES = 8;
// Maximum values of ReSTIRPTSPMISUnionDomainSettings::neighbor_count, visibility_candidate_count and mis_group_size
static constexpr int RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS				= 8;
static constexpr int RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_CANDIDATES = 4;
static constexpr int RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_MIS_GROUP_SIZE		= 3;
// Maximum value of ReSTIRPTSPMISUnionDomainSettings::visibility_pool_size
static constexpr int RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE = 8;
// Maximum number of probes (canonical_group_count x mis_group_size) for which the canonical MIS weight of the union domain reuse is computed as a
// U-statistic over all the groups of probes. Above that, the canonical MIS weight averages canonical_group_count disjoint groups
static constexpr int RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES = 8;
// Number of floats of the neighbor record of a pixel (two float4, see ReSTIR_PT_SPMIS_BuildNeighborRecords)
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_NEIGHBOR_RECORD_FLOAT_COUNT = 8;
// Per pixel in the union domain cache: the number of cached cells then one representative pixel per cell
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE_STRIDE = 1 + RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS;

struct ReSTIRPTCommonSPMISCompatibilityGuidedCellSelectionSettings
{
	// If true, uses the heuristic from [Compatibility-Guided Neighbor Selection for ReSTIR, Junkins, 2026] to weight the selection of the neighboring cell.
	// Disabled by default because not that good, a bit worse than hard rejection actually
	bool do_compatibility_guided_selection = false;

	// How much the distance to the center pixel is scaled when computing the weight of a neighboring cell. The lower this factor, the more closer cells are
	// preferred. 0.0f turns off distance scaling.
	float solid_angle_omega = 0.05f;
};

/**
 * Settings of the SPMIS "union domain" spatial reuse, see SpatialSPMISUnionDomain.h:
 *	- the reuse domain of a pixel is the union of up to RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS distinct SPMIS cells found by the SPMIS cell search
 *	- the neighbors are sampled with a two-stage selection: candidates drawn approximately proportionally to the visibility-free shifted integrand
 *	  (closed form on compact neighbor records), then one visible candidate resampled per neighbor
 *	- the MIS weights are stochastic group-wise MIS weights (generalization of the stochastic pairwise MIS weights, see SpatialGroupwiseMISWeight.h)
 *
 * Only used when ReSTIRPTSettings::do_spmis_union_domain_reuse() returns true: the group-wise MIS weights are only an exact partition of unity when all
 * the input reservoirs have the same confidence, which requires no temporal reuse and a single spatial reuse pass. The legacy SPMIS spatial reuse runs
 * otherwise
 */
struct ReSTIRPTSPMISUnionDomainSettings
{
	bool do_union_domain_reuse = true;

	// How many neighbors are resampled from the union domain. Replaces ReSTIRPTSpatialPassSettings::reuse_neighbor_count. Clamped to
	// [1, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS]
	int neighbor_count = 3;
	// How many candidates are drawn per neighbor before tracing their visibility and resampling one visible candidate. One shadow ray per candidate.
	// Clamped to [1, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_CANDIDATES]
	int visibility_candidate_count = 3;
	// Number of pixels per group of the stochastic group-wise MIS weights. 1 is pairwise MIS. Clamped to [1, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_MIS_GROUP_SIZE]
	int mis_group_size = 2;
	// How many groups of mis_group_size pixels (one shadow ray each) estimate the MIS weight of the canonical sample
	int canonical_group_count = 4;
	// Domains with at most that many non-zero neighbors are sampled exactly (streaming over all their neighbors). Larger domains draw that many
	// stratified candidates in total, split over the neighbors, through the CDFs of the cells
	int selection_candidate_count = 64;
	// If > 0, the neighbors are taken among the visible candidates of ONE pool of visibility_pool_size candidates for the pixel (one shadow ray each,
	// whatever neighbor_count) instead of visibility_candidate_count candidates per neighbor, see spmis_union_domain_select_neighbors_from_pool().
	// Clamped to RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE
	int visibility_pool_size = 0;

	// For every pixel with a non-zero spatial reuse input reservoir, RESTIR_PT_SPMIS_UNION_DOMAIN_NEIGHBOR_RECORD_FLOAT_COUNT floats at the slot of the
	// pixel in its cell (same indexing as pixel_indices_sorted), built every frame by ReSTIR_PT_SPMIS_BuildNeighborRecords
	float* neighbor_records = nullptr;
	// Per-pixel cache of the union domain (RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE_STRIDE unsigned ints per pixel) so that the cell search only runs when the cache
	// is empty. Emptied whenever the reuse cell cache all_pixels_reuse_cell_pixel_index is
	unsigned int* domain_cache = nullptr;
};

struct ReSTIRPTSPMISSettings
{
	// Screen space tile size
	int tile_size = 32;

	// Initial radius for finding a good neighboring cell
	float initial_search_radius = 10.0f;
	// How much the search radius grows by after each step
	float neighboring_cell_search_radius_increment = 1.25f;
	// How many iterations to search for a neighboring cell
	int neighboring_cell_max_search_iterations = 8;
	// When searching for a neighboring cell to reuse from, cells further away are downweighted by 1.0f / distance_to_center_pixel to improve variance (since we
	// will then be reusing from closer pixels). However, directly weighting by the inverse distance isn't enough so we're further scaling by a controllable
	// factor. The lower this factor, the more closer cells are preferred. 0.0f turns off distance scaling.
	float distance_scaling			 = 8.0f;
	bool variance_aware_reuse_radius = true;

	ReSTIRPTCommonSPMISCompatibilityGuidedCellSelectionSettings compatibility_guided_cell_selection;

	int hash_normal_precision		  = 2;
	float hash_normal_jitter_strength = 0.2f;

	// How many pixels to stream from a cell to produce one non-canonical neighbor
	int ris_neighbor_count = 8;
	// If true, ris_neighbor_count is ignored and all non-zero pixels are importance sampled at once through inverse CDF sampling
	bool ris_neighbor_cdf	   = true;
	int ris_neighbor_cdf_count = 8;

	// Whether or not to scale non-canonical candidates confidence during resampling, section 4.3 of the SPMIS paper
	bool do_non_canonical_confidence_adjustement = false;

	// How many neighboring pixels to sample to estimate the canonical weight
	int canonical_weight_estimation_count = 2;

	// Union domain reuse. The settings above that only apply to the selection and MIS weights of the single reuse cell (ris_neighbor_count,
	// ris_neighbor_cdf, ris_neighbor_cdf_count, do_non_canonical_confidence_adjustement, canonical_weight_estimation_count) are not used by it
	ReSTIRPTSPMISUnionDomainSettings union_domain;

	// Size of the fullscreen buffers
	unsigned int pixel_hashes_count = 0;

	// Fullscreen buffer that contains the hash cell index of a given pixel
	unsigned int* all_pixel_hashes						 = nullptr;
	AtomicType<unsigned int>* all_pixel_hashes_checksums = nullptr;

	// For each pixel, the index in its hash cell
	unsigned int* all_pixels_index_in_cell = nullptr;
	// For each pixel, the indices of pixels whose reuse cells can be reused from. These are precomputed to avoid having to do an expensive reuse cell search
	// each frame.
	// The buffer contains RESTIR_PT_SPMIS_REUSE_CELL_CACHE_SIZE entries per pixel.
	unsigned int* all_pixels_reuse_cell_pixel_index = nullptr;
	// A fullscreen buffer which contains, for each cell, the list of pixel indices that belongs to that cell. Pixel indices in each cell are sorted with
	// important pixels (non-zero contribution reservoirs) first and non-important pixels after that. This buffer should be indexed as [cell_ffset +
	// index_in_cell] with cell_offset coming from the cell_offsets buffer and index_in_cell in [0, cell_pixels_counts[cell_index]], with the first
	// cell_non_zero_reservoir_counters[cell_index] pixels indices of the cell being the important pixels and the remaining ones being the non-important pixels.
	unsigned int* pixel_indices_sorted = nullptr;

	// How many **pixels** are in the cells, containing non-zero reservoirs or not
	// TODO unsigned char is enough for 8 * 8 cells
	AtomicType<unsigned short int>* cell_pixels_counters = nullptr;
	// For each cell, how many pixels have a non-zero reservoir (important pixels) in it.
	AtomicType<unsigned short int>* cell_non_zero_reservoir_counters = nullptr;
	// Cell counters but prefixed scanned so that we can know the offset of each cell
	unsigned int* cell_offsets = nullptr;
	// A global counter used to compute the offsets of each cell
	AtomicType<unsigned int>* cell_global_offset_counter = nullptr;
	// Counter of how many different cells have at least one pixel in them. This is used to know how many cells we need to build CDFs for.
	AtomicType<unsigned int>* cell_total_count_counter = nullptr;
	// For each cell, whether or not it has at least one pixel in it. This is used to increment the cell total count counter without counting multiple times the
	// same cell
	AtomicType<unsigned char>* cell_occupied = nullptr;
	// A list of size cell_total_count_counter that contains the indices of all cells that have at least one pixel in them. This is used to build CDFs for all
	// cells that have at least one pixel in them.
	unsigned int* cell_alive_list = nullptr;
	// Sum of the confidence weights of all pixels of a given cell
	AtomicType<unsigned int>* cell_confidence_sums = nullptr;
	// CDF built on the important pixels of each cell to be able to sample a pixel from a cell according to its confidence weight directly without RISing over
	// everything
	// TODO fp16
	float* cell_cdfs				  = nullptr;
	unsigned short int* cell_cdf_luts = nullptr;
	// Fullscreen buffer that contains, for each cell, the offset in the cell_cdf_luts buffer of the CDF LUT of that cell.
	unsigned int* cell_cdf_lut_offsets = nullptr;

	// Per-cell coefficient of variation of the contributions UCW * target_function.
	//
	// Normalized variation coefficient:
	//		coeff = sqrt(Var(X) / Mean^2)
	//		normalized coeff = coeff / (1 + coeff) to be in [0, 1]
	// Computed inside BuildCDFs.
	float* cell_variance = nullptr;
};

#endif // #ifndef DEVICE_INCLUDES_RESTIR_PT_COMMON_SPMIS_SETTINGS_H
