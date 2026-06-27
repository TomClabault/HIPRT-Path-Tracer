/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_COMMON_SPMIS_SETTINGS_H
#define DEVICE_INCLUDES_RESTIR_COMMON_SPMIS_SETTINGS_H

#include "HostDeviceCommon/AtomicType.h"

struct ReSTIRCommonSPMISCompatibilityGuidedCellSelectionSettings
{
	// If true, uses the heuristic from [Compatibility-Guided Neighbor Selection for ReSTIR, Junkins, 2026] to weight the selection of the neighboring cell.
	// Disabled by default because not that good, a bit worse than hard rejection actually
	bool do_compatibility_guided_selection = false;

	// How much the distance to the center pixel is scaled when computing the weight of a neighboring cell. The lower this factor, the more closer cells are
	// preferred. 0.0f turns off distance scaling.
	float solid_angle_omega = 0.05f;
};

struct ReSTIRCommonSPMISSettings
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
	float distance_scaling = 8.0f;

	ReSTIRCommonSPMISCompatibilityGuidedCellSelectionSettings compatibility_guided_cell_selection;

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

	// Size of the fullscreen buffers
	unsigned int pixel_hashes_count = 0;

	// Fullscreen buffer that contains the hash cell index of a given pixel
	unsigned int* all_pixel_hashes						 = nullptr;
	AtomicType<unsigned int>* all_pixel_hashes_checksums = nullptr;

	// For each pixel, the index in its hash cell
	unsigned int* all_pixels_index_in_cell = nullptr;
	// For each pixel, the index of the pixel whose reuse cell to reuse from. This is precomputed to avoid having to do an expensive reuse cell search each
	// frame
	unsigned int* all_pixels_reuse_cell_pixel_index = nullptr;
	// A fullscreen buffer which contains, for each cell, the list of pixel indices that belongs to that cell. Pixel indices in each cell are sorted with
	// important pixels (non-zero contribution reservoirs) first and non-important pixels after that. This buffer should be indexed as [cell_ffset +
	// index_in_cell] with cell_offset coming from the cell_offsets buffer and index_in_cell in [0, cell_pixels_counts[cell_index]], with the first
	// cell_non_zero_reservoir_counters[cell_index] pixels indices of the cell being the important pixels and the remaining ones being the non-important pixels.
	unsigned int* pixel_indices_sorted = nullptr;

	// How many **pixels** are in the cells, containing non-zero reservoirs or not
	// TODO unsigned char is enough for 8 * 8 cells
	AtomicType<unsigned int>* cell_pixels_counters = nullptr;
	// For each cell, how many pixels have a non-zero reservoir (important pixels) in it.
	AtomicType<unsigned int>* cell_non_zero_reservoir_counters = nullptr;
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
};

#endif
