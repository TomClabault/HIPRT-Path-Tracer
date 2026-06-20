/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_COMMON_SPMIS_SETTINGS_H
#define DEVICE_INCLUDES_RESTIR_COMMON_SPMIS_SETTINGS_H

#include "HostDeviceCommon/AtomicType.h"

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

	// How many pixels to stream from a cell to produce one non-canonical neighbor
	int ris_neighbor_count = 8;

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
	// Sum of the confidence weights of all pixels of a given cell
	AtomicType<unsigned int>* cell_confidence_sums = nullptr;
};

#endif
