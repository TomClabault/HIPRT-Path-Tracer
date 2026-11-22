/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_RESTIR_PT_SPMIS_UNION_DOMAIN_SETTINGS_H
#define HOST_DEVICE_RESTIR_PT_SPMIS_UNION_DOMAIN_SETTINGS_H

#include "HostDeviceCommon/KernelOptions/ReSTIRPTOptions.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRPTSPMISUnionDomainShadowSplitBuffers.h"

/**
 * Settings of the SPMIS "union domain" spatial reuse, see SpatialSPMISUnionDomain.h:
 *	- the reuse domain of a pixel is the union of up to RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS distinct SPMIS cells found by the SPMIS cell search
 *	- the neighbors are sampled with a two-stage selection: one pool of candidates drawn approximately proportionally to the visibility-free shifted
 *	  integrand (closed form on compact neighbor records), then the neighbors chosen among the visible candidates of the pool
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
	// Number of pixels per group of the stochastic group-wise MIS weights. 1 is pairwise MIS. Clamped to [1, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_MIS_GROUP_SIZE]
	int mis_group_size = 2;
	// How many groups of mis_group_size pixels (one shadow ray each) estimate the MIS weight of the canonical sample. canonical_group_count x mis_group_size
	// is at most RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES (probes beyond that are not traced)
	int canonical_group_count = 2;

	// Domains with at most that many non-zero neighbors are sampled exactly (streaming over all their neighbors). Larger domains draw that many
	// stratified candidates in total, split over the neighbors, through the alias tables of the cells.
	//
	// A candidate costs no ray: 32 candidates with a visibility pool of 5 and 2 canonical groups measured faster and at least as good as 16 candidates
	// with a pool of 6 and 3 canonical groups
	int total_neighbor_budget = 32;
	// The neighbors are taken among the visible candidates of ONE pool of visibility_pool_size candidates for the pixel (one shadow ray each, whatever
	// neighbor_count), see spmis_union_domain_select_neighbors(). Clamped to [1, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE]
	int visibility_pool_size = 5;
	// Whether the spatial reuse runs as the shadow split kernels of SpatialSPMISUnionDomainShadowSplit.h (the shadow rays traced by lean kernels)
	// instead of the single union domain kernel. Same estimator. GPU only, and only used when
	// ReSTIRPTSettings::use_spmis_union_domain_shadow_split_kernels() is true: the single kernel runs otherwise
	bool use_shadow_split_kernels = true;

	// For every pixel with a non-zero spatial reuse input reservoir, RESTIR_PT_SPMIS_UNION_DOMAIN_NEIGHBOR_RECORD_FLOAT_COUNT floats at the slot of the
	// pixel in its cell (same indexing as pixel_indices_sorted), built every frame by ReSTIR_PT_SPMIS_BuildNeighborRecords
	float* neighbor_records = nullptr;
	// Alias table of every cell, RESTIR_PT_SPMIS_UNION_DOMAIN_ALIAS_TABLE_FLOAT_COUNT floats per non-zero pixel of the cell (same slots as
	// neighbor_records): same distribution as the CDF of the cell, sampled by stage 1 of the neighbor selection. Built every frame by
	// ReSTIR_PT_SPMIS_BuildAliasTables
	float* cell_alias_tables = nullptr;
	// Per-pixel cache of the union domain (RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE_STRIDE unsigned ints per pixel) so that the cell search only runs when the cache
	// is empty. Emptied whenever the reuse cell cache all_pixels_reuse_cell_pixel_index is
	unsigned int* domain_cache = nullptr;
	// State of the shadow split kernels, set by the render pass while they run
	ReSTIRPTSPMISUnionDomainShadowSplitBuffers shadow_split_buffers;
};

#endif // #ifndef HOST_DEVICE_RESTIR_PT_SPMIS_UNION_DOMAIN_SETTINGS_H
