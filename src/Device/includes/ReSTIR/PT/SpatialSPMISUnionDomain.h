/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_SPMIS_UNION_DOMAIN_H
#define DEVICE_RESTIR_PT_SPATIAL_SPMIS_UNION_DOMAIN_H

#include "Device/includes/BSDFs/BSDFDispatcher.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/PT/UtilsSpatialSPMIS.h"
#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/RenderData.h"

/**
 * SPMIS "union domain" spatial reuse (ReSTIRPTSPMISUnionDomainSettings), the reuse domain and the neighbor selection. The MIS weights are in
 * SpatialGroupwiseMISWeight.h and the spatial reuse itself in ReSTIR_PT_SPMIS_union_domain_spatial_reuse() (SpatialReuseSPMIS.h).
 *
 *	- The reuse domain of a pixel is the union of up to RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS distinct SPMIS cells, each found by the SPMIS cell search
 *	  with the cells already in the domain excluded. It is searched once and cached per pixel (one representative pixel per cell), as the single reuse cell
 *	  of the legacy SPMIS path is.
 *	- The neighbors are sampled in two stages. Stage 1 draws candidates approximately proportionally to the visibility-free shifted integrand of the
 *	  non-zero neighbors of the domain, evaluated in closed form from compact per-pixel records (ReSTIR_PT_SPMIS_BuildNeighborRecords). Stage 2 traces
 *	  the visibility of every candidate at the center pixel and resamples one visible candidate per neighbor.
 */

/**
 * Reuse domain of the union domain reuse: up to RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS distinct SPMIS cells and the sums over all their pixels, zero
 * reservoirs included.
 *
 * The cells are kept in scalars (no runtime-indexed register array, which would spill to scratch)
 */
struct ReSTIRPTSPMISUnionDomain
{
	HIPRT_DEVICE unsigned int get_cell(int index) const
	{
		return index == 0 ? cell_0 : (index == 1 ? cell_1 : (index == 2 ? cell_2 : cell_3));
	}

	/**
	 * Adds the cell to the domain and its pixels to the sums of the domain
	 */
	HIPRT_DEVICE void add_cell(const ReSTIRPTSPMISSettings& spmis_settings, unsigned int cell_index)
	{
		if (cell_count == 0)
			cell_0 = cell_index;
		else if (cell_count == 1)
			cell_1 = cell_index;
		else if (cell_count == 2)
			cell_2 = cell_index;
		else
			cell_3 = cell_index;

		cell_count++;

		confidence_sum += spmis_settings.cell_confidence_sums[cell_index];
		pixel_count += spmis_settings.cell_pixels_counters[cell_index];
		non_zero_count += spmis_settings.cell_non_zero_reservoir_counters[cell_index];
		if (spmis_settings.cell_non_zero_reservoir_counters[cell_index] > 0)
			// SPMIS CDF layout: cdf[0] holds the sum of the importances c W p_hat of the cell
			importance_sum += spmis_settings.cell_cdfs[spmis_settings.cell_offsets[cell_index]];
	}

	unsigned int cell_0 = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	unsigned int cell_1 = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	unsigned int cell_2 = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	unsigned int cell_3 = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;

	int cell_count = 0;

	// Sum of the confidences of all the pixels of the domain
	int confidence_sum = 0;
	// Number of pixels of the domain
	unsigned int pixel_count = 0;
	// Number of pixels of the domain whose reservoir is non-zero, the only pixels that can be selected as neighbors
	unsigned int non_zero_count = 0;
	// Sum over the cells of their CDF importance sums (c W p_hat summed over the non-zero pixels of the domain)
	float importance_sum = 0.0f;
};

/**
 * Searches the union domain of the center pixel: up to RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS distinct cells, each found by the SPMIS cell search
 * (spmis_search_reuse_cell_index(), same radius / iteration / distance scaling settings as spmis_get_reuse_cell_index()) with the cells already in the
 * domain excluded. At most RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELL_SEARCHES searches (a search may not find a new cell).
 *
 * The domain depends on the hash cells, the confidences and the random numbers of the search only, not on the samples of the pixels: the MIS weights
 * computed over the domain stay valid.
 *
 * 'out_representative_pixels' (RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS entries) receives, for every cell of the domain in order, the pixel through which
 * the search found the cell
 */
HIPRT_DEVICE ReSTIRPTSPMISUnionDomain spmis_search_union_domain(const HIPRTRenderData& render_data,
																unsigned int center_pixel_index,
																int2_t center_pixel_coords,
																const ReSTIRPTSurface& center_pixel_surface,
																unsigned int* out_representative_pixels,
																Xorshift32Generator& rng)
{
	const ReSTIRPTSPMISSettings& spmis_settings = render_data.render_settings.restir_pt_settings.spmis_settings;

	ReSTIRPTSPMISUnionDomain domain;

	unsigned int center_cell_index = spmis_settings.all_pixel_hashes[center_pixel_index];
	if (center_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX || center_cell_index >= spmis_settings.pixel_hashes_count)
		// Can happen if hash collision resolution fails
		return domain;

	unsigned int center_cell_weight = spmis_settings.cell_confidence_sums[center_cell_index];

	// Same search settings as the single reuse cell search of spmis_get_reuse_cell_index()
	float radius;
	unsigned int max_search_iterations;
	float distance_scaling;
	spmis_get_cell_search_settings(spmis_settings, center_cell_index, radius, max_search_iterations, distance_scaling);

	// Cells already in the domain, excluded from the next searches
	unsigned int domain_cells[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS];
	for (int search = 0; search < RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELL_SEARCHES && domain.cell_count < RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS; search++)
	{
		unsigned int selected_pixel_index;
		unsigned int selected_cell_confidence_sum;
		unsigned int cell_index = spmis_search_reuse_cell_index(render_data, center_pixel_index, center_pixel_coords, center_pixel_surface, center_cell_index,
																center_cell_weight, radius, max_search_iterations, distance_scaling, selected_pixel_index,
																selected_cell_confidence_sum, rng, domain_cells, domain.cell_count);
		if (cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX || spmis_settings.cell_pixels_counters[cell_index] == 0)
			// This search didn't find a new cell
			continue;

		out_representative_pixels[domain.cell_count] = selected_pixel_index;

		domain_cells[domain.cell_count] = cell_index;
		domain.add_cell(spmis_settings, cell_index);
	}

	return domain;
}

/**
 * spmis_search_union_domain() with the per-pixel cache ReSTIRPTSPMISUnionDomainSettings::domain_cache: the scheme of the reuse cell cache of
 * spmis_get_reuse_cell_index() applied to the whole domain.
 *
 * The cache of a pixel (RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE_STRIDE unsigned ints: the number of cells, then one representative pixel per cell) is emptied
 * (cell count = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX) whenever the reuse cell cache is. While it is empty, or when none of the cached pixels has a
 * defined, non-empty cell this frame, the cells are searched again and their representative pixels cached. Otherwise the domain is the union of the
 * cells of the cached pixels in this frame (the cell indices are rebuilt every frame), duplicates and undefined cells skipped. After the search, the
 * domain is then a function of the pixel and of the cells of this frame, not of the random numbers of the frame
 */
HIPRT_DEVICE ReSTIRPTSPMISUnionDomain spmis_get_union_domain(const HIPRTRenderData& render_data,
															 unsigned int center_pixel_index,
															 int2_t center_pixel_coords,
															 const ReSTIRPTSurface& center_pixel_surface,
															 Xorshift32Generator& rng)
{
	const ReSTIRPTSPMISSettings& spmis_settings = render_data.render_settings.restir_pt_settings.spmis_settings;

	unsigned int pixel_count	   = render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y;
	unsigned int* domain_cache	   = spmis_settings.union_domain.domain_cache + center_pixel_index * RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE_STRIDE;
	unsigned int cached_cell_count = domain_cache[0];

	ReSTIRPTSPMISUnionDomain domain;
	if (cached_cell_count != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX && cached_cell_count <= RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS)
	{
		unsigned int domain_cells[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS];
		for (unsigned int cached_index = 0; cached_index < cached_cell_count; cached_index++)
		{
			unsigned int cached_pixel_index = domain_cache[1 + cached_index];
			if (cached_pixel_index >= pixel_count)
				continue;

			unsigned int cell_index = spmis_settings.all_pixel_hashes[cached_pixel_index];
			if (cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX || cell_index >= spmis_settings.pixel_hashes_count)
				continue;
			if (spmis_settings.cell_pixels_counters[cell_index] == 0)
				continue;
			if (spmis_cell_in_list(cell_index, domain_cells, domain.cell_count))
				// Two cached pixels can fall in the same cell this frame
				continue;

			domain_cells[domain.cell_count] = cell_index;
			domain.add_cell(spmis_settings, cell_index);
		}

		if (domain.cell_count > 0)
			return domain;
	}

	// Empty cache: fresh search, the representative pixels of the cells found are cached
	unsigned int representative_pixels[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS];
	domain = spmis_search_union_domain(render_data, center_pixel_index, center_pixel_coords, center_pixel_surface, representative_pixels, rng);

	domain_cache[0] = static_cast<unsigned int>(domain.cell_count);
	for (int cell = 0; cell < domain.cell_count; cell++)
		domain_cache[1 + cell] = representative_pixels[cell];

	return domain;
}

/**
 * The pixel at 'pixel_list_index' (in [0, domain.pixel_count - 1]) in the pixel list of the domain: all the pixels of the cells of the domain, zero
 * reservoirs included, cell after cell in the order of pixel_indices_sorted
 */
HIPRT_DEVICE unsigned int spmis_union_domain_pixel_at_index(const HIPRTRenderData& render_data,
															const ReSTIRPTSPMISUnionDomain& domain,
															unsigned int pixel_list_index)
{
	const ReSTIRPTSPMISSettings& spmis_settings = render_data.render_settings.restir_pt_settings.spmis_settings;

	for (int cell = 0; cell < domain.cell_count; cell++)
	{
		unsigned int cell_index		  = domain.get_cell(cell);
		unsigned int cell_pixel_count = spmis_settings.cell_pixels_counters[cell_index];
		if (pixel_list_index < cell_pixel_count)
			return spmis_settings.pixel_indices_sorted[spmis_settings.cell_offsets[cell_index] + pixel_list_index];

		pixel_list_index -= cell_pixel_count;
	}

	// Floating point / count safety: last pixel of the last cell
	unsigned int last_cell = domain.get_cell(domain.cell_count - 1);
	return spmis_settings.pixel_indices_sorted[spmis_settings.cell_offsets[last_cell] + spmis_settings.cell_pixels_counters[last_cell] - 1];
}

/**
 * One pixel drawn uniformly over all the pixels of the domain's cells, zero reservoirs included (as the canonical MIS weight estimator of the legacy SPMIS
 * path does over one cell). The probability is 1 / domain.pixel_count
 */
HIPRT_DEVICE unsigned int spmis_union_domain_uniform_pixel(const HIPRTRenderData& render_data, const ReSTIRPTSPMISUnionDomain& domain, Xorshift32Generator& rng)
{
	return spmis_union_domain_pixel_at_index(render_data, domain, rng.random_index(domain.pixel_count));
}

/**
 * The pixel at the position 'uniform' (in [0, 1)) of the pixel list of the domain (see spmis_union_domain_pixel_at_index()). A uniform drawn uniformly in
 * [0, 1) gives a pixel drawn uniformly in the domain, a uniform drawn in a sub-interval gives a pixel of that stratum of the list (stratified draws)
 */
HIPRT_DEVICE unsigned int spmis_union_domain_pixel_at_uniform(const HIPRTRenderData& render_data, const ReSTIRPTSPMISUnionDomain& domain, float uniform)
{
	unsigned int pixel_list_index = hippt::min(domain.pixel_count - 1u, static_cast<unsigned int>(uniform * domain.pixel_count));

	return spmis_union_domain_pixel_at_index(render_data, domain, pixel_list_index);
}

/**
 * Neighbor records (ReSTIRPTSPMISUnionDomainSettings::neighbor_records), written by ReSTIR_PT_SPMIS_BuildNeighborRecords for every pixel with a non-zero
 * spatial reuse input reservoir, at the slot of the pixel in its SPMIS cell (cell_offsets[cell] + index_in_cell, the same slot as in pixel_indices_sorted).
 * RESTIR_PT_SPMIS_UNION_DOMAIN_NEIGHBOR_RECORD_FLOAT_COUNT floats, read as two float4:
 *	[0..2] reconnection vertex position (the direction of the path for an envmap path)
 *	[3..4] octahedral encoding of the geometric normal of the reconnection vertex. [3] = 2 marks an envmap path
 *	[5]	   Q_k: luminance of the radiance leaving the reconnection vertex toward the shading point of the neighbor, times the inverse of the solid angle to
 *		   area conversion of the neighbor (d^2 / cosine at the reconnection vertex). Just the luminance of the radiance for an envmap path. The sign is
 *		   negative for a DI sample (reconnection vertex on a light)
 *	[6]	   c_i W_i p_hat_i(X_i): the importance of the pixel in the CDF of its cell (same value as ReSTIR_PT_SPMIS_BuildCDFs)
 *	[7]	   C_cell p_hat_i(X_i), with C_cell the confidence sum of the cell of the pixel
 */
HIPRT_HOST_DEVICE float2_t spmis_neighbor_record_encode_normal(const float3_t& normal)
{
	float l1_norm	  = hippt::abs(normal.x) + hippt::abs(normal.y) + hippt::abs(normal.z);
	float projected_x = normal.x / l1_norm;
	float projected_y = normal.y / l1_norm;
	if (normal.z < 0.0f)
		return make_float2((1.0f - hippt::abs(projected_y)) * (projected_x >= 0.0f ? 1.0f : -1.0f),
						   (1.0f - hippt::abs(projected_x)) * (projected_y >= 0.0f ? 1.0f : -1.0f));

	return make_float2(projected_x, projected_y);
}

HIPRT_HOST_DEVICE float3_t spmis_neighbor_record_decode_normal(float x, float y)
{
	float3_t normal = make_float3(x, y, 1.0f - hippt::abs(x) - hippt::abs(y));
	if (normal.z < 0.0f)
	{
		float decoded_x = (1.0f - hippt::abs(y)) * (x >= 0.0f ? 1.0f : -1.0f);
		float decoded_y = (1.0f - hippt::abs(x)) * (y >= 0.0f ? 1.0f : -1.0f);
		normal.x		= decoded_x;
		normal.y		= decoded_y;
	}

	return hippt::normalize(normal);
}

/**
 * Selection target t_i of a neighbor record at the center pixel, in closed form:
 *
 *		t_i = c_i W_i p_hat_i(X_i) T_i / (C p_hat_i(X_i) + c_c T_i)
 *
 * with T_i the visibility-free target function of the neighbor's sample shifted to the center pixel times the Jacobian of the shift (center BSDF luminance
 * x cosine at the center x |cosine at the reconnection vertex| / d^2 x Q_k) and C the confidence sum of the domain ('record_second.w' must be scaled
 * from the confidence sum of the record's cell to C by the caller). This is the full visibility-free shifted integrand of the neighbor: the pairwise MIS
 * weight x shifted target function x UCW x Jacobian.
 *
 * This target only drives the selection, the selection probabilities are exact for it: any error of the approximations (Lambertian center BSDF, no
 * visibility) costs variance, not bias
 */
HIPRT_DEVICE float spmis_neighbor_record_target(
	const float4_t& record_first, const float4_t& record_second, const ReSTIRPTSurface& center_surface, float center_bsdf_luminance, float center_confidence)
{
	float3_t rc_vertex = make_float3(record_first.x, record_first.y, record_first.z);
	float signed_qk	   = record_second.y;
	float qk		   = hippt::abs(signed_qk);

	float shifted_target_times_jacobian;
	if (record_first.w == 2.0f)
	{
		// Envmap path, 'rc_vertex' is the direction of the path
		float cosine_center			  = hippt::dot(center_surface.shading_normal, rc_vertex);
		shifted_target_times_jacobian = cosine_center > 0.0f ? center_bsdf_luminance * cosine_center * qk : 0.0f;
	}
	else
	{
		float3_t rc_normal	   = spmis_neighbor_record_decode_normal(record_first.w, record_second.x);
		float3_t to_rc_vertex  = rc_vertex - center_surface.shading_point;
		float distance_2	   = hippt::dot(to_rc_vertex, to_rc_vertex);
		float inverse_distance = 1.0f / hippt::sqrt(distance_2);
		float cosine_center	   = hippt::dot(center_surface.shading_normal, to_rc_vertex) * inverse_distance;
		float cosine_rc_signed = -hippt::dot(rc_normal, to_rc_vertex) * inverse_distance;
		bool backfacing_light  = signed_qk < 0.0f && cosine_rc_signed <= 0.0f;

		shifted_target_times_jacobian = (cosine_center > 0.0f && !backfacing_light)
											? center_bsdf_luminance * cosine_center * hippt::abs(cosine_rc_signed) * qk * (inverse_distance * inverse_distance)
											: 0.0f;
	}

	float denominator = record_second.w + center_confidence * shifted_target_times_jacobian;

	return denominator > 0.0f ? record_second.z * shifted_target_times_jacobian / denominator : 0.0f;
}

/**
 * Inverse of the CDF of a cell (SPMIS layout: cdf[0] holds the sum of the importances, cdf[k > 0] the normalized prefix sums) at a given uniform, with
 * the LUT acceleration of CDFDeviceWithLUT::sample(). The uniform is given (instead of drawn) so that stratified uniforms can be used
 */
HIPRT_DEVICE unsigned int spmis_sample_cell_cdf(const float* cdf, const unsigned short int* cdf_lut, unsigned int size, float random_value)
{
	unsigned int lut_bin_index = hippt::min(ReSTIR_PT_SPMISCDFLUTSize - 1u, static_cast<unsigned int>(random_value * ReSTIR_PT_SPMISCDFLUTSize));
	unsigned int left		   = cdf_lut[lut_bin_index];
	unsigned int right		   = (lut_bin_index < ReSTIR_PT_SPMISCDFLUTSize - 1u ? cdf_lut[lut_bin_index + 1] : size - 1) + 1;

	while (left < right)
	{
		unsigned int middle = (left + right) / 2;
		float cdf_value		= middle == 0 ? 0.0f : cdf[middle];
		if (cdf_value < random_value)
			left = middle + 1;
		else
			right = middle;
	}

	return hippt::min(left > 0 ? left - 1 : 0, size - 1);
}

/**
 * Importance sum (SPMIS CDF layout: cdf[0]) of the cell 'cell' (in [0, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS - 1]) of the domain, 0 for a cell without
 * any non-zero reservoir or beyond the cells of the domain
 */
HIPRT_DEVICE float spmis_union_domain_cell_importance(const ReSTIRPTSPMISSettings& spmis_settings, const ReSTIRPTSPMISUnionDomain& domain, int cell)
{
	if (cell >= domain.cell_count)
		return 0.0f;

	unsigned int cell_index = domain.get_cell(cell);
	if (spmis_settings.cell_non_zero_reservoir_counters[cell_index] == 0)
		return 0.0f;

	return spmis_settings.cell_cdfs[spmis_settings.cell_offsets[cell_index]];
}

/**
 * Stage 1 of the neighbor selection of the union domain reuse: 'candidates_per_neighbor' draws per neighbor slot, each one individually distributed
 * approximately proportionally to the selection target t of spmis_neighbor_record_target() (the draws of a slot are correlated, which stage 2 allows):
 *	- domains with at most ReSTIRPTSPMISUnionDomainSettings::total_neighbor_budget non-zero neighbors: exact streaming over all the records of the
 *	  domain into neighbor_count x candidates_per_neighbor weighted reservoirs that share one acceptance uniform per record, rotated per reservoir (virtual
 *	  slot j = candidate x neighbor_count + slot: the candidates of one slot are spread over the CDF). Exact probabilities P = t_i / sum(t)
 *	- larger domains: RIS. Every slot draws its own ceil(K / neighbor_count) stratified candidates: the cell proportionally to its importance sum, then
 *	  the record in the cell through its CDF (every non-zero record has a non-zero importance c W p_hat). The candidates_per_neighbor reservoirs of the slot
 *	  resample that candidate set with rotated copies of one acceptance uniform. P = t_i / sum(RIS weights of the slot), the inverse of the RIS unbiased
 *	  contribution weight
 *
 * Outputs, slot-major (index = slot x candidates_per_neighbor + candidate): the pixel, its selection "probability" P (1 / unbiased contribution weight) and
 * its selection target t. Slots without any non-zero candidate are compacted away. Returns the number of slots written
 *
 * 'MaxCandidatesPerNeighbor' is the maximum of candidates_per_neighbor (size of the per-slot reservoir arrays)
 */
template <int MaxCandidatesPerNeighbor>
HIPRT_DEVICE int spmis_union_domain_select_candidates(const HIPRTRenderData& render_data,
													  ReSTIRPTSurface& center_surface,
													  const ReSTIRPTSPMISUnionDomain& domain,
													  int center_pixel_reservoir_confidence,
													  int neighbor_count,
													  int candidates_per_neighbor,
													  unsigned int* out_pixel_indices,
													  float* out_selection_probabilities,
													  float* out_selection_targets,
													  Xorshift32Generator& rng)
{
	const ReSTIRPTSPMISSettings& spmis_settings					  = render_data.render_settings.restir_pt_settings.spmis_settings;
	const ReSTIRPTSPMISUnionDomainSettings& union_domain_settings = spmis_settings.union_domain;

	neighbor_count			= hippt::min(neighbor_count, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS);
	candidates_per_neighbor = hippt::clamp(1, MaxCandidatesPerNeighbor, candidates_per_neighbor);
	if (domain.non_zero_count == 0 || domain.importance_sum <= 0.0f || neighbor_count <= 0)
		return 0;

	int total_neighbor_budget = hippt::max(1, union_domain_settings.total_neighbor_budget);
	bool exact				  = domain.non_zero_count <= static_cast<unsigned int>(total_neighbor_budget);

	// Luminance of the center BSDF, evaluated once (exact for a Lambertian BSDF, albedo / pi, the cosine is applied per neighbor)
	float bsdf_pdf;
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
	BSDFContext center_context(center_surface.view_direction, center_surface.shading_normal, center_surface.geometric_normal, center_surface.shading_normal,
							   incident_light_info, center_surface.ray_volume_state, false, center_surface.material, 0.0f,
							   MicrofacetRegularization::RegularizationMode::NO_REGULARIZATION);
	float center_bsdf_luminance = bsdf_dispatcher_eval(render_data, center_context, bsdf_pdf, rng).luminance();
	float center_confidence		= static_cast<float>(center_pixel_reservoir_confidence);
	float domain_confidence_sum = static_cast<float>(domain.confidence_sum);

	const float4_t* neighbor_records	  = reinterpret_cast<const float4_t*>(union_domain_settings.neighbor_records);
	float inverse_candidates_per_neighbor = 1.0f / candidates_per_neighbor;

	if (exact)
	{
		// Virtual slot j = candidate x neighbor_count + slot, its acceptance uniform is rotated by j / (neighbor_count x candidates_per_neighbor)
		int virtual_slot_count			 = neighbor_count * candidates_per_neighbor;
		float inverse_virtual_slot_count = 1.0f / virtual_slot_count;

		unsigned int selected_pixel_indices[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS * MaxCandidatesPerNeighbor];
		float selected_targets[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS * MaxCandidatesPerNeighbor];
		for (int virtual_slot = 0; virtual_slot < virtual_slot_count; virtual_slot++)
		{
			selected_pixel_indices[virtual_slot] = 0;
			selected_targets[virtual_slot]		 = 0.0f;
		}

		float target_sum = 0.0f;
		for (int cell = 0; cell < domain.cell_count; cell++)
		{
			unsigned int cell_index			= domain.get_cell(cell);
			unsigned int cell_start_index	= spmis_settings.cell_offsets[cell_index];
			unsigned int non_zero_cell_size = spmis_settings.cell_non_zero_reservoir_counters[cell_index];
			// The records store the confidence sum of their own cell, the selection target needs the one of the whole domain
			float denominator_scale = domain_confidence_sum / hippt::max(1.0f, static_cast<float>(spmis_settings.cell_confidence_sums[cell_index]));

			for (unsigned int record = 0; record < non_zero_cell_size; record++)
			{
				float4_t record_first  = neighbor_records[(cell_start_index + record) * 2 + 0];
				float4_t record_second = neighbor_records[(cell_start_index + record) * 2 + 1];
				record_second.w *= denominator_scale;

				float target = spmis_neighbor_record_target(record_first, record_second, center_surface, center_bsdf_luminance, center_confidence);
				if (target <= 0.0f)
					continue;

				target_sum += target;
				float acceptance		 = target / target_sum;
				float acceptance_uniform = rng();
				unsigned int pixel_index = spmis_settings.pixel_indices_sorted[cell_start_index + record];

				for (int virtual_slot = 0; virtual_slot < virtual_slot_count; virtual_slot++)
				{
					float rotated_uniform = acceptance_uniform + virtual_slot * inverse_virtual_slot_count;
					rotated_uniform		  = rotated_uniform >= 1.0f ? rotated_uniform - 1.0f : rotated_uniform;
					if (rotated_uniform < acceptance)
					{
						selected_pixel_indices[virtual_slot] = pixel_index;
						selected_targets[virtual_slot]		 = target;
					}
				}
			}
		}

		if (!(target_sum > 0.0f))
			return 0;

		for (int slot = 0; slot < neighbor_count; slot++)
		{
			for (int candidate = 0; candidate < candidates_per_neighbor; candidate++)
			{
				int virtual_slot = candidate * neighbor_count + slot;
				int output_index = slot * candidates_per_neighbor + candidate;

				out_pixel_indices[output_index]			  = selected_pixel_indices[virtual_slot];
				out_selection_probabilities[output_index] = selected_targets[virtual_slot] / target_sum;
				out_selection_targets[output_index]		  = selected_targets[virtual_slot];
			}
		}

		return neighbor_count;
	}

	// Independent candidate sets: every neighbor slot resamples its own ceil(K / N) stratified candidates
	int per_neighbor_candidate_count		   = (total_neighbor_budget + neighbor_count - 1) / neighbor_count;
	float inverse_per_neighbor_candidate_count = 1.0f / per_neighbor_candidate_count;
	float inverse_importance_sum			   = 1.0f / domain.importance_sum;

	// The importance sums of the cells of the domain do not depend on the candidate: loaded once per pixel instead of once per candidate and per
	// cell by the cell selection below
	float cell_importance_0 = spmis_union_domain_cell_importance(spmis_settings, domain, 0);
	float cell_importance_1 = spmis_union_domain_cell_importance(spmis_settings, domain, 1);
	float cell_importance_2 = spmis_union_domain_cell_importance(spmis_settings, domain, 2);
	float cell_importance_3 = spmis_union_domain_cell_importance(spmis_settings, domain, 3);

	int written_slot_count = 0;

	for (int slot = 0; slot < neighbor_count; slot++)
	{
		float slot_weight_sum = 0.0f;

		// Slot of the selected record in pixel_indices_sorted (cell_offsets[cell] + index in the cell): the pixel index is only loaded for the selected
		// candidates, after the resampling. The initial value is never output: the first candidate with a non-zero weight is accepted by every reservoir
		unsigned int slot_selected_record_slots[MaxCandidatesPerNeighbor];
		float slot_selected_targets[MaxCandidatesPerNeighbor];

		UNROLL_LOOP
		for (int candidate = 0; candidate < MaxCandidatesPerNeighbor; candidate++)
		{
			slot_selected_record_slots[candidate] = 0;
			slot_selected_targets[candidate]	  = 0.0f;
		}

		for (int stratum = 0; stratum < per_neighbor_candidate_count; stratum++)
		{
			float stratified_uniform = hippt::min((stratum + rng()) * inverse_per_neighbor_candidate_count, 0.99999994f);

			// Cell proportionally to its importance sum, the uniform is remapped into the interval of the chosen cell (the stratification carries over)
			unsigned int cell_index		= domain.cell_0;
			float cell_importance		= 0.0f;
			float cumulative_before		= 0.0f;
			float cumulative_importance = 0.0f;
			bool cell_found				= false;

			for (int cell = 0; cell < domain.cell_count; cell++)
			{
				unsigned int candidate_cell_index = domain.get_cell(cell);
				float importance = cell == 0 ? cell_importance_0 : (cell == 1 ? cell_importance_1 : (cell == 2 ? cell_importance_2 : cell_importance_3));

				float next_cumulative_importance = cumulative_importance + importance * inverse_importance_sum;

				bool take_cell = !cell_found && importance > 0.0f && (stratified_uniform < next_cumulative_importance || cell == domain.cell_count - 1);
				if (take_cell)
				{
					cell_index		  = candidate_cell_index;
					cell_importance	  = importance;
					cumulative_before = cumulative_importance;
					cell_found		  = true;
				}

				cumulative_importance = next_cumulative_importance;
			}

			if (!cell_found || cell_importance <= 0.0f)
				continue;

			float cell_share		 = cell_importance * inverse_importance_sum;
			float cell_local_uniform = hippt::clamp(0.0f, 0.99999994f, (stratified_uniform - cumulative_before) / cell_share);

			unsigned int cell_start_index	   = spmis_settings.cell_offsets[cell_index];
			unsigned int non_zero_cell_size	   = spmis_settings.cell_non_zero_reservoir_counters[cell_index];
			const float* cell_cdf			   = spmis_settings.cell_cdfs + cell_start_index;
			const unsigned short int* cell_lut = spmis_settings.cell_cdf_luts + spmis_settings.cell_cdf_lut_offsets[cell_index];
			unsigned int record_index		   = spmis_sample_cell_cdf(cell_cdf, cell_lut, non_zero_cell_size, cell_local_uniform);

			float4_t record_first  = neighbor_records[(cell_start_index + record_index) * 2 + 0];
			float4_t record_second = neighbor_records[(cell_start_index + record_index) * 2 + 1];
			// Probability of the candidate: probability of its cell x probability of the record in the CDF of the cell (record_second.z is the CDF
			// importance c W p_hat of the record)
			float candidate_probability = cell_share * record_second.z / cell_importance;
			// The records store the confidence sum of their own cell, the selection target needs the one of the whole domain
			record_second.w *= domain_confidence_sum / hippt::max(1.0f, static_cast<float>(spmis_settings.cell_confidence_sums[cell_index]));

			float target = spmis_neighbor_record_target(record_first, record_second, center_surface, center_bsdf_luminance, center_confidence);
			if (target <= 0.0f)
				continue;

			float weight = target * inverse_per_neighbor_candidate_count / candidate_probability;
			slot_weight_sum += weight;

			// One acceptance uniform, rotated for every reservoir of the slot: each reservoir is a weighted reservoir sampling of the same candidate set
			// with uniform acceptance tests
			float acceptance		 = weight / slot_weight_sum;
			float acceptance_uniform = rng();
			unsigned int record_slot = cell_start_index + record_index;

			UNROLL_LOOP
			for (int candidate = 0; candidate < MaxCandidatesPerNeighbor; candidate++)
			{
				float rotated_uniform = acceptance_uniform + candidate * inverse_candidates_per_neighbor;
				rotated_uniform		  = rotated_uniform >= 1.0f ? rotated_uniform - 1.0f : rotated_uniform;

				bool accept = candidate < candidates_per_neighbor && rotated_uniform < acceptance;

				slot_selected_record_slots[candidate] = accept ? record_slot : slot_selected_record_slots[candidate];
				slot_selected_targets[candidate]	  = accept ? target : slot_selected_targets[candidate];
			}
		}

		if (slot_weight_sum > 0.0f)
		{
			for (int candidate = 0; candidate < candidates_per_neighbor; candidate++)
			{
				int output_index = written_slot_count * candidates_per_neighbor + candidate;

				out_pixel_indices[output_index]			  = spmis_settings.pixel_indices_sorted[slot_selected_record_slots[candidate]];
				out_selection_probabilities[output_index] = slot_selected_targets[candidate] / slot_weight_sum;
				out_selection_targets[output_index]		  = slot_selected_targets[candidate];
			}

			written_slot_count++;
		}
	}

	return written_slot_count;
}

/**
 * Two-stage visibility-aware neighbor selection of the union domain reuse (ReSTIRPTSPMISUnionDomainSettings::visibility_candidate_count = M).
 *
 * Stage 1: spmis_union_domain_select_candidates(), M draws (i_k, W_k = 1 / P_k) per neighbor slot, each with E[W_k g(i_k)] = sum over the domain of g
 * for any function g of the neighbors that is zero where the selection target t is zero.
 *
 * Stage 2: the visibility V_k of the sample of every candidate at the center pixel, with the same tests and shadow ray as the visibility part of
 * ReSTIR_PT_evaluate_target_function<true>() (ReSTIR_PT_sample_visible_at_surface()). RIS with target t x V: w_k = t(i_k) V_k W_k, k* drawn with
 * probability w_k / sum(w), W' = (sum(w) / M) / t(i_k*). The returned "probability" is P' = 1 / W' = M t(i_k*) / sum(w).
 *
 * Unbiasedness: V_k = 0 implies a zero contribution of the candidate (its target function at the center, visibility included, is zero) so
 * E[W' g(i_k*)] = (1 / M) sum over k of E[W_k g(i_k)] = sum over the domain of g. M = 1 is a single draw with its visibility test.
 *
 * Every selected neighbor's sample is visible at the center pixel: the caller evaluates its target function at the center without the visibility ray. A
 * slot whose candidates are all occluded gets the pixel index HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX, the caller skips it (zero contribution).
 * Returns the number of slots written
 */
HIPRT_DEVICE int spmis_union_domain_select_neighbors(const HIPRTRenderData& render_data,
													 ReSTIRPTSurface& center_surface,
													 const ReSTIRPTSPMISUnionDomain& domain,
													 int center_pixel_reservoir_confidence,
													 int neighbor_count,
													 int candidates_per_neighbor,
													 unsigned int* out_pixel_indices,
													 float* out_selection_probabilities,
													 Xorshift32Generator& rng)
{
	neighbor_count			= hippt::min(neighbor_count, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS);
	candidates_per_neighbor = hippt::clamp(1, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_CANDIDATES, candidates_per_neighbor);

	unsigned int candidate_pixel_indices[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS * RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_CANDIDATES];
	float candidate_selection_probabilities[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS * RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_CANDIDATES];
	float candidate_selection_targets[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS * RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_CANDIDATES];

	int slot_count = spmis_union_domain_select_candidates<RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_CANDIDATES>(
		render_data, center_surface, domain, center_pixel_reservoir_confidence, neighbor_count, candidates_per_neighbor, candidate_pixel_indices,
		candidate_selection_probabilities, candidate_selection_targets, rng);

	const ReSTIRPTReservoir* input_reservoirs = render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs;
	for (int slot = 0; slot < slot_count; slot++)
	{
		float second_stage_weight_sum	= 0.0f;
		unsigned int chosen_pixel_index = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
		float chosen_selection_target	= 0.0f;

		for (int candidate = 0; candidate < candidates_per_neighbor; candidate++)
		{
			int candidate_index				   = slot * candidates_per_neighbor + candidate;
			unsigned int candidate_pixel_index = candidate_pixel_indices[candidate_index];
			float selection_target			   = candidate_selection_targets[candidate_index];
			float selection_probability		   = candidate_selection_probabilities[candidate_index];
			if (selection_target <= 0.0f || selection_probability <= 0.0f)
				continue;

			const ReSTIRPTReservoir& candidate_reservoir = input_reservoirs[candidate_pixel_index];
			bool candidate_visible						 = false;
			if (candidate_reservoir.UCW > 0.0f)
				candidate_visible = ReSTIR_PT_sample_visible_at_surface(render_data, candidate_reservoir.sample, center_surface, rng);

			if (!candidate_visible)
				// V_k = 0
				continue;

			// w_k = t x V_k x W_k with W_k = 1 / P_k
			float second_stage_weight = selection_target / selection_probability;
			second_stage_weight_sum += second_stage_weight;
			if (rng() < second_stage_weight / second_stage_weight_sum)
			{
				chosen_pixel_index		= candidate_pixel_index;
				chosen_selection_target = selection_target;
			}
		}

		if (second_stage_weight_sum > 0.0f)
		{
			out_pixel_indices[slot]			  = chosen_pixel_index;
			out_selection_probabilities[slot] = candidates_per_neighbor * chosen_selection_target / second_stage_weight_sum;
		}
		else
		{
			out_pixel_indices[slot]			  = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
			out_selection_probabilities[slot] = 1.0f;
		}
	}

	return slot_count;
}

/**
 * Visibility pool variant of the neighbor selection of the union domain reuse (ReSTIRPTSPMISUnionDomainSettings::visibility_pool_size = L > 0): every
 * pixel traces exactly L shadow rays, whatever the number of neighbors N.
 *
 * Stage 1: ONE pool for the pixel, spmis_union_domain_select_candidates() with a single slot: the K stratified candidates of the whole domain resampled
 * by L reservoirs with rotated copies of one acceptance uniform. Every pick l is marginally drawn proportionally to the RIS weights w of the pool,
 * P_l = t_l / W with W the sum of the RIS weights, so that for any function g of the neighbors that is zero where t is zero:
 *
 *		E[(1 / L) x sum over the L picks of g(i_l) / P_l] = E[sum over the pool of w_k g(c_k) / t_k] = sum over the domain of g
 *
 * Stage 2: the visibility V_l of every pick at the center pixel (one ray each, same test as ReSTIR_PT_sample_visible_at_surface()). g contains the
 * visibility (an occluded sample has a zero contribution at the center) so only the n_vis visible picks contribute. min(N, n_vis) of them are chosen
 * uniformly without replacement, each visible pick with the inclusion probability pi = min(1, N / n_vis), which only depends on the visibility of the
 * picks, not on their values (Horvitz-Thompson):
 *
 *		E[sum over the chosen picks of g(i_l) / (pi L P_l)] = E[(1 / L) x sum over the visible picks of g(i_l) / P_l]
 *
 * The spatial reuse divides every neighbor contribution by N x P (see ReSTIRPTSpatialGroupwiseMISWeight::get_resampling_MIS_weight_non_canonical()),
 * so the returned "probability" of a chosen pick is P = pi L P_l / N. The slots beyond min(N, n_vis) get HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX
 * (skipped by the caller). Duplicate picks are distinct terms of the estimator: each one is traced and can be chosen.
 *
 * Returns the number of slots written: N, or 0 when the pool is empty
 */
HIPRT_DEVICE int spmis_union_domain_select_neighbors_from_pool(const HIPRTRenderData& render_data,
															   ReSTIRPTSurface& center_surface,
															   const ReSTIRPTSPMISUnionDomain& domain,
															   int center_pixel_reservoir_confidence,
															   int neighbor_count,
															   int pool_size,
															   unsigned int* out_pixel_indices,
															   float* out_selection_probabilities,
															   Xorshift32Generator& rng)
{
	neighbor_count = hippt::min(neighbor_count, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS);
	pool_size	   = hippt::clamp(1, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE, pool_size);

	unsigned int pool_pixel_indices[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE];
	float pool_selection_probabilities[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE];
	float pool_selection_targets[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE];

	// A single slot: one pool of L picks
	int pool_slot_count = spmis_union_domain_select_candidates<RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE>(
		render_data, center_surface, domain, center_pixel_reservoir_confidence, 1, pool_size, pool_pixel_indices, pool_selection_probabilities,
		pool_selection_targets, rng);
	if (pool_slot_count == 0)
		return 0;

	for (int slot = 0; slot < neighbor_count; slot++)
	{
		out_pixel_indices[slot]			  = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
		out_selection_probabilities[slot] = 1.0f;
	}

	// Stage 2: exactly one visibility ray per pick
	const ReSTIRPTReservoir* input_reservoirs = render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs;
	int visible_picks[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE];
	int visible_pick_count = 0;
	for (int pick = 0; pick < pool_size; pick++)
	{
		if (pool_selection_targets[pick] <= 0.0f || pool_selection_probabilities[pick] <= 0.0f)
			continue;

		const ReSTIRPTReservoir& pick_reservoir = input_reservoirs[pool_pixel_indices[pick]];
		if (pick_reservoir.UCW <= 0.0f)
			continue;

		if (ReSTIR_PT_sample_visible_at_surface(render_data, pick_reservoir.sample, center_surface, rng))
			visible_picks[visible_pick_count++] = pick;
	}

	// min(N, n_vis) visible picks chosen uniformly without replacement (partial Fisher-Yates shuffle)
	int chosen_count			= hippt::min(neighbor_count, visible_pick_count);
	float inclusion_probability = visible_pick_count > neighbor_count ? neighbor_count / static_cast<float>(visible_pick_count) : 1.0f;
	for (int chosen = 0; chosen < chosen_count; chosen++)
	{
		int swapped_index			 = chosen + static_cast<int>(rng.random_index(visible_pick_count - chosen));
		int chosen_pick				 = visible_picks[swapped_index];
		visible_picks[swapped_index] = visible_picks[chosen];
		visible_picks[chosen]		 = chosen_pick;

		out_pixel_indices[chosen]			= pool_pixel_indices[chosen_pick];
		out_selection_probabilities[chosen] = inclusion_probability * pool_size * pool_selection_probabilities[chosen_pick] / neighbor_count;
	}

	return neighbor_count;
}

#endif // #ifndef DEVICE_RESTIR_PT_SPATIAL_SPMIS_UNION_DOMAIN_H
