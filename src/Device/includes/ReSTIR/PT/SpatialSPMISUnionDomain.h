/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_SPMIS_UNION_DOMAIN_H
#define DEVICE_RESTIR_PT_SPATIAL_SPMIS_UNION_DOMAIN_H

#include "Device/includes/BSDFs/BSDFDispatcher.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/ReSTIR/PT/SpatialTargetFunctionBSDFApproximation.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/PT/UtilsSpatialSPMIS.h"
#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/RenderData.h"

/**
 * SPMIS "union domain" spatial reuse (ReSTIRPTSPMISUnionDomainSettings), the reuse domain and the neighbor selection. The MIS weights are in
 * SpatialGroupwiseMISWeight.h and the kernels of the spatial reuse itself are described in SpatialSPMISUnionDomainShadowSplit.h.
 *
 *	- The reuse domain of a pixel is the union of up to RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS distinct SPMIS cells, each found by the SPMIS cell search
 *	  with the cells already in the domain excluded. It is searched once and cached per pixel (one representative pixel per cell), as the single reuse cell
 *	  of the legacy SPMIS path is.
 *
 *	- The neighbors are sampled in two stages. Stage 1 draws one pool of candidates approximately proportionally to the visibility-free shifted
 *	  integrand of the non-zero neighbors of the domain, evaluated in closed form from compact per-pixel records (ReSTIR_PT_SPMIS_BuildNeighborRecords).
 *	  Stage 2 traces the visibility of every candidate of the pool at the center pixel and chooses the neighbors among the visible ones.
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
	 * Offset of the cell 'index' of the domain in pixel_indices_sorted (cell_offsets of the cell)
	 */
	HIPRT_DEVICE unsigned int get_cell_offset(int index) const
	{
		return index == 0 ? cell_offset_0 : (index == 1 ? cell_offset_1 : (index == 2 ? cell_offset_2 : cell_offset_3));
	}

	/**
	 * Number of pixels of the cell 'index' of the domain, zero reservoirs included (cell_pixels_counters of the cell)
	 */
	HIPRT_DEVICE unsigned int get_cell_pixel_count(int index) const
	{
		return index == 0 ? cell_pixel_count_0 : (index == 1 ? cell_pixel_count_1 : (index == 2 ? cell_pixel_count_2 : cell_pixel_count_3));
	}

	/**
	 * Adds the cell to the domain and its pixels to the sums of the domain
	 */
	HIPRT_DEVICE void add_cell(const ReSTIRPTSPMISSettings& spmis_settings, unsigned int cell_index)
	{
		unsigned int cell_offset	  = spmis_settings.cell_offsets[cell_index];
		unsigned int cell_pixel_count = spmis_settings.cell_pixels_counters[cell_index];

		if (cell_count == 0)
		{
			cell_0			   = cell_index;
			cell_offset_0	   = cell_offset;
			cell_pixel_count_0 = cell_pixel_count;
		}
		else if (cell_count == 1)
		{
			cell_1			   = cell_index;
			cell_offset_1	   = cell_offset;
			cell_pixel_count_1 = cell_pixel_count;
		}
		else if (cell_count == 2)
		{
			cell_2			   = cell_index;
			cell_offset_2	   = cell_offset;
			cell_pixel_count_2 = cell_pixel_count;
		}
		else
		{
			cell_3			   = cell_index;
			cell_offset_3	   = cell_offset;
			cell_pixel_count_3 = cell_pixel_count;
		}

		cell_count++;

		confidence_sum += spmis_settings.cell_confidence_sums[cell_index];
		pixel_count += cell_pixel_count;
		non_zero_count += spmis_settings.cell_non_zero_reservoir_counters[cell_index];

		if (spmis_settings.cell_non_zero_reservoir_counters[cell_index] > 0)
			// SPMIS CDF layout: cdf[0] holds the sum of the importances c W p_hat of the cell
			importance_sum += spmis_settings.cell_cdfs[cell_offset];
	}

	unsigned int cell_0 = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	unsigned int cell_1 = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	unsigned int cell_2 = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	unsigned int cell_3 = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;

	// Offsets and pixel counts of the cells, read once when the cell is added: the uniform pixel draws over the domain and the neighbor selection
	// would otherwise load them again for every draw and every candidate, as dependent loads before the pixel or the record
	unsigned int cell_offset_0 = 0;
	unsigned int cell_offset_1 = 0;
	unsigned int cell_offset_2 = 0;
	unsigned int cell_offset_3 = 0;

	unsigned int cell_pixel_count_0 = 0;
	unsigned int cell_pixel_count_1 = 0;
	unsigned int cell_pixel_count_2 = 0;
	unsigned int cell_pixel_count_3 = 0;

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
 * (cell count = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX) whenever the reuse cell cache is.
 *
 * While it is empty, or when none of the cached pixels has a defined, non-empty cell this frame, the cells are searched again and their representative
 * pixels cached. Otherwise the domain is the union of the cells of the cached pixels in this frame (the cell indices are rebuilt every frame), duplicates
 * and undefined cells skipped.
 *
 * After the search, the domain is then a function of the pixel and of the cells of this frame, not of the random numbers of the frame
 *
 * Forced inline for the prepare kernel of the shadow split union domain spatial reuse, for the reason given above
 * spmis_union_domain_shadow_split_canonical_probe_value() in SpatialReuseSPMISUnionDomainShadowSplitPrepare.h
 */
HIPRT_DEVICE HIPRT_FORCEINLINE ReSTIRPTSPMISUnionDomain spmis_get_union_domain(const HIPRTRenderData& render_data,
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

	// Cell of the pixel and index in that cell. No early exit: the pixel counts of all the cells are read up front, so their reads do not wait on
	// one another, and only the final read of the pixel depends on them
	int last_cell	  = domain.cell_count - 1;
	int selected_cell = last_cell;

	unsigned int index_in_cell		 = domain.get_cell_pixel_count(last_cell) - 1;
	unsigned int pixels_before_cells = 0;

	bool cell_found = false;

	UNROLL_LOOP
	for (int cell = 0; cell < RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_CELLS; cell++)
	{
		unsigned int cell_pixel_count = domain.get_cell_pixel_count(cell);

		bool in_cell =
			!cell_found && cell < domain.cell_count && pixel_list_index >= pixels_before_cells && pixel_list_index < pixels_before_cells + cell_pixel_count;
		if (in_cell)
		{
			selected_cell = cell;
			index_in_cell = pixel_list_index - pixels_before_cells;
			cell_found	  = true;
		}

		pixels_before_cells += cell_pixel_count;
	}

	// If no cell was found (floating point / count safety): last pixel of the last cell
	return spmis_settings.pixel_indices_sorted[domain.get_cell_offset(selected_cell) + index_in_cell];
}

/**
 * One pixel drawn uniformly over all the pixels of the domain's cells, zero reservoirs included (as the canonical MIS weight estimator of the legacy SPMIS
 * path does over one cell). The probability is 1 / domain.pixel_count
 *
 * Forced inline for the prepare kernel of the shadow split union domain spatial reuse, for the reason given above
 * spmis_union_domain_shadow_split_canonical_probe_value() in SpatialReuseSPMISUnionDomainShadowSplitPrepare.h
 */
HIPRT_DEVICE HIPRT_FORCEINLINE unsigned int spmis_union_domain_uniform_pixel(const HIPRTRenderData& render_data,
																			 const ReSTIRPTSPMISUnionDomain& domain,
																			 Xorshift32Generator& rng)
{
	return spmis_union_domain_pixel_at_index(render_data, domain, rng.random_index(domain.pixel_count));
}

/**
 * Neighbor records (ReSTIRPTSPMISUnionDomainSettings::neighbor_records), written by ReSTIR_PT_SPMIS_BuildNeighborRecords for every pixel with a non-zero
 * spatial reuse input reservoir, at the slot of the pixel in its SPMIS cell (cell_offsets[cell] + index_in_cell, the same slot as in pixel_indices_sorted).
 *
 * RESTIR_PT_SPMIS_UNION_DOMAIN_NEIGHBOR_RECORD_FLOAT_COUNT floats, read as two float4:
 *	[0..2] reconnection vertex position (the direction of the path for an envmap path)
 *	[3]	   bits of the Octahedral24BitNormalPadded32b geometric normal of the reconnection vertex (the reservoir's packed normal, copied as is). Its padding
 *		   byte is 1 for an envmap path (no normal), 0 otherwise, see spmis_neighbor_record_read_normal()
 *	[4]	   unused
 *	[5]	   Q_k: luminance of the radiance leaving the reconnection vertex toward the shading point of the neighbor, times the inverse of the solid angle to
 *		   area conversion of the neighbor (d^2 / cosine at the reconnection vertex). Just the luminance of the radiance for an envmap path. The sign is
 *		   negative for a DI sample (reconnection vertex on a light)
 *	[6]	   c_i W_i p_hat_i(X_i): the importance of the pixel in the CDF of its cell (same value as ReSTIR_PT_SPMIS_BuildCDFs)
 *	[7]	   C_cell p_hat_i(X_i), with C_cell the confidence sum of the cell of the pixel
 */
HIPRT_DEVICE float spmis_neighbor_record_write_normal(Octahedral24BitNormalPadded32b packed_normal, bool envmap_path)
{
	packed_normal.set_padding(envmap_path ? 1 : 0);

	float normal_bits;
	memcpy(&normal_bits, &packed_normal, sizeof(float));

	return normal_bits;
}

HIPRT_DEVICE Octahedral24BitNormalPadded32b spmis_neighbor_record_read_normal(float normal_bits)
{
	Octahedral24BitNormalPadded32b packed_normal;
	memcpy(&packed_normal, &normal_bits, sizeof(float));

	return packed_normal;
}

/**
 * BSDF of the center pixel in the selection target of the neighbors (spmis_neighbor_record_target()), set up once per pixel by
 * spmis_union_domain_center_bsdf_setup().
 *
 * With the principled BSDF and a center material in an analytic class of SpatialTargetFunctionBSDFApproximation.h, the BSDF term toward a reconnection
 * vertex is the analytic lobe of the center material at the view direction of the pixel (specular GGX lobe + diffuse): the selection then favors the
 * neighbors whose reconnection vertex lies in the glossy lobe of the center. Otherwise it is the Lambertian proxy: the center BSDF evaluated once with
 * the incident direction along the shading normal, which ignores the direction of the reconnection vertex.
 *
 * The analytic lobe is non-zero for every incident direction above the shading normal, except for a perfectly smooth metal where it is zero everywhere.
 * The Lambertian proxy of such a metal is zero too (a perfectly smooth lobe evaluated without incident light information is zero off its delta peak): the
 * analytic lobe never selects fewer neighbors than the Lambertian proxy would
 */
struct ReSTIRPTSPMISUnionDomainCenterBSDF
{
	ReSTIRPTTargetFunctionBSDFApproximationLobe lobe;

	// Luminance of the center BSDF evaluated with the incident direction along the shading normal, only used without the analytic lobe
	float lambertian_proxy_luminance = 0.0f;
	// Cosine between the view direction and the shading normal of the center pixel, only used with the analytic lobe
	float view_cosine  = 0.0f;
	bool analytic_lobe = false;
};

HIPRT_DEVICE ReSTIRPTSPMISUnionDomainCenterBSDF spmis_union_domain_center_bsdf_setup(const HIPRTRenderData& render_data,
																					 ReSTIRPTSurface& center_surface,
																					 Xorshift32Generator& rng)
{
	ReSTIRPTSPMISUnionDomainCenterBSDF center_bsdf;

#if BSDFKernelModel == BSDF_PRINCIPLED
	center_bsdf.view_cosine = hippt::dot(center_surface.view_direction, center_surface.shading_normal);

	bool smooth_lobe;
	int material_class = ReSTIR_PT_target_function_bsdf_approximation_material_class(center_surface.material, smooth_lobe);

	// From inside an object, the principled BSDF only evaluates the glass lobe, which has no analytic class
	center_bsdf.analytic_lobe = material_class != RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE &&
								center_bsdf.view_cosine > RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_THRESHOLD &&
								!center_surface.ray_volume_state.inside_material;

	if (center_bsdf.analytic_lobe)
	{
		center_bsdf.lobe = ReSTIR_PT_target_function_bsdf_approximation_lobe_from_material(render_data, center_surface.material, material_class, smooth_lobe,
																						   true, center_bsdf.view_cosine);

		return center_bsdf;
	}
#endif // #if BSDFKernelModel == BSDF_PRINCIPLED

	// Exact for a Lambertian BSDF, albedo / pi, the cosine is applied per neighbor
	float bsdf_pdf;
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
	BSDFContext center_context(center_surface.view_direction, center_surface.shading_normal, center_surface.geometric_normal, center_surface.shading_normal,
							   incident_light_info, center_surface.ray_volume_state, false, center_surface.material, 0.0f,
							   MicrofacetRegularization::RegularizationMode::NO_REGULARIZATION);
	center_bsdf.lambertian_proxy_luminance = bsdf_dispatcher_eval(render_data, center_context, bsdf_pdf, rng).luminance();

	return center_bsdf;
}

/**
 * BSDF x cosine of the center pixel toward 'incident_direction' (cosine 'cosine_center' > 0 with the shading normal) in the selection target
 */
HIPRT_DEVICE float spmis_union_domain_center_bsdf_value(const HIPRTRenderData& render_data,
														const ReSTIRPTSPMISUnionDomainCenterBSDF& center_bsdf,
														const ReSTIRPTSurface& center_surface,
														const float3_t& incident_direction,
														float cosine_center)
{
	if (!center_bsdf.analytic_lobe)
		return center_bsdf.lambertian_proxy_luminance * cosine_center;

	return ReSTIR_PT_target_function_bsdf_approximation_lobe_value(render_data, center_bsdf.lobe, center_surface.shading_normal, center_surface.view_direction,
																   incident_direction, center_bsdf.view_cosine, cosine_center, false);
}

/**
 * Selection target t_i of a neighbor record at the center pixel, in closed form:
 *
 *		t_i = c_i W_i p_hat_i(X_i) T_i / (C p_hat_i(X_i) + c_c T_i)
 *
 * with T_i the visibility-free target function of the neighbor's sample shifted to the center pixel times the Jacobian of the shift (center BSDF term
 * of spmis_union_domain_center_bsdf_value(), cosine at the center included, x |cosine at the reconnection vertex| / d^2 x Q_k) and C the confidence
 * sum of the domain ('record_second.w' must be scaled from the confidence sum of the record's cell to C by the caller). This is the full visibility-free
 * shifted integrand of the neighbor: the pairwise MIS weight x shifted target function x UCW x Jacobian.
 *
 * This target only drives the selection, the selection probabilities are exact for it: any error of the approximations (analytic or Lambertian center
 * BSDF, no BSDF at the reconnection vertex, no visibility) costs variance, not bias, as long as the target is non-zero wherever the shifted contribution is
 */
HIPRT_DEVICE float spmis_neighbor_record_target(const HIPRTRenderData& render_data,
												const float4_t& record_first,
												const float4_t& record_second,
												const ReSTIRPTSurface& center_surface,
												const ReSTIRPTSPMISUnionDomainCenterBSDF& center_bsdf,
												float center_confidence)
{
	float3_t rc_vertex = make_float3(record_first.x, record_first.y, record_first.z);
	float signed_qk	   = record_second.y;
	float qk		   = hippt::abs(signed_qk);

	Octahedral24BitNormalPadded32b packed_rc_normal = spmis_neighbor_record_read_normal(record_first.w);

	float shifted_target_times_jacobian;

	if (packed_rc_normal.get_padding() == 1)
	{
		// Envmap path, 'rc_vertex' is the direction of the path
		float cosine_center = hippt::dot(center_surface.shading_normal, rc_vertex);
		shifted_target_times_jacobian =
			cosine_center > 0.0f ? spmis_union_domain_center_bsdf_value(render_data, center_bsdf, center_surface, rc_vertex, cosine_center) * qk : 0.0f;
	}
	else
	{
		float3_t rc_normal	   = packed_rc_normal.unpack();
		float3_t to_rc_vertex  = rc_vertex - center_surface.shading_point;
		float distance_2	   = hippt::dot(to_rc_vertex, to_rc_vertex);
		float inverse_distance = 1.0f / hippt::sqrt(distance_2);
		float cosine_center	   = hippt::dot(center_surface.shading_normal, to_rc_vertex) * inverse_distance;
		float cosine_rc_signed = -hippt::dot(rc_normal, to_rc_vertex) * inverse_distance;
		bool backfacing_light  = signed_qk < 0.0f && cosine_rc_signed <= 0.0f;

		shifted_target_times_jacobian = 0.0f;

		if (cosine_center > 0.0f && !backfacing_light)
		{
			float3_t direction_to_rc_vertex = to_rc_vertex * inverse_distance;
			float center_bsdf_value = spmis_union_domain_center_bsdf_value(render_data, center_bsdf, center_surface, direction_to_rc_vertex, cosine_center);

			shifted_target_times_jacobian = center_bsdf_value * hippt::abs(cosine_rc_signed) * qk * (inverse_distance * inverse_distance);
		}
	}

	float denominator = record_second.w + center_confidence * shifted_target_times_jacobian;

	return denominator > 0.0f ? record_second.z * shifted_target_times_jacobian / denominator : 0.0f;
}

/**
 * Samples the alias table of a cell (ReSTIRPTSPMISUnionDomainSettings::cell_alias_tables, built by ReSTIR_PT_SPMIS_BuildAliasTables) at a given uniform
 *
 * The integer part of 'random_value' x 'size' picks the bin, its fractional part chooses between the bin and its alias, so that one uniform is enough
 * (and stratified uniforms stay stratified over the bins). Returns the index of the record in the cell
 */
HIPRT_DEVICE unsigned int spmis_sample_cell_alias_table(const float2_t* cell_alias_table, unsigned int size, float random_value)
{
	float scaled_random_value = random_value * size;
	unsigned int bin_index	  = hippt::min(static_cast<unsigned int>(scaled_random_value), size - 1);
	float bin_fraction		  = scaled_random_value - bin_index;

	// x: probability of keeping the bin, y: index of its alias in the cell (exact as a float)
	float2_t alias_table_entry = cell_alias_table[bin_index];

	return bin_fraction < alias_table_entry.x ? bin_index : static_cast<unsigned int>(alias_table_entry.y);
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

	return spmis_settings.cell_cdfs[domain.get_cell_offset(cell)];
}

/**
 * Scale from the confidence sum of the cell 'cell' of the domain to the confidence sum of the whole domain, 0 beyond the cells of the domain
 */
HIPRT_DEVICE float spmis_union_domain_cell_confidence_scale(const ReSTIRPTSPMISSettings& spmis_settings,
															const ReSTIRPTSPMISUnionDomain& domain,
															int cell,
															float domain_confidence_sum)
{
	if (cell >= domain.cell_count)
		return 0.0f;

	return domain_confidence_sum / hippt::max(1.0f, static_cast<float>(spmis_settings.cell_confidence_sums[domain.get_cell(cell)]));
}

/**
 * Stage 1 of the neighbor selection of the union domain reuse: ONE pool of 'pool_size' picks for the pixel, each one individually distributed
 * approximately proportionally to the selection target t of spmis_neighbor_record_target() (the picks are correlated, which stage 2 allows):
 *	- domains with at most ReSTIRPTSPMISUnionDomainSettings::total_neighbor_budget non-zero neighbors: exact streaming over all the records of the
 *	  domain into pool_size weighted reservoirs that share one acceptance uniform per record, rotated per reservoir (the picks are spread over the CDF).
 *	  Exact probabilities P = t_i / sum(t)
 *
 *	- larger domains: RIS. The pool draws K = total_neighbor_budget stratified candidates: the cell proportionally to its importance sum, then the record
 *	  in the cell with the alias table of the cell, same distribution as its CDF (every non-zero record has a non-zero importance c W p_hat). The
 *	  pool_size reservoirs resample that candidate set with rotated copies of one acceptance uniform. P = t_i / sum(RIS weights), the inverse of
 *	  the RIS unbiased contribution weight
 *
 * Outputs, per pick: the pixel, its selection "probability" P (1 / unbiased contribution weight) and its selection target t. 'pool_size' must be in
 * [1, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE].
 *
 * Returns false (nothing written) when the domain has no candidate with a non-zero selection target
 *
 * Forced inline for the prepare kernel of the shadow split union domain spatial reuse, for the reason given above
 * spmis_union_domain_shadow_split_canonical_probe_value() in SpatialReuseSPMISUnionDomainShadowSplitPrepare.h
 */
HIPRT_DEVICE HIPRT_FORCEINLINE bool spmis_union_domain_select_pool_candidates(const HIPRTRenderData& render_data,
																			  ReSTIRPTSurface& center_surface,
																			  const ReSTIRPTSPMISUnionDomain& domain,
																			  int center_pixel_reservoir_confidence,
																			  int pool_size,
																			  unsigned int* out_pixel_indices,
																			  float* out_selection_probabilities,
																			  float* out_selection_targets,
																			  Xorshift32Generator& rng)
{
	const ReSTIRPTSPMISSettings& spmis_settings					  = render_data.render_settings.restir_pt_settings.spmis_settings;
	const ReSTIRPTSPMISUnionDomainSettings& union_domain_settings = spmis_settings.union_domain;

	if (domain.non_zero_count == 0 || domain.importance_sum <= 0.0f)
		return false;

	int total_neighbor_budget = hippt::max(1, union_domain_settings.total_neighbor_budget);
	bool exact				  = domain.non_zero_count <= static_cast<unsigned int>(total_neighbor_budget);

	// Center BSDF of the selection target, set up once for all the candidates
	ReSTIRPTSPMISUnionDomainCenterBSDF center_bsdf = spmis_union_domain_center_bsdf_setup(render_data, center_surface, rng);
	float center_confidence						   = static_cast<float>(center_pixel_reservoir_confidence);
	float domain_confidence_sum					   = static_cast<float>(domain.confidence_sum);

	const float4_t* neighbor_records  = reinterpret_cast<const float4_t*>(union_domain_settings.neighbor_records);
	const float2_t* cell_alias_tables = reinterpret_cast<const float2_t*>(union_domain_settings.cell_alias_tables);
	float inverse_pool_size			  = 1.0f / pool_size;

	if (exact)
	{
		// The acceptance uniform of pick j is rotated by j / pool_size
		for (int pick = 0; pick < pool_size; pick++)
		{
			out_pixel_indices[pick]		= 0;
			out_selection_targets[pick] = 0.0f;
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

				float target = spmis_neighbor_record_target(render_data, record_first, record_second, center_surface, center_bsdf, center_confidence);
				if (target <= 0.0f)
					continue;

				target_sum += target;
				float acceptance		 = target / target_sum;
				float acceptance_uniform = rng();
				unsigned int pixel_index = spmis_settings.pixel_indices_sorted[cell_start_index + record];

				for (int pick = 0; pick < pool_size; pick++)
				{
					float rotated_uniform = acceptance_uniform + pick * inverse_pool_size;
					rotated_uniform		  = rotated_uniform >= 1.0f ? rotated_uniform - 1.0f : rotated_uniform;
					if (rotated_uniform < acceptance)
					{
						out_pixel_indices[pick]		= pixel_index;
						out_selection_targets[pick] = target;
					}
				}
			}
		}

		if (!(target_sum > 0.0f))
			return false;

		for (int pick = 0; pick < pool_size; pick++)
			out_selection_probabilities[pick] = out_selection_targets[pick] / target_sum;

		return true;
	}

	// RIS over K stratified candidates of the whole domain
	float inverse_total_neighbor_budget = 1.0f / total_neighbor_budget;
	float inverse_importance_sum		= 1.0f / domain.importance_sum;

	// The importance sums of the cells of the domain do not depend on the candidate: loaded once per pixel instead of once per candidate and per
	// cell by the cell selection below
	float cell_importance_0 = spmis_union_domain_cell_importance(spmis_settings, domain, 0);
	float cell_importance_1 = spmis_union_domain_cell_importance(spmis_settings, domain, 1);
	float cell_importance_2 = spmis_union_domain_cell_importance(spmis_settings, domain, 2);
	float cell_importance_3 = spmis_union_domain_cell_importance(spmis_settings, domain, 3);

	// Same for the number of non-zero reservoirs of the cells and the scale from the confidence sum of a cell to the one of the whole domain (the
	// records store the confidence sum of their own cell, the selection target needs the one of the whole domain)
	unsigned int non_zero_cell_size_0 = domain.cell_count > 0 ? static_cast<unsigned int>(spmis_settings.cell_non_zero_reservoir_counters[domain.cell_0]) : 0;
	unsigned int non_zero_cell_size_1 = domain.cell_count > 1 ? static_cast<unsigned int>(spmis_settings.cell_non_zero_reservoir_counters[domain.cell_1]) : 0;
	unsigned int non_zero_cell_size_2 = domain.cell_count > 2 ? static_cast<unsigned int>(spmis_settings.cell_non_zero_reservoir_counters[domain.cell_2]) : 0;
	unsigned int non_zero_cell_size_3 = domain.cell_count > 3 ? static_cast<unsigned int>(spmis_settings.cell_non_zero_reservoir_counters[domain.cell_3]) : 0;

	float confidence_scale_0 = spmis_union_domain_cell_confidence_scale(spmis_settings, domain, 0, domain_confidence_sum);
	float confidence_scale_1 = spmis_union_domain_cell_confidence_scale(spmis_settings, domain, 1, domain_confidence_sum);
	float confidence_scale_2 = spmis_union_domain_cell_confidence_scale(spmis_settings, domain, 2, domain_confidence_sum);
	float confidence_scale_3 = spmis_union_domain_cell_confidence_scale(spmis_settings, domain, 3, domain_confidence_sum);

	// Offsets of the cells in pixel_indices_sorted, copied out of the domain once per pixel: the domain lives in scratch memory (it is kept across
	// the out-of-line calls of the kernel that follow this selection), so reading the offset from the domain for every candidate would be a scratch
	// load before the alias table and record loads of that candidate
	unsigned int cell_offset_0 = domain.get_cell_offset(0);
	unsigned int cell_offset_1 = domain.get_cell_offset(1);
	unsigned int cell_offset_2 = domain.get_cell_offset(2);
	unsigned int cell_offset_3 = domain.get_cell_offset(3);

	float weight_sum = 0.0f;

	// Slot of the selected record in pixel_indices_sorted (cell_offsets[cell] + index in the cell): the pixel index is only loaded for the selected
	// candidates, after the resampling. The initial value is never output: the first candidate with a non-zero weight is accepted by every reservoir
	unsigned int selected_record_slots[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE];
	float selected_targets[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE];

	UNROLL_LOOP
	for (int pick = 0; pick < RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE; pick++)
	{
		selected_record_slots[pick] = 0;
		selected_targets[pick]		= 0.0f;
	}

	for (int stratum = 0; stratum < total_neighbor_budget; stratum++)
	{
		float stratified_uniform = hippt::min((stratum + rng()) * inverse_total_neighbor_budget, 0.99999994f);

		// Cell proportionally to its importance sum, the uniform is remapped into the interval of the chosen cell (the stratification carries over)
		int selected_cell			= 0;
		float cell_importance		= 0.0f;
		float cumulative_before		= 0.0f;
		float cumulative_importance = 0.0f;
		bool cell_found				= false;

		for (int cell = 0; cell < domain.cell_count; cell++)
		{
			float importance = cell == 0 ? cell_importance_0 : (cell == 1 ? cell_importance_1 : (cell == 2 ? cell_importance_2 : cell_importance_3));

			float next_cumulative_importance = cumulative_importance + importance * inverse_importance_sum;

			bool take_cell = !cell_found && importance > 0.0f && (stratified_uniform < next_cumulative_importance || cell == domain.cell_count - 1);
			if (take_cell)
			{
				selected_cell	  = cell;
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

		unsigned int cell_start_index =
			selected_cell == 0 ? cell_offset_0 : (selected_cell == 1 ? cell_offset_1 : (selected_cell == 2 ? cell_offset_2 : cell_offset_3));
		unsigned int non_zero_cell_size =
			selected_cell == 0 ? non_zero_cell_size_0
							   : (selected_cell == 1 ? non_zero_cell_size_1 : (selected_cell == 2 ? non_zero_cell_size_2 : non_zero_cell_size_3));
		// Same distribution as the CDF of the cell: one alias table load picks the record
		unsigned int record_index = spmis_sample_cell_alias_table(cell_alias_tables + cell_start_index, non_zero_cell_size, cell_local_uniform);

		float4_t record_first  = neighbor_records[(cell_start_index + record_index) * 2 + 0];
		float4_t record_second = neighbor_records[(cell_start_index + record_index) * 2 + 1];
		// Probability of the candidate: probability of its cell x probability of the record in the CDF of the cell, which the alias table samples
		// (record_second.z is the CDF importance c W p_hat of the record)
		float candidate_probability = cell_share * record_second.z / cell_importance;
		// The records store the confidence sum of their own cell, the selection target needs the one of the whole domain
		record_second.w *= selected_cell == 0 ? confidence_scale_0
											  : (selected_cell == 1 ? confidence_scale_1 : (selected_cell == 2 ? confidence_scale_2 : confidence_scale_3));

		float target = spmis_neighbor_record_target(render_data, record_first, record_second, center_surface, center_bsdf, center_confidence);
		if (target <= 0.0f)
			continue;

		float weight = target * inverse_total_neighbor_budget / candidate_probability;
		weight_sum += weight;

		// One acceptance uniform, rotated for every reservoir of the pool: each reservoir is a weighted reservoir sampling of the same candidate set
		// with uniform acceptance tests
		float acceptance		 = weight / weight_sum;
		float acceptance_uniform = rng();
		unsigned int record_slot = cell_start_index + record_index;

		UNROLL_LOOP
		for (int pick = 0; pick < RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE; pick++)
		{
			float rotated_uniform = acceptance_uniform + pick * inverse_pool_size;
			rotated_uniform		  = rotated_uniform >= 1.0f ? rotated_uniform - 1.0f : rotated_uniform;

			bool accept = pick < pool_size && rotated_uniform < acceptance;

			selected_record_slots[pick] = accept ? record_slot : selected_record_slots[pick];
			selected_targets[pick]		= accept ? target : selected_targets[pick];
		}
	}

	if (!(weight_sum > 0.0f))
		return false;

	for (int pick = 0; pick < pool_size; pick++)
	{
		out_pixel_indices[pick]			  = spmis_settings.pixel_indices_sorted[selected_record_slots[pick]];
		out_selection_probabilities[pick] = selected_targets[pick] / weight_sum;
		out_selection_targets[pick]		  = selected_targets[pick];
	}

	return true;
}

/**
 * Neighbor selection of the union domain reuse: two-stage and visibility-aware, with a visibility pool
 * (ReSTIRPTSPMISUnionDomainSettings::visibility_pool_size = L >= 1). Every pixel traces exactly L shadow rays, whatever the number of neighbors N.
 *
 * Stage 1: ONE pool for the pixel, spmis_union_domain_select_pool_candidates() (prepare kernel, SpatialReuseSPMISUnionDomainShadowSplitPrepare.h): the
 * K stratified candidates of the whole domain resampled by L reservoirs with rotated copies of one acceptance uniform. Every pick l is marginally drawn
 * proportionally to the RIS weights w of the pool, P_l = t_l / W with W the sum of the RIS weights, so that for any function g of the neighbors that is
 * zero where t is zero:
 *
 *		E[(1 / L) x sum over the L picks of g(i_l) / P_l] = E[sum over the pool of w_k g(c_k) / t_k] = sum over the domain of g
 *
 * Stage 2: the visibility V_l of every pick at the center pixel (one ray each, traced by the trace selection kernel, same test as
 * ReSTIR_PT_sample_visible_at_surface()). g contains the visibility (an occluded sample has a zero contribution at the center) so only the n_vis visible
 * picks contribute. min(N, n_vis) of them are chosen uniformly without replacement by the weights kernel, each visible pick with the inclusion
 * probability pi = min(1, N / n_vis), which only depends on the visibility of the picks, not on their values (Horvitz-Thompson):
 *
 *		E[sum over the chosen picks of g(i_l) / (pi L P_l)] = E[(1 / L) x sum over the visible picks of g(i_l) / P_l]
 *
 * The spatial reuse divides every neighbor contribution by N x P (see ReSTIRPTSpatialGroupwiseMISWeight::get_MIS_weight_non_canonical_from_group_sum()),
 * so the "probability" of a chosen pick is P = pi L P_l / N. The slots beyond min(N, n_vis) have no neighbor. Duplicate picks are distinct terms of the
 * estimator: each one can be chosen.
 */

#endif // #ifndef DEVICE_RESTIR_PT_SPATIAL_SPMIS_UNION_DOMAIN_H
