/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_GROUPWISE_MIS_WEIGHT_H
#define DEVICE_RESTIR_PT_SPATIAL_GROUPWISE_MIS_WEIGHT_H

#include "Device/includes/ReSTIR/Jacobian.h"
#include "Device/includes/ReSTIR/PT/SpatialSPMISUnionDomain.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"

#include "HostDeviceCommon/RenderData.h"

/**
 * p_j(y): the target function (visibility included) of a path y of the center pixel at pixel j of the reuse domain, in the measure of the center pixel
 * (times the Jacobian of the center -> j shift). 'sample' is the canonical sample or the sample of a neighbor (the reconnection data is the same whichever
 * pixel the sample comes from). Same evaluation as the canonical MIS weight of the stochastic pairwise MIS (get_resampling_MIS_weight_canonical())
 */
HIPRT_DEVICE float spmis_target_function_at_domain_pixel(const HIPRTRenderData& render_data,
														 const ReSTIRPTReservoirSample& sample,
														 const ReSTIRPTSurface& center_pixel_surface,
														 unsigned int pixel_index,
														 Xorshift32Generator& random_number_generator)
{
	float3_t pixel_shading_point = render_data.g_buffer.primary_hit_position[pixel_index];

	float target_function = ReSTIR_PT_evaluate_target_function_at_pixel<true>(render_data, sample, pixel_index, pixel_shading_point, random_number_generator);
	if (target_function > 0.0f && !sample.is_envmap_path())
	{
		// If this is an envmap path the jacobian is just 1
		float jacobian	= get_jacobian_determinant_reconnection_shift(sample.rc_vertex, sample.rc_vertex_geometric_normal.unpack(), pixel_shading_point,
																	  center_pixel_surface.shading_point,
																	  render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());
		target_function = jacobian == 0.0f ? 0.0f : target_function * jacobian;
	}

	return target_function;
}

/**
 * Sum of p_j(y) over 'draw_count' pixels j drawn uniformly (with replacement) in the reuse domain
 */
HIPRT_DEVICE float spmis_union_domain_group_target_function_sum(const HIPRTRenderData& render_data,
																const ReSTIRPTReservoirSample& sample,
																const ReSTIRPTSurface& center_pixel_surface,
																const ReSTIRPTSPMISUnionDomain& domain,
																int draw_count,
																Xorshift32Generator& random_number_generator)
{
	float target_function_sum = 0.0f;
	for (int draw = 0; draw < draw_count; draw++)
	{
		unsigned int pixel_index = spmis_union_domain_uniform_pixel(render_data, domain, random_number_generator);

		target_function_sum += spmis_target_function_at_domain_pixel(render_data, sample, center_pixel_surface, pixel_index, random_number_generator);
	}

	return target_function_sum;
}

/**
 * Stochastic group-wise MIS weights of the SPMIS union domain reuse (ReSTIRPTSPMISUnionDomainSettings::mis_group_size = g), a generalization of the
 * stochastic pairwise MIS weights of "Stochastic Pairwise MIS for Unbiased Large-Kernel Reuse in Real-Time, Hedstrom et al. 2026".
 *
 * The generalized balance heuristic over the canonical pixel c and all the pixels j of the reuse domain D has the denominator
 *
 *		c_c p_c(y) + sum over j in D of c_j p_j(y)
 *
 * whose sum over D is estimated with g pixels drawn uniformly in D: sum over j in D of c_j p_j(y) ~= (C / g) x sum over the g pixels of p_j(y), with C the
 * confidence sum of D. g = 1 is pairwise MIS, larger groups get closer to the balance heuristic for g more shadow rays per weight.
 *
 * Partition of unity: when all the confidences of D are equal (c_j = C / |D|), for every path y, the expectation over the group draws of the canonical
 * weight plus the sum over D of the neighbor weights is exactly 1. With S the sum of the p_j(y) of g uniform pixels, by symmetry of the g draws:
 *
 *		sum over i in D of E[c_i p_i / (c_c p_c + (C / g)(p_i + S'))] = (C / g) E[S / (c_c p_c + (C / g) S)]	(S' = the g - 1 other pixels)
 *
 * so that E[m_c(y)] + sum over i of E[m_i(y)] = E[(c_c p_c + (C / g) S) / (c_c p_c + (C / g) S)] = 1. The group draws are independent of the selection of
 * the neighbors: the spatial reuse stays unbiased. ReSTIRPTSettings::do_spmis_union_domain_reuse() only enables these weights when the confidences are
 * equal (initial candidates reservoirs as the only input)
 */
struct ReSTIRPTSpatialGroupwiseMISWeight
{
	/**
	 * MIS weight of the canonical sample y of the center pixel:
	 *
	 *		m_c(y) = c_c p_c(y) / (c_c p_c(y) + (C / g) x sum over a group of g uniform pixels of p_j(y))
	 *
	 * estimated with 'canonical_group_count' x g probes (one shadow ray each) drawn i.i.d. uniformly in the domain. Up to
	 * RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES probes (and g <= 3), the bracket is averaged over ALL the subsets of size g of the probes (U-statistic):
	 * every subset is a group of g i.i.d. uniform pixels so every bracket is an unbiased estimate of E[m_c], with the same rays as canonical_group_count
	 * disjoint groups. Above that, the bracket is averaged over canonical_group_count disjoint groups
	 *
	 * 'stratified_probes' (ReSTIRPTSPMISUnionDomainSettings::stratified_canonical_probes, g = 2 and 4 to RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES
	 * probes only): the pixel list of the domain is cut in S = probes / 2 strata of equal probability, with two independent probes per stratum. For two
	 * independent uniform pixels a and b of the domain:
	 *
	 *		E[bracket(p_a + p_b)] = (1 / S^2) x sum over the ordered pairs of strata (k, k') of E[bracket(p_a + p_b) | a in k, b in k']
	 *
	 * The term k = k' is estimated by the two probes of the stratum and a term k != k' by the mean over the 4 pairs (one probe of k, one probe of k').
	 * Every term is unbiased so the estimate has the same expectation as the U-statistic, with a lower variance when the visibility of the canonical
	 * sample is coherent over the domain (the list is ordered cell after cell)
	 */
	HIPRT_DEVICE float get_resampling_MIS_weight_canonical(const HIPRTRenderData& render_data,
														   const ReSTIRPTReservoirSample& center_pixel_reservoir_sample,
														   int center_pixel_reservoir_confidence,
														   float center_pixel_reservoir_target_function,
														   const ReSTIRPTSurface& center_pixel_surface,
														   const ReSTIRPTSPMISUnionDomain& domain,
														   int group_size,
														   int canonical_group_count,
														   bool stratified_probes,
														   Xorshift32Generator& random_number_generator)
	{
		float canonical_numerator = center_pixel_reservoir_target_function * center_pixel_reservoir_confidence;
		// The g pixels of a group stand for the whole domain
		float group_scale = static_cast<float>(domain.confidence_sum) / group_size;

		int probe_count = canonical_group_count * group_size;

		if (stratified_probes && group_size == 2 && probe_count >= 4 && probe_count <= RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES)
		{
			int strata_count	 = probe_count / 2;
			float strata_inverse = 1.0f / strata_count;

			float probe_target_functions[RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES];
			for (int probe = 0; probe < probe_count; probe++)
			{
				// Probes 2k and 2k + 1 are drawn independently in the stratum k
				float probe_uniform		 = hippt::min(((probe / 2) + random_number_generator()) * strata_inverse, 0.99999994f);
				unsigned int pixel_index = spmis_union_domain_pixel_at_uniform(render_data, domain, probe_uniform);

				probe_target_functions[probe] = spmis_target_function_at_domain_pixel(render_data, center_pixel_reservoir_sample, center_pixel_surface,
																					  pixel_index, random_number_generator);
			}

			float same_stratum_bracket_sum = 0.0f;
			float cross_strata_bracket_sum = 0.0f;
			for (int first_stratum = 0; first_stratum < strata_count; first_stratum++)
			{
				float first_probe_0 = probe_target_functions[first_stratum * 2 + 0];
				float first_probe_1 = probe_target_functions[first_stratum * 2 + 1];

				same_stratum_bracket_sum += canonical_bracket(canonical_numerator, group_scale, first_probe_0 + first_probe_1);

				for (int second_stratum = first_stratum + 1; second_stratum < strata_count; second_stratum++)
				{
					float second_probe_0 = probe_target_functions[second_stratum * 2 + 0];
					float second_probe_1 = probe_target_functions[second_stratum * 2 + 1];

					cross_strata_bracket_sum += canonical_bracket(canonical_numerator, group_scale, first_probe_0 + second_probe_0);
					cross_strata_bracket_sum += canonical_bracket(canonical_numerator, group_scale, first_probe_0 + second_probe_1);
					cross_strata_bracket_sum += canonical_bracket(canonical_numerator, group_scale, first_probe_1 + second_probe_0);
					cross_strata_bracket_sum += canonical_bracket(canonical_numerator, group_scale, first_probe_1 + second_probe_1);
				}
			}

			// An unordered pair of distinct strata stands for 2 ordered pairs, each one the mean of its 4 brackets
			return (same_stratum_bracket_sum + 0.5f * cross_strata_bracket_sum) * strata_inverse * strata_inverse;
		}

		// The subsets of the U-statistic are enumerated by up to 3 nested loops below
		if (group_size <= 3 && probe_count <= RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES && probe_count >= group_size)
		{
			float probe_target_functions[RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES];
			for (int probe = 0; probe < probe_count; probe++)
				probe_target_functions[probe] = spmis_union_domain_group_target_function_sum(render_data, center_pixel_reservoir_sample, center_pixel_surface,
																							 domain, 1, random_number_generator);

			float bracket_sum		  = 0.0f;
			unsigned int subset_count = 0;
			for (int first = 0; first < probe_count; first++)
			{
				if (group_size == 1)
				{
					bracket_sum += canonical_bracket(canonical_numerator, group_scale, probe_target_functions[first]);
					subset_count++;

					continue;
				}

				for (int second = first + 1; second < probe_count; second++)
				{
					if (group_size == 2)
					{
						bracket_sum += canonical_bracket(canonical_numerator, group_scale, probe_target_functions[first] + probe_target_functions[second]);
						subset_count++;

						continue;
					}

					for (int third = second + 1; third < probe_count; third++)
					{
						bracket_sum += canonical_bracket(canonical_numerator, group_scale,
														 probe_target_functions[first] + probe_target_functions[second] + probe_target_functions[third]);
						subset_count++;
					}
				}
			}

			return subset_count > 0 ? bracket_sum / subset_count : 0.0f;
		}

		float mis_weight = 0.0f;
		for (int group = 0; group < canonical_group_count; group++)
		{
			float group_target_function_sum = spmis_union_domain_group_target_function_sum(render_data, center_pixel_reservoir_sample, center_pixel_surface,
																						   domain, group_size, random_number_generator);

			mis_weight += canonical_bracket(canonical_numerator, group_scale, group_target_function_sum) / canonical_group_count;
		}

		return mis_weight;
	}

	/**
	 * MIS weight of the sample y_i of the neighbor i selected for one of the 'reused_neighbors_count' neighbor slots with the selection "probability"
	 * 'neighbor_selection_probability' (1 / unbiased contribution weight of the selection, see spmis_union_domain_select_neighbors()):
	 *
	 *		m_i(y_i) = c_i p_i(y_i) / (c_c p_c(y_i) + (C / g) x (p_i(y_i) + sum over g - 1 uniform pixels of p_j(y_i)))
	 *
	 * The neighbor itself and g - 1 uniformly drawn pixels stand for the whole domain. The returned weight includes the 1 / (N P_i) factor of the stochastic
	 * MIS: each of the N slots draws one neighbor with P_i, so that (1 / N) x sum over the slots of m_i f / P_i is an unbiased estimate of the sum over
	 * the domain of m_i f.
	 *
	 * 'target_function_at_neighbor' is p_i(y_i) in the measure of the center pixel (target function of the sample at its pixel / Jacobian of the shift)
	 * and 'target_function_at_center' is p_c(y_i). Both must be > 0 (the caller skips the neighbor otherwise, its contribution is zero)
	 */
	HIPRT_DEVICE float get_resampling_MIS_weight_non_canonical(const HIPRTRenderData& render_data,
															   const ReSTIRPTReservoirSample& neighbor_reservoir_sample,
															   float neighbor_reservoir_confidence,
															   float target_function_at_neighbor,
															   int center_pixel_reservoir_confidence,
															   float target_function_at_center,
															   const ReSTIRPTSurface& center_pixel_surface,
															   const ReSTIRPTSPMISUnionDomain& domain,
															   int group_size,
															   int reused_neighbors_count,
															   float neighbor_selection_probability,
															   Xorshift32Generator& random_number_generator)
	{
		float group_target_function_sum =
			target_function_at_neighbor + spmis_union_domain_group_target_function_sum(render_data, neighbor_reservoir_sample, center_pixel_surface, domain,
																					   group_size - 1, random_number_generator);

		float denominator = target_function_at_center * center_pixel_reservoir_confidence;
		denominator += static_cast<float>(domain.confidence_sum) / group_size * group_target_function_sum;

		if (denominator > 0.0f)
			return neighbor_reservoir_confidence * target_function_at_neighbor / denominator / (reused_neighbors_count * neighbor_selection_probability);
		else
			return 0.0f;
	}

private:
	/**
	 * One bracket c_c p_c / (c_c p_c + (C / g) x sum of the p_j of a group) of the canonical MIS weight
	 */
	HIPRT_DEVICE float canonical_bracket(float canonical_numerator, float group_scale, float group_target_function_sum)
	{
		float denominator = canonical_numerator + group_scale * group_target_function_sum;

		return denominator > 0.0f ? canonical_numerator / denominator : 0.0f;
	}
};

#endif // #ifndef DEVICE_RESTIR_PT_SPATIAL_GROUPWISE_MIS_WEIGHT_H
