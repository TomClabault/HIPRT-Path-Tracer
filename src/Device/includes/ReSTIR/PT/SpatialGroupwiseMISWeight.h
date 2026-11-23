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
 * p_j(y): the target function of a path y of the center pixel at pixel j of the reuse domain, in the measure of the center pixel (times the Jacobian of
 * the center -> j shift), with the pixel j assumed visible: no shadow ray. 'sample' is the canonical sample or the sample of a neighbor (the reconnection
 * data is the same whichever pixel the sample comes from)
 *
 * The MIS target function BSDF approximation (SpatialMISTargetFunctionBSDFApproximation.h) replaces the target function, the Jacobian is unchanged
 *
 * The kernels of the union domain spatial reuse (SpatialSPMISUnionDomainShadowSplit.h) evaluate the probes with it and trace the ray of a probe in
 * another kernel when its value is non-zero
 *
 * Forced inline: out of line, the sample passed by reference lives in scratch memory in the caller and every field read here is a scratch load. The
 * prepare kernel calls it through the out-of-line spmis_union_domain_shadow_split_canonical_probe_value() instead (see there)
 */
HIPRT_DEVICE HIPRT_FORCEINLINE float spmis_target_function_at_domain_pixel(const HIPRTRenderData& render_data,
																		   const ReSTIRPTReservoirSample& sample,
																		   float3_t center_pixel_shading_point,
																		   unsigned int pixel_index,
																		   Xorshift32Generator& random_number_generator)
{
	float3_t pixel_shading_point = render_data.g_buffer.primary_hit_position[pixel_index];

	float target_function =
		ReSTIR_PT_evaluate_target_function_at_pixel<false, true>(render_data, sample, pixel_index, pixel_shading_point, random_number_generator);
	if (target_function > 0.0f && !sample.is_envmap_path())
	{
		// If this is an envmap path the jacobian is just 1
		float jacobian	= get_jacobian_determinant_reconnection_shift(sample.rc_vertex, sample.rc_vertex_geometric_normal.unpack(), pixel_shading_point,
																	  center_pixel_shading_point,
																	  render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());
		target_function = jacobian == 0.0f ? 0.0f : target_function * jacobian;
	}

	return target_function;
}

/**
 * Stochastic group-wise MIS weights of the SPMIS union domain reuse (groups of g = RESTIR_PT_SPMIS_UNION_DOMAIN_MIS_GROUP_SIZE pixels), a generalization
 * of the stochastic pairwise MIS weights of "Stochastic Pairwise MIS for Unbiased Large-Kernel Reuse in Real-Time, Hedstrom et al. 2026".
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
 *
 * The partition of unity holds for any functions p_j with the support of the target functions. Every p_j(y) of these weights is the MIS target function
 * BSDF approximation of SpatialMISTargetFunctionBSDFApproximation.h, the own terms passed by the caller included
 * (ReSTIR_PT_MIS_target_function_bsdf_approximation_own_term())
 *
 * The visibility of the probes is traced in separate kernels (SpatialSPMISUnionDomainShadowSplit.h): these functions take the probe values with their
 * visibility already applied (zero for an occluded probe)
 */
struct ReSTIRPTSpatialGroupwiseMISWeight
{
	/**
	 * MIS weight of the canonical sample y of the center pixel:
	 *
	 *		m_c(y) = c_c p_c(y) / (c_c p_c(y) + (C / g) x sum over a group of g uniform pixels of p_j(y))
	 *
	 * estimated with 'probe_count' probes (canonical_group_count x g, one shadow ray each) drawn i.i.d. uniformly in the domain, at most
	 * RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES probes. The bracket is averaged over ALL the subsets of size g of the probes (U-statistic): every
	 * subset is a group of g i.i.d. uniform pixels so every bracket is an unbiased estimate of E[m_c], with the same rays as canonical_group_count disjoint
	 * groups. The U-statistic is valid for any number of probes >= g, not only for the multiples of g.
	 *
	 * 'probe_target_functions' holds the 'probe_count' values p_j(y) (zero for an occluded probe), 'canonical_numerator' is c_c p_c with p_c the MIS
	 * target function BSDF approximation of the canonical sample at the center pixel and 'group_scale' is C / g
	 */
	HIPRT_DEVICE float get_canonical_MIS_weight_from_probes(float canonical_numerator, float group_scale, const float* probe_target_functions, int probe_count)
	{
		static_assert(RESTIR_PT_SPMIS_UNION_DOMAIN_MIS_GROUP_SIZE == 2, "The subsets of the U-statistic are enumerated as pairs");

		float bracket_sum		  = 0.0f;
		unsigned int subset_count = 0;

		for (int first = 0; first < probe_count; first++)
		{
			for (int second = first + 1; second < probe_count; second++)
			{
				bracket_sum += canonical_bracket(canonical_numerator, group_scale, probe_target_functions[first] + probe_target_functions[second]);
				subset_count++;
			}
		}

		return subset_count > 0 ? bracket_sum / subset_count : 0.0f;
	}

	/**
	 * MIS weight of the sample y_i of the neighbor i selected for one of the 'reused_neighbors_count' neighbor slots with the selection "probability"
	 * 'neighbor_selection_probability' (1 / unbiased contribution weight of the selection, see the neighbor selection of SpatialSPMISUnionDomain.h):
	 *
	 *		m_i(y_i) = c_i p_i(y_i) / (c_c p_c(y_i) + (C / g) x (p_i(y_i) + sum over g - 1 uniform pixels of p_j(y_i)))
	 *
	 * The neighbor itself and g - 1 uniformly drawn pixels (the companions of the slot) stand for the whole domain. The returned weight includes the
	 * 1 / (N P_i) factor of the stochastic MIS: each of the N slots draws one neighbor with P_i, so that (1 / N) x sum over the slots of m_i f / P_i is an
	 * unbiased estimate of the sum over the domain of m_i f.
	 *
	 * 'target_function_at_neighbor' is p_i(y_i) in the measure of the center pixel (target function of the sample at its pixel / Jacobian of the shift)
	 * and 'target_function_at_center' is p_c(y_i), both the MIS target function BSDF approximations of the neighbor's sample. 'group_target_function_sum'
	 * is the sum of p_i(y_i) and of the g - 1 companions of the group (zero for an occluded companion). 'domain_confidence_sum' is C.
	 */
	HIPRT_DEVICE float get_MIS_weight_non_canonical_from_group_sum(float neighbor_reservoir_confidence,
																   float target_function_at_neighbor,
																   int center_pixel_reservoir_confidence,
																   float target_function_at_center,
																   float domain_confidence_sum,
																   int reused_neighbors_count,
																   float neighbor_selection_probability,
																   float group_target_function_sum)
	{
		float denominator = target_function_at_center * center_pixel_reservoir_confidence;
		denominator += domain_confidence_sum / RESTIR_PT_SPMIS_UNION_DOMAIN_MIS_GROUP_SIZE * group_target_function_sum;
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
