/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_BSDFS_MICROFACET_MULTIPLE_SCATTERING_CUI2023_H
#define DEVICE_INCLUDES_BSDFS_MICROFACET_MULTIPLE_SCATTERING_CUI2023_H

#include "Device/includes/BSDFs/Fresnel.h"
#include "Device/includes/BSDFs/MicrofacetCommon.h"
#include "Device/includes/BSDFs/MicrofacetGGX.h"
#include "Device/includes/BSDFs/MicrofacetMultipleScatteringCui2023Macros.h"
#include "Device/includes/BSDFs/ThinFilm.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material/MaterialUtils.h"

HIPRT_DEVICE static float microfacet_GGX_pdf_reflect(float material_roughness,
													 float material_anisotropy,
													 const float3_t& local_view_direction,
													 const float3_t& local_to_light_direction,
													 const float3_t& local_halfway_vector,
													 SpecularDeltaReflectionSampled incident_light_direction_is_from_GGX_sample,
													 bool zero_below_surface = true);

HIPRT_DEVICE static ColorRGB32F principled_metallic_fresnel(const DeviceUnpackedEffectiveMaterial& material,
															float incident_ior,
															float3_t local_to_light_direction,
															float3_t local_half_vector);

// TODO perf:
// Variable number of bounces depending on roughness and anisotropy

/**
 * Implementation of [Multiple-bounce Smith Microfacet BRDFs using the Invariance Principle, Cui et al., 2023]
 */
class SegmentTerm
{
public:
	HIPRT_DEVICE SegmentTerm(float lambda_0) : lambda_init(lambda_0) {}

	HIPRT_DEVICE void add_bounce(fp16 lambda_k)
	{
		if (lambda_k < (fp16)0.0f)
		{
			// Ray is going down into the microsurface

			set_l(N, -lambda_k);
			set_g(N, 0.0f);
			m *= get_e(N);

			N++;
		}
		else
		{
			if (m == (fp16)0.0f)
				set_g(N - 1, get_g(N - 1) / (lambda_k + get_lambda(N - 1)));
			else
			{
				set_g(N - 1, (fp16)(1.0f) / (lambda_k + get_lambda(N - 1)));
				m = (fp16)0.0f;
			}

			for (int i = N - 2; i >= 0; i--)
				set_g(i, (get_g(i) + get_g(i + 1)) / (lambda_k + get_lambda(i)));
		}
	}

	HIPRT_DEVICE float get_sk() const
	{
		if (m != (fp16)0.0f)
			return m;

		fp16 s = 0.0f;

		for (int i = N - 1; i >= 0; i--)
			s = get_e(i) * (s + get_g(i));

		return (float)s;
	}

private:
	HIPRT_DEVICE fp16 get_e(int i) const
	{
		return (fp16)1.0f / (lambda_init + get_lambda(i));
	}

	HIPRT_DEVICE fp16 get_g(int i) const
	{
		MS_CUI2023_GET_G_BODY;
	}

	HIPRT_DEVICE fp16 get_lambda(int i) const
	{
		MS_CUI2023_GET_L_BODY;
	}

	HIPRT_DEVICE void set_g(int i, fp16 value)
	{
		MS_CUI2023_SET_G_BODY;
	}

	HIPRT_DEVICE void set_l(int i, fp16 value)
	{
		MS_CUI2023_SET_L_BODY;
	}

private:
	int N  = 0;
	fp16 m = 1.0f;

	fp16 lambda_init = 0.0f;

	MS_CUI2023_DECLARE_G;
	MS_CUI2023_DECLARE_LAMBDAS;
};

// TODO do we need 2022 and 2023? Are they not the same when developing?
/**
 * Returns 1.0f + Lambda if the direction is below the surface and Lambda iif the direction is above the surface, and then that result is multiplied by the sign
 * of the direction.z (1.0f if above the surface and -1.0f if below the surface) to match the convention of the paper where Lambda is negative when the ray is
 * going down into the microsurface and positive when it's going up from the microsurface
 */
HIPRT_DEVICE float G1_Smith_lambda_signed_2023(float alpha_x, float alpha_y, const float3_t& local_direction)
{
	// 1.0f + Lambda if the direction is below the surface and Lambda iif the direction is above the surface
	//
	// And then that result is multiplied by the sign of the direction.z (1.0f if above the surface and -1.0f if below the surface) to match the convention
	// of the paper where Lambda is negative when the ray is going down into the microsurface and positive when it's going up from the microsurface
	return (G1_Smith_lambda(alpha_x, alpha_y, local_direction) + (local_direction.z < 0.0f ? 1.0f : 0.0f)) * (local_direction.z > 0.0f ? 1.0f : -1.0f);
}

HIPRT_DEVICE ColorRGB32F Cui_2023_vertex_term(const DeviceUnpackedEffectiveMaterial& material,
											  float incident_ior,
											  float alpha_x,
											  float alpha_y,
											  const float3_t& local_view_direction,
											  const float3_t& local_to_light_direction)
{
	float local_half_vector_length = hippt::length(local_view_direction + local_to_light_direction);
	if (local_half_vector_length == 0.0f)
		return ColorRGB32F(0.0f);

	float3_t local_half_vector = (local_view_direction + local_to_light_direction) / local_half_vector_length;
	ColorRGB32F F			   = principled_metallic_fresnel(material, incident_ior, local_to_light_direction, local_half_vector);

	return F * GGX_anisotropic(alpha_x, alpha_y, local_half_vector) / (4.0f * hippt::abs(local_view_direction.z));
}

/**
 * local_view_direction and local_to_light_direction should bot be pointing outward the surface here
 */
HIPRT_DEVICE ColorRGB32F
torrace_sparrow_GGX_multiple_scattering_invariance_eval_reflect(const HIPRTRenderData& render_data,
																const DeviceUnpackedEffectiveMaterial& material,
																float material_roughness,
																float material_anisotropy,
																float incident_ior,
																ColorRGB32F F,
																float3_t local_view_direction,	   // w_i in the paper
																float3_t local_to_light_direction, // w_o in the paper
																Xorshift32Generator& rng,
																float& out_pdf,
																SpecularDeltaReflectionSampled incident_light_direction_is_from_GGX_sample)
{
	if (local_to_light_direction.z < 0.0f || local_view_direction.z < 0.0f)
	{
		// A direction that is below the surface is invalid for a microfacet ** BRDF **
		out_pdf = 0.0f;

		return ColorRGB32F(0.0f);
	}

	// TODO can we remove this somehow? This is annoying to manage. The target functions in ReSTIR would basically need to be 0 if the surface is specular.
	// ReSTIR spatial/temporal reuse is basically the only reason this exists
	if (MaterialUtils::is_perfectly_smooth(material_roughness) && PrincipledBSDFDeltaDistributionEvaluationOptimization == KERNEL_OPTION_TRUE)
	{
		// Fast path for perfectly specular BRDF
		if (incident_light_direction_is_from_GGX_sample == SpecularDeltaReflectionSampled::SPECULAR_PEAK_NOT_SAMPLED)
			// For a perfectly smooth GGX distribution (a delta distribution), anything other than a
			// perfectly sampled reflection direction is going to yield 0 contribution
			return ColorRGB32F(0.0f);
		else
		{
			if (hippt::dot(reflect_ray(local_view_direction, make_float3(0.0f, 0.0f, 1.0f)), local_to_light_direction) <
				MaterialConstants::DELTA_DISTRIBUTION_ALIGNEMENT_THRESHOLD)
			{
				// Just an additional check that we indeed have the incident light
				// direction aligned with the perfect reflection direction
				//
				// This additional check is mainly useful for ReSTIR where we need
				// to evaluate the BRDF with a sample that may have been sampled from
				// a delta distribution at a neighbor (so it checks all the boxes for the shortcut
				// and we could just quickly return MaterialConstants::DELTA_DISTRIBUTION_HIGH_VALUE
				// but because that sample wasn't sampled at the current pixel, there
				// is a good chance that it doesn't actually align with the perfect
				// reflection direction = it doesn't align with the specular peak = 0 contribution

				out_pdf = 0.0f;

				return ColorRGB32F(0.0f);
			}

			out_pdf = MaterialConstants::DELTA_DISTRIBUTION_HIGH_VALUE;

			return ColorRGB32F(MaterialConstants::DELTA_DISTRIBUTION_HIGH_VALUE) * F / hippt::abs(local_to_light_direction.z);
		}
	}

	float alpha_x;
	float alpha_y;
	MaterialUtils::get_alphas(material_roughness, material_anisotropy, alpha_x, alpha_y);

	SegmentTerm s(G1_Smith_lambda_signed_2023(alpha_x, alpha_y, local_to_light_direction));

	float g1v_accum = G1_Smith_lambda_signed_2023(alpha_x, alpha_y, -local_view_direction);
	s.add_bounce(g1v_accum);

	ColorRGB32F weight = ColorRGB32F(1.0f);
	ColorRGB32F multiple_scattering_contribution =
							Cui_2023_vertex_term(material, incident_ior, alpha_x, alpha_y, local_view_direction, local_to_light_direction) * s.get_sk();
	float3_t current_view_direction		= local_view_direction;
	float3_t current_to_light_direction = local_to_light_direction;

	bool rough_enough = material_roughness > render_data.bsdfs_data.energy_compensation_roughness_threshold;
	if (rough_enough)
	{
		for (int bounce = 1; bounce < PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces; ++bounce)
		{
			current_to_light_direction = microfacet_GGX_sample_reflection<false>(material_roughness, material_anisotropy, current_view_direction, rng, false);

			float half_vector_length = hippt::length(current_view_direction + current_to_light_direction);
			if (half_vector_length == 0.0f)
				break;

			float3_t half_vector = (current_view_direction + current_to_light_direction) / half_vector_length;
			/*out_pdf *= microfacet_GGX_pdf_reflect(material_roughness, material_anisotropy, current_view_direction, current_to_light_direction, half_vector,
												  SpecularDeltaReflectionSampled::SPECULAR_PEAK_NOT_SAMPLED);*/

			// Here weight "should" be multiplied by the vertex term and divided by the VNDF PDF but this simplifies to F/G1V. So we're only just multiplying by
			// the fresnel term here and the G1V terms are accounted for by "g1v_accum"
			weight *= principled_metallic_fresnel(material, incident_ior, current_view_direction, half_vector);

			float lambda = G1_Smith_lambda_signed_2023(alpha_x, alpha_y, current_to_light_direction);
			s.add_bounce(lambda);
			float s_k = s.get_sk();

#if PrincipledBSDFMultipleScatteringCuiDoRussianRoulette == KERNEL_OPTION_TRUE
			if (bounce >= render_data.bsdfs_data.multiple_scattering_cui_2023_min_bounce_russian_roulette)
			{
				// Russian roulette
				float q = hippt::max(hippt::min(s_k, 0.95f), 0.3f);
				if (rng() >= q)
					break;

				weight /= q;
			}
#endif // PrincipledBSDFMultipleScatteringCuiDoRussianRoulette

			current_view_direction = -current_to_light_direction;
			multiple_scattering_contribution +=
									Cui_2023_vertex_term(material, incident_ior, alpha_x, alpha_y, current_view_direction, local_to_light_direction) * weight *
									hippt::abs(g1v_accum) * s_k;

			g1v_accum *= lambda;
		}
	}

	out_pdf = microfacet_GGX_pdf_reflect(material_roughness, material_anisotropy, local_view_direction, local_to_light_direction,
										 hippt::normalize(local_view_direction + local_to_light_direction), incident_light_direction_is_from_GGX_sample);

	return multiple_scattering_contribution / local_to_light_direction.z;
}

#endif
