/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_BSDF_MICROFACET_TORRANCE_SPARROW_H
#define DEVICE_INCLUDES_BSDF_MICROFACET_TORRANCE_SPARROW_H

#include "Device/includes/BSDFs/MicrofacetCommon.h"
#include "Device/includes/BSDFs/MicrofacetEnergyCompensation.h"
#include "Device/includes/BSDFs/MicrofacetMultipleScatteringCui2023.h"
#include "Device/includes/Sampling.h"

/**
 * 'incident_light_direction_is_from_GGX_sample' should be true if the 'local_to_light_direction' given comes from
 * sampling the microfacet distribution that is being evaluated by this function call
 *
 * false otherwise (if 'local_to_light_direction' comes from light sampling NEE, or sampling another lobe of the BSDF, ...).
 * This parameter only matters if the BRDF is perfectly smooth: roughness < MaterialConstants::ROUGHNESS_CLAMP
 */
template <bool useMultipleScatteringEnergyCompensation>
HIPRT_DEVICE static ColorRGB32F torrance_sparrow_GGX_eval_reflect(const HIPRTRenderData& render_data,
																  const DeviceUnpackedEffectiveMaterial& material,
																  float material_roughness,
																  float material_anisotropy,
																  float incident_ior,
																  bool do_energy_compensation,
																  const ColorRGB32F& F,
																  const float3& local_view_direction,
																  const float3& local_to_light_direction,
																  const float3& local_halfway_vector,
																  float& out_pdf,
																  SpecularDeltaReflectionSampled incident_light_direction_is_from_GGX_sample,
																  int current_bounce,
																  Xorshift32Generator& rng)
{
	out_pdf = -1.0f;
	return ColorRGB32F(-1.0f);
}

/**
 * Evaluates the Torrance Sparrow BRDF 'FDG / 4.NoL.NoV' with the
 * GGX as the microfacet distribution
 * function with single scattering (no energy compensation)
 *
 * 'incident_light_direction_is_from_GGX_sample' should be true if the 'local_to_light_direction' given comes from
 * sampling the microfacet distribution that is being evaluated by this function call
 *
 * false otherwise (if 'local_to_light_direction' comes from light sampling NEE, or sampling another lobe of the BSDF, ...).
 * This parameter only matters if the BRDF is perfectly smooth: roughness < MaterialConstants::ROUGHNESS_CLAMP
 *
 * Reference: [Sampling the GGX Distribution of Visible Normals, Heitz, 2018]
 * Equation 15
 */
template <>
HIPRT_DEVICE ColorRGB32F torrance_sparrow_GGX_eval_reflect<0>(const HIPRTRenderData& render_data,
															  const DeviceUnpackedEffectiveMaterial& material,
															  float material_roughness,
															  float material_anisotropy,
															  float incident_ior,
															  bool do_energy_compensation,
															  const ColorRGB32F& F,
															  const float3& local_view_direction,
															  const float3& local_to_light_direction,
															  const float3& local_halfway_vector,
															  float& out_pdf,
															  SpecularDeltaReflectionSampled incident_light_direction_is_from_GGX_sample,
															  int current_bounce,
															  Xorshift32Generator& rng)
{
	out_pdf = 0.0f;

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

	if (local_to_light_direction.z < 0.0f)
		// A direction that is below the surface is invalid for a microfacet ** BRDF **
		return ColorRGB32F(0.0f);

	float alpha_x;
	float alpha_y;
	MaterialUtils::get_alphas(material_roughness, material_anisotropy, alpha_x, alpha_y);

	// GGX normal distribution
	float D = GGX_anisotropic(alpha_x, alpha_y, local_halfway_vector);

	// GGX visible normal distribution for evaluating the PDF
	float lambda_V = G1_Smith_lambda(alpha_x, alpha_y, local_view_direction);
	float G1V	   = 1.0f / (1.0f + lambda_V);
	float Dvisible = GGX_anisotropic_vndf(D, G1V, local_view_direction, local_halfway_vector);

	// Maxing to GGX_DOT_PRODUCTS_CLAMP here to avoid zeros and numerical imprecisions
	// TODO note that we shouldn't need abs() here because we cannot have the view direction or to light direction below the surface
	float NoV = hippt::max(GGX_DOT_PRODUCTS_CLAMP, hippt::abs(local_view_direction.z));
	float NoL = hippt::max(GGX_DOT_PRODUCTS_CLAMP, hippt::abs(local_to_light_direction.z));

	// Because we're exactly sampling the visible normals distribution function,
	// that's exactly our PDF.
	//
	// Additionally, because we need to take into account the reflection operator
	// that we're going to apply to get our final 'to light direction' and so the
	// jacobian determinant of that reflection operator is the (4.0f * HoV) in the
	// denominator
	out_pdf = Dvisible / (4.0f * hippt::dot(local_view_direction, local_halfway_vector));
	if (out_pdf == 0.0f)
		return ColorRGB32F(0.0f);
	else
	{
		float lambda_L = G1_Smith_lambda(alpha_x, alpha_y, local_to_light_direction);

		if (render_data.bsdfs_data.GGX_masking_shadowing == GGXMaskingShadowingFlavor::HeightUncorrelated)
		{
			float G1L = 1.0f / (1.0f + lambda_L);
			float G2  = G1V * G1L;

			return F * D * G2 / (4.0f * NoL * NoV);
		}
		else // Default to GGXMaskingShadowingFlavor::HeightCorrelated
		{
			float G2HeightCorrelated = 1.0f / (1.0f + lambda_V + lambda_L);

			return F * D * G2HeightCorrelated / (4.0f * NoL * NoV);
		}
	}
}

/**
 * 'incident_light_direction_is_from_GGX_sample' should be true if the 'local_to_light_direction' given comes from
 * sampling the microfacet distribution that is being evaluated by this function call
 *
 * false otherwise(if 'local_to_light_direction' comes from light sampling NEE, or sampling another lobe of the BSDF, ...).
 * This parameter only matters if the BRDF is perfectly smooth: roughness < MaterialConstants::ROUGHNESS_CLAMP
 */
template <>
HIPRT_DEVICE ColorRGB32F torrance_sparrow_GGX_eval_reflect<1>(const HIPRTRenderData& render_data,
															  const DeviceUnpackedEffectiveMaterial& material,
															  float material_roughness,
															  float material_anisotropy,
															  float incident_ior,
															  bool do_energy_compensation,
															  const ColorRGB32F& F,
															  const float3& local_view_direction,
															  const float3& local_to_light_direction,
															  const float3& local_halfway_vector,
															  float& out_pdf,
															  SpecularDeltaReflectionSampled incident_light_direction_is_from_GGX_sample,
															  int current_bounce,
															  Xorshift32Generator& rng)
{
#if PrincipledBSDFEnergyCompensationMode == ENERGY_COMPENSATION_MODE_INVARIANCE_CUI
	return torrace_sparrow_GGX_multiple_scattering_invariance_eval_reflect(material, material_roughness, material_anisotropy, incident_ior, F,
																		   local_view_direction, local_to_light_direction, rng, out_pdf,
																		   incident_light_direction_is_from_GGX_sample);
#else
	ColorRGB32F ms_compensation_term = get_GGX_energy_compensation_conductors(render_data, F, material_roughness, do_energy_compensation, local_view_direction,
																			  current_bounce);
	ColorRGB32F single_scattering	 = torrance_sparrow_GGX_eval_reflect<0>(
							   render_data, material, material_roughness, material_anisotropy, incident_ior, do_energy_compensation, F, local_view_direction,
							   local_to_light_direction, local_halfway_vector, out_pdf, incident_light_direction_is_from_GGX_sample, current_bounce, rng);

	return single_scattering * ms_compensation_term;
#endif
}

/**
 * Returns the PDF of the Torrance Sparrow BRDF 'FDG / 4.NoL.NoV' with the
 * GGX as the microfacet distribution
 *
 * 'incident_light_direction_is_from_GGX_sample' should be true if the 'local_to_light_direction' given comes from
 * sampling the microfacet distribution that is being evaluated by this function call
 *
 * false otherwise (if 'local_to_light_direction' comes from light sampling NEE, or sampling another lobe of the BSDF, ...).
 * This parameter only matters if the BRDF is perfectly smooth: roughness < MaterialConstants::ROUGHNESS_CLAMP
 *
 * Reference: [Sampling the GGX Distribution of Visible Normals, Heitz, 2018]
 * Equation 15
 */
HIPRT_DEVICE float microfacet_GGX_pdf_reflect(float material_roughness,
											  float material_anisotropy,
											  const float3& local_view_direction,
											  const float3& local_to_light_direction,
											  const float3& local_halfway_vector,
											  SpecularDeltaReflectionSampled incident_light_direction_is_from_GGX_sample)
{
	if (MaterialUtils::is_perfectly_smooth(material_roughness) && PrincipledBSDFDeltaDistributionEvaluationOptimization == KERNEL_OPTION_TRUE)
	{
		// Fast path for perfectly specular BRDF
		if (incident_light_direction_is_from_GGX_sample == SpecularDeltaReflectionSampled::SPECULAR_PEAK_NOT_SAMPLED)
			// For a perfectly smooth GGX distribution (a delta distribution), anything other than a
			// perfectly sampled reflection direction is going to yield 0 contribution
			return 0.0f;
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

				return 0.0f;
			}

			return MaterialConstants::DELTA_DISTRIBUTION_HIGH_VALUE;
		}
	}

	// if (local_to_light_direction.z < 0.0f)
	//	// A direction that is below the surface is invalid for a microfacet ** BRDF **
	//	return 0.0f;

	float pdf = 0.0f;

	float alpha_x;
	float alpha_y;
	MaterialUtils::get_alphas(material_roughness, material_anisotropy, alpha_x, alpha_y);

	// GGX normal distribution
	float D = GGX_anisotropic(alpha_x, alpha_y, local_halfway_vector);

	// GGX visible normal distribution for evaluating the PDF
	float lambda_V = G1_Smith_lambda(alpha_x, alpha_y, local_view_direction);
	float G1V	   = 1.0f / (1.0f + lambda_V);
	// Using abs()
	float Dvisible = GGX_anisotropic_vndf(D, G1V, local_view_direction, local_halfway_vector);

	// Because we're exactly sampling the visible normals distribution function,
	// that's exactly our PDF.
	//
	// Additionally, because we need to take into account the reflection operator
	// that we're going to apply to get our final 'to light direction' and so the
	// jacobian determinant of that reflection operator is the (4.0f * HoV) in the
	// denominator
	return Dvisible / (4.0f * hippt::abs(hippt::dot(local_view_direction, local_halfway_vector)));
}

HIPRT_DEVICE static ColorRGB32F torrance_sparrow_GGX_eval_refract(const DeviceUnpackedEffectiveMaterial& material,
																  float roughness,
																  float relative_eta,
																  ColorRGB32F fresnel_reflectance,
																  const float3& local_view_direction,
																  const float3& local_to_light_direction,
																  const float3& local_halfway_vector,
																  float& out_pdf,
																  BSDFIncidentLightInfo incident_light_info)
{
	float NoL = local_to_light_direction.z;
	float NoV = local_view_direction.z;
	float HoL = hippt::dot(local_to_light_direction, local_halfway_vector);
	float HoV = hippt::dot(local_view_direction, local_halfway_vector);

	ColorRGB32F color;
	if (MaterialUtils::is_perfectly_smooth(roughness) && PrincipledBSDFDeltaDistributionEvaluationOptimization == KERNEL_OPTION_TRUE)
	{
		// Fast path for specular glass
		bool incident_direction_is_perfect_refraction = hippt::dot(refract_ray(local_view_direction, make_float3(0.0f, 0.0f, 1.0f), relative_eta),
																   local_to_light_direction) > MaterialConstants::DELTA_DISTRIBUTION_ALIGNEMENT_THRESHOLD;
		if (incident_light_info == BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFRACT_LOBE && incident_direction_is_perfect_refraction)
		{
			// When the glass is perfectly smooth i.e. delta distribution, our only hope is to sample
			// directly from the glass lobe. If we didn't sample from the glass lobe, this is going to be 0
			// contribution

			// Just some high value because this is a delta distribution
			// And also, take fresnel into account
			color = ColorRGB32F(MaterialConstants::DELTA_DISTRIBUTION_HIGH_VALUE) * (ColorRGB32F(1.0f) - fresnel_reflectance) * material.base_color;
			color /= hippt::abs(NoL);
			out_pdf = MaterialConstants::DELTA_DISTRIBUTION_HIGH_VALUE;
		}
		else
		{
			color	= ColorRGB32F(0.0f);
			out_pdf = 0.0f;
		}
	}
	else
	{
		float dot_prod	= HoL + HoV / relative_eta;
		float dot_prod2 = dot_prod * dot_prod;
		float denom		= dot_prod2 * NoL * NoV;

		float alpha_x;
		float alpha_y;
		MaterialUtils::get_alphas(roughness, material.anisotropy, alpha_x, alpha_y);

		float D	   = GGX_anisotropic(alpha_x, alpha_y, local_halfway_vector);
		float G1_V = G1_Smith(alpha_x, alpha_y, local_view_direction);
		float G1_L = G1_Smith(alpha_x, alpha_y, local_to_light_direction);
		float G2   = G1_V * G1_L;

		float dwm_dwi = hippt::abs(HoL) / dot_prod2;
		float D_pdf	  = G1_V / hippt::abs(NoV) * D * hippt::abs(HoV);
		out_pdf		  = dwm_dwi * D_pdf;

		// We added a check a few lines above to "avoid dividing by 0 later on". This is where.
		// When NoL is 0, denom is 0 too and we're dividing by 0.
		// The PDF of this case is as low as 1.0e-9 (light direction sampled perpendicularly to the normal)
		// so this is an extremely rare case.
		// The PDF being non-zero, we could actualy compute it, it's valid but not with floats :D
		color = material.base_color * D * (ColorRGB32F(1.0f) - fresnel_reflectance) * G2 * hippt::abs(HoL * HoV / denom);
	}

	// Account for non-symmetric scattering when refracting
	// Reference: https://www.pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF#Non-SymmetricScatteringandRefraction
	color /= hippt::square(relative_eta);

	return color;
}

HIPRT_DEVICE static float torrance_sparrow_GGX_pdf_refract(const DeviceUnpackedEffectiveMaterial& material,
														   float roughness,
														   float relative_eta,
														   const float3& local_view_direction,
														   const float3& local_to_light_direction,
														   const float3& local_halfway_vector,
														   BSDFIncidentLightInfo incident_light_info)
{
	float NoL = local_to_light_direction.z;
	float NoV = local_view_direction.z;
	float HoL = hippt::dot(local_to_light_direction, local_halfway_vector);
	float HoV = hippt::dot(local_view_direction, local_halfway_vector);

	ColorRGB32F color;
	if (MaterialUtils::is_perfectly_smooth(roughness) && PrincipledBSDFDeltaDistributionEvaluationOptimization == KERNEL_OPTION_TRUE)
	{
		// Fast path for specular glass
		bool incident_direction_is_perfect_refraction = hippt::dot(refract_ray(local_view_direction, make_float3(0.0f, 0.0f, 1.0f), relative_eta),
																   local_to_light_direction) > MaterialConstants::DELTA_DISTRIBUTION_ALIGNEMENT_THRESHOLD;
		if (incident_light_info == BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFRACT_LOBE && incident_direction_is_perfect_refraction)
			return MaterialConstants::DELTA_DISTRIBUTION_HIGH_VALUE;
		else
			return 0.0f;
	}
	else
	{
		float dot_prod	= HoL + HoV / relative_eta;
		float dot_prod2 = dot_prod * dot_prod;
		float denom		= dot_prod2 * NoL * NoV;

		float alpha_x;
		float alpha_y;
		MaterialUtils::get_alphas(roughness, material.anisotropy, alpha_x, alpha_y);

		float D	   = GGX_anisotropic(alpha_x, alpha_y, local_halfway_vector);
		float G1_V = G1_Smith(alpha_x, alpha_y, local_view_direction);
		float G1_L = G1_Smith(alpha_x, alpha_y, local_to_light_direction);
		float G2   = G1_V * G1_L;

		float dwm_dwi = hippt::abs(HoL) / dot_prod2;
		float D_pdf	  = G1_V / hippt::abs(NoV) * D * hippt::abs(HoV);

		return dwm_dwi * D_pdf;
	}
}

#endif
