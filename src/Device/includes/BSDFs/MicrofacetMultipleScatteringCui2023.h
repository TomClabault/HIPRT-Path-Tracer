/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_BSDFS_MICROFACET_MULTIPLE_SCATTERING_CUI2023_H
#define DEVICE_INCLUDES_BSDFS_MICROFACET_MULTIPLE_SCATTERING_CUI2023_H

#include "Device/includes/BSDFs/Fresnel.h"
#include "Device/includes/BSDFs/MicrofacetCommon.h"
#include "Device/includes/BSDFs/MicrofacetGGX.h"
#include "Device/includes/BSDFs/ThinFilm.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material/MaterialUtils.h"

HIPRT_DEVICE static float microfacet_GGX_pdf_reflect(float material_roughness,
													 float material_anisotropy,
													 const float3& local_view_direction,
													 const float3& local_to_light_direction,
													 const float3& local_halfway_vector,
													 SpecularDeltaReflectionSampled incident_light_direction_is_from_GGX_sample);

HIPRT_DEVICE static ColorRGB32F principled_metallic_fresnel(const DeviceUnpackedEffectiveMaterial& material,
															float incident_ior,
															float3 local_to_light_direction,
															float3 local_half_vector);

// TODO perf:
// Auto macro for get_g, get_l because that seems to be improving perf.
// Russian roulette
// Variable number of bounces depending on roughness and anisotropy
// Read GPT again after that and gemini and grok

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
		switch (i)
		{
		case 0:
			return g0;

		case 1:
			return g1;

		case 2:
			return g2;

		case 3:
			return g3;

		default:
			return g0;
		}
	}

	HIPRT_DEVICE fp16 get_lambda(int i) const
	{
		switch (i)
		{
		case 0:
			return lambda_0;

		case 1:
			return lambda_1;

		case 2:
			return lambda_2;

		case 3:
			return lambda_3;

		default:
			return lambda_0;
		}
	}

	HIPRT_DEVICE void set_g(int i, fp16 value)
	{
		switch (i)
		{
		case 0:
			g0 = value;
			break;

		case 1:
			g1 = value;
			break;

		case 2:
			g2 = value;
			break;

		case 3:
			g3 = value;
			break;

		default:
			g0 = value;
			break;
		}
	}

	HIPRT_DEVICE void set_l(int i, fp16 value)
	{
		switch (i)
		{
		case 0:
			lambda_0 = value;
			break;

		case 1:
			lambda_1 = value;
			break;

		case 2:
			lambda_2 = value;
			break;

		case 3:
			lambda_3 = value;
			break;

		default:
			lambda_0 = value;
			break;
		}
	}

private:
	int N  = 0;
	fp16 m = 1.0f;

	fp16 lambda_init = 0.0f;

	fp16 g0, g1, g2, g3;
	fp16 lambda_0, lambda_1, lambda_2, lambda_3;
};

// TODO do we need 2022 and 2023? Are they not the same when developing?
HIPRT_DEVICE float G1_Smith_lambda_signed_2023(float alpha_x, float alpha_y, const float3& local_direction)
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
											  const float3& local_view_direction,
											  const float3& local_to_light_direction)
{
	float local_half_vector_length = hippt::length(local_view_direction + local_to_light_direction);
	if (local_half_vector_length == 0.0f)
		return ColorRGB32F(0.0f);

	float3 local_half_vector = (local_view_direction + local_to_light_direction) / local_half_vector_length;
	ColorRGB32F F			 = principled_metallic_fresnel(material, incident_ior, local_to_light_direction, local_half_vector);

	return F * GGX_anisotropic(alpha_x, alpha_y, local_half_vector) / (4.0f * hippt::abs(local_view_direction.z));
}

/**
 * local_view_direction and local_to_light_direction should bot be pointing outward the surface here
 */
HIPRT_DEVICE ColorRGB32F
torrace_sparrow_GGX_multiple_scattering_invariance_eval_reflect(const DeviceUnpackedEffectiveMaterial& material,
																float material_roughness,
																float material_anisotropy,
																float incident_ior,
																ColorRGB32F F,
																float3 local_view_direction,	 // w_i in the paper
																float3 local_to_light_direction, // w_o in the paper
																Xorshift32Generator& rng,
																float& out_pdf,
																SpecularDeltaReflectionSampled incident_light_direction_is_from_GGX_sample)
{
	if (local_to_light_direction.z < 0.0f || local_view_direction.z < 0.0f)
		// A direction that is below the surface is invalid for a microfacet ** BRDF **
		return ColorRGB32F(0.0f);

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

	out_pdf = 1.0f;

	float alpha_x;
	float alpha_y;
	MaterialUtils::get_alphas(material_roughness, material_anisotropy, alpha_x, alpha_y);

	SegmentTerm s(G1_Smith_lambda_signed_2023(alpha_x, alpha_y, local_to_light_direction));

	float inverse_pdf = G1_Smith_lambda_signed_2023(alpha_x, alpha_y, -local_view_direction);
	s.add_bounce(inverse_pdf);

	ColorRGB32F weight = ColorRGB32F(1.0f);
	ColorRGB32F multiple_scattering_contribution =
							Cui_2023_vertex_term(material, incident_ior, alpha_x, alpha_y, local_view_direction, local_to_light_direction) * s.get_sk();
	float3 current_view_direction	  = local_view_direction;
	float3 current_to_light_direction = local_to_light_direction;

	for (int i = 1; i < PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces; ++i)
	{
		current_to_light_direction = microfacet_GGX_sample_reflection<false>(material_roughness, material_anisotropy, current_view_direction, rng, false);

		ColorRGB32F vertex_term = Cui_2023_vertex_term(material, incident_ior, alpha_x, alpha_y, current_view_direction, current_to_light_direction);

		float half_vector_length = hippt::length(current_view_direction + current_to_light_direction);
		if (half_vector_length == 0.0f)
			break;

		float3 half_vector = (current_view_direction + current_to_light_direction) / half_vector_length;
		weight *= principled_metallic_fresnel(material, incident_ior, current_view_direction, half_vector);

		float lambda = G1_Smith_lambda_signed_2023(alpha_x, alpha_y, current_to_light_direction);
		s.add_bounce(lambda);
		float s_k = s.get_sk();

		current_view_direction = -current_to_light_direction;
		multiple_scattering_contribution += Cui_2023_vertex_term(material, incident_ior, alpha_x, alpha_y, current_view_direction, local_to_light_direction) *
											weight * hippt::abs(inverse_pdf) * s_k;

		inverse_pdf *= lambda;
	}

	out_pdf = microfacet_GGX_pdf_reflect(material_roughness, material_anisotropy, local_view_direction, local_to_light_direction,
										 hippt::normalize(local_view_direction + local_to_light_direction), incident_light_direction_is_from_GGX_sample);

	return multiple_scattering_contribution / local_to_light_direction.z;
}

// HIPRT_DEVICE float G1_Smith_lambda_signed_2022(float alpha_x, float alpha_y, const float3& local_direction)
//{
//	float theta = acosf(local_direction.z);
//	float cosTheta = local_direction.z;
//	float sinTheta = sinf(theta);
//	float tanTheta = sinTheta / cosTheta;
//	const float invSinTheta2 = 1.0f / (1.0f - local_direction.z * local_direction.z);
//	const float cosPhi2 = local_direction.x * local_direction.x * invSinTheta2;
//	const float sinPhi2 = local_direction.y * local_direction.y * invSinTheta2;
//	float alpha = sqrtf(cosPhi2 * alpha_x * alpha_x + sinPhi2 * alpha_y * alpha_y);
//	float Lambda;
//	if (local_direction.z > 0.9999f)
//		Lambda = 0.0f;
//	else if (local_direction.z < -0.9999f)
//		Lambda = -1.0f;
//	else
//	{
//		const float a = 1.0f / tanTheta / alpha;
//		Lambda = 0.5f * (-1.0f + ((a > 0) ? 1.0f : -1.0f) * sqrtf(1 + 1 / (a * a)));
//	}
//
//	return Lambda;
// }
//
// HIPRT_DEVICE float computeG2_cor_middle(float alpha_x, float alpha_y, const float3& wi, const float3& wo)
//{
//	float inLambda	= G1_Smith_lambda_signed_2022(alpha_x, alpha_y, wi);
//	float outLambda = G1_Smith_lambda_signed_2022(alpha_x, alpha_y, wo);
//
//	const float Gtemp  = 1.0f / (abs(1.0f + inLambda) + outLambda);
//	const float Gtemp2 = 1.0f / (abs(1.0f + inLambda));
//	float G			   = Gtemp2 - Gtemp;
//
//	return G;
// }
//// height-correlated G2 for the last bounce
// HIPRT_DEVICE float computeG2_cor_last(float alpha_x, float alpha_y, const float3& wi, const float3& wo)
//{
//	float inLambda	= G1_Smith_lambda_signed_2022(alpha_x, alpha_y, wi);
//	float outLambda = G1_Smith_lambda_signed_2022(alpha_x, alpha_y, wo);
//	float temp		= (abs(1.0f + inLambda) + outLambda);
//	float G			= abs(temp) < 1e-10 ? 0.0 : 1.0f / temp;
//
//	return G;
// }
//
// HIPRT_DEVICE float computeG1(float alpha_x, float alpha_y, const float3& wi)
//{
//	float lambda = G1_Smith_lambda_signed_2022(alpha_x, alpha_y, wi);
//	float G11	 = 1.0f / abs(1.0f + lambda);
//	return G11;
// }
//
// HIPRT_DEVICE float computeG(float alpha_x, float alpha_y, const float3& wi, const float3& wo, bool outShadow)
//{
//	float G;
//
//	if (outShadow)
//		G = computeG2_cor_last(alpha_x, alpha_y, wi, wo);
//	else
//	{
//		if (wo.z < 0)
//			G = computeG1(alpha_x, alpha_y, wi);
//		else
//			G = computeG2_cor_middle(alpha_x, alpha_y, wi, wo);
//	}
//
//	return G;
// }
//
// HIPRT_DEVICE inline ColorRGB32F computeD_F(
//						const DeviceUnpackedEffectiveMaterial& material, float incident_ior, float alpha_x, float alpha_y, const float3& wi, const float3&
// wo)
//{
//	if (hippt::length(wo + wi) <= 0.0f)
//		return ColorRGB32F(0.0f);
//
//	/* Calculate the reflection half-vector */
//	float3 H = hippt::normalize(wo + wi);
//	float D	 = GGX_anisotropic(alpha_x, alpha_y, H);
//	if (D == 0)
//		return ColorRGB32F(0.0f);
//
//	/* Fresnel factor */
//	ColorRGB32F F = principled_metallic_fresnel(material, incident_ior, wo, H);
//
//	return D * F;
// }
//
// HIPRT_DEVICE inline ColorRGB32F computeD_F_withoutD(
//						const DeviceUnpackedEffectiveMaterial& material, float incident_ior, float alpha_x, float alpha_y, const float3& wi, const float3&
// wo)
//{
//	if (hippt::length(wo + wi) <= 0.0f)
//		return ColorRGB32F(0.0f);
//
//	float3 H = hippt::normalize(wo + wi);
//
//	/* Fresnel factor */
//	return principled_metallic_fresnel(material, incident_ior, wo, H);
// }
//
// HIPRT_DEVICE ColorRGB32F
// evalBounceLast(const DeviceUnpackedEffectiveMaterial& material, float incident_ior, float alpha_x, float alpha_y, const float3& wi, const float3& wo)
//{
//	ColorRGB32F result = computeD_F(material, incident_ior, alpha_x, alpha_y, wi, wo);
//
//	float G = computeG(alpha_x, alpha_y, wi, wo, true);
//	if (G == 0.0 || !hippt::is_finite(G))
//		return ColorRGB32F(0.0f);
//
//	result *= G / (4 * abs(wi.z));
//	return result;
// }
//
// HIPRT_DEVICE ColorRGB32F
// evalBounceSample(const DeviceUnpackedEffectiveMaterial& material, float incident_ior, float alpha_x, float alpha_y, const float3& wi, const float3& wo)
//{
//	ColorRGB32F result = computeD_F_withoutD(material, incident_ior, alpha_x, alpha_y, wi, wo);
//
//	float G = computeG(alpha_x, alpha_y, wi, wo, false);
//
//	if (G == 0.0 || !hippt::is_finite(G))
//		return ColorRGB32F(0.0f);
//
//	// the Jacbian term is alrady included in the sample
//	result *= G;
//
//	return result;
// }

// HIPRT_DEVICE ColorRGB32F
// torrace_sparrow_GGX_multiple_scattering_invariance_eval_reflect(const DeviceUnpackedEffectiveMaterial& material,
//																float material_roughness,
//																float material_anisotropy,
//																float incident_ior,
//																float3 local_view_direction,	 // w_i in the paper
//																float3 local_to_light_direction, // w_o in the paper
//																Xorshift32Generator& rng,
//																float& out_pdf,
//																SpecularDeltaReflectionSampled incident_light_direction_is_from_GGX_sample)
//{
//	/* Stop if this component was not requested */
//	if (local_view_direction.z <= 0 || local_to_light_direction.z <= 0)
//	{
//		out_pdf = 0.0f;
//
//		return ColorRGB32F(0.0f);
//	}
//
//	float alpha_x;
//	float alpha_y;
//	MaterialUtils::get_alphas(material_roughness, material_anisotropy, alpha_x, alpha_y);
//
//	float3 wi = local_view_direction;
//	float3 wo = local_to_light_direction;
//
//	float pdf  = 1;
//	float3 w0  = local_to_light_direction;
//	float3 woN = local_view_direction;
//	ColorRGB32F result(0.0f);
//
//	float3 bRec_wi = local_view_direction;
//	float3 bRec_wo = local_to_light_direction;
//
//	// for the single scattering
//	result += evalBounceLast(material, incident_ior, alpha_x, alpha_y, wi, wo);
//
//	ColorRGB32F weight = ColorRGB32F(1.0f);
//
//	for (int i = 1; i < PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces; i++)
//	{
//		bRec_wo			 = microfacet_GGX_sample_reflection<false>(material_roughness, material_anisotropy, bRec_wi, rng, false);
//		// Multiply by the Jacobian here because they don't include to Jacobian in the PDF in the paper code
//		float currentPDF = microfacet_GGX_pdf_reflect(material_roughness, material_anisotropy, bRec_wi, bRec_wo, hippt::normalize(bRec_wi + bRec_wo),
//													  incident_light_direction_is_from_GGX_sample) *
//						   (4.0f * hippt::abs(hippt::dot(hippt::normalize(bRec_wi + bRec_wo), bRec_wo)));
//		if (currentPDF < 1e-10f)
//			hippt::debugbreak();
//
//		pdf *= currentPDF;
//
//		weight *= evalBounceSample(material, incident_ior, alpha_x, alpha_y, bRec_wi, bRec_wo);
//		if (weight.has_nan_or_inf())
//			hippt::debugbreak();
//		if (pdf <= 1e-4f)
//			break;
//
//		// next event estimation
//		bRec_wi = -bRec_wo;
//		bRec_wo = woN;
//
//		ColorRGB32F currentWight = weight * evalBounceLast(material, incident_ior, alpha_x, alpha_y, bRec_wi, bRec_wo);
//		result += currentWight / pdf;
//
//		if (result.has_nan_or_inf())
//			hippt::debugbreak();
//
//		/*if (i + 1 >= m_rrDepth)
//		{
//
//			Float q = std::min(currentWight.max(), (Float)0.95f);
//			if (generateRandomNumber() >= q)
//				break;
//			weight /= q;
//		}*/
//	}
//
//	out_pdf = microfacet_GGX_pdf_reflect(material_roughness, material_anisotropy, local_view_direction, local_to_light_direction,
//										 hippt::normalize(local_view_direction + local_to_light_direction), incident_light_direction_is_from_GGX_sample);
//
//	return result / local_to_light_direction.z;
// }

// HIPRT_DEVICE float pdfVNDF(const float3& wi, const float3& wo, float alpha_x, float alpha_y)
//{
//	float D = GGX_anisotropic(alpha_x, alpha_y, hippt::normalize(wi + wo));
//	return D / (4 * hippt::abs(G1_Smith_lambda_signed_2023(alpha_x, alpha_y, -wi) * wi.z));
// }
//
// HIPRT_DEVICE float pdfWi(const float3& wi, const float3& wo, float inLamda, float alpha_x, float alpha_y)
//{
//	float3 m = hippt::normalize(wo + wi);
//
//	float G1 = 1.0f / (1 + inLamda);
//	float D	 = GGX_anisotropic(alpha_x, alpha_y, m);
//
//	float lambda = G1_Smith_lambda_signed_2023(alpha_x, alpha_y, -wi);
//
//	float pdf = D / (4 * hippt::abs(lambda) * hippt::abs(wi.z));
//
//	return pdf;
// }
//
// HIPRT_DEVICE float3 microfacet_GGX_multiple_scattering_invariance_sample_reflection(const float3& local_view_direction, // w_i in the paper
//																					float material_roughness,
//																					float material_anisotropy,
//																					Xorshift32Generator& rng)
//{
//	if (local_view_direction.z < 0.0f)
//		// A direction that is below the surface is invalid for a microfacet ** BRDF **
//		return float3(0.0f);
//
//	float alpha_x;
//	float alpha_y;
//	MaterialUtils::get_alphas(material_roughness, material_anisotropy, alpha_x, alpha_y);
//
//	float pdf = 1.0f;
//	ColorRGB32F weightAcc(1.0f);
//	float3 current_view_direction  = local_view_direction;
//	float3 next_to_light_direction = microfacet_GGX_sample_reflection<false>(material_roughness, material_anisotropy, current_view_direction, rng);
//	// float inLamda				   = G1_Smith_lambda_signed_2023(alpha_x, alpha_y, current_view_direction);
//
//	for (int i = 1; i < PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces; i++)
//	{
//		// float currentPDF = pdfVNDF(current_view_direction, next_to_light_direction, alpha_x, alpha_y);
//		float currentPDF = microfacet_GGX_pdf_reflect(material_roughness, material_anisotropy, current_view_direction, next_to_light_direction,
//													  hippt::normalize(current_view_direction + next_to_light_direction),
//													  SpecularDeltaReflectionSampled::SPECULAR_PEAK_SAMPLED);
//		// pdf *= currentPDF;
//
//		if (currentPDF < 0.0f)
//		{
//			hippt::debugbreak();
//
//			// TODO is this hit often?
//			// TODO eval should work even wityhouth that fancy sampling routuibne
//			return float3(0.0f);
//		}
//
//		// ColorRGB32F weight;
//		// float3 to_light_direction = next_to_light_direction;
//		// float outLamda			  = G1_Smith_lambda_signed(alpha_x, alpha_y, to_light_direction);
//		// float mapOutLamda		  = abs(outLamda + 1) - 1;
//
//		// TODO pass these as arguments
//		// DeviceUnpackedEffectiveMaterial material;
//		// float incident_ior = 1.0f;
//
//		// weight = Cui_2023_vertex_term(material, incident_ior, alpha_x, alpha_y, current_view_direction, to_light_direction) / currentPDF;
//		// weightAcc *= weight;
//
//		// to_light_direction	   = current_view_direction;
//		// current_view_direction = next_to_light_direction;
//
//		current_view_direction = next_to_light_direction;
//		// inLamda				   = abs(outLamda) - 1; //
//
//		next_to_light_direction = microfacet_GGX_sample_reflection<false>(material_roughness, material_anisotropy, current_view_direction, rng);
//
//		/*if (m_rrDepth > -1 && i + 1 >= m_rrDepth)
//		{
//			Float q = std::min(weightAcc.max(), (Float)0.95f);
//
//			if (generateRandomNumber() > q)
//			{
//				path.add(0.0);
//				break;
//			}
//			else
//			{
//				weightAcc /= q;
//			}
//		}*/
//	}
//
//	return next_to_light_direction;
// }

#endif
