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

HIPRT_DEVICE float hapke_isotropic(float HoX, float alpha_hapke)
{
	return (1.0f + 2.0f * HoX) / (1.0f + 2.0f * hippt::sqrt(1.0f - alpha_hapke) * HoX);
}

HIPRT_DEVICE float pdf_multiscatter_approx(float material_roughness,
										   float material_anisotropy,
										   float3_t local_view_direction,
										   float3_t local_to_light_direction,
										   float3_t local_half_vector,
										   SpecularDeltaReflectionSampled incident_light_direction_is_from_GGX_sample)

{
	float alpha_x;
	float alpha_y;
	MaterialUtils::get_alphas(material_roughness, material_anisotropy, alpha_x, alpha_y);

	float pdf_VNDF = microfacet_GGX_pdf_reflect(material_roughness, material_anisotropy, local_view_direction, local_to_light_direction,
												hippt::normalize(local_view_direction + local_to_light_direction), incident_light_direction_is_from_GGX_sample);

	float alpha_hapke	   = (alpha_x + alpha_y) / 2.0f;
	float multiscatter_pdf = alpha_hapke / (4.0f * hippt::M_Pi) *
							 (hapke_isotropic(local_view_direction.z, alpha_hapke) * hapke_isotropic(local_to_light_direction.z, alpha_hapke) - 1.0f) /
							 (local_view_direction.z + local_to_light_direction.z);

	if (pdf_VNDF + multiscatter_pdf <= 0.0f)
		hippt::debugbreak();

	return pdf_VNDF + multiscatter_pdf;
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

#if PrincipledBSDFMultipleScatteringCuiSampleMultiscatter == KERNEL_OPTION_TRUE
	out_pdf = pdf_multiscatter_approx(material_roughness, material_anisotropy, local_view_direction, local_to_light_direction,
									  hippt::normalize(local_view_direction + local_to_light_direction), incident_light_direction_is_from_GGX_sample);
#else
	out_pdf = microfacet_GGX_pdf_reflect(material_roughness, material_anisotropy, local_view_direction, local_to_light_direction,
										 hippt::normalize(local_view_direction + local_to_light_direction), incident_light_direction_is_from_GGX_sample);
#endif // PrincipledBSDFMultipleScatteringCuiSampleMultiscatter

	return multiple_scattering_contribution / local_to_light_direction.z;
}

/**
 * This function samples a path in the microsurface along with its weight (what eval() would have returned for that path) and its PDF. This function does all
 * the sampling + eval + PDFat once because of the nature of the integrator. We can't just sample() one path here and then feed it to the eval() function
 * because the eval() function is stochastic and there is no chance that it will replay the same path, leading to an eval() that is different from the true
 * value of the sampled path. One solution could be to return literally a full path from this function and then re-evaluate that given path with an eval_path()
 * function but storing paths like that on the GPU is way too costly. So instead, that sample function only returns a 'to_light_direction'. And because it only
 * returns a to_light_direction, we lose the information of what bounces in the microsurface we got between the view_direction and the to_light_direction such
 * that the eval() cannot replay the path properly. This is why this function does the eval() itself. It also does the PDF itself because the PDF of this
 * sampled path is a marginal PDF that we cannot estimate reliably outside of the sampling function.
 */
HIPRT_DEVICE float3_t microfacet_GGX_multiple_scattering_invariance_sample_reflection(const float3_t& local_view_direction, // w_i in the paper
																					  float material_roughness,
																					  float material_anisotropy,
																					  Xorshift32Generator& rng,
																					  float* DEBUGOUTPDF,
																					  ColorRGB32F* DEBUGOUTPUTEVAL,
																					  DeviceUnpackedEffectiveMaterial* material,
																					  float* incident_ior)
{
	float alpha_x;
	float alpha_y;
	MaterialUtils::get_alphas(material_roughness, material_anisotropy, alpha_x, alpha_y);

	// TODO how to compute the final weight with s_k while not getting 0 at roughness 0? Are we missing 1 add_bounce?
	ColorRGB32F w = ColorRGB32F(1.0f); // accumulated vertex weights
	float p		  = 1.0f;			   // accumulated path probability

	std::vector<float3_t> directions(PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces + 1);
	directions[0] = -local_view_direction;

	// DEBUG block just to reproduce when debugging
	{
		unsigned int seed_before = rng.m_state.seed;
		rng.m_state.seed		 = seed_before;
	}

#define LOGGING 0

#if LOGGING == 1
#define LOG(...) std::cerr << __VA_ARGS__
#else
#define LOG(...)
#endif

	LOG("Input view-direction to the function (facing away from surface): (" << local_view_direction.x << ", " << local_view_direction.y << ","
																			 << local_view_direction.z << ")\n"
																			 << std::endl);

	int actual_bounces = 0;
	float3_t current   = local_view_direction;
	for (int bounce = 1; bounce <= PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces; ++bounce)
	{
		// Sample next direction + its VNDF pdf + its vertex weight wk
		float3_t dk = microfacet_GGX_sample_reflection<false>(material_roughness, material_anisotropy, current, rng, false);

		float half_vec_length;
		float3_t half_vector = current + dk;
		if ((half_vec_length = hippt::length(half_vector)) <= 1.0e-8f)
		{
			*DEBUGOUTPDF = 0.0f;

			return make_float3(0.0f, 0.0f, 0.0f);
		}
		float pk = microfacet_GGX_pdf_reflect(material_roughness, material_anisotropy, current, dk, half_vector / half_vec_length,
											  SpecularDeltaReflectionSampled::SPECULAR_PEAK_SAMPLED, false);
		if (pk <= 0.0f)
			hippt::debugbreak();

		ColorRGB32F wk = Cui_2023_vertex_term(*material, *incident_ior, alpha_x, alpha_y, current, dk);

		LOG("Bounce " << bounce << ":\n\tCurrent view direction: (" << current.x << ", " << current.y << "," << current.z << ")\n\tSampled direction: (" << dk.x
					  << ", " << dk.y << ", " << dk.z << "),\n\n\tPDF: " << pk << ",\n\tvertex weight: (" << wk.r << ", " << wk.g << ", " << wk.b << ")\n"
					  << std::endl);
		LOG("\tCurrent weight / pdf: " << w.r << ", " << w.g << ", " << w.b << " / " << p << " = (" << w.r / p << ", " << w.g / p << ", " << w.b / p << ")\n\n"
									   << std::endl);

		w *= wk;
		p *= pk;

		// segment.add_bounce(G1_Smith_lambda_signed_2023(alpha_x, alpha_y, dk));
		directions[++actual_bounces] = dk;

		if (dk.z > 0.0f)
		{
			LOG("\tRay is leaving the microgeometry at bounce " << bounce << ". G1 is: ");

			// Ray is leaving the microgeometry, continue only with the probability that the bounce is occluded/shadowed by the microgeometry
			float G1 = G1_Smith(alpha_x, alpha_y, dk);
			LOG(G1 << std::endl);

			float r = rng();
			if (r < G1)
			{
				p *= G1;

				current = dk;
				LOG("\'tCurrent' becomes: (" << current.x << ", " << current.y << "," << current.z << ")\n" << std::endl);

				LOG("\tRay not shadowed by the microgeometry. Leaving loop." << std::endl);

				break;
			}
			else
			{
				LOG("\tRay is leaving the microgeometry at bounce " << bounce << " but is shadowed by the microgeometry. Continuing loop with next bounce."
																	<< std::endl);

				p *= (1.0f - G1);
			}
		}

		current = dk;
		LOG("\tCurrent becomes: (" << current.x << ", " << current.y << "," << current.z << ")\n" << std::endl);
	}

	float3_t final_half_vector = hippt::normalize(current + local_view_direction);
	LOG("\nExited loop after " << actual_bounces - 1 << " bounces.\n\tFinal sampled direction: (" << current.x << ", " << current.y << "," << current.z
							   << ").\n\tFinal half vector: "
							   << "(" << final_half_vector.x << ", " << final_half_vector.y << "," << final_half_vector.z << ")\n"
							   << std::endl);

	LOG("Initializing SegmentTerm with lambda: " << G1_Smith_lambda_signed_2023(alpha_x, alpha_y, current) << std::endl);

	float Sk;

	if (current.z < 0.0f)
	{
		// If the last bouce of the path is pointing inside the surface, that's 0 contribution, Eq. 14 of the paper
		*DEBUGOUTPDF = 0.0f;

		LOG("Last bounce is going below the surface, returning 0 contribution." << std::endl);

		return make_float3(0.0f, 0.0f, 0.0f);
	}
	else
	{
		SegmentTerm segment(G1_Smith_lambda_signed_2023(alpha_x, alpha_y, current));
		for (int i = actual_bounces - 1; i >= 0; --i) // add d_{k-1} ... d_0
		{
			LOG("Adding bounce to SegmentTerm with lambda: " << G1_Smith_lambda_signed_2023(alpha_x, alpha_y, directions.at(i)) << std::endl);

			segment.add_bounce(G1_Smith_lambda_signed_2023(alpha_x, alpha_y, directions.at(i)));
		}
		/*for (int i = 0; i < actual_bounces; i++)
		{
			LOG("Adding bounce to SegmentTerm with lambda: " << G1_Smith_lambda_signed_2023(alpha_x, alpha_y, directions.at(i)) << std::endl);

			segment.add_bounce(G1_Smith_lambda_signed_2023(alpha_x, alpha_y, directions.at(i)));
		}*/
		Sk = segment.get_sk();
	}

	// Final step
	ColorRGB32F final_weight = w / p * Sk;
	LOG("Final weight = w / p * Sk = (" << w.r << ", " << w.g << ", " << w.b << ") / " << p << " * " << Sk << " = (" << final_weight.r << ", " << final_weight.g
										<< ", " << final_weight.b << ")" << std::endl);

	if (!final_weight.is_finite() || !hippt::is_finite(p) || !ColorRGB32F(w / current.z).is_finite())
		hippt::debugbreak();

	// Division by current.z here because the integrator expects BRDF to return their values without cos_theta included
	*DEBUGOUTPUTEVAL = final_weight / current.z;
	// The integrator divides by the PDF but the PDF is already included in final_weight so we don't want to divide by anything / divide by 1.0f in the
	// integrator
	*DEBUGOUTPDF = 1.0f;

	// And return final to light direction
	return current;
}

// HIPRT_DEVICE float G1_Smith_lambda_signed_2022(float alpha_x, float alpha_y, const float3_t& local_direction)
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
// HIPRT_DEVICE float computeG2_cor_middle(float alpha_x, float alpha_y, const float3_t& wi, const float3_t& wo)
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
// HIPRT_DEVICE float computeG2_cor_last(float alpha_x, float alpha_y, const float3_t& wi, const float3_t& wo)
//{
//	float inLambda	= G1_Smith_lambda_signed_2022(alpha_x, alpha_y, wi);
//	float outLambda = G1_Smith_lambda_signed_2022(alpha_x, alpha_y, wo);
//	float temp		= (abs(1.0f + inLambda) + outLambda);
//	float G			= abs(temp) < 1e-10 ? 0.0 : 1.0f / temp;
//
//	return G;
// }
//
// HIPRT_DEVICE float computeG1(float alpha_x, float alpha_y, const float3_t& wi)
//{
//	float lambda = G1_Smith_lambda_signed_2022(alpha_x, alpha_y, wi);
//	float G11	 = 1.0f / abs(1.0f + lambda);
//	return G11;
// }
//
// HIPRT_DEVICE float computeG(float alpha_x, float alpha_y, const float3_t& wi, const float3_t& wo, bool outShadow)
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
//						const DeviceUnpackedEffectiveMaterial& material, float incident_ior, float alpha_x, float alpha_y, const float3_t& wi, const float3_t&
// wo)
//{
//	if (hippt::length(wo + wi) <= 0.0f)
//		return ColorRGB32F(0.0f);
//
//	/* Calculate the reflection half-vector */
//	float3_t H = hippt::normalize(wo + wi);
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
//						const DeviceUnpackedEffectiveMaterial& material, float incident_ior, float alpha_x, float alpha_y, const float3_t& wi, const float3_t&
// wo)
//{
//	if (hippt::length(wo + wi) <= 0.0f)
//		return ColorRGB32F(0.0f);
//
//	float3_t H = hippt::normalize(wo + wi);
//
//	/* Fresnel factor */
//	return principled_metallic_fresnel(material, incident_ior, wo, H);
// }
//
// HIPRT_DEVICE ColorRGB32F
// evalBounceLast(const DeviceUnpackedEffectiveMaterial& material, float incident_ior, float alpha_x, float alpha_y, const float3_t& wi, const float3_t& wo)
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
// evalBounceSample(const DeviceUnpackedEffectiveMaterial& material, float incident_ior, float alpha_x, float alpha_y, const float3_t& wi, const float3_t& wo)
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
//																float3_t local_view_direction,	 // w_i in the paper
//																float3_t local_to_light_direction, // w_o in the paper
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
//	float3_t wi = local_view_direction;
//	float3_t wo = local_to_light_direction;
//
//	float pdf  = 1;
//	float3_t w0  = local_to_light_direction;
//	float3_t woN = local_view_direction;
//	ColorRGB32F result(0.0f);
//
//	float3_t bRec_wi = local_view_direction;
//	float3_t bRec_wo = local_to_light_direction;
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

// HIPRT_DEVICE float pdfVNDF(const float3_t& wi, const float3_t& wo, float alpha_x, float alpha_y)
//{
//	float D = GGX_anisotropic(alpha_x, alpha_y, hippt::normalize(wi + wo));
//	return D / (4 * hippt::abs(G1_Smith_lambda_signed_2023(alpha_x, alpha_y, -wi) * wi.z));
// }
//
// HIPRT_DEVICE float pdfWi(const float3_t& wi, const float3_t& wo, float inLamda, float alpha_x, float alpha_y)
//{
//	float3_t m = hippt::normalize(wo + wi);
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

#endif
