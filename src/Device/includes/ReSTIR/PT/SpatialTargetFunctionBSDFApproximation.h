/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_TARGET_FUNCTION_BSDF_APPROXIMATION_H
#define DEVICE_RESTIR_PT_SPATIAL_TARGET_FUNCTION_BSDF_APPROXIMATION_H

#include "Device/includes/BSDFs/BSDFDispatcher.h"
#include "Device/includes/GBufferDevice.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/Sampling.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * Approximation of the BSDF in the target function of the union domain reuse of ReSTIR PT. It is used by:
 *	- the MIS weights: canonical and reverse probes (ReSTIR_PT_evaluate_target_function_at_pixel()) and own terms
 *	  (ReSTIR_PT_target_function_bsdf_approximation_own_term(): a sample at its own pixel, a neighbor's sample at the center pixel, the canonical sample at
 *	  the center pixel)
 *	- the resampling weights, times the luminance of the radiance of the sample (ReSTIR_PT_resampling_target_function_bsdf_approximation()): the neighbors'
 *	  samples are only evaluated with the full BSDFs at the center pixel for the pairs that take the fallback
 *	- the selection target of stage 1 of the neighbor selection: the analytic lobe of the center pixel (spmis_union_domain_center_bsdf_setup() in
 *	  SpatialSPMISUnionDomain.h)
 *
 * The lobes are stored quantized so that the cheap path below reads no material: one 8-byte record per pixel and the padding bytes of the sample (see
 * ReSTIR_PT_target_function_bsdf_approximation_encode_pixel_lobe() and ReSTIR_PT_target_function_bsdf_approximation_encode_sample_lobe())
 *
 * INVARIANT: for every sample and every pixel, the approximation is > 0 exactly when the target function is > 0, and the probes and the own terms use the same
 * function of (sample, pixel). The group-wise MIS weights then stay a partition of unity and the spatial reuse stays unbiased, whatever the approximation
 * values are on that support.
 *
 * The flags below (pixel flags in the padding byte of GBufferDevice::shading_normals, written by CameraRays; sample flags in
 * ReSTIRPTReservoirSample::target_function_bsdf_approximation_flags) select the "cheap path" for a pair (sample, pixel): both the pixel material and the
 * reconnection vertex material are in an analytic class whose BSDF is strictly positive for every pair of directions above the shading normal with cosines
 *above RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_THRESHOLD (the support of the target function is then decided by the geometry alone).
 *
 * The approximation is then:
 *	- Lambertian BSDF: the cosine at the pixel, zero when the pixel is below the shading normal of the reconnection vertex
 *	- principled BSDF: the analytic lobe value at the pixel (one 8-byte record per pixel, GBufferDevice::target_function_bsdf_approximation_lobe_records)
 * times the analytic lobe value at the reconnection vertex (stored in the sample), 1 for DI samples. No material fetch, no BSDF evaluation
 *	- other BSDF models: no material is in the analytic class
 *
 * Every other pair takes the "fallback": the full target function evaluation, the value is then the full target function divided by the luminance of the
 * radiance of the sample (principled) or the cosine at the pixel on the support of the full target function (Lambertian).
 *
 * The analytic classes and lobes MIRROR THE ZERO CASES AND THE VALUES OF THE BSDF CODE BELOW. Any change to one of these functions or kernel options must
 * be checked against ReSTIR_PT_target_function_bsdf_approximation_material_class(), ReSTIR_PT_target_function_bsdf_approximation_lobe_from_material(),
 * ReSTIR_PT_target_function_bsdf_approximation_lobe_value() and ReSTIR_PT_target_function_bsdf_approximation_delta_peak_possible() (an approximation
 * that is zero where the target function is not, or the opposite, biases the spatial reuse):
 *	- lambertian_brdf_eval()
 *	- principled_bsdf_eval(), internal_eval_glossy_base(), internal_eval_specular_layer()
 *	- principled_specular_eval(), principled_specular_fresnel(), principled_specular_compute_darkening()
 *	- get_principled_energy_compensation_glossy_base()
 *	- internal_eval_metal_layer(), principled_metallic_eval(), principled_metallic_fresnel()
 *	- torrance_sparrow_GGX_eval_reflect_single(), torrace_sparrow_GGX_multiple_scattering_invariance_eval_reflect()
 *	- DeviceUnpackedPrincipledFullMaterial::is_specular_delta_reflection_sampled()
 *	- compute_principled_lobe_weights()
 *	- kernel options: BSDFKernelModel, PrincipledBSDFDiffuseLobe, PrincipledBSDFDeltaDistributionEvaluationOptimization, PrincipledBSDFDoEnergyCompensation,
 *	  PrincipledBSDFDoSpecularEnergyCompensation, PrincipledBSDFDoMetallicEnergyCompensation, PrincipledBSDFEnergyCompensationMode
 *
 * Analytic classes of the principled BSDF:
 *	- dielectric: opaque dielectric without coat, sheen or thin film (metallic, specular and diffuse transmission 0), ior >= 1, specular in [0, 1], base
 *	  color > 0 on the three channels, anisotropy 0, Lambertian diffuse lobe, perfectly smooth or not
 *	- metal: metallic 1, no second roughness, no retro reflection, no coat / sheen / thin film / transmission and, when not perfectly smooth, anisotropy 0,
 *	  F82 = F90 = (1, 1, 1), F90 falloff exponent 5 and every base color channel >= 1e-6 (the metal Fresnel is then Schlick with F0 = base color)
 *	- metallic blend: metallic in (0, 1), the conditions of the dielectric class, no second roughness, no retro reflection and, when not perfectly smooth,
 *	  the default metal Fresnel of the metal class. Its lobe has the metal form, see ReSTIR_PT_target_function_bsdf_approximation_lobe_from_material()
 *
 * Lobe value at a pair of directions (V, L) around the shading normal n, NoV and NoL above RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_THRESHOLD:
 *
 *		value = NoL * [Cs * F(LoH) D(H) G2 / (4 NoV NoL) + Cd0 + Cd1 * (1 - F_d(NoL)) * (reconnection side: (1 - F_d(NoV)), pixel side: 1)]
 *
 * with the isotropic GGX D and Smith G2 of the code (alpha = roughness^2), F = full_fresnel_dielectric(., ior) (dielectric) or the Schlick luminance
 * lumF0 + (1 - lumF0) (1 - LoH)^5 (metal).
 *
 * Pixel side (V fixed, folded): Cs = s lum(tint) / E, Cd0 = lum(base) (1 - s) / (pi E), Cd1 = lum(base darkening) s (1 - F_d(NoV)) / (pi E) with E the
 * glossy base energy compensation at the NoV of the pixel. Reconnection side (L fixed): the same without E and without the (1 - F_d(NoV)) factor.
 * Metal: Cs = 1, Cd0 = Cd1 = 0.
 *
 * A perfectly smooth lobe has Cs = 0 (its value is 0 off the delta peak), a pair whose directions may hit the delta peak goes to the fallback
 * (ReSTIR_PT_target_function_bsdf_approximation_delta_peak_possible())
 */

// Bit 0 of the pixel flags and of the sample flags: the material is in an analytic class (and the other conditions of
// ReSTIR_PT_target_function_bsdf_approximation_pixel_flags() / ReSTIR_PT_target_function_bsdf_approximation_sample_flags() hold)
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_FLAG 1u
// Bit 1 of the pixel flags and of the sample flags: the analytic lobe is perfectly smooth (its delta peak can return the delta value)
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_SMOOTH_LOBE_FLAG 2u

// The cosines of the cheap path (NoV and NoL at both BSDFs) must be above this threshold, smaller positive cosines go to the fallback. 0 for the Lambertian
// BRDF (exact zero test), 1.0e-3 for the principled BSDF (keeps the Fresnel transmission of the specular layer away from its float rounding to 0 at grazing
// angles)
#if BSDFKernelModel == BSDF_PRINCIPLED
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_THRESHOLD 1.0e-3f
#else
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_THRESHOLD 0.0f
#endif // #if BSDFKernelModel == BSDF_PRINCIPLED

// Analytic classes of the materials, see the top of this file
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE			  0
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_DIELECTRIC	  1
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_METAL		  2
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_METALLIC_BLEND 3

// Decisions of ReSTIR_PT_target_function_bsdf_approximation_decision() for a pair (sample, pixel)
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_FALLBACK	0
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CHEAP_PATH 1
// Cheap path with a zero approximation: the pixel is below the shading normal of the reconnection vertex
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_ZERO 2

// The cheap path is left for the fallback when the light direction is within this cosine of the mirror direction of a perfectly smooth lobe that can
// return its delta value (wider than MaterialConstants::DELTA_DISTRIBUTION_ALIGNEMENT_THRESHOLD = 0.99999 of the code, world space here against the local
// frame of the code)
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_DELTA_ALIGNMENT_THRESHOLD 0.9999f
// Incident light infos for which DeviceUnpackedPrincipledFullMaterial::is_specular_delta_reflection_sampled() can answer SPECULAR_PEAK_SAMPLED for a
// perfectly smooth base roughness: glass reflection, first / second metal, retro reflection, coat, specular (all of them, whatever the lobe evaluated)
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_DELTA_TRIGGER_INFOS                                                                                       \
	(BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_COAT_LOBE | BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_FIRST_METAL_LOBE |                    \
	 BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_SECOND_METAL_LOBE | BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_RETRO_REFLECTION_LOBE |       \
	 BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_SPECULAR_LOBE | BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFLECT_LOBE)

// The principled approximation values are floored at this value where the target function is known to be > 0 so that their support stays exact
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_VALUE_FLOOR 1.0e-30f
// The cosine of the Lambertian own terms is floored at this value where the target function is known to be > 0 (a stored target function evaluated with a
// cosine that rounds to 0 here)
#define RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_FLOOR 1.0e-8f

/**
 * Analytic lobe of a material (quantized or not), see the top of this file. 'fresnel_parameter' is the ior of a dielectric, the luminance of F0 of a metal
 */
struct ReSTIRPTTargetFunctionBSDFApproximationLobe
{
	float specular_scale	= 0.0f;
	float diffuse_constant	= 0.0f;
	float diffuse_fresnel	= 0.0f;
	float alpha				= 1.0f;
	float fresnel_parameter = 1.0f;
	bool metal				= false;
};

/**
 * Whether the union domain reuse can take the cheap path of the approximation for this render: the spatial reuse takes the union domain path of ReSTIR PT
 * (and, with the principled BSDF, the lobe records are allocated). CameraRays and the initial candidates only write the flags and the lobes when this is
 * true, they stay zero otherwise
 *
 * The union domain spatial reuse kernel does not call this function: it is only launched when the spatial reuse takes the union domain path. Without the
 * flags and the lobes (this function returned false when they were written), no pair (sample, pixel) is on the cheap path: every pair takes the fallback
 * and the lobe records are not read
 */
HIPRT_DEVICE bool ReSTIR_PT_target_function_bsdf_approximation_enabled(const HIPRTRenderData& render_data)
{
#if PathSamplingStrategyIsReSTIRPT
	bool enabled = render_data.render_settings.restir_pt_settings.do_spmis_union_domain_reuse(ReSTIR_PT_MISWeightsType);

#if BSDFKernelModel == BSDF_PRINCIPLED
	enabled = enabled && render_data.g_buffer.target_function_bsdf_approximation_lobe_records != nullptr;
#endif // #if BSDFKernelModel == BSDF_PRINCIPLED

	return enabled;
#else
	return false;
#endif // #if PathSamplingStrategyIsReSTIRPT
}

/**
 * Whether the glossy base of the principled BSDF of 'material' (internal_eval_glossy_base(): specular layer over the diffuse layer) is strictly positive for
 * every pair of directions above the shading normal with cosines above RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_THRESHOLD, outside of the
 * object (volume state checked by the callers, no refraction): an opaque dielectric base without coat, sheen or thin film.
 *
 * The diffuse layer (Lambertian, PrincipledBSDFDiffuseLobe) is then base_color / pi times its diffuse weight (1 - metallic, no transmission) times the layers
 * throughput of the specular layer only, lerp(1, (1 - F(NoL)) (1 - F(NoV)) darkening, specular): >= 1 - specular for specular < 1, and > 0 for specular = 1
 * because the dielectric Fresnel of an IOR >= 1 (the relative IOR of the specular layer without a coat is the material IOR) stays below 1 above the cosine
 * threshold.
 *
 * The specular GGX lobe adds a non-negative term, the glossy base energy compensation divides by a positive LUT value, the clearcoat compensation is 1
 * without a coat
 *
 * Forced inline: out of line, the material passed by reference would live in scratch memory in the initial candidates kernels, and every field read here
 * would be a scratch load
 */
HIPRT_DEVICE HIPRT_FORCEINLINE bool ReSTIR_PT_target_function_bsdf_approximation_strictly_positive_glossy_base(
	const DeviceUnpackedPrincipledFullMaterial& material)
{
#if PrincipledBSDFDiffuseLobe != PRINCIPLED_DIFFUSE_LOBE_LAMBERTIAN
	return false;
#endif // #if PrincipledBSDFDiffuseLobe != PRINCIPLED_DIFFUSE_LOBE_LAMBERTIAN

	if (material.coat != 0.0f || material.sheen != 0.0f)
		return false;

	if (material.specular_transmission != 0.0f || material.diffuse_transmission != 0.0f || material.thin_film != 0.0f)
		return false;

	if (material.ior < 1.0f || material.specular < 0.0f || material.specular > 1.0f)
		return false;

	return material.base_color.r > 0.0f && material.base_color.g > 0.0f && material.base_color.b > 0.0f && material.anisotropy == 0.0f;
}

/**
 * Analytic class of 'material' (RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE when it is not in one) and whether its lobe is perfectly smooth
 * ('out_smooth_lobe', only meaningful for a class). Mirrors the zero cases of the BSDF model compiled in, see the top of this file
 */
HIPRT_DEVICE int ReSTIR_PT_target_function_bsdf_approximation_material_class(const DeviceUnpackedPrincipledFullMaterial& material, bool& out_smooth_lobe)
{
	out_smooth_lobe = false;

#if BSDFKernelModel == BSDF_LAMBERTIAN
	// lambertian_brdf_eval(): 0 if NoV <= 0 or NoL <= 0, base_color / pi otherwise
	bool positive_base_color = material.base_color.r > 0.0f && material.base_color.g > 0.0f && material.base_color.b > 0.0f;

	return positive_base_color ? RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_DIELECTRIC : RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE;
#elif BSDFKernelModel == BSDF_PRINCIPLED
	bool smooth_lobe = MaterialUtils::is_perfectly_smooth(material.roughness);
	if (smooth_lobe && PrincipledBSDFDeltaDistributionEvaluationOptimization != KERNEL_OPTION_TRUE)
		// The perfectly smooth lobes are then evaluated as GGX lobes with a tiny alpha, not handled
		return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE;

	if (material.metallic == 0.0f)
	{
		if (!ReSTIR_PT_target_function_bsdf_approximation_strictly_positive_glossy_base(material))
			return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE;

		out_smooth_lobe = smooth_lobe;

		return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_DIELECTRIC;
	}

	// The metal class (metallic 1) and the metallic blend class (metallic in (0, 1)) share the conditions of the metal layer below
	if (material.metallic < 0.0f || material.metallic > 1.0f || material.second_roughness_weight != 0.0f || material.retro_reflection != 0.0f)
		return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE;

	if (material.coat != 0.0f || material.sheen != 0.0f || material.thin_film != 0.0f)
		return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE;

	// Zero weights for metallic = 1 in compute_principled_lobe_weights(), excluded anyway (volume state of the target function). A metallic blend with a
	// transmission lobe has no analytic lobe
	if (material.specular_transmission != 0.0f || material.diffuse_transmission != 0.0f)
		return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE;

	if (!smooth_lobe)
	{
		if (material.anisotropy != 0.0f)
			return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE;

		bool default_fresnel = material.metallic_F82.r == 1.0f && material.metallic_F82.g == 1.0f && material.metallic_F82.b == 1.0f &&
							   material.metallic_F90.r == 1.0f && material.metallic_F90.g == 1.0f && material.metallic_F90.b == 1.0f &&
							   material.metallic_F90_falloff_exponent == 5.0f;
		if (!default_fresnel)
			return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE;

		if (material.base_color.r < 1.0e-6f || material.base_color.g < 1.0e-6f || material.base_color.b < 1.0e-6f)
			return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE;
	}

	out_smooth_lobe = smooth_lobe;

	if (material.metallic == 1.0f)
		return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_METAL;

	// Blend of the metal layer (weight metallic) and of the glossy base (specular weight (1 - metallic) specular, diffuse weight 1 - metallic) in
	// compute_principled_lobe_weights(): the glossy base alone is strictly positive under the conditions of the dielectric class, scaled by
	// 1 - metallic > 0, and the metal layer adds a non-negative term (Schlick Fresnel of a base color > 0, or 0 off the peak of a perfectly smooth
	// lobe). The metal conditions above only keep its value close to the Schlick lobe of the approximation
	if (!ReSTIR_PT_target_function_bsdf_approximation_strictly_positive_glossy_base(material))
		return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE;

	return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_METALLIC_BLEND;
#else
	return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE;
#endif // #if BSDFKernelModel == BSDF_LAMBERTIAN
}

/**
 * Unquantized lobe of 'material' (class 'material_class'). 'pixel_side': V is the view direction of the pixel with cosine 'view_cosine', the (1 - F_d(NoV))
 * factor and the 1 / E glossy base energy compensation are folded in. Only used with the principled BSDF
 */
HIPRT_DEVICE ReSTIRPTTargetFunctionBSDFApproximationLobe
ReSTIR_PT_target_function_bsdf_approximation_lobe_from_material(const HIPRTRenderData& render_data,
																const DeviceUnpackedPrincipledFullMaterial& material,
																int material_class,
																bool smooth_lobe,
																bool pixel_side,
																float view_cosine)
{
	ReSTIRPTTargetFunctionBSDFApproximationLobe lobe;

#if BSDFKernelModel == BSDF_PRINCIPLED
	lobe.alpha = hippt::clamp(0.0f, 1.0f, material.roughness * material.roughness);

	if (material_class == RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_METAL)
	{
		lobe.metal			   = true;
		lobe.fresnel_parameter = material.base_color.luminance();

		if (!smooth_lobe)
			lobe.specular_scale = 1.0f;

		return lobe;
	}

	// Dielectric glossy base: specular weight s, diffuse weight 1. Without a coat, the relative ior of the specular layer is the material ior
	float specular		   = material.specular;
	lobe.fresnel_parameter = material.ior;

	float energy_compensation_divisor = 1.0f;
	if (pixel_side)
		// The incident medium ior only enters through the coat (none here)
		energy_compensation_divisor = get_principled_energy_compensation_glossy_base(render_data, material, 1.0f, view_cosine);

	ColorRGB32F tint = hippt::lerp(ColorRGB32F(1.0f), material.specular_tint * material.specular_color, specular);

	if (!smooth_lobe)
		lobe.specular_scale = specular * tint.luminance() / energy_compensation_divisor;

	ColorRGB32F darkening = principled_specular_compute_darkening(material, material.ior, 0.0f);
	lobe.diffuse_constant = material.base_color.luminance() * (1.0f - specular) * hippt::M_INV_PI / energy_compensation_divisor;
	lobe.diffuse_fresnel  = (material.base_color * darkening).luminance() * specular * hippt::M_INV_PI / energy_compensation_divisor;

	if (pixel_side)
		lobe.diffuse_fresnel *= 1.0f - full_fresnel_dielectric(view_cosine, material.ior);

	if (material_class == RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_METALLIC_BLEND)
	{
		// principled_bsdf_eval() is m metal(V, L) + (1 - m) glossy_base(V, L) / E with m the metallic weight: the glossy base lobe above scaled by
		// 1 - m, plus the metal lobe scaled by m. Both specular lobes share the roughness and the GGX D and G2 terms, so the specular part is
		// D G2 / (4 NoV NoL) times
		//
		//		Cs_d F_d(LoH) + Cs_m F_m(LoH),	Cs_d = (1 - m) Cs of the glossy base, Cs_m = m, F_m Schlick with F0_m = lum(base)
		//
		// With F_d replaced by the Schlick Fresnel of F0_d = ((ior - 1) / (ior + 1))^2, that sum is exactly (Cs_d + Cs_m) times the Schlick Fresnel of
		// F0 = (Cs_d F0_d + Cs_m F0_m) / (Cs_d + Cs_m): the metal form of the lobe
		float metallic		= material.metallic;
		float dielectric_F0 = hippt::square((material.ior - 1.0f) / (material.ior + 1.0f));
		float metal_F0		= material.base_color.luminance();

		float dielectric_specular_scale = (1.0f - metallic) * lobe.specular_scale;
		float specular_scale			= dielectric_specular_scale + metallic;

		lobe.metal = true;
		// A perfectly smooth lobe has no specular term and keeps the Fresnel of the metal (never read)
		lobe.fresnel_parameter = metal_F0;
		if (!smooth_lobe)
		{
			lobe.specular_scale	   = specular_scale;
			lobe.fresnel_parameter = (dielectric_specular_scale * dielectric_F0 + metallic * metal_F0) / specular_scale;
		}

		// The metal form has no Fresnel transmission term: the (1 - F_d) factors of Cd1 not folded above are replaced by their cosine-weighted
		// hemispherical average with the Schlick Fresnel, 1 - F0_d - (1 - F0_d) / 21: (1 - F_d(NoL)) on both sides, (1 - F_d(NoV)) on the reconnection
		// side. Cd0 > 0 (positive base color, 1 - m > 0, transmissions > 0) keeps the approximation > 0 everywhere above the cosine threshold, as the BSDF
		float average_transmission = 1.0f - dielectric_F0 - (1.0f - dielectric_F0) / 21.0f;

		float folded_transmission = average_transmission;
		if (!pixel_side)
			folded_transmission *= average_transmission;

		lobe.diffuse_constant = (1.0f - metallic) * (lobe.diffuse_constant + lobe.diffuse_fresnel * folded_transmission);
		lobe.diffuse_fresnel  = 0.0f;
	}
#endif // #if BSDFKernelModel == BSDF_PRINCIPLED

	return lobe;
}

/**
 * (sin theta, 0, cos theta) of 'direction' around 'normal' (cosine 'cosine'), the tangential part from the projection for precision near the normal. The
 * isotropic GGX functions of the code only read x^2 + y^2 and z
 */
HIPRT_DEVICE float3_t ReSTIR_PT_target_function_bsdf_approximation_isotropic_local_direction(const float3_t& direction, const float3_t& normal, float cosine)
{
	float3_t tangential = direction - cosine * normal;

	return make_float3(sqrtf(hippt::dot(tangential, tangential)), 0.0f, cosine);
}

/**
 * Value of a lobe at a pair of directions (see the top of this file). 'view_cosine' and 'light_cosine' are > 0. 'reconnection_side': the
 * (1 - F_d(NoV)) factor is evaluated here instead of being folded in the lobe
 */
HIPRT_DEVICE float ReSTIR_PT_target_function_bsdf_approximation_lobe_value(const HIPRTRenderData& render_data,
																		   const ReSTIRPTTargetFunctionBSDFApproximationLobe& lobe,
																		   const float3_t& shading_normal,
																		   const float3_t& view_direction,
																		   const float3_t& light_direction,
																		   float view_cosine,
																		   float light_cosine,
																		   bool reconnection_side)
{
	float value = lobe.diffuse_constant;

#if BSDFKernelModel == BSDF_PRINCIPLED
	if (lobe.specular_scale > 0.0f)
	{
		float3_t half_vector	= hippt::normalize(view_direction + light_direction);
		float half_cosine		= hippt::dot(shading_normal, half_vector);
		float light_half_cosine = hippt::dot(light_direction, half_vector);

		float alpha = lobe.alpha;
		float distribution =
			GGX_anisotropic(alpha, alpha, ReSTIR_PT_target_function_bsdf_approximation_isotropic_local_direction(half_vector, shading_normal, half_cosine));

		float lambda_view =
			G1_Smith_lambda(alpha, alpha, ReSTIR_PT_target_function_bsdf_approximation_isotropic_local_direction(view_direction, shading_normal, view_cosine));
		float lambda_light = G1_Smith_lambda(
			alpha, alpha, ReSTIR_PT_target_function_bsdf_approximation_isotropic_local_direction(light_direction, shading_normal, light_cosine));

		float masking_shadowing;
		if (render_data.bsdfs_data.GGX_masking_shadowing == GGXMaskingShadowingFlavor::HeightUncorrelated)
			masking_shadowing = 1.0f / ((1.0f + lambda_view) * (1.0f + lambda_light));
		else
			masking_shadowing = 1.0f / (1.0f + lambda_view + lambda_light);

		float fresnel;
		if (lobe.metal)
			// adobe_f82_tint_fresnel() with F82 = F90 = 1 and exponent 5: Schlick, linear in F0 (luminance of F0)
			fresnel = lobe.fresnel_parameter + (1.0f - lobe.fresnel_parameter) * hippt::pow_5(1.0f - hippt::clamp(1.0e-8f, 1.0f, light_half_cosine));
		else
			fresnel = full_fresnel_dielectric(light_half_cosine, lobe.fresnel_parameter);

		float clamped_view_cosine  = hippt::max(GGX_DOT_PRODUCTS_CLAMP, view_cosine);
		float clamped_light_cosine = hippt::max(GGX_DOT_PRODUCTS_CLAMP, light_cosine);

		value += lobe.specular_scale * fresnel * distribution * masking_shadowing / (4.0f * clamped_view_cosine * clamped_light_cosine);
	}

	if (lobe.diffuse_fresnel > 0.0f)
	{
		float transmission = 1.0f - full_fresnel_dielectric(light_cosine, lobe.fresnel_parameter);
		if (reconnection_side)
			transmission *= 1.0f - full_fresnel_dielectric(view_cosine, lobe.fresnel_parameter);

		value += lobe.diffuse_fresnel * transmission;
	}
#endif // #if BSDFKernelModel == BSDF_PRINCIPLED

	return light_cosine * value;
}

/**
 * True when a perfectly smooth lobe can return its delta value for this incident light info and pair of directions: the pair is then left for the
 * fallback. 'view_direction' and 'light_direction' point away from the surface
 */
HIPRT_DEVICE bool ReSTIR_PT_target_function_bsdf_approximation_delta_peak_possible(BSDFIncidentLightInfo incident_light_info,
																				   const float3_t& view_direction,
																				   const float3_t& shading_normal,
																				   const float3_t& light_direction)
{
	if ((incident_light_info & RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_DELTA_TRIGGER_INFOS) == 0)
		return false;

	return hippt::dot(reflect_ray(view_direction, shading_normal), light_direction) >= RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_DELTA_ALIGNMENT_THRESHOLD;
}

/**
 * 'bits'-bit unorm code of 'value' in [0, 1], rounded to the nearest or up
 */
HIPRT_DEVICE unsigned int ReSTIR_PT_target_function_bsdf_approximation_quantize(float value, unsigned int bits, bool round_up)
{
	float maximum_code = static_cast<float>((1u << bits) - 1u);
	float scaled	   = hippt::clamp(0.0f, maximum_code, value * maximum_code);

	return static_cast<unsigned int>(round_up ? ceilf(scaled) : scaled + 0.5f);
}

HIPRT_DEVICE float ReSTIR_PT_target_function_bsdf_approximation_dequantize(unsigned int code, unsigned int bits)
{
	return code / static_cast<float>((1u << bits) - 1u);
}

/**
 * 8-bit log code of alpha: alpha = 1e-4 ^ (1 - code / 255)
 */
HIPRT_DEVICE unsigned int ReSTIR_PT_target_function_bsdf_approximation_encode_alpha(float alpha)
{
	float normalized = 1.0f + log10f(hippt::max(alpha, 1.0e-4f)) * 0.25f;

	return ReSTIR_PT_target_function_bsdf_approximation_quantize(normalized, 8, false);
}

HIPRT_DEVICE float ReSTIR_PT_target_function_bsdf_approximation_decode_alpha(unsigned int code)
{
	// log2(1e-4)
	return exp2f(-13.287712379549449f * (1.0f - code * (1.0f / 255.0f)));
}

/**
 * Fresnel code: dielectric sqrt(F0) = (ior - 1) / (ior + 1) uniform on [0, 0.5], metal sqrt(lumF0) uniform on [0, 1]
 */
HIPRT_DEVICE unsigned int ReSTIR_PT_target_function_bsdf_approximation_encode_fresnel(const ReSTIRPTTargetFunctionBSDFApproximationLobe& lobe,
																					  unsigned int bits)
{
	if (lobe.metal)
		return ReSTIR_PT_target_function_bsdf_approximation_quantize(sqrtf(hippt::clamp(0.0f, 1.0f, lobe.fresnel_parameter)), bits, false);

	float F0_sqrt = (lobe.fresnel_parameter - 1.0f) / (lobe.fresnel_parameter + 1.0f);

	return ReSTIR_PT_target_function_bsdf_approximation_quantize(F0_sqrt * 2.0f, bits, false);
}

HIPRT_DEVICE float ReSTIR_PT_target_function_bsdf_approximation_decode_fresnel(unsigned int code, unsigned int bits, bool metal)
{
	float normalized = ReSTIR_PT_target_function_bsdf_approximation_dequantize(code, bits);

	if (metal)
		return normalized * normalized;

	float F0_sqrt = normalized * 0.5f;

	return (1.0f + F0_sqrt) / (1.0f - F0_sqrt);
}

/**
 * Pixel record of a lobe (layout in GBufferDevice.h). A positive Cs keeps a non-zero code, Cd0 and Cd1 are rounded up: the support of the decoded lobe is
 * the one of the unquantized lobe
 */
HIPRT_DEVICE ReSTIRPTTargetFunctionBSDFApproximationPixelLobeRecord
ReSTIR_PT_target_function_bsdf_approximation_encode_pixel_lobe(const ReSTIRPTTargetFunctionBSDFApproximationLobe& lobe)
{
	unsigned int specular_code = ReSTIR_PT_target_function_bsdf_approximation_quantize(lobe.specular_scale * 0.25f, 16, false);
	if (lobe.specular_scale > 0.0f)
		specular_code = hippt::max(specular_code, 1u);

	unsigned int diffuse_constant_code = ReSTIR_PT_target_function_bsdf_approximation_quantize(lobe.diffuse_constant, 16, true);
	unsigned int diffuse_fresnel_code  = ReSTIR_PT_target_function_bsdf_approximation_quantize(lobe.diffuse_fresnel, 16, true);

	ReSTIRPTTargetFunctionBSDFApproximationPixelLobeRecord record;
	record.coefficients = specular_code | (diffuse_constant_code << 16);
	record.parameters	= diffuse_fresnel_code | (ReSTIR_PT_target_function_bsdf_approximation_encode_alpha(lobe.alpha) << 16) |
						(ReSTIR_PT_target_function_bsdf_approximation_encode_fresnel(lobe, 7) << 24) | (lobe.metal ? (1u << 31) : 0u);

	return record;
}

HIPRT_DEVICE ReSTIRPTTargetFunctionBSDFApproximationLobe
ReSTIR_PT_target_function_bsdf_approximation_decode_pixel_lobe(const ReSTIRPTTargetFunctionBSDFApproximationPixelLobeRecord& record)
{
	ReSTIRPTTargetFunctionBSDFApproximationLobe lobe;
	lobe.metal			   = (record.parameters >> 31) != 0;
	lobe.specular_scale	   = ReSTIR_PT_target_function_bsdf_approximation_dequantize(record.coefficients & 0xFFFFu, 16) * 4.0f;
	lobe.diffuse_constant  = ReSTIR_PT_target_function_bsdf_approximation_dequantize(record.coefficients >> 16, 16);
	lobe.diffuse_fresnel   = ReSTIR_PT_target_function_bsdf_approximation_dequantize(record.parameters & 0xFFFFu, 16);
	lobe.alpha			   = ReSTIR_PT_target_function_bsdf_approximation_decode_alpha((record.parameters >> 16) & 0xFFu);
	lobe.fresnel_parameter = ReSTIR_PT_target_function_bsdf_approximation_decode_fresnel((record.parameters >> 24) & 0x7Fu, 7, lobe.metal);

	return lobe;
}

/**
 * Stores the reconnection-side lobe in the lobe bytes of the sample (layout in Reservoir.h) and returns its flag byte (class bit set, smooth bit
 * 'smooth_lobe', bits 2..7 the 6-bit Fresnel code). Cs' sqrt-coded on [0, 4] (8 bits, a positive value keeps a non-zero code), Cd0' sqrt-coded on [0, 1]
 * (7 bits, rounded up), Cd1' sqrt-coded on [0, 1] (8 bits, rounded up)
 */
HIPRT_DEVICE unsigned char ReSTIR_PT_target_function_bsdf_approximation_encode_sample_lobe(ReSTIRPTReservoirSample& sample,
																						   const ReSTIRPTTargetFunctionBSDFApproximationLobe& lobe,
																						   bool smooth_lobe)
{
	unsigned int specular_code = ReSTIR_PT_target_function_bsdf_approximation_quantize(sqrtf(hippt::clamp(0.0f, 1.0f, lobe.specular_scale * 0.25f)), 8, false);
	if (lobe.specular_scale > 0.0f)
		specular_code = hippt::max(specular_code, 1u);

	unsigned int diffuse_constant_code = ReSTIR_PT_target_function_bsdf_approximation_quantize(sqrtf(hippt::clamp(0.0f, 1.0f, lobe.diffuse_constant)), 7, true);
	unsigned int diffuse_fresnel_code  = ReSTIR_PT_target_function_bsdf_approximation_quantize(sqrtf(hippt::clamp(0.0f, 1.0f, lobe.diffuse_fresnel)), 8, true);

	sample.target_function_bsdf_approximation_lobe_alpha	= static_cast<unsigned char>(ReSTIR_PT_target_function_bsdf_approximation_encode_alpha(lobe.alpha));
	sample.target_function_bsdf_approximation_lobe_specular = static_cast<unsigned char>(specular_code);
	sample.rc_vertex_shading_normal.set_padding(static_cast<unsigned char>(diffuse_fresnel_code));
	sample.rc_vertex_geometric_normal.set_padding(static_cast<unsigned char>(diffuse_constant_code | (lobe.metal ? 0x80u : 0u)));

	unsigned int flags = RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_FLAG |
						 (smooth_lobe ? RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_SMOOTH_LOBE_FLAG : 0u) |
						 (ReSTIR_PT_target_function_bsdf_approximation_encode_fresnel(lobe, 6) << 2);

	return static_cast<unsigned char>(flags);
}

HIPRT_DEVICE ReSTIRPTTargetFunctionBSDFApproximationLobe ReSTIR_PT_target_function_bsdf_approximation_decode_sample_lobe(const ReSTIRPTReservoirSample& sample)
{
	unsigned int geometric_normal_byte = sample.rc_vertex_geometric_normal.get_padding();

	float specular_normalized		  = ReSTIR_PT_target_function_bsdf_approximation_dequantize(sample.target_function_bsdf_approximation_lobe_specular, 8);
	float diffuse_constant_normalized = ReSTIR_PT_target_function_bsdf_approximation_dequantize(geometric_normal_byte & 0x7Fu, 7);
	float diffuse_fresnel_normalized  = ReSTIR_PT_target_function_bsdf_approximation_dequantize(sample.rc_vertex_shading_normal.get_padding(), 8);

	ReSTIRPTTargetFunctionBSDFApproximationLobe lobe;
	lobe.metal			   = (geometric_normal_byte & 0x80u) != 0;
	lobe.specular_scale	   = 4.0f * specular_normalized * specular_normalized;
	lobe.diffuse_constant  = diffuse_constant_normalized * diffuse_constant_normalized;
	lobe.diffuse_fresnel   = diffuse_fresnel_normalized * diffuse_fresnel_normalized;
	lobe.alpha			   = ReSTIR_PT_target_function_bsdf_approximation_decode_alpha(sample.target_function_bsdf_approximation_lobe_alpha);
	lobe.fresnel_parameter = ReSTIR_PT_target_function_bsdf_approximation_decode_fresnel(
		static_cast<unsigned int>(sample.target_function_bsdf_approximation_flags) >> 2, 6, lobe.metal);

	return lobe;
}

/**
 * Flags of a pixel (padding byte of GBufferDevice::shading_normals), from the values the target function reads back from the G-buffer: 'material' is the
 * unpacked G-buffer material, 'shading_normal' the unpacked G-buffer shading normal and the view direction comes from the camera position and
 * 'shading_point'.
 *
 * Class bit: the material is in an analytic class, the view direction is above the shading normal with a cosine above
 * RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_THRESHOLD and (principled BSDF) the camera is outside of the object.
 * Smooth bit: the lobe is perfectly smooth.
 *
 * With the principled BSDF, also writes the lobe record of the pixel when the class bit is set
 */
HIPRT_DEVICE unsigned char ReSTIR_PT_target_function_bsdf_approximation_pixel_flags(const HIPRTRenderData& render_data,
																					unsigned int pixel_index,
																					const DeviceUnpackedPrincipledFullMaterial& material,
																					const float3_t& shading_point,
																					const float3_t& shading_normal,
																					int primitive_index)
{
	float3_t view_direction = hippt::normalize(render_data.current_camera.position - shading_point);
	float view_cosine		= hippt::dot(view_direction, shading_normal);

	bool smooth_lobe;

	int material_class = ReSTIR_PT_target_function_bsdf_approximation_material_class(material, smooth_lobe);
	if (material_class == RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE ||
		view_cosine <= RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_THRESHOLD)
		return 0;

#if BSDFKernelModel == BSDF_PRINCIPLED
	// The principled BSDF evaluates only the glass lobe from inside an object: the volume state that the target function reconstructs for this pixel must
	// be outside. No random number is used for an opaque material (no dispersion wavelength)
	Xorshift32Generator unused_random_number_generator(1);
	RayVolumeState first_hit_volume_state;
	first_hit_volume_state.reconstruct_first_hit(material, render_data.buffers.material_indices, primitive_index, unused_random_number_generator);

	if (first_hit_volume_state.inside_material)
		return 0;

	ReSTIRPTTargetFunctionBSDFApproximationLobe lobe =
		ReSTIR_PT_target_function_bsdf_approximation_lobe_from_material(render_data, material, material_class, smooth_lobe, true, view_cosine);
	render_data.g_buffer.target_function_bsdf_approximation_lobe_records[pixel_index] = ReSTIR_PT_target_function_bsdf_approximation_encode_pixel_lobe(lobe);
#endif // #if BSDFKernelModel == BSDF_PRINCIPLED

	return static_cast<unsigned char>(RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_FLAG |
									  (smooth_lobe ? RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_SMOOTH_LOBE_FLAG : 0u));
}

/**
 * Material part of the flags of a sample whose reconnection vertex has the material 'rc_vertex_material' (the material that the target function fetches
 * for the reconnection vertex), and with the principled BSDF the reconnection-side lobe of the sample. Written when the reconnection vertex is recorded,
 * completed by ReSTIR_PT_target_function_bsdf_approximation_sample_flags() once the sample is final
 */
HIPRT_DEVICE void ReSTIR_PT_target_function_bsdf_approximation_fill_rc_vertex(const HIPRTRenderData& render_data,
																			  const DeviceUnpackedPrincipledFullMaterial& rc_vertex_material,
																			  ReSTIRPTReservoirSample& sample)
{
	bool smooth_lobe;

	int material_class = ReSTIR_PT_target_function_bsdf_approximation_material_class(rc_vertex_material, smooth_lobe);
	if (material_class == RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_NONE)
	{
		sample.target_function_bsdf_approximation_flags = 0;

		return;
	}

#if BSDFKernelModel == BSDF_PRINCIPLED
	ReSTIRPTTargetFunctionBSDFApproximationLobe lobe =
		ReSTIR_PT_target_function_bsdf_approximation_lobe_from_material(render_data, rc_vertex_material, material_class, smooth_lobe, false, 0.0f);

	sample.target_function_bsdf_approximation_flags = ReSTIR_PT_target_function_bsdf_approximation_encode_sample_lobe(sample, lobe, smooth_lobe);
#else
	sample.target_function_bsdf_approximation_flags = static_cast<unsigned char>(RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_FLAG);
#endif // #if BSDFKernelModel == BSDF_PRINCIPLED
}

/**
 * Final flag byte of a sample (0 = not in an analytic class) from the material part written by
 *ReSTIR_PT_target_function_bsdf_approximation_fill_rc_vertex().
 *
 * For any pixel that passes the zero tests of the target function and sees the reconnection vertex from above its shading normal, the target function is
 * then non-zero as soon as the pixel BSDF is:
 *	- the radiance has no negative channel and a positive luminance (a product of strictly positive colors with it has a positive luminance)
 *	- DI sample: no BSDF at the reconnection vertex in the target function, only the class bit
 *	- otherwise not an envmap path, no transmission lobe at the visible point (the volume state of the reconnection vertex BSDF evaluation would then
 *	  depend on the refraction at the pixel), the stored incident light direction is above the shading normal of the reconnection vertex with a cosine
 *	  above RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_THRESHOLD and the reconnection vertex material is in an analytic class
 */
HIPRT_DEVICE unsigned char ReSTIR_PT_target_function_bsdf_approximation_sample_flags(const ReSTIRPTReservoirSample& sample)
{
	ColorRGB32F radiance = sample.rc_vertex_incident_radiance;
	if (radiance.r < 0.0f || radiance.g < 0.0f || radiance.b < 0.0f || radiance.luminance() <= 0.0f)
		return 0;

	if (sample.di_sample)
		// No BSDF at the reconnection vertex in the target function
		return static_cast<unsigned char>(RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_FLAG);

	if (sample.is_envmap_path() || bsdf_incident_light_info_transmission_lobe(sample.incident_light_info_at_visible_point))
		return 0;

	float3_t rc_vertex_shading_normal = sample.rc_vertex_shading_normal.unpack();
	if (hippt::dot(sample.rc_vertex_incident_light_direction, rc_vertex_shading_normal) <= RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_THRESHOLD)
		return 0;

	return sample.target_function_bsdf_approximation_flags;
}

/**
 * Pixel-side value of the principled approximation from the lobe record of the pixel (one 8-byte load), view direction from the camera position
 *
 * Forced inline, as the other functions of the cheap path below: out of line, the sample and the vectors passed by reference live in scratch memory in
 * the caller, every field read here is a scratch load and the caller saves its callee-saved registers around the call
 */
HIPRT_DEVICE HIPRT_FORCEINLINE float ReSTIR_PT_target_function_bsdf_approximation_pixel_side_value(const HIPRTRenderData& render_data,
																								   int pixel_index,
																								   const float3_t& shading_point,
																								   const float3_t& shading_normal,
																								   const float3_t& incident_light_direction,
																								   float cosine_term)
{
	float3_t view_direction = hippt::normalize(render_data.current_camera.position - shading_point);
	float view_cosine		= hippt::dot(shading_normal, view_direction);

	ReSTIRPTTargetFunctionBSDFApproximationLobe lobe =
		ReSTIR_PT_target_function_bsdf_approximation_decode_pixel_lobe(render_data.g_buffer.target_function_bsdf_approximation_lobe_records[pixel_index]);

	return ReSTIR_PT_target_function_bsdf_approximation_lobe_value(render_data, lobe, shading_normal, view_direction, incident_light_direction, view_cosine,
																   cosine_term, false);
}

/**
 * Reconnection-side value of the principled approximation from the lobe stored in the sample, view direction from the reconnection vertex to the pixel at
 * 'shading_point'. 1 for a DI sample. No material fetch
 */
HIPRT_DEVICE HIPRT_FORCEINLINE float ReSTIR_PT_target_function_bsdf_approximation_rc_side_value(const HIPRTRenderData& render_data,
																								const ReSTIRPTReservoirSample& sample,
																								const float3_t& shading_point)
{
	if (sample.di_sample)
		return 1.0f;

	float3_t rc_vertex_shading_normal = sample.rc_vertex_shading_normal.unpack();
	float3_t view_direction			  = hippt::normalize(shading_point - sample.rc_vertex);
	float view_cosine				  = hippt::dot(view_direction, rc_vertex_shading_normal);
	float light_cosine				  = hippt::dot(sample.rc_vertex_incident_light_direction, rc_vertex_shading_normal);

	ReSTIRPTTargetFunctionBSDFApproximationLobe lobe = ReSTIR_PT_target_function_bsdf_approximation_decode_sample_lobe(sample);

	return ReSTIR_PT_target_function_bsdf_approximation_lobe_value(render_data, lobe, rc_vertex_shading_normal, view_direction,
																   sample.rc_vertex_incident_light_direction, view_cosine, light_cosine, true);
}

/**
 * True when a perfectly smooth lobe of the pair can return its delta value: the pixel's lobe (pixel smooth bit, incident light info at the visible point,
 * view direction of the pixel, direction to the reconnection vertex) or the reconnection vertex lobe (sample smooth bit, incident light info at the sample
 * point, direction to the pixel, stored outgoing direction)
 */
HIPRT_DEVICE HIPRT_FORCEINLINE bool ReSTIR_PT_target_function_bsdf_approximation_pair_delta_peak_possible(const HIPRTRenderData& render_data,
																										  const ReSTIRPTReservoirSample& sample,
																										  unsigned int pixel_flags,
																										  const float3_t& shading_point,
																										  const float3_t& shading_normal,
																										  const float3_t& incident_light_direction)
{
	if ((pixel_flags & RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_SMOOTH_LOBE_FLAG) != 0)
	{
		float3_t view_direction = hippt::normalize(render_data.current_camera.position - shading_point);
		if (ReSTIR_PT_target_function_bsdf_approximation_delta_peak_possible(sample.incident_light_info_at_visible_point, view_direction, shading_normal,
																			 incident_light_direction))
			return true;
	}

	if (!sample.di_sample && (sample.target_function_bsdf_approximation_flags & RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_SMOOTH_LOBE_FLAG) != 0)
	{
		float3_t rc_vertex_view_direction = hippt::normalize(shading_point - sample.rc_vertex);
		if (ReSTIR_PT_target_function_bsdf_approximation_delta_peak_possible(sample.incident_light_info_at_sample_point, rc_vertex_view_direction,
																			 sample.rc_vertex_shading_normal.unpack(),
																			 sample.rc_vertex_incident_light_direction))
			return true;
	}

	return false;
}

/**
 * Cheap path or fallback for a sample at a pixel whose flags are 'pixel_flags', for a pair that passed the zero tests of the target function (degenerate
 * distance, backfacing light of a DI sample, cosine <= 0 outside of a transmission lobe). The same decision in the probes and in the own terms.
 *
 * Returns RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_FALLBACK, RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CHEAP_PATH or
 * RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_ZERO (cheap path, the pixel is below the shading normal of the reconnection vertex: every lobe of the
 * reconnection vertex BSDF is zero)
 *
 * Forced inline for the reason given above ReSTIR_PT_target_function_bsdf_approximation_pixel_side_value()
 */
HIPRT_DEVICE HIPRT_FORCEINLINE int ReSTIR_PT_target_function_bsdf_approximation_decision(const HIPRTRenderData& render_data,
																						 const ReSTIRPTReservoirSample& sample,
																						 unsigned int pixel_flags,
																						 const float3_t& shading_point,
																						 const float3_t& shading_normal,
																						 const float3_t& incident_light_direction,
																						 float cosine_term)
{
	bool analytic_pixel = (pixel_flags & RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CLASS_FLAG) != 0;

	if (!analytic_pixel || sample.target_function_bsdf_approximation_flags == 0 || cosine_term <= RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_THRESHOLD)
		return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_FALLBACK;

	if (!sample.di_sample)
	{
		// Side test of the reconnection vertex BSDF: the same view direction as in ReSTIR_PT_evaluate_target_function_bsdf(). Between 0 and the cosine
		// threshold, the fallback decides
		float3_t rc_vertex_view_direction = hippt::normalize(shading_point - sample.rc_vertex);

		float rc_vertex_view_cosine = hippt::dot(rc_vertex_view_direction, sample.rc_vertex_shading_normal.unpack());
		if (rc_vertex_view_cosine <= 0.0f)
			return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_ZERO;

		if (rc_vertex_view_cosine <= RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_THRESHOLD)
			return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_FALLBACK;
	}

#if BSDFKernelModel != BSDF_LAMBERTIAN
	// A perfectly smooth lobe that can return its delta value for this pair goes to the fallback
	if (ReSTIR_PT_target_function_bsdf_approximation_pair_delta_peak_possible(render_data, sample, pixel_flags, shading_point, shading_normal,
																			  incident_light_direction))
		return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_FALLBACK;
#endif // #if BSDFKernelModel != BSDF_LAMBERTIAN

	return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CHEAP_PATH;
}

/**
 * Approximation value of a pair (sample, pixel) on the cheap path (ReSTIR_PT_target_function_bsdf_approximation_decision() returned
 * RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CHEAP_PATH)
 *
 * Forced inline for the reason given above ReSTIR_PT_target_function_bsdf_approximation_pixel_side_value()
 */
HIPRT_DEVICE HIPRT_FORCEINLINE float ReSTIR_PT_target_function_bsdf_approximation_cheap_path_value(const HIPRTRenderData& render_data,
																								   const ReSTIRPTReservoirSample& sample,
																								   int pixel_index,
																								   const float3_t& shading_point,
																								   const float3_t& shading_normal,
																								   const float3_t& incident_light_direction,
																								   float cosine_term)
{
#if BSDFKernelModel == BSDF_LAMBERTIAN
	return cosine_term;
#else
	float pixel_side_value = ReSTIR_PT_target_function_bsdf_approximation_pixel_side_value(render_data, pixel_index, shading_point, shading_normal,
																						   incident_light_direction, cosine_term);

	return pixel_side_value * ReSTIR_PT_target_function_bsdf_approximation_rc_side_value(render_data, sample, shading_point);
#endif // #if BSDFKernelModel == BSDF_LAMBERTIAN
}

/**
 * Approximation value of a pair (sample, pixel) on the fallback, from the full target function 'full_target_function' of the pair: the cosine at the pixel on
 * the support of the full target function (Lambertian BSDF), the full target function without the radiance of the sample otherwise (floored on its
 * support)
 *
 * Forced inline for the reason given above ReSTIR_PT_target_function_bsdf_approximation_pixel_side_value()
 */
HIPRT_DEVICE HIPRT_FORCEINLINE float ReSTIR_PT_target_function_bsdf_approximation_fallback_value(const ReSTIRPTReservoirSample& sample,
																								 float full_target_function,
																								 float cosine_term)
{
	if (full_target_function <= 0.0f)
		return 0.0f;

#if BSDFKernelModel == BSDF_LAMBERTIAN
	return hippt::abs(cosine_term);
#else
	return hippt::max(full_target_function / hippt::max(sample.rc_vertex_incident_radiance.luminance(), 1.0e-30f),
					  RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_VALUE_FLOOR);
#endif // #if BSDFKernelModel == BSDF_LAMBERTIAN
}

/**
 * Approximation of an own term of the MIS weights: the sample 'sample' at the G-buffer pixel 'pixel_index' (its own pixel, or the center pixel for a neighbor's
 * sample or for the canonical sample), whose full target function 'own_full_target_function' is known (no visibility test: the sample is visible from
 * that pixel). The same function of (sample, pixel) as the probes of ReSTIR_PT_evaluate_target_function_at_pixel(), with no BSDF evaluation: the fallback
 * value comes from the known full target function.
 *
 * A known full target function can be > 0 where a re-evaluation at the current G-buffer is 0 (a stored delta-peak target of a perfectly smooth lobe whose
 * mirror direction moved). The support of the own term must be the one of 'own_full_target_function': a zero cheap value then takes the fallback value
 *
 * Forced inline for the reason given above ReSTIR_PT_target_function_bsdf_approximation_pixel_side_value()
 */
HIPRT_DEVICE HIPRT_FORCEINLINE float ReSTIR_PT_target_function_bsdf_approximation_own_term(const HIPRTRenderData& render_data,
																						   const ReSTIRPTReservoirSample& sample,
																						   int pixel_index,
																						   float own_full_target_function)
{
	if (own_full_target_function <= 0.0f)
		return 0.0f;

#if BSDFKernelModel == BSDF_LAMBERTIAN
	// Probes: the cosine at the pixel on the cheap path or on the support of the full target function. On the support of 'own_full_target_function', that
	// is the cosine whatever the decision, floored so that it never rounds to 0
	float3_t shading_point = render_data.g_buffer.primary_hit_position[pixel_index];

	float3_t incident_light_direction;
	if (sample.is_envmap_path())
		incident_light_direction = sample.rc_vertex;
	else
	{
		incident_light_direction = sample.rc_vertex - shading_point;

		float distance_to_sample_point = hippt::length(incident_light_direction);
		if (distance_to_sample_point <= 1.0e-6f)
			return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_FLOOR;

		incident_light_direction /= distance_to_sample_point;
	}

	float cosine = hippt::abs(hippt::dot(incident_light_direction, render_data.g_buffer.shading_normals[pixel_index].unpack()));

	return hippt::max(cosine, RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_COSINE_FLOOR);
#else
	float3_t shading_point								 = render_data.g_buffer.primary_hit_position[pixel_index];
	Octahedral24BitNormalPadded32b packed_shading_normal = render_data.g_buffer.shading_normals[pixel_index];
	float3_t shading_normal								 = packed_shading_normal.unpack();

	float3_t incident_light_direction;
	if (sample.is_envmap_path())
		incident_light_direction = sample.rc_vertex;
	else
	{
		incident_light_direction = sample.rc_vertex - shading_point;

		float distance_to_sample_point = hippt::length(incident_light_direction);
		if (distance_to_sample_point <= 1.0e-6f)
			return RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_VALUE_FLOOR;

		incident_light_direction /= distance_to_sample_point;
	}

	float cosine_term = hippt::dot(incident_light_direction, shading_normal);

	int decision = ReSTIR_PT_target_function_bsdf_approximation_decision(render_data, sample, packed_shading_normal.get_padding(), shading_point,
																		 shading_normal, incident_light_direction, cosine_term);

	float own_term_value = 0.0f;
	if (decision == RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_CHEAP_PATH)
		own_term_value = ReSTIR_PT_target_function_bsdf_approximation_cheap_path_value(render_data, sample, pixel_index, shading_point, shading_normal,
																					   incident_light_direction, cosine_term);

	if (decision == RESTIR_PT_TARGET_FUNCTION_BSDF_APPROXIMATION_FALLBACK || !(own_term_value > 0.0f))
		// The fallback, or a zero cheap value with a known full target function > 0
		own_term_value = ReSTIR_PT_target_function_bsdf_approximation_fallback_value(sample, own_full_target_function, cosine_term);

	return own_term_value;
#endif // #if BSDFKernelModel == BSDF_LAMBERTIAN
}

/**
 * Resampling target function of the union domain reuse: the approximation 'approximation_value' of the sample at a pixel
 * (ReSTIR_PT_evaluate_target_function_at_pixel() with the approximation, or an own term) times the luminance of the radiance of the sample.
 *
 * Principled BSDF: on the fallback, the approximation is the full target function divided by that luminance (floored), this is the full target function.
 * On the cheap path, the analytic lobes times that luminance. Lambertian BSDF: the cosine at the pixel times that luminance.
 *
 * Zero exactly where the target function is, which keeps the resampling unbiased: the UCW of the output sample is computed with this same function and
 * the shading pass evaluates the full integrand. The output reservoir stores this function, which no later pass reads as a target function: the union
 * domain reuse only runs without temporal reuse and with a single spatial reuse pass (ReSTIRPTSettings::do_spmis_union_domain_reuse())
 */
HIPRT_DEVICE float ReSTIR_PT_resampling_target_function_bsdf_approximation(const ReSTIRPTReservoirSample& sample, float approximation_value)
{
	return approximation_value * hippt::max(sample.rc_vertex_incident_radiance.luminance(), 1.0e-30f);
}

#endif // #ifndef DEVICE_RESTIR_PT_SPATIAL_TARGET_FUNCTION_BSDF_APPROXIMATION_H
