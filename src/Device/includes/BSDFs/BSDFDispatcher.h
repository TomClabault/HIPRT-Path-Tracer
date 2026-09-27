/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_BSDF_DISPATCHER_H
#define DEVICE_BSDF_DISPATCHER_H

#include "Device/includes/BSDFs/Lambertian.h"
#include "Device/includes/BSDFs/OrenNayar.h"
#include "Device/includes/BSDFs/Principled.h"
#include "Device/includes/RayPayload.h"
#include "HostDeviceCommon/Material/MaterialTraits.h"

/**
 * The 'random_number_generator' passed here is used only in case
 * monte-carlo integration of the directional albedo is enabled
 *
 * If 'update_ray_volume_state' is passed as true, the givenargument is passed as nullptr, the volume state of the ray won't
 * be updated by this sample call (i.e. the ray won't track if this sample call made it exit/enter a new material)
 */
template <typename MaterialType>
HIPRT_DEVICE static ColorRGB32F bsdf_dispatcher_eval(const HIPRTRenderData& render_data,
													 BSDFContextT<MaterialType>& bsdf_context,
													 float& pdf,
													 Xorshift32Generator& random_number_generator)
{
#if !defined(BSDF_MODEL) || BSDF_MODEL == BSDF_PRINCIPLED
	/*switch (brdf_type)
	{
	...
	...
	default:
		break;
	}*/
	if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationDiffuse)
		return principled_compact_diffuse_eval(bsdf_context, pdf);
	else if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationGlass)
		return principled_compact_glass_eval(render_data, bsdf_context, pdf, random_number_generator);
	else if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationSingleMetallic)
		return principled_compact_metallic_eval(render_data, bsdf_context, pdf, random_number_generator);
	else if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationSpecularDiffuse)
		return principled_compact_specular_diffuse_eval(render_data, bsdf_context, pdf, random_number_generator);
	else
		return principled_bsdf_eval(render_data, bsdf_context, pdf, random_number_generator);
#elif BSDF_MODEL == BSDF_LAMBERTIAN // #if !defined(BSDF_MODEL) || BSDF_MODEL == BSDF_PRINCIPLED
	return lambertian_brdf_eval(bsdf_context.material, hippt::dot(bsdf_context.to_light_direction, bsdf_context.shading_normal), pdf);
#elif BSDF_MODEL == BSDF_OREN_NAYAR // #if !defined(BSDF_MODEL) || BSDF_MODEL == BSDF_PRINCIPLED
	return oren_nayar_brdf_eval(bsdf_context.material, bsdf_context.view_direction, bsdf_context.shading_normal, bsdf_context.to_light_direction, pdf);
#endif								// #if !defined(BSDF_MODEL) || BSDF_MODEL == BSDF_PRINCIPLED
}

template <typename MaterialType>
HIPRT_DEVICE static float bsdf_dispatcher_pdf(const HIPRTRenderData& render_data, BSDFContextT<MaterialType>& bsdf_context)
{
#if !defined(BSDF_MODEL) || BSDF_MODEL == BSDF_PRINCIPLED
	/*switch (brdf_type)
	{
	...
	...
	default:
		break;
	}*/
	if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationDiffuse)
		return principled_compact_diffuse_pdf(bsdf_context);
	else if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationGlass)
		return principled_compact_glass_pdf(render_data, bsdf_context);
	else if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationSingleMetallic)
		return principled_compact_metallic_pdf(render_data, bsdf_context);
	else if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationSpecularDiffuse)
		return principled_compact_specular_diffuse_pdf(render_data, bsdf_context);
	else
		return principled_bsdf_pdf(render_data, bsdf_context);
#elif BSDF_MODEL == BSDF_LAMBERTIAN // #if !defined(BSDF_MODEL) || BSDF_MODEL == BSDF_PRINCIPLED
	return lambertian_brdf_pdf(hippt::dot(bsdf_context.to_light_direction, bsdf_context.shading_normal));
#elif BSDF_MODEL == BSDF_OREN_NAYAR // #if !defined(BSDF_MODEL) || BSDF_MODEL == BSDF_PRINCIPLED
	return oren_nayar_brdf_pdf(bsdf_context.to_light_direction);
#endif								// #if !defined(BSDF_MODEL) || BSDF_MODEL == BSDF_PRINCIPLED
}

/**
 * If the 'ray_volume_state' argument is passed as nullptr, the volume state of the ray won't
 * be updated by this sample call (i.e. the ray won't track if this sample call made it exit/enter a new material)
 *
 * If sampleDirectionOnly is 'true',, this function samples only the BSDF without
 * evaluating the contribution or the PDF of the BSDF. This function will then always return
 * ColorRGB32F(0.0f) and the 'pdf' out parameter will always be set to 0.0f
 */
template <bool sampleDirectionOnly = false, typename MaterialType>
HIPRT_DEVICE static ColorRGB32F bsdf_dispatcher_sample(const HIPRTRenderData& render_data,
													   BSDFContextT<MaterialType>& bsdf_context,
													   float3_t& sampled_direction,
													   float& pdf,
													   Xorshift32Generator& random_number_generator)
{
#if !defined(BSDF_MODEL) || BSDF_MODEL == BSDF_PRINCIPLED
	/*switch (brdf_type)
	{
	...
	...
	default:
		break;
	}*/
	if constexpr (MaterialTraits<MaterialType>::family == KernelMaterialSpecializationDiffuse ||
				  MaterialTraits<MaterialType>::family == KernelMaterialSpecializationGlass ||
				  MaterialTraits<MaterialType>::family == KernelMaterialSpecializationSingleMetallic ||
				  MaterialTraits<MaterialType>::family == KernelMaterialSpecializationSpecularDiffuse)
		return principled_compact_family_sample<sampleDirectionOnly>(render_data, bsdf_context, sampled_direction, pdf, random_number_generator);
	else
		return principled_bsdf_sample<sampleDirectionOnly>(render_data, bsdf_context, sampled_direction, pdf, random_number_generator);
#elif BSDF_MODEL == BSDF_LAMBERTIAN // #if !defined(BSDF_MODEL) || BSDF_MODEL == BSDF_PRINCIPLED
	return lambertian_brdf_sample<sampleDirectionOnly>(bsdf_context.material, bsdf_context.geometric_normal, bsdf_context.shading_normal, sampled_direction,
													   pdf, random_number_generator, bsdf_context.incident_light_info);
#elif BSDF_MODEL == BSDF_OREN_NAYAR // #if !defined(BSDF_MODEL) || BSDF_MODEL == BSDF_PRINCIPLED
	return oren_nayar_brdf_sample<sampleDirectionOnly>(bsdf_context.material, bsdf_context.view_direction, bsdf_context.geometric_normal,
													   bsdf_context.shading_normal, sampled_direction, pdf, random_number_generator,
													   bsdf_context.incident_light_info);
#endif								// #if !defined(BSDF_MODEL) || BSDF_MODEL == BSDF_PRINCIPLED
}

#endif // #ifndef DEVICE_DISPATCHER_H
