/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_DISPATCHER_H
#define DEVICE_DISPATCHER_H

#include "Device/includes/BSDFs/Lambertian.h"
#include "Device/includes/BSDFs/OrenNayar.h"
#include "Device/includes/BSDFs/Principled.h"
#include "Device/includes/RayPayload.h"

/**
 * The 'random_number_generator' passed here is used only in case 
 * monte-carlo integration of the directional albedo is enabled
 * 
 * If 'update_ray_volume_state' is passed as true, the givenargument is passed as nullptr, the volume state of the ray won't
 * be updated by this sample call (i.e. the ray won't track if this sample call made it exit/enter a new material)
 */
HIPRT_DEVICE HIPRT_INLINE ColorRGB32F bsdf_dispatcher_eval(const HIPRTRenderData& render_data, BSDFContext& bsdf_context, float& pdf, Xorshift32Generator& random_number_generator)
{
#if BSDFOverride == BSDF_NONE || BSDFOverride == BSDF_PRINCIPLED
	/*switch (brdf_type)
	{
	...
	...
	default:
		break;
	}*/
	return principled_bsdf_eval(render_data, bsdf_context, pdf);
#elif BSDFOverride == BSDF_LAMBERTIAN
	return lambertian_brdf_eval(bsdf_context.material, hippt::dot(bsdf_context.to_light_direction, bsdf_context.shading_normal), pdf);
#elif BSDFOverride == BSDF_OREN_NAYAR
	return oren_nayar_brdf_eval(bsdf_context.material, bsdf_context.view_direction, bsdf_context.shading_normal, bsdf_context.to_light_direction, pdf);
#endif
}

HIPRT_DEVICE HIPRT_INLINE float bsdf_dispatcher_pdf(const HIPRTRenderData& render_data, BSDFContext& bsdf_context)
{
#if BSDFOverride == BSDF_NONE || BSDFOverride == BSDF_PRINCIPLED
	/*switch (brdf_type)
	{
	...
	...
	default:
		break;
	}*/
	return principled_bsdf_pdf(render_data, bsdf_context);
#elif BSDFOverride == BSDF_LAMBERTIAN
	return lambertian_brdf_pdf(bsdf_context.material, hippt::dot(bsdf_context.to_light_direction, bsdf_context.shading_normal));
#elif BSDFOverride == BSDF_OREN_NAYAR
	return oren_nayar_brdf_pdf(bsdf_context.material, bsdf_context.view_direction, bsdf_context.shading_normal, bsdf_context.to_light_direction);
#endif
}

/**
 * If the 'ray_volume_state' argument is passed as nullptr, the volume state of the ray won't
 * be updated by this sample call (i.e. the ray won't track if this sample call made it exit/enter a new material)
 * 
 * If sampleDirectionOnly is 'true',, this function samples only the BSDF without 
 * evaluating the contribution or the PDF of the BSDF. This function will then always return
 * ColorRGB32F(0.0f) and the 'pdf' out parameter will always be set to 0.0f
 */
template <bool sampleDirectionOnly = false>
HIPRT_DEVICE HIPRT_INLINE ColorRGB32F bsdf_dispatcher_sample(const HIPRTRenderData& render_data, BSDFContext& bsdf_context, float3& sampled_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
#if BSDFOverride == BSDF_NONE || BSDFOverride == BSDF_PRINCIPLED
	/*switch (brdf_type)
	{
	...
	...
	default:
		break;
	}*/
    return principled_bsdf_sample<sampleDirectionOnly>(render_data, bsdf_context, sampled_direction, pdf, random_number_generator);
#elif BSDFOverride == BSDF_LAMBERTIAN
	return lambertian_brdf_sample<sampleDirectionOnly>(bsdf_context.material, bsdf_context.shading_normal, sampled_direction, pdf, random_number_generator, bsdf_context.incident_light_info);
#elif BSDFOverride == BSDF_OREN_NAYAR
	return oren_nayar_brdf_sample<sampleDirectionOnly>(material, view_direction, surface_normal, sampled_direction, pdf, random_number_generator, out_sampled_light_info);
#endif
}

#endif
