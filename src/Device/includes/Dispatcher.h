/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_DISPATCHER_H
#define DEVICE_DISPATCHER_H

#include "Device/includes/Disney.h"
#include "Device/includes/Lambertian.h"
#include "Device/includes/OrenNayar.h"
#include "Device/includes/RayPayload.h"

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F bsdf_dispatcher_eval(const RendererMaterial* materials_buffer, const RendererMaterial& material, RayVolumeState& ray_volume_state, const float3& view_direction, const float3& surface_normal, const float3& to_light_direction, float& pdf)
{
#if BSDFOverride == BSDF_NONE
	/*switch (switch_on)
	{
	...
	...
	default:
		break;
	}*/
    return disney_bsdf_eval(materials_buffer, material, ray_volume_state, view_direction, surface_normal, to_light_direction, pdf);
#elif BSDFOverride == BSDF_LAMBERTIAN
	return lambertian_brdf_eval(material, view_direction, surface_normal, to_light_direction, pdf);
#elif BSDFOverride == BSDF_OREN_NAYAR
	return oren_nayar_brdf_eval(material, view_direction, surface_normal, to_light_direction, pdf);
#elif BSDFOverride == BSDF_DISNEY
    return disney_bsdf_eval(materials_buffer, material, ray_volume_state, view_direction, surface_normal, to_light_direction, pdf);
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F bsdf_dispatcher_sample(const RendererMaterial* materials_buffer, const RendererMaterial& material, RayVolumeState& ray_volume_state, const float3& view_direction, const float3& surface_normal, const float3& geometric_normal, float3& sampled_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
#if BSDFOverride == BSDF_NONE
	/*switch (switch_on)
	{
	...
	...
	default:
		break;
	}*/
    return disney_bsdf_sample(materials_buffer, material, ray_volume_state, view_direction, surface_normal, geometric_normal, sampled_direction, pdf, random_number_generator);
#elif BSDFOverride == BSDF_LAMBERTIAN
	return lambertian_brdf_sample(material, view_direction, surface_normal, sampled_direction, pdf, random_number_generator);
#elif BSDFOverride == BSDF_OREN_NAYAR
	return oren_nayar_brdf_sample(material, view_direction, surface_normal, sampled_direction, pdf, random_number_generator);
#elif BSDFOverride == BSDF_DISNEY
    return disney_bsdf_sample(materials_buffer, material, ray_volume_state, view_direction, surface_normal, geometric_normal, sampled_direction, pdf, random_number_generator);
#endif

}

#endif
