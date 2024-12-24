/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_DISPATCHER_H
#define DEVICE_DISPATCHER_H

#include "Device/includes/BSDFs/Lambertian.h"
#include "Device/includes/BSDFs/OrenNayar.h"
#include "Device/includes/BSDFs/Principled.h"
#include "Device/includes/RayPayload.h"

/**
 * The random number generator passed here is used in case monte-carlo integration of the directional albedo
 * is enabled
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F bsdf_dispatcher_eval(const HIPRTRenderData& render_data, const DeviceEffectiveMaterial& material, RayVolumeState& ray_volume_state,
	const float3& view_direction, const float3& shading_normal, const float3& geometric_normal, const float3& to_light_direction, 
	float& pdf, Xorshift32Generator& random_number_generator)
{
#if BSDFOverride == BSDF_NONE
	/*switch (brdf_type)
	{
	...
	...
	default:
		break;
	}*/
#if PrincipledBSDFEnforceStrongEnergyConservation == KERNEL_OPTION_TRUE
    return principled_bsdf_eval_energy_compensated(render_data, material, ray_volume_state, view_direction, shading_normal, geometric_normal, to_light_direction, pdf, random_number_generator);
#else
    return principled_bsdf_eval(render_data, material, ray_volume_state, view_direction, shading_normal, to_light_direction, pdf);
#endif

#elif BSDFOverride == BSDF_LAMBERTIAN
	return lambertian_brdf_eval(material, hippt::dot(to_light_direction, shading_normal), pdf);
#elif BSDFOverride == BSDF_OREN_NAYAR
	return oren_nayar_brdf_eval<0>(material, view_direction, shading_normal, to_light_direction, pdf);
#elif BSDFOverride == BSDF_PRINCIPLED
    return principled_bsdf_eval(render_data, material, ray_volume_state, view_direction, shading_normal, to_light_direction, pdf);
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F bsdf_dispatcher_sample(const HIPRTRenderData& render_data, const DeviceEffectiveMaterial& material, RayVolumeState& ray_volume_state, const float3& view_direction, const float3& surface_normal, const float3& geometric_normal, float3& sampled_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
#if BSDFOverride == BSDF_NONE
	/*switch (brdf_type)
	{
	...
	...
	default:
		break;
	}*/
#if PrincipledBSDFEnforceStrongEnergyConservation == KERNEL_OPTION_TRUE
    return principled_bsdf_sample_energy_compensated(render_data, material, ray_volume_state, view_direction, surface_normal, geometric_normal, sampled_direction, pdf, random_number_generator);
#else
    return principled_bsdf_sample(render_data, material, ray_volume_state, view_direction, surface_normal, geometric_normal, sampled_direction, pdf, random_number_generator);
#endif

#elif BSDFOverride == BSDF_LAMBERTIAN
	return lambertian_brdf_sample(material, view_direction, surface_normal, sampled_direction, pdf, random_number_generator);
#elif BSDFOverride == BSDF_OREN_NAYAR
	return oren_nayar_brdf_sample(material, view_direction, surface_normal, sampled_direction, pdf, random_number_generator);
#elif BSDFOverride == BSDF_PRINCIPLED
    return principled_bsdf_sample(render_data, material, ray_volume_state, view_direction, surface_normal, geometric_normal, sampled_direction, pdf, random_number_generator);
#endif
}

#endif
