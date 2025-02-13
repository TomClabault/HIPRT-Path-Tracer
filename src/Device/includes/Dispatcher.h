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
 * The 'random_number_generator' passed here is used only in case 
 * monte-carlo integration of the directional albedo is enabled
 * 
 * If 'update_ray_volume_state' is passed as true, the givenargument is passed as nullptr, the volume state of the ray won't
 * be updated by this sample call (i.e. the ray won't track if this sample call made it exit/enter a new material)
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F bsdf_dispatcher_eval(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, 
	RayVolumeState& ray_volume_state, bool update_ray_volume_state,
	const float3& view_direction, const float3& shading_normal, const float3& geometric_normal, const float3& to_light_direction, 
	float& pdf, Xorshift32Generator& random_number_generator,
	int current_bounce, BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO)
{
#if BSDFOverride == BSDF_NONE || BSDFOverride == BSDF_PRINCIPLED
	/*switch (brdf_type)
	{
	...
	...
	default:
		break;
	}*/
#if PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE && PrincipledBSDFEnforceStrongEnergyConservation == KERNEL_OPTION_TRUE
    return principled_bsdf_eval_energy_compensated(render_data, material, ray_volume_state, update_ray_volume_state, 
												   view_direction, shading_normal, geometric_normal, to_light_direction, 
												   pdf, random_number_generator, BSDFIncidentLightInfo::NO_INFO, current_bounce);
#else
    return principled_bsdf_eval(render_data, material, ray_volume_state, update_ray_volume_state, 
								view_direction, shading_normal, to_light_direction, 
								pdf, current_bounce, incident_light_info);
#endif

#elif BSDFOverride == BSDF_LAMBERTIAN
	return lambertian_brdf_eval(material, hippt::dot(to_light_direction, shading_normal), pdf);
#elif BSDFOverride == BSDF_OREN_NAYAR
	return oren_nayar_brdf_eval<0>(material, view_direction, shading_normal, to_light_direction, pdf);
#endif
}

/**
 * If the 'ray_volume_state' argument is passed as nullptr, the volume state of the ray won't
 * be updated by this sample call (i.e. the ray won't track if this sample call made it exit/enter a new material)
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F bsdf_dispatcher_sample(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, 
	RayVolumeState& ray_volume_state, bool update_ray_volume_state,
	const float3& view_direction, const float3& surface_normal, const float3& geometric_normal, float3& sampled_direction, 
	float& pdf, Xorshift32Generator& random_number_generator,
	int current_bounce, BSDFIncidentLightInfo* out_sampled_light_info = nullptr)
{
#if BSDFOverride == BSDF_NONE || BSDFOverride == BSDF_PRINCIPLED
	/*switch (brdf_type)
	{
	...
	...
	default:
		break;
	}*/
#if PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE && PrincipledBSDFEnforceStrongEnergyConservation == KERNEL_OPTION_TRUE
    return principled_bsdf_sample_energy_compensated(render_data, material, ray_volume_state, update_ray_volume_state, 
													 view_direction, surface_normal, geometric_normal, sampled_direction, 
													 pdf, random_number_generator, current_bounce);
#else
    return principled_bsdf_sample(render_data, material, ray_volume_state, update_ray_volume_state, 
								  view_direction, surface_normal, geometric_normal, sampled_direction, 
								  pdf, random_number_generator, current_bounce, out_sampled_light_info);
#endif

#elif BSDFOverride == BSDF_LAMBERTIAN
	return lambertian_brdf_sample(render_data, material, surface_normal, sampled_direction, pdf, random_number_generator, out_sampled_light_info);
#elif BSDFOverride == BSDF_OREN_NAYAR
	return oren_nayar_brdf_sample(material, view_direction, surface_normal, sampled_direction, pdf, random_number_generator, out_sampled_light_info);
#endif
}

#endif
