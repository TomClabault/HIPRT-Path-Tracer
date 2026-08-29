/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_PATH_GUIDING_PATH_SAMPLING_H
#define DEVICE_INCLUDES_PATH_GUIDING_PATH_SAMPLING_H

#include "Device/includes/BSDFs/BSDFIncidentLightInfo.h"
#include "Device/includes/HitInfo.h"

HIPRT_DEVICE void path_guiding_update_volume_stack_after_sampling(const float3_t& bounce_direction, const HitInfo& closest_hit_info, RayPayload& ray_payload)
{
	if (hippt::dot(bounce_direction, closest_hit_info.shading_normal) < 0.0f && ray_payload.material.allows_refraction())
	{
		// Sampled a refraction
		if (ray_payload.material.thin_walled)
			// If thin walled, there is no interior so we're always popping the stack
			ray_payload.volume_state.interior_stack.pop(ray_payload.volume_state.inside_material);
		else
			// If not thin walled, we're going to pop the stack only if refracting out of the material and this is going to be done when evaluating the BSDF,
			// not sampling it so there is nothing to do here
			;
	}
	else
		// Always popping for reflections
		ray_payload.volume_state.interior_stack.pop(ray_payload.volume_state.inside_material);
}

HIPRT_DEVICE void path_guiding_update_volume_stack_after_eval(const float3_t& bounce_direction, const HitInfo& closest_hit_info, RayPayload& ray_payload)
{
	if (hippt::dot(bounce_direction, closest_hit_info.shading_normal) < 0.0f && ray_payload.material.allows_refraction())
	{
		// We only need to worry about popping if we're refracting out of a material

		if (ray_payload.volume_state.inside_material || ray_payload.material.thin_walled)
		{
			// We changed volume so we're resetting the distance
			ray_payload.volume_state.distance_in_volume = 0.0f;
			// We refracting out of a volume so we're poping the stack
			ray_payload.volume_state.interior_stack.pop(ray_payload.volume_state.inside_material);
		}
	}
}

HIPRT_DEVICE void path_guiding_compute_sampled_lobe(HIPRTRenderData& render_data,
													BSDFContext& bsdf_context,
													RayPayload& ray_payload,
													float3_t sampled_direction,
													BSDFIncidentLightInfo& out_sampled_light_info,
													Xorshift32Generator& rng)
{
#if BSDFOverride == BSDF_NONE || BSDFOverride == BSDF_PRINCIPLED
	// Stochastically simulating which lobe the BSDF would have sampled

	float coat_sampling_proba, sheen_sampling_proba, metal_1_sampling_proba;
	float metal_2_sampling_proba, retro_reflection_sampling_proba, specular_sampling_proba, diffuse_sampling_proba;
	float glass_sampling_proba, diffuse_transmission_sampling_proba;
	principled_bsdf_get_lobes_sampling_proba(render_data, bsdf_context.material, hippt::dot(bsdf_context.view_direction, bsdf_context.shading_normal),
											 bsdf_context.volume_state, coat_sampling_proba, sheen_sampling_proba, metal_1_sampling_proba,
											 metal_2_sampling_proba, retro_reflection_sampling_proba, specular_sampling_proba, diffuse_sampling_proba,
											 glass_sampling_proba, diffuse_transmission_sampling_proba);

	float rand_value = rng();
	if (rand_value < coat_sampling_proba)
		out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_COAT_LOBE;
	else if (rand_value < coat_sampling_proba + sheen_sampling_proba)
		out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_DIFFUSE_LOBE;
	else if (rand_value < coat_sampling_proba + sheen_sampling_proba + metal_1_sampling_proba)
		out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_FIRST_METAL_LOBE;
	else if (rand_value < coat_sampling_proba + sheen_sampling_proba + metal_1_sampling_proba + metal_2_sampling_proba)
		out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_SECOND_METAL_LOBE;
	else if (rand_value < coat_sampling_proba + sheen_sampling_proba + metal_1_sampling_proba + metal_2_sampling_proba + retro_reflection_sampling_proba)
		out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_RETRO_REFLECTION_LOBE;
	else if (rand_value < coat_sampling_proba + sheen_sampling_proba + metal_1_sampling_proba + metal_2_sampling_proba + retro_reflection_sampling_proba +
							  specular_sampling_proba)
		out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_SPECULAR_LOBE;
	else if (rand_value < coat_sampling_proba + sheen_sampling_proba + metal_1_sampling_proba + metal_2_sampling_proba + retro_reflection_sampling_proba +
							  specular_sampling_proba + diffuse_sampling_proba)
		out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_DIFFUSE_LOBE;
	else if (rand_value < coat_sampling_proba + sheen_sampling_proba + metal_1_sampling_proba + metal_2_sampling_proba + retro_reflection_sampling_proba +
							  specular_sampling_proba + diffuse_transmission_sampling_proba)
		out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_DIFFUSE_TRANSMISSION_LOBE;
	else if (rand_value < coat_sampling_proba + sheen_sampling_proba + metal_1_sampling_proba + metal_2_sampling_proba + retro_reflection_sampling_proba +
							  specular_sampling_proba + diffuse_sampling_proba + diffuse_transmission_sampling_proba + glass_sampling_proba)
	{
		if (hippt::dot(sampled_direction, bsdf_context.shading_normal) < 0.0f)
			out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFRACT_LOBE;
		else
			out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFLECT_LOBE;
	}
#elif BSDFOverride == BSDF_LAMBERTIAN || BSDFOverride == BSDF_OREN_NAYAR // #if BSDFOverride == BSDF_NONE || BSDFOverride == BSDF_PRINCIPLED
	out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_DIFFUSE_LOBE;
#endif // #if BSDFOverride == BSDF_NONE || BSDFOverride == BSDF_PRINCIPLED

	ray_payload.accumulate_roughness(out_sampled_light_info);
}

#endif // #ifndef DEVICE_INCLUDES_PATH_GUIDING_PATH_SAMPLING_H
