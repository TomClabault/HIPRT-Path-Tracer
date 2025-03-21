/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_BSDF_CONTEXT_H
#define DEVICE_INCLUDES_BSDF_CONTEXT_H

#include "Device/includes/BSDFs/BSDFIncidentLightInfo.h"
#include "Device/includes/RayVolumeState.h"

struct BSDFContext
{
	float3 view_direction = make_float3(-1.0f, -1.0f, -1.0f);
	float3 shading_normal = make_float3(-1.0f, -1.0f, -1.0f);
	float3 geometric_normal = make_float3(-1.0f, -1.0f, -1.0f);
	float3 to_light_direction = make_float3(-1.0f, -1.0f, -1.0f);

	BSDFIncidentLightInfo* incident_light_info = nullptr;

	// Whether or not to modify the volume state of the ray as the BSDF is sampled / evaluated.
	//
	// For example, if the ray is currently refracting out of a glass material, and 'update_ray_volume_state' == true,
	// the ray volume state of the ray will be updated and the glass object will be popped out of the
	// nested dielectrics stack
	bool update_ray_volume_state = false;
	RayVolumeState* volume_state = nullptr;
	DeviceUnpackedEffectiveMaterial material;

	int current_bounce = 0;
	float accumulated_path_roughness = 0.0f;

	/**
	 * 'to_light_direction' is only needed if evaluating the BSDF // TODO create a separate eval context and sampling context
	 * 'incident_light_info' can be passed nullptr if you don't care about what lobe the BSDF sampled of if you don't have the information about
	 * what lobe the 'to_light_direction' comes from (during NEE light sampling for example)
	 */
	HIPRT_HOST_DEVICE static BSDFContext create_context(float3 view_direction, float3 shading_normal, float3 geometric_normal, float3 to_light_direction,
		BSDFIncidentLightInfo* incident_light_info,
		RayVolumeState* ray_volume_state, bool update_ray_volume_state,
		DeviceUnpackedEffectiveMaterial material,
		int current_bounce, float accumulated_path_roughness)
	{
		BSDFContext context;

		context.view_direction = view_direction;
		context.geometric_normal = geometric_normal;
		context.shading_normal = shading_normal;
		context.to_light_direction = to_light_direction;

		context.incident_light_info = incident_light_info;

		context.volume_state = ray_volume_state;
		context.update_ray_volume_state = update_ray_volume_state;

		context.material = material;
		context.current_bounce = current_bounce;
		context.accumulated_path_roughness = accumulated_path_roughness;

		return context;
	}
};

#endif