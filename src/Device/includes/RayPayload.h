/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RAY_PAYLOAD_H
#define DEVICE_RAY_PAYLOAD_H

#include "Device/includes/RayVolumeState.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/Material/MaterialUnpacked.h"

enum RayState
{
	BOUNCE,
	MISSED
};

struct RayPayload
{
	// Energy left in the ray after it bounces around the scene
	ColorRGB32F throughput = ColorRGB32F(1.0f);
	// Final color of the ray
	ColorRGB32F ray_color = ColorRGB32F(0.0f);
	// Camera ray is "Bounce" to give it a chance to hit the scene
	RayState next_ray_state = RayState::BOUNCE;

	// What bounce we're currently at
	int bounce = 0;
	// Roughness accumulated by the bounces of the ray along the path. In [0, 1]
	//
	// If a ray bounced on a Lambertian surface along its path for example, the
	// accumulated roughness is going to be 1.0f.
	//
	// If the camera ray bounced on a mirror, the accumulated roughness is going to be 0.0f at bounce == 1.
	//
	// If the ray bounced on a specular  diffuse surface, the accumulated roughness is going to be that
	// of which lobe was sampled between the specular or diffuse
	//
	// The accumulated roughness is computed as the maximum between the current accumulated roughness
	// and the roughness of the lobe that was sampled to get the next bounce direction
	float accumulated_roughness = 0.0f;
	
	// Material of the current hit
	DeviceUnpackedEffectiveMaterial material;

	RayVolumeState volume_state;
	
	HIPRT_HOST_DEVICE void accumulate_roughness(BSDFIncidentLightInfo sampled_lobe)
	{
		switch (sampled_lobe)
		{
		case LIGHT_DIRECTION_SAMPLED_FROM_DIFFUSE_LOBE:
		case LIGHT_DIRECTION_SAMPLED_FROM_DIFFUSE_TRANSMISSION_LOBE:
			accumulated_roughness = 1.0f;
			break;

		case LIGHT_DIRECTION_SAMPLED_FROM_COAT_LOBE:
			accumulated_roughness = hippt::max(material.coat_roughness, accumulated_roughness);
			break;

		case LIGHT_DIRECTION_SAMPLED_FROM_FIRST_METAL_LOBE:
			accumulated_roughness = hippt::max(material.roughness, accumulated_roughness);
			break;

		case LIGHT_DIRECTION_SAMPLED_FROM_SECOND_METAL_LOBE:
			accumulated_roughness = hippt::max(material.second_roughness, accumulated_roughness);
			break;

		case LIGHT_DIRECTION_SAMPLED_FROM_SPECULAR_LOBE:
			// The specular roughness is just material.roughness
			// accumulated_roughness = hippt::max(material.roughness, accumulated_roughness);
			accumulated_roughness = hippt::max(1.0f, accumulated_roughness);
			break;

		case LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFLECT_LOBE:
			// The glass roughness is just material.roughness
			accumulated_roughness = hippt::max(material.roughness, accumulated_roughness);
			break;

		case LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFRACT_LOBE:
			// The glass roughness is just material.roughness
			accumulated_roughness = hippt::max(material.roughness, accumulated_roughness);
			break;

		default:
			break;
		}
	}
};

#endif
