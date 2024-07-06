/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RAY_PAYLOAD_H
#define DEVICE_RAY_PAYLOAD_H

#include "Device/includes/NestedDielectrics.h"
#include "HostDeviceCommon/Color.h"

enum RayState
{
	BOUNCE,
	MISSED
};

struct RayVolumeState
{
	// How far has the ray traveled in the current volume.
	float distance_in_volume = 0.0f;
	// The stack of materials being traversed. Used for nested dielectrics handling
	InteriorStackImpl<InteriorStackStrategy> interior_stack;
	// Indices of the material we were in before hitting the current dielectric surface
	int incident_mat_index = -1, outgoing_mat_index = -1;
	// Whether or not we're exiting a material
	bool leaving_mat = false;
};

struct RayPayload
{
	// Energy left in the ray after it bounces around the scene
	ColorRGB throughput = ColorRGB(1.0f);
	// Final color of the ray
	ColorRGB ray_color = ColorRGB(0.0f);
	// Camera ray is "Bounce" to give it a chance to hit the scene
	RayState next_ray_state = RayState::BOUNCE;
	// Type of BRDF found at the last intersection
	BRDF last_brdf_hit_type = BRDF::Uninitialized;

	// Material of the last hit
	RendererMaterial material;

	RayVolumeState volume_state;

	HIPRT_HOST_DEVICE bool is_inside_volume() const
	{
		return volume_state.interior_stack.stack_position > 0;
	}
};

#endif
