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

	// Material index of the last hit
	//unsigned int material_index;
	// Material of the last hit
	DeviceUnpackedEffectiveMaterial material;

	RayVolumeState volume_state;

	HIPRT_HOST_DEVICE bool is_inside_volume() const
	{
		// TODO this is not general and calling this function in
		// the principled BSDF sample function before poping the stack
		// (when sampling a reflection) would return true even if we're out of any volumes
		return volume_state.interior_stack.stack_position > 0;
	}
};

#endif
