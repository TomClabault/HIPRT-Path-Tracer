#ifndef RAY_PAYLOAD_H
#define RAY_PAYLOAD_H

#include "HostDeviceCommon/Color.h"

enum RayState
{
	BOUNCE,
	MISSED
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
	// How has the ray traveled in the current volume.
	float distance_in_volume = 0.0f;
	// Are we currently inside a volume?
	bool inside_volume = false;
};

#endif
