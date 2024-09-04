/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RAY_VOLUME_STATE_H
#define DEVICE_RAY_VOLUME_STATE_H

#include "Device/includes/NestedDielectrics.h"

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

#endif
