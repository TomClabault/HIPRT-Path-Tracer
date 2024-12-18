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
	bool inside_material = false;

	// For spectral dispersion. A random wavelength is sampled and replaces this value
	// when a glass object is hit. This wavelength can then be used to determine the IOR
	// that should be used for refractions/reflections on the dielectric object. 
	// 
	// The wavelength is also used to apply a throughput filter on the ray such that only the
	// sampled wavelength's color travels around the scene.
	//
	// If this value is negative, this is because the ray throughput filter hasn't been applied
	// yet. If the value is positive, the filter has been applied
	float sampled_wavelength = 0.0f;
};

#endif
