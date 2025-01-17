/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef LIGHT_PRESAMPLING_KERNEL_PARAMETERS_H
#define LIGHT_PRESAMPLING_KERNEL_PARAMETERS_H

#include "Device/includes/ReSTIR/DI/PresampledLight.h"
#include "Device/includes/ReSTIR/DI/Reservoir.h"

#include "HostDeviceCommon/Material/MaterialPackedSoA.h"
#include "HostDeviceCommon/WorldSettings.h"

struct LightPresamplingParameters
{
	/**
	 * Parameters specific to the kernel
	 */

	// From all the lights of the scene, how many subsets to presample
	int number_of_subsets = 128;
	// How many lights to presample in each subset
	int subset_size = 1024;
	// Buffer that holds the presampled lights
	ReSTIRDIPresampledLight* out_light_samples;

	/**
	 * Generic parameters needed by the kernel
	 */
	int emissive_triangles_count = 0;
	int* emissive_triangles_indices = nullptr;
	int* triangles_indices = nullptr;
	float3* vertices_positions = nullptr;
	int* material_indices = nullptr;
	DevicePackedTexturedMaterialSoA materials;

	// World settings for sampling the envmap
	WorldSettings world_settings;

	bool freeze_random = false;
	int sample_number = 0;
	// Random seed for varying the randomness between frames
	int random_seed = 42;

	// For each presampled light, the probability that this is going to be an envmap sample
	float envmap_sampling_probability;
};

#endif
