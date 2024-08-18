/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_RESTIR_DI_SETTINGS_H
#define HOST_DEVICE_RESTIR_DI_SETTINGS_H

#include "Device/includes/ReSTIR/ReSTIR_DI_Reservoir.h"

struct InitialCandidatesSettings
{
	// How many light candidates to resamples during the initial candidates sampling pass
	int number_of_initial_light_candidates = 4;
	// How many BSDF candidates to resamples during the initial candidates sampling pass
	int number_of_initial_bsdf_candidates = 1;

	// Buffer that contains the reservoirs that will hold the reservoir
	// for the initial candidates generated
	ReSTIRDIReservoir* output_reservoirs = nullptr;
};

struct SpatialPassSettings
{
	// How many spatial reuse pass to perform
	int number_of_passes = 2;
	// The radius within which neighbor are going to be reused spatially
	int spatial_reuse_radius = 32;
	// How many neighbors to reuse during the spatial pass
	int spatial_reuse_neighbor_count = 5;

	// Buffer that contains the input reservoirs for the spatial reuse pass
	ReSTIRDIReservoir* input_reservoirs = nullptr;
	// Buffer that contains the output reservoir of the spatial reuse pass
	ReSTIRDIReservoir* output_reservoirs = nullptr;
};

struct ReSTIRDITargetFunctionSettings
{
	// Whether or not to include the geometry term in the target function when resampling neighbors
	// Defaults to false because of numeric instability when dividing by very small distance to light
	bool geometry_term_in_target_function = false;
};

struct ReSTIRDISettings
{
	// Settings for the initial candidates generation pass
	InitialCandidatesSettings initial_candidates;
	// Settings for the spatial reuse pass
	SpatialPassSettings spatial_pass;
	// Settings for the target function used in all passes of ReSTIR DI
	ReSTIRDITargetFunctionSettings target_function;
};

#endif
