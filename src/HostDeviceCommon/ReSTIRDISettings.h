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

struct TemporalPassSettings
{
	bool do_temporal_reuse_pass = true;

	// The temporal reuse pass resamples the initial candidates as well as the last frame reservoirs which
	// are accessed through this pointer
	ReSTIRDIReservoir* input_reservoirs = nullptr;
	// Buffer that holds the output of the temporal reuse pass
	ReSTIRDIReservoir* output_reservoirs = nullptr;
};

struct SpatialPassSettings
{
	// How many spatial reuse pass to perform
	int number_of_passes = 1;
	// The radius within which neighbor are going to be reused spatially
	int spatial_reuse_radius = 32;
	// How many neighbors to reuse during the spatial pass
	int spatial_reuse_neighbor_count = 5;

	// When finalizing the reservoir in the spatial reuse pass, what value
	// to cap the reservoirs's M value to.
	//
	// The point of this parameter is to avoid too much correlation between frames if using
	// a bias correction that uses confidence weights. Without M-capping, the M value of a reservoir
	// will keep growing exponentially through temporal and spatial reuse and when that exponentially
	// grown M value is used in confidence weights, it results in new samples being very unlikely 
	// to be chosen which in turn results in non-convergence since always the same sample is evaluated
	// for a given pixel.
	//
	// A M-cap value between 5 - 30 is usually good
	int m_cap = 10;

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
	// Settings for the temporal reuse pass
	TemporalPassSettings temporal_pass;
	// Settings for the spatial reuse pass
	SpatialPassSettings spatial_pass;
	// Settings for the target function used in all passes of ReSTIR DI
	ReSTIRDITargetFunctionSettings target_function;
};

#endif
