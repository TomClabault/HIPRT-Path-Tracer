/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RESTIR_DI_RESERVOIRS_H
#define RESTIR_DI_RESERVOIRS_H

#include "HIPRT-Orochi/OrochiBuffer.h"

class GPURenderer;
class ReSTIRDIReservoir;

struct ReSTIR_DI_State
{
	// ReSTIR reservoirs for the initial candidates
	OrochiBuffer<ReSTIRDIReservoir> initial_candidates_reservoirs;
	// ReSTIR reservoirs for the output of the spatial reuse pass
	OrochiBuffer<ReSTIRDIReservoir> spatial_output_reservoirs_1;
	// ReSTIR DI final reservoirs of the frame. 
	// This the output of the spatial reuse passes.
	// Those are the reservoirs that are carried over between frames for
	// the temporal reuse pass to feed upon
	OrochiBuffer<ReSTIRDIReservoir> spatial_output_reservoirs_2;

	// Buffer that holds the presampled lights if light presampling is enabled 
	// (GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHTS_PRESAMPLING)
	//
	// Implementation from the paper
	// [Rearchitecting Spatiotemporal Resampling for Production] https://research.nvidia.com/publication/2021-07_rearchitecting-spatiotemporal-resampling-production
	OrochiBuffer<ReSTIRDIPresampledLight> presampled_lights_buffer;

	// Whether or not we're currently rendering an odd frame.
	// This is used to adjust which buffers are used as input/outputs
	// and ping-pong between them
	bool odd_frame = false;

	// Events for timing the cumulated render time of all the spatial reuses passes
	oroEvent_t spatial_reuse_time_start;
	oroEvent_t spatial_reuse_time_stop;
};

#endif