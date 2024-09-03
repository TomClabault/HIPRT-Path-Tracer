/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RESTIR_DI_RESERVOIRS_H
#define RESTIR_DI_RESERVOIRS_H

#include "Device/includes/ReSTIR/ReSTIR_DI_Reservoir.h"
#include "HIPRT-Orochi/OrochiBuffer.h"

class GPURenderer;

struct ReSTIR_DI_State
{
	// ReSTIR reservoirs for the initial candidates
	OrochiBuffer<ReSTIRDIReservoir> initial_candidates_reservoirs;
	// ReSTIR reservoirs for the output of the spatial reuse pass
	OrochiBuffer<ReSTIRDIReservoir> spatial_reuse_output_1;
	// ReSTIR DI final reservoirs of the frame. 
	// This the output of the spatial reuse passes.
	// Those are the reservoirs that are carried over between frames for
	// the temporal reuse pass to feed upon
	OrochiBuffer<ReSTIRDIReservoir> spatial_reuse_output_2;

	// Whether or not we're currently rendering an odd frame.
	// This is used to adjust which buffers are used as input/outputs
	// and ping-pong between them
	bool odd_frame = false;
};

#endif