/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RESTIR_DI_RESERVOIRS_H
#define RESTIR_DI_RESERVOIRS_H

#include "Device/includes/ReSTIR/ReSTIR_DI_Reservoir.h"
#include "HIPRT-Orochi/OrochiBuffer.h"

struct ReSTIR_DI_Reservoirs
{
	// ReSTIR reservoirs for the initial candidates
	OrochiBuffer<ReSTIRDIReservoir> initial_candidates_reservoirs;
	// ReSTIR reservoirs for the output of the spatial reuse pass
	OrochiBuffer<ReSTIRDIReservoir> spatial_reuse_output_2;
	// ReSTIR DI final reservoirs of the frame. 
	// This the output of the spatial reuse passes.
	// Those are the reservoirs that are carried over between frames for
	// the temporal reuse pass to feed upon
	OrochiBuffer<ReSTIRDIReservoir> spatial_reuse_output_1;
};

#endif