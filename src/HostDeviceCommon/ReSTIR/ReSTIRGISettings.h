/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_RESTIR_GI_SETTINGS_H
#define HOST_DEVICE_RESTIR_GI_SETTINGS_H

struct ReSTIRGIReservoir;
struct ReSTIRGISample;

struct ReSTIRGIInitialCandidatesPassSettings
{
	// Buffer that contains the reservoirs that will hold the reservoir
	// for the initial candidates generated
	ReSTIRGISample* initial_candidates_buffer = nullptr;
};

struct ReSTIRGISpatialPassSettings
{
	// Buffer that contains the input reservoirs for the spatial reuse pass
	ReSTIRGIReservoir* input_reservoirs = nullptr;
	// Buffer that contains the output reservoir of the spatial reuse pass
	ReSTIRGIReservoir* output_reservoirs = nullptr;
};

struct ReSTIRGISettings
{
	ReSTIRGIInitialCandidatesPassSettings initial_candidates;
	ReSTIRGISpatialPassSettings spatial_pass;
};

#endif
