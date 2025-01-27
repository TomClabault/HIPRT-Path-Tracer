/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_RESTIR_GI_SETTINGS_H
#define HOST_DEVICE_RESTIR_GI_SETTINGS_H

struct ReSTIRGIReservoir;

struct ReSTIRGIInitialCandidatesPassSettings
{
	// Buffer that contains the reservoirs that will hold the reservoir
	// for the initial candidates generated
	ReSTIRGIReservoir* initial_candidates_buffer = nullptr;
};

struct ReSTIRGITemporalPassSettings
{
	bool temporal_buffer_clear_requested = true;

	// Buffer that contains the input reservoirs for the temporal reuse pass
	ReSTIRGIReservoir* input_reservoirs = nullptr;
	// Buffer that contains the output reservoir of the temporal reuse pass
	ReSTIRGIReservoir* output_reservoirs = nullptr;
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
	ReSTIRGITemporalPassSettings temporal_pass;
	ReSTIRGISpatialPassSettings spatial_pass;
};

#endif
