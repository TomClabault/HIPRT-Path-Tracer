/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_RESTIR_GI_SETTINGS_H
#define HOST_DEVICE_RESTIR_GI_SETTINGS_H

#include "HostDeviceCommon/ReSTIR/ReSTIRGIDefaultSettings.h"

struct ReSTIRGIReservoir;

struct ReSTIRGIInitialCandidatesPassSettings
{
	// Buffer that contains the reservoirs that will hold the reservoir
	// for the initial candidates generated
	ReSTIRGIReservoir* initial_candidates_buffer = nullptr;
};

struct ReSTIRGITemporalPassSettings
{
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

enum ReSTIRGIDebugView
{
	NO_DEBUG = 0,
	FINAL_RESERVOIR_UCW = 1,
	TARGET_FUNCTION = 2,
	WEIGHT_SUM = 3,
};

struct ReSTIRGISettings : public ReSTIRCommonSettings
{
	ReSTIRGISettings() : ReSTIRCommonSettings(RESTIR_GI_DEFAULT_COMMON_SETTINGS) 
	{
		// Doing this one manually because we can't have complex evaluation in the declaration of
		// 'RESTIR_DI_DEFAULT_COMMON_SETTINGS'
		common_spatial_pass.neighbor_visibility_count = common_spatial_pass.do_disocclusion_reuse_boost ? common_spatial_pass.disocclusion_reuse_count : common_spatial_pass.reuse_neighbor_count;
	}

	ReSTIRGIInitialCandidatesPassSettings initial_candidates;
	ReSTIRGITemporalPassSettings temporal_pass;
	ReSTIRGISpatialPassSettings spatial_pass;
	
	ReSTIRGIReservoir* restir_output_reservoirs = nullptr;

	ReSTIRGIDebugView debug_view = ReSTIRGIDebugView::NO_DEBUG;
	float debug_view_scale_factor = 0.04f;
};

#endif
