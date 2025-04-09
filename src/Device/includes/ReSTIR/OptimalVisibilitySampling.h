/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_OPTIMAL_VISIBILITY_SAMPLING_H
#define DEVICE_RESTIR_OPTIMAL_VISIBILITY_SAMPLING_H

#include "Device/includes/ReSTIR/MISWeightsCommon.h" // For the ReSTIRReservoirType
#include "Device/includes/ReSTIR/Utils.h" // For the ReSTIRReservoirType

#include "HostDeviceCommon/KernelOptions/ReSTIRDIOptions.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRGIOptions.h"
#include "HostDeviceCommon/RenderData.h"

template <bool IsReSTIRGI>
HIPRT_HOST_DEVICE bool ReSTIR_optimal_visibility_sampling(HIPRTRenderData& render_data, 
	ReSTIRReservoirType<IsReSTIRGI>& spatial_reuse_output_reservoir, ReSTIRSurface& center_pixel_surface, 
	int neighbor_index, int reused_neighbors_count, 
	Xorshift32Generator& random_number_generator)
{
#if ReSTIR_DI_DoOptimalVisibilitySampling == KERNEL_OPTION_TRUE || ReSTIR_GI_DoOptimalVisibilitySampling == KERNEL_OPTION_TRUE
	bool at_least_one_neighbor_resampled = spatial_reuse_output_reservoir.weight_sum > 0.0f;
	bool last_neighbor_before_canonical = neighbor_index == reused_neighbors_count - 1;
	constexpr bool ovs_enabled = (!IsReSTIRGI && ReSTIR_DI_DoOptimalVisibilitySampling == KERNEL_OPTION_TRUE) || (IsReSTIRGI && ReSTIR_GI_DoOptimalVisibilitySampling == KERNEL_OPTION_TRUE);
	if (at_least_one_neighbor_resampled && last_neighbor_before_canonical && ovs_enabled)
	{
		// If the spatial neighbors resampled up until now are occluded, they will be discarded by this
		// visiblity test and so the canonical sample will be the resulting reservoir

		bool reservoir_killed;
		if constexpr (IsReSTIRGI)
			reservoir_killed = ReSTIR_GI_visibility_validation(render_data, spatial_reuse_output_reservoir, center_pixel_surface.shading_point, center_pixel_surface.last_hit_primitive_index, random_number_generator);
		else
			reservoir_killed = ReSTIR_DI_visibility_test_kill_reservoir(render_data, spatial_reuse_output_reservoir, center_pixel_surface.shading_point, center_pixel_surface.last_hit_primitive_index, random_number_generator);

		if (reservoir_killed)
			spatial_reuse_output_reservoir.weight_sum = 0.0f;

		return reservoir_killed;
	}
#endif

	return false;
}

#endif
