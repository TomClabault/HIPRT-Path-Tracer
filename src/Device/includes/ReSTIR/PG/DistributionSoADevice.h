/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_PG_DISTRIBUTION_SOA_DEVICE_H
#define DEVICE_INCLUDES_RESTIR_PG_DISTRIBUTION_SOA_DEVICE_H

#include "Device/includes/ReSTIR/PG/Distribution.h"
#include "Device/includes/VMF.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRPGOptions.h"

struct ReSTIRPGDistributionSoADevice
{
	// The buffers are size [HashGridCellCount * ReSTIRPGDistributionComponentCount], all components 0 of all distributions are stored in the first part of the
	// buffers, and then all components 1, and then all components 2, ..... to allow for coalesced memory access when all threads of a warp want to fetch their
	// component of their distribution
	//
	// So the buffers are indexed as [hash_grid_cell_index * ReSTIRPGDistributionComponentCount + component_index]
	float3_t* axis			= nullptr;
	float* sharpness		= nullptr;
	float* component_weight = nullptr;

	HIPRT_DEVICE ReSTIRPGDistribution get_distribution(unsigned int hash_grid_cell_index) const
	{
		ReSTIRPGDistribution out_distribution;

		for (int i = 0; i < ReSTIRPGDistributionComponentCount; ++i)
		{
			unsigned int component_index = index_buffer(hash_grid_cell_index, i);

			out_distribution.distribution_components[i].vmf.axis	  = axis[component_index];
			out_distribution.distribution_components[i].vmf.sharpness = sharpness[component_index];
			out_distribution.distribution_components[i].weight		  = component_weight[component_index];
		}

		return out_distribution;
	}

	HIPRT_DEVICE void set_distribution_component_vmf(unsigned int hash_grid_cell_index, unsigned int component_index, const VMF& vmf)
	{
		unsigned int buffer_index = index_buffer(hash_grid_cell_index, component_index);

		axis[buffer_index]		= vmf.axis;
		sharpness[buffer_index] = vmf.sharpness;
	}

	HIPRT_DEVICE void set_distribution_component_weight(unsigned int hash_grid_cell_index, unsigned int component_index, float weight)
	{
		unsigned int buffer_index = index_buffer(hash_grid_cell_index, component_index);

		component_weight[buffer_index] = weight;
	}

private:
	HIPRT_DEVICE unsigned int index_buffer(unsigned int hash_grid_cell_index, unsigned int component_index) const
	{
		return hash_grid_cell_index * ReSTIRPGDistributionComponentCount + component_index;
	}
};

#endif
