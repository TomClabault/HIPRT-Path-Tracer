#ifndef DEVICE_INCLUDES_RESTIR_PG_DISTRIBUTION_SUFFICIENT_STATISTICS_H
#define DEVICE_INCLUDES_RESTIR_PG_DISTRIBUTION_SUFFICIENT_STATISTICS_H

#include "Device/includes/ReSTIR/PG/SplattingSample.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRPGOptions.h"
#include "HostDeviceCommon/Maths/VecTypes.h"

struct ReSTIRPGDistributionSufficientStatisticsSoADevice
{
	HIPRT_DEVICE float3_t read_directions_sum(unsigned int component_index, unsigned int hash_grid_cell_index, unsigned int total_number_of_cells)
	{
		unsigned int index = component_index * total_number_of_cells + hash_grid_cell_index;

		return make_float3(directions_sum_x[index], directions_sum_y[index], directions_sum_z[index]);
	}

	// Number of cells * ReSTIRPGDistributionComponentCount in size
	//
	// Should be indexed with [component_index * total_number_of_cells + hash_grid_cell_index]
	//
	// Directions sums * responsibility, accumulated by the splatting pass
	AtomicType<float>* directions_sum_x = nullptr;
	AtomicType<float>* directions_sum_y = nullptr;
	AtomicType<float>* directions_sum_z = nullptr;

	// Should be indexed with [component_index * total_number_of_cells + hash_grid_cell_index]
	AtomicType<float>* responsibility_weights_sum = nullptr;
	// How many samples were accumulated in this sufficient statistics. This is the same for all components
	// Number of cells in size
	//
	// Should be indexed with [hash_grid_cell_index]
	AtomicType<unsigned int>* sample_count = nullptr;
};

#endif // #ifndef DEVICE_INCLUDES_RESTIR_PG_DISTRIBUTION_SUFFICIENT_STATISTICS_H
