/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTIR_PG_SETTINGS_H
#define HOST_DEVICE_COMMON_RESTIR_PG_SETTINGS_H

#include "Device/includes/ReSTIR/PG/DistributionSoADevice.h"
#include "Device/includes/ReSTIR/PG/DistributionSufficientStatisticsSoADevice.h"
#include "Device/includes/ReSTIR/PG/SplattingSample.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRPGOptions.h"

struct ReSTIRPGSettings
{
	// Splatting samples buffer. This is used to store the samples that are going to be splatted into the grid in the splatting pass of ReSTIR PG.
	//
	// Screen space size * (number of bounces - 1)
	ReSTIRPGSplattingSample* splatting_samples = nullptr;

	ReSTIRPGDistributionSoADevice hash_grid_distributions_soa;
	AtomicType<unsigned int>* hash_grid_checksums	= nullptr;
	AtomicType<unsigned int>* grid_cell_alive		= nullptr;
	AtomicType<unsigned int>* grid_cell_alive_count = nullptr;
	unsigned int* grid_cell_alive_list				= nullptr;

	// Buffers used during the splatting phase to accumulate sample data (expectation phase of the EM algorithm)
	AtomicType<unsigned int>* hash_grid_distributions_sufficient_statistics_lock = nullptr;
	ReSTIRPGDistributionSufficientStatisticsSoADevice hash_grid_distributions_sufficient_statistics_soa;

	unsigned int hash_grid_total_number_of_cells = 0;
	float hash_grid_target_projected_size		 = 10.0f;
	float hash_grid_cell_min_size				 = 0.1f;

	// For RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_DIRECTION
	int debug_distribution_component_direction_number = 0;

	HIPRT_DEVICE unsigned int get_hash_grid_cell_index_from_position_data(float3_t position,
																		  float3_t surface_normal,
																		  const HIPRTCamera& current_camera,
																		  unsigned int& out_checksum) const
	{
		return hash_pos_distance_to_camera(position, surface_normal, current_camera, hash_grid_target_projected_size, hash_grid_cell_min_size, 2,
										   out_checksum) %
			   hash_grid_total_number_of_cells;
	}

	HIPRT_DEVICE ReSTIRPGDistribution get_distribution_from_position_data(float3_t position, float3_t surface_normal, const HIPRTCamera& current_camera) const
	{
		unsigned int checksum;
		unsigned int cell_index = get_hash_grid_cell_index_from_position_data(position, surface_normal, current_camera, checksum);
		if (!HashGrid::resolve_collision<ReSTIRPGHashGridCollisionResolveSteps, false>(hash_grid_checksums, hash_grid_total_number_of_cells, cell_index,
																					   checksum))
			return ReSTIRPGDistribution();

		return hash_grid_distributions_soa.get_distribution(cell_index);
	}

	HIPRT_DEVICE void invalidate_sample(unsigned int sample_index) const
	{
		splatting_samples[sample_index].normal = make_float3(0.0f, 0.0f, 0.0f);
	}
};

#endif
