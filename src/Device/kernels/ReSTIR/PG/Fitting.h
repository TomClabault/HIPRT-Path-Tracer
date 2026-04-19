/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_RESTIR_PG_FITTING_H
#define DEVICE_KERNELS_RESTIR_PG_FITTING_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * Kernel dispatched in 1D with 1 thread per cell
 */
#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReSTIR_PG_Fitting(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PG_Fitting(HIPRTRenderData render_data, int x)
#endif
{
#ifdef __KERNELCC__
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	ReSTIRPGSettings& restir_pg_settings = render_data.render_settings.restir_pg_settings;

	unsigned int cell_index = x;
	if (cell_index >= hippt::atomic_load(restir_pg_settings.grid_cell_alive_count))
		return;

	unsigned int hash_grid_cell_index = restir_pg_settings.grid_cell_alive_list[cell_index];
	if (hash_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
		// Should never happen
		return;

	ReSTIRPGDistributionSufficientStatisticsSoADevice sufficient_statistics_soa = restir_pg_settings.hash_grid_distributions_sufficient_statistics_soa;
	ReSTIRPGDistribution current_distribution = restir_pg_settings.hash_grid_distributions_soa.get_distribution(hash_grid_cell_index);

	float sum_responsibilities_weight_sum = 0.0f;
	for (int component_index = 0; component_index < ReSTIRPGDistributionComponentCount; component_index++)
		sum_responsibilities_weight_sum +=
			sufficient_statistics_soa.responsibility_weights_sum[component_index * restir_pg_settings.hash_grid_total_number_of_cells + hash_grid_cell_index];

	for (int component_index = 0; component_index < ReSTIRPGDistributionComponentCount; component_index++)
	{
		float3_t directions_sum =
			sufficient_statistics_soa.read_directions_sum(component_index, hash_grid_cell_index, restir_pg_settings.hash_grid_total_number_of_cells);

		float directions_sum_length = hippt::length(directions_sum);
		float responsibility_weights_sum =
			sufficient_statistics_soa.responsibility_weights_sum[component_index * restir_pg_settings.hash_grid_total_number_of_cells + hash_grid_cell_index];

		if (responsibility_weights_sum <= 1e-25f || directions_sum_length <= 1e-25f)
			continue;

		float3_t new_mean_vmf_direction	  = directions_sum / directions_sum_length;
		float normalized_resultant_length = directions_sum_length / responsibility_weights_sum;

		constexpr float Nprior					  = 0.2f;
		float current_mixture_component_weight	  = current_distribution.distribution_components[component_index].weight;
		unsigned int current_mixture_sample_count = sufficient_statistics_soa.sample_count[hash_grid_cell_index];
		float rpriork							  = (normalized_resultant_length * current_mixture_component_weight * current_mixture_sample_count) /
						(current_mixture_component_weight * current_mixture_sample_count + Nprior);

		float new_vmf_sharpness = hippt::clamp(1.0e-2f, 1.0e4f, (3.0f * rpriork - hippt::pow_3(rpriork)) / (1.0f - hippt::square(rpriork)));

		constexpr float prior = 1.0e-2f;
		float new_mixture_component_weight =
			(responsibility_weights_sum + prior) / (sum_responsibilities_weight_sum + prior * ReSTIRPGDistributionComponentCount);

		VMFMixtureComponent new_vmf;
		new_vmf.vmf.axis	  = new_mean_vmf_direction;
		new_vmf.vmf.sharpness = new_vmf_sharpness;
		new_vmf.weight		  = new_mixture_component_weight;

		/*if ((!hippt::is_finite(new_vmf.vmf.axis.x) || !hippt::is_finite(new_vmf.vmf.axis.y) || !hippt::is_finite(new_vmf.vmf.axis.z) ||
			 !hippt::is_finite(new_vmf.vmf.sharpness) || !hippt::is_finite(new_vmf.weight)) &&
			hash_grid_cell_index < 1000)
		{
			printf("Not finite: new_vmf.sharpness = %f, new_vmf.weight = %f @ %u\n\tSum responsibilities: %f, "
				   "\n\tresponsibility_weights_sum = %f, \n\tdirections_sum_length: %f\n\tnormalized_resultant_length: %f, "
				   "\n\tcurrent_mixture_component_weight = %f, \n\tcurrent_mixture_sample_count = "
				   "%u\n\t, rpriork = %f\n",
				   new_vmf.vmf.sharpness, new_vmf.weight, hash_grid_cell_index, sum_responsibilities_weight_sum, responsibility_weights_sum,
				   directions_sum_length, normalized_resultant_length, current_mixture_component_weight, current_mixture_sample_count, rpriork);
		}*/

		restir_pg_settings.hash_grid_distributions_soa.set_distribution_component_vmf(hash_grid_cell_index, component_index, new_vmf.vmf);
		restir_pg_settings.hash_grid_distributions_soa.set_distribution_component_weight(hash_grid_cell_index, component_index, new_vmf.weight);
	}
}

#endif
