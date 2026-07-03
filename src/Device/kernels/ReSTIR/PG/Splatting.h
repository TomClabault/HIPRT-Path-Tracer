/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_RESTIR_PG_SPLATTING_H
#define DEVICE_KERNELS_RESTIR_PG_SPLATTING_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ReSTIR/PG/SplattingSample.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE float compute_sample_responsibility(const ReSTIRPGDistribution& hash_grid_cell_distribution,
												 float3_t sample_direction,
												 int distribution_component,
												 const ReSTIRPGSettings& restir_pg_settings)
{
	const VMFMixtureComponent& vmf_mixture_component = hash_grid_cell_distribution.distribution_components[distribution_component];

	float component_weight = vmf_mixture_component.weight;
	return component_weight * vmf_mixture_component.vmf.density_evaluation(sample_direction);
}

HIPRT_DEVICE void atomic_accumulate_sample(
	const ReSTIRPGSettings& restir_pg_settings, float3_t sample_direction, float responsibility, int distribution_component, unsigned int hash_grid_cell_index)
{
	const ReSTIRPGDistributionSufficientStatisticsSoADevice& sufficient_statistics = restir_pg_settings.hash_grid_distributions_sufficient_statistics_soa;

	unsigned int index = distribution_component * restir_pg_settings.hash_grid_total_number_of_cells + hash_grid_cell_index;

#ifdef __KERNELCC__

	// Warp aggregation for the GPU path
	unsigned int same_index_mask = hippt::warp_match_any_sync(0xFFFFFFFF, index);

	float sum_x				 = sample_direction.x * responsibility;
	float sum_y				 = sample_direction.y * responsibility;
	float sum_z				 = sample_direction.z * responsibility;
	float sum_responsibility = responsibility;

	unsigned int first_active_in_mask = hippt::ffs(same_index_mask) - 1;
	unsigned int linear_thread_index  = threadIdx.x + threadIdx.y * blockDim.x;
	unsigned int lane_index			  = linear_thread_index & 31;

	// The leader thread sums up the contributions of all the threads in the warp that have the same index and then atomically adds the sum to memory.
	// There could be a faster way to do this but good enough
	for (int src = 0; src < 32; ++src)
	{
		float other_lane_sum_x				= hippt::warp_shfl(sample_direction.x * responsibility, src);
		float other_lane_sum_y				= hippt::warp_shfl(sample_direction.y * responsibility, src);
		float other_lane_sum_z				= hippt::warp_shfl(sample_direction.z * responsibility, src);
		float other_lane_sum_responsibility = hippt::warp_shfl_sync(same_index_mask, responsibility, src);

		if (lane_index == first_active_in_mask && src != first_active_in_mask && (same_index_mask & (1u << src)) != 0)
		{
			sum_x += other_lane_sum_x;
			sum_y += other_lane_sum_y;
			sum_z += other_lane_sum_z;
			sum_responsibility += other_lane_sum_responsibility;
		}
	}

	if (lane_index == first_active_in_mask && hash_grid_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
	{
		hippt::atomic_fetch_add(&sufficient_statistics.directions_sum_x[index], sum_x);
		hippt::atomic_fetch_add(&sufficient_statistics.directions_sum_y[index], sum_y);
		hippt::atomic_fetch_add(&sufficient_statistics.directions_sum_z[index], sum_z);
		hippt::atomic_fetch_add(&sufficient_statistics.responsibility_weights_sum[index], sum_responsibility);
	}

#else

	// CPU
	hippt::atomic_fetch_add(&sufficient_statistics.directions_sum_x[index], sample_direction.x * responsibility);
	hippt::atomic_fetch_add(&sufficient_statistics.directions_sum_y[index], sample_direction.y * responsibility);
	hippt::atomic_fetch_add(&sufficient_statistics.directions_sum_z[index], sample_direction.z * responsibility);
	hippt::atomic_fetch_add(&sufficient_statistics.responsibility_weights_sum[index], responsibility);

#endif
}

// Dispatched as 1D render_resolution.x * render_resolution.y threads to facilitate mapping thread indices to proper warps for coalescing
#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReSTIR_PG_Splatting(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PG_Splatting(HIPRTRenderData render_data, uint32_t index)
#endif
{
#ifdef __KERNELCC__
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
#endif

	const uint32_t x = index % render_data.render_settings.render_resolution.x;
	const uint32_t y = index / render_data.render_settings.render_resolution.x;

	bool thread_valid = true;
	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		thread_valid = false;

	uint32_t pixel_index = y * render_data.render_settings.render_resolution.x + x;
	uint32_t pixel_count = render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y;
	uint32_t restir_reservoir_pixel_index;
	if (thread_valid)
	{
		// Reading from the output of ReSTIR GI or PT depending on which one is being used

#if PathSamplingStrategy == PATH_SAMPLING_RESTIR_GI
		restir_reservoir_pixel_index = render_data.render_settings.restir_gi_settings.restir_output_reservoirs[pixel_index].sample.pixel_index;
#elif PathSamplingStrategy == PATH_SAMPLING_RESTIR_PT
		restir_reservoir_pixel_index = render_data.render_settings.restir_pt_settings.restir_output_reservoirs[pixel_index].sample.pixel_index;
#else
		restir_reservoir_pixel_index = -1;

#if ReSTIRPGEnable == KERNEL_OPTION_TRUE
#error "Unknown PathSamplingStrategy"
#endif
#endif
	}
	if (restir_reservoir_pixel_index == static_cast<unsigned int>(-1))
		// That means potentially no spatial reuse / temporal so our reservoir didn't move
		restir_reservoir_pixel_index = pixel_index;
	uint32_t reservoir_x = restir_reservoir_pixel_index % render_data.render_settings.render_resolution.x;
	uint16_t reservoir_y = restir_reservoir_pixel_index / render_data.render_settings.render_resolution.x;

	const ReSTIRPGSettings& restir_pg_settings = render_data.render_settings.restir_pg_settings;

	// For each bounce, splatting the sample of that bounce (for the current pixel) into the hash grid
	for (int bounce = 0; bounce < render_data.render_settings.nb_bounces; bounce++)
	{
		ReSTIRPGSplattingSample sample;
		if (thread_valid)
		{
			if (bounce == 0)
			{
				unsigned int soa_index =
					restir_pg_settings.splatting_samples_soa.get_soa_index(render_data.render_settings.render_resolution, reservoir_x, reservoir_y, bounce);
				// Only reading the incident direction here, the position and normal at bounce 0 are filled below from the gbuffer
				sample.incident_direction = restir_pg_settings.splatting_samples_soa.incident_direction[soa_index];

				// At bounce 0, the samples do not come from the reservoirs but from the G-Buffer because we're using the reconnection shift. So paths only come
				// from
				// the reservoir starting at the sample point which is not the first hit
				sample.position = render_data.g_buffer.primary_hit_position[pixel_index];

				if (render_data.g_buffer.first_hit_prim_index[pixel_index] == -1)
					// That means that the primary ray didn't hit anything, so we don't have a valid sample and we can just skip the rest of the loop and
					// not insert it into the hash grid
					sample.normal = make_float3(0.0f, 0.0f, 0.0f);
				else
					sample.normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();
			}
			else
				sample = restir_pg_settings.splatting_samples_soa.read_sample(render_data.render_settings.render_resolution, reservoir_x, reservoir_y, bounce);
		}

		bool sample_valid		= sample.valid_sample();
		bool should_participate = sample_valid && thread_valid;

		unsigned int checksum;
		unsigned int sample_hash_grid_index;
		if (should_participate)
		{
			sample_hash_grid_index =
				restir_pg_settings.get_hash_grid_cell_index_from_position_data(sample.position, sample.normal, render_data.current_camera, checksum);

			if (!HashGrid::resolve_collision<ReSTIRPGHashGridCollisionResolveSteps, true>(
					restir_pg_settings.hash_grid_checksums, restir_pg_settings.hash_grid_total_number_of_cells, sample_hash_grid_index, checksum))
				// If the collision resolution failed, then we just skip this sample and don't insert it into the hash grid
				should_participate = false;
		}
		else
			sample_hash_grid_index = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;

		if (should_participate && !hippt::atomic_compare_exchange(&restir_pg_settings.grid_cell_alive[sample_hash_grid_index], 0u, 1u))
		{
			// Setting the grid cell as alive
			unsigned int grid_cell_alive_index							   = hippt::atomic_fetch_add(restir_pg_settings.grid_cell_alive_count, 1u);
			restir_pg_settings.grid_cell_alive_list[grid_cell_alive_index] = sample_hash_grid_index;
		}

		/**
		 * Inserting the sample into the sufficient statistics of the hash grid cell, in each component for the expectation step of the EM algorithm
		 */
		ReSTIRPGDistribution distribution;
		if (should_participate)
			distribution = restir_pg_settings.hash_grid_distributions_soa.get_distribution(sample_hash_grid_index);

		// Begin by computing the responsibility of this sample for each component of the distribution of the hash grid cell it maps to
		float sum_responsibilities = 1.0e-8f;
		float responsibilities[ReSTIRPGDistributionComponentCount];
		if (should_participate)
		{
			for (int component = 0; component < ReSTIRPGDistributionComponentCount; component++)
			{
				float responsibility = compute_sample_responsibility(distribution, sample.incident_direction, component, restir_pg_settings);

				responsibilities[component] = responsibility;
				sum_responsibilities += responsibility;
			}
		}

		// Normalize responsibilities
		for (int component = 0; component < ReSTIRPGDistributionComponentCount; component++)
			responsibilities[component] /= sum_responsibilities;

#ifndef __KERNELCC__
		// On the GPU, we're going to do some warp intrinsic stuff to accumulate samples so all threads need to go in there otherwise that's going to be UB. On
		// the CPU though we only want to accumulate samples for actually valid sample so we do check for should_participate
		if (should_participate)
#endif
		{
			for (int component = 0; component < ReSTIRPGDistributionComponentCount; component++)
			{
				float responsibility = responsibilities[component];

				atomic_accumulate_sample(restir_pg_settings, sample.incident_direction, responsibility, component, sample_hash_grid_index);
			}
		}

		if (should_participate)
			hippt::atomic_fetch_add(&restir_pg_settings.hash_grid_distributions_sufficient_statistics_soa.sample_count[sample_hash_grid_index], 1u);
	}
}

#endif // DEVICE_KERNELS_RESTIR_PG_SPLATTING_H
