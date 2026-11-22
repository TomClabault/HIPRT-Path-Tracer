/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_H
#define DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_H

// The shadow rays of this kernel are mostly visible and are not faster with the shadow ray traversal hint (Intersect.h): they keep the default hint.
// Defined before the includes so that Intersect.h sees it. GPU only: the CPU renderer compiles every kernel in one translation unit
#ifdef __KERNELCC__
#define SHADOW_RAY_TRAVERSAL_HINT hiprtTraversalHintDefault
#endif

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ReSTIR/PT/SpatialSPMISUnionDomainShadowSplit.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * Resolve kernel for the center pixel whose resolve flags are 'resolve_flags' (pixel not done) once its reverse probe rays are traced
 * ('companion_visible_mask': the slots whose companion is visible), see the top of SpatialSPMISUnionDomainShadowSplit.h.
 *
 * Merges the candidate slots in slot order then the canonical sample with streaming resampling. One uniform is drawn
 * per candidate whatever its weight (a zero weight is never selected): same distribution as ReSTIRPTReservoir::combine_with(), but only the sample
 * of the selected candidate is loaded, once, at the end. Then the normalization, the M-cap, the output reservoir and the random seed of the pixel
 */
HIPRT_DEVICE void spmis_union_domain_shadow_split_resolve_pixel(HIPRTRenderData& render_data,
																unsigned int center_pixel_index,
																unsigned int resolve_flags,
																unsigned int companion_visible_mask)
{
	const ReSTIRPTSPMISUnionDomainSettings& union_domain_settings		   = render_data.render_settings.restir_pt_settings.spmis_settings.union_domain;
	const ReSTIRPTSPMISUnionDomainShadowSplitBuffers& shadow_split_buffers = union_domain_settings.shadow_split_buffers;
	ReSTIRPTReservoir* input_reservoir_buffer							   = render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs;
	unsigned int pixel_count											   = shadow_split_buffers.pixel_count;

	// Continues the generator chain of the weights kernel
	Xorshift32Generator random_number_generator(shadow_split_buffers.random_states[center_pixel_index]);

	// Slots that traced a reverse probe ray whose companion is occluded
	unsigned int reverse_ray_mask =
		(resolve_flags >> RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_REVERSE_RAY_SHIFT) & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_NEIGHBOR_SLOT_MASK;
	unsigned int companion_occluded_mask = reverse_ray_mask & ~companion_visible_mask;

	// The canonical sample is the candidate after the neighbor slots
	constexpr int canonical_candidate = RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS;

	float weight_sum = 0.0f;
	// -1 = nothing selected
	int selected_candidate = -1;

	unsigned int candidate_mask = resolve_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_NEIGHBOR_SLOT_MASK;
	while (candidate_mask != 0)
	{
		unsigned int slot = hippt::lowest_set_bit(candidate_mask);
		candidate_mask &= candidate_mask - 1u;

		// w = w0 + (1 - V) x correction
		unsigned int slot_entry_index = slot * pixel_count + center_pixel_index;

		float weight = shadow_split_buffers.neighbor_weights[slot_entry_index];
		if ((companion_occluded_mask >> slot) & 1u)
			weight += shadow_split_buffers.neighbor_weight_corrections[slot_entry_index];

		float acceptance_uniform = random_number_generator();

		if (!(weight > 0.0f))
			continue;

		weight_sum += weight;
		if (acceptance_uniform < weight / weight_sum)
			selected_candidate = static_cast<int>(slot);
	}

	if (resolve_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_CANONICAL_BIT)
	{
		float weight			 = shadow_split_buffers.canonical_weights[center_pixel_index];
		float acceptance_uniform = random_number_generator();

		if (weight > 0.0f)
		{
			weight_sum += weight;
			if (acceptance_uniform < weight / weight_sum)
				selected_candidate = canonical_candidate;
		}
	}

	ReSTIRPTReservoir spatial_reuse_output_reservoir;

	if (selected_candidate == canonical_candidate)
	{
		// The canonical sample with the resampling target function its resampling weight was computed with
		const ReSTIRPTReservoir& center_pixel_reservoir = input_reservoir_buffer[center_pixel_index];

		spatial_reuse_output_reservoir.sample				  = center_pixel_reservoir.sample;
		spatial_reuse_output_reservoir.sample.target_function = spmis_union_domain_shadow_split_canonical_resampling_target_function(
			center_pixel_reservoir, shadow_split_buffers.canonical_numerators[center_pixel_index]);
	}
	else if (selected_candidate >= 0)
	{
		// The sample of the selected neighbor with its target function at the center pixel
		unsigned int slot_entry_index						  = selected_candidate * pixel_count + center_pixel_index;
		spatial_reuse_output_reservoir.sample				  = input_reservoir_buffer[shadow_split_buffers.neighbor_pixels[slot_entry_index]].sample;
		spatial_reuse_output_reservoir.sample.target_function = shadow_split_buffers.neighbor_targets[slot_entry_index];
	}
	spatial_reuse_output_reservoir.weight_sum = weight_sum;

	int reused_neighbors_count = hippt::clamp(1, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS, union_domain_settings.neighbor_count);

	spatial_reuse_output_reservoir.confidence = reused_neighbors_count + input_reservoir_buffer[center_pixel_index].confidence;
	spatial_reuse_output_reservoir.end_with_normalization(1.0f, 1.0f);

	// M-capping so that we don't have to M-cap when reading reservoirs on the next frame. The union domain reuse only runs with a single spatial
	// reuse pass, this is the last one
	bool m_cap_enabled = render_data.render_settings.restir_pt_settings.m_cap > 0;
	if (m_cap_enabled)
		spatial_reuse_output_reservoir.confidence = hippt::min(spatial_reuse_output_reservoir.confidence, render_data.render_settings.restir_pt_settings.m_cap);

	render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs[center_pixel_index] = spatial_reuse_output_reservoir;
	render_data.store_updated_random_seed(center_pixel_index, random_number_generator.m_state.seed);
}

/**
 * Resolve kernel of the shadow split union domain spatial reuse (see SpatialSPMISUnionDomainShadowSplit.h): the reverse probe rays recorded by the
 * weights kernel, block-distributed (spmis_union_domain_shadow_split_trace_rays_block()), then the final weights, the merge, the normalization and the
 * output reservoir. One thread per pixel over the 8 x 8 blocks of the render resolution.
 */
#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char RESTIR_PT_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_PT_SpatialReuseSPMISUnionDomainShadowSplitResolve()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PT_SpatialReuseSPMISUnionDomainShadowSplitResolve(HIPRTRenderData render_data, int x, int y)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_PT_RENDER_DATA);
#endif

	// Only compiling this whole kernel if we're using stochastic pairwise MIS: the union domain reuse is not used with the other MIS weights
#if ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS

	const ReSTIRPTSPMISUnionDomainShadowSplitBuffers& shadow_split_buffers =
		render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.shadow_split_buffers;

#ifdef __KERNELCC__
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
#endif

	// No early return before the block-distributed tracing (barriers): a thread without pixel (border blocks) or whose pixel is done has no ray
	bool thread_has_pixel			= x < render_data.render_settings.render_resolution.x && y < render_data.render_settings.render_resolution.y;
	unsigned int center_pixel_index = thread_has_pixel ? x + y * render_data.render_settings.render_resolution.x : 0;
	unsigned int resolve_flags =
		thread_has_pixel ? shadow_split_buffers.resolve_flags[center_pixel_index] : RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_DONE_BIT;

	// The neighbor slots that trace a reverse probe ray
	unsigned int reverse_ray_slots =
		(resolve_flags >> RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_REVERSE_RAY_SHIFT) & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_NEIGHBOR_SLOT_MASK;

	bool pixel_done		  = (resolve_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_DONE_BIT) != 0;
	unsigned int ray_mask = pixel_done ? 0u : reverse_ray_slots;

	unsigned int companion_visible_mask = spmis_union_domain_shadow_split_trace_rays_block<true>(render_data, center_pixel_index, ray_mask, resolve_flags);

	if (!pixel_done)
		spmis_union_domain_shadow_split_resolve_pixel(render_data, center_pixel_index, resolve_flags, companion_visible_mask);
#endif // #if ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS
}

#endif // #ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_H
