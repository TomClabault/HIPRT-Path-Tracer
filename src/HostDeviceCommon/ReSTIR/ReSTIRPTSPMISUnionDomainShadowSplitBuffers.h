/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_BUFFERS_H
#define HOST_DEVICE_RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_BUFFERS_H

/**
 * Per-pixel state passed between the kernels of the union domain spatial reuse (see SpatialSPMISUnionDomainShadowSplit.h). All the arrays live in one
 * allocation of the renderer (see layout()), made only while the union domain reuse runs (nullptr otherwise).
 *
 * Every per-slot array is slot-major: the entry of slot s of pixel p is at s x pixel_count + p, so that the lanes of a wave load one slot of
 * consecutive pixels. A ray record is 4 floats: the reconnection vertex (or the envmap direction) and the bits of the alpha test seed of the sample.
 *
 * Kernels: prepare (domain, stage 1 of the neighbor selection, canonical probes), trace selection (stage-2 and canonical probe rays), weights (choice
 * of the neighbors, shifts, MIS weights without the reverse probe visibility), resolve (reverse probe rays, merge, output)
 */
struct ReSTIRPTSPMISUnionDomainShadowSplitBuffers
{
	// Number of pixels of the render resolution, set by the render pass: the stride between two slots of the per-slot arrays
	unsigned int pixel_count = 0;

	// Per-pixel flags and random state (prepare -> later kernels)

	// One unsigned int per pixel, the RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_* prepare flags (ray bits of the stage-2 picks and canonical probes,
	// envmap bits, pool valid, canonical present, pixel done, duplicate rays). Written by the prepare kernel. Read by the trace selection kernel (which rays to
	// trace) and by the weights kernel (which picks and probes to use, pixels already done)
	unsigned int* prepare_flags = nullptr;

	// One unsigned int per pixel, the state of the random generator of the pixel. Written by the prepare kernel, read and rewritten by the weights
	// kernel which continues the chain, read by the resolve kernel for the merge uniforms
	unsigned int* random_states = nullptr;

	// Prepare -> trace selection

	// One ray record per stage-2 pool pick: RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE slots per pixel of 4 floats, slot-major. Written by
	// the prepare kernel for the picks with a ray bit. Read by the trace selection kernel: the endpoint and seed of the ray from the center pixel
	float* selection_rays = nullptr;

	// One ray record per pixel (4 floats): the canonical sample, shared by all the canonical probe rays of the pixel. Written by the prepare kernel
	// when at least one canonical probe has a ray bit. Read by the trace selection kernel: the endpoint and seed of every canonical probe ray
	float* canonical_rays = nullptr;

	// One pixel index per canonical probe: RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES slots per pixel, slot-major. Written by the prepare kernel
	// for the probes with a ray bit. Read by the trace selection kernel: the origin of the probe ray (primary hit of that pixel)
	unsigned int* canonical_viewpoints = nullptr;

	// Trace selection -> weights

	// One unsigned int per pixel: the visibility bits of the stage-2 picks and canonical probes (bit positions of the ray bits of the prepare flags,
	// 0 = occluded or no ray), the bits of the duplicate rays copied from the rays they repeat. Written by the trace selection kernel for the pixels
	// not done. Read by the weights kernel: the visible picks of the selection, the visible canonical probes.
	//
	// Before that, for the pixels with the duplicate rays flag only: the duplicate map of the pixel (bit layout in SpatialSPMISUnionDomainShadowSplit.h),
	// written by the prepare kernel and read by the trace selection kernel, which then overwrites it with the visibility bits
	unsigned int* selection_visibility = nullptr;

	// Prepare -> weights

	// One pixel index per stage-2 pool pick: RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE slots per pixel, slot-major. Written by the
	// prepare kernel for the picks that need a ray (traced or duplicate). Read by the weights kernel: the neighbor pixel of a chosen pick
	unsigned int* pick_pixels = nullptr;

	// One float per stage-2 pool pick, same layout as pick_pixels: the stage-1 selection probability P_l of the pick. Written by the prepare kernel
	// for the picks that need a ray (traced or duplicate). Read by the weights kernel: the selection probability of a chosen neighbor
	float* pick_probabilities = nullptr;

	// One pixel index per neighbor slot: RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS slots per pixel, slot-major. The uniform companion
	// (reverse probe pixel) of the MIS weight of the slot, written by the prepare kernel when the pool is valid. Read by the weights kernel: the
	// companion's value without visibility and the origin of its reverse probe ray
	unsigned int* companions = nullptr;

	// One float per canonical probe, same layout as canonical_viewpoints: the value of the probe without visibility. Written by the prepare kernel for
	// the probes with a non-zero value (traced or duplicate). Read by the weights kernel for the visible probes: the canonical MIS weight
	float* canonical_values = nullptr;

	// One float per pixel: the numerator c_c p_c of the canonical MIS weight. Written by the prepare kernel when the canonical sample is resampled.
	// Read by the weights kernel: the canonical MIS weight and the resampling target function of the canonical sample. Read by the resolve kernel when
	// the canonical sample is selected: its resampling target function
	float* canonical_numerators = nullptr;

	// One float per pixel: the confidence sum C of the domain of the pixel. Written by the prepare kernel. Read by the weights kernel: the group
	// scale C / g of the MIS weights
	float* domain_confidence_sums = nullptr;

	// Weights -> resolve

	// One unsigned int per pixel, the RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_* resolve flags (candidate slots, reverse ray bits, envmap bits,
	// canonical candidate, pixel done). Written by the weights kernel, and by the prepare kernel for the pixels it finishes. Read by the resolve
	// kernel: which reverse rays to trace and which candidates to merge
	unsigned int* resolve_flags = nullptr;

	// One pixel index per neighbor slot, same layout as companions: the neighbor chosen for the slot. Written by the weights kernel for the candidate
	// slots. Read by the resolve kernel: the sample of the selected neighbor
	unsigned int* neighbor_pixels = nullptr;

	// One float per neighbor slot, same layout as companions: the resampling target function of the neighbor's sample at the center pixel
	// (ReSTIR_PT_resampling_target_function_bsdf_approximation()). Written by the weights kernel for the candidate slots. Read by the resolve kernel: the
	// target function of the output sample when that neighbor is selected
	float* neighbor_targets = nullptr;

	// One float per neighbor slot, same layout as companions: the resampling weight of the neighbor with its companion visible. Written by the
	// weights kernel for the candidate slots. Read by the resolve kernel: the merge weight
	float* neighbor_weights = nullptr;

	// One float per neighbor slot, same layout as companions: the correction of the resampling weight when the companion is occluded. Written by the
	// weights kernel for the candidate slots. Read by the resolve kernel, added to the merge weight when the reverse probe ray is occluded
	float* neighbor_weight_corrections = nullptr;

	// One ray record per neighbor slot: RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS slots per pixel of 4 floats, slot-major (the sample of
	// the neighbor). Written by the weights kernel for the slots with a reverse ray bit. Read by the resolve kernel: the endpoint and seed of the
	// reverse probe ray
	float* reverse_rays = nullptr;

	// One pixel index per neighbor slot, same layout as companions: the companion of the slot. Written by the weights kernel for the slots with a
	// reverse ray bit. Read by the resolve kernel: the origin of the reverse probe ray (primary hit of that pixel)
	unsigned int* reverse_viewpoints = nullptr;

	// One float per pixel: the resampling weight of the canonical sample. Written by the weights kernel when the canonical sample is resampled. Read
	// by the resolve kernel: the merge weight of the canonical sample
	float* canonical_weights = nullptr;

#ifndef __KERNELCC__
	/**
	 * Layout of all the arrays of ReSTIRPTSPMISUnionDomainShadowSplitBuffers in one buffer for 'pixel_count' pixels (about 460 bytes per pixel with the
	 * maximum pool size and probe count of ReSTIRPTOptions.h, plus alignment). Sets the pointers of 'shadow_split_buffers' into the buffer
	 * 'buffer_pointer' (nullptr pointers when 'buffer_pointer' is nullptr) and returns the byte size of the buffer
	 */
	static size_t layout(unsigned char* buffer_pointer, unsigned int pixel_count, ReSTIRPTSPMISUnionDomainShadowSplitBuffers& shadow_split_buffers)
	{
		// Entries of the per-pixel and per-slot arrays. A ray record is 4 floats
		size_t pixels		  = pixel_count;
		size_t pool_slots	  = RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE * pixels;
		size_t probe_slots	  = RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES * pixels;
		size_t neighbor_slots = RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS * pixels;

		size_t buffer_byte_size = 0;

		// Per-pixel flags and random state
		reserve_array(buffer_pointer, buffer_byte_size, pixels, shadow_split_buffers.prepare_flags);
		reserve_array(buffer_pointer, buffer_byte_size, pixels, shadow_split_buffers.random_states);

		// Prepare -> trace selection
		reserve_array(buffer_pointer, buffer_byte_size, pool_slots * 4, shadow_split_buffers.selection_rays);
		reserve_array(buffer_pointer, buffer_byte_size, pixels * 4, shadow_split_buffers.canonical_rays);
		reserve_array(buffer_pointer, buffer_byte_size, probe_slots, shadow_split_buffers.canonical_viewpoints);

		// Trace selection -> weights
		reserve_array(buffer_pointer, buffer_byte_size, pixels, shadow_split_buffers.selection_visibility);

		// Prepare -> weights
		reserve_array(buffer_pointer, buffer_byte_size, pool_slots, shadow_split_buffers.pick_pixels);
		reserve_array(buffer_pointer, buffer_byte_size, pool_slots, shadow_split_buffers.pick_probabilities);
		reserve_array(buffer_pointer, buffer_byte_size, neighbor_slots, shadow_split_buffers.companions);
		reserve_array(buffer_pointer, buffer_byte_size, probe_slots, shadow_split_buffers.canonical_values);
		reserve_array(buffer_pointer, buffer_byte_size, pixels, shadow_split_buffers.canonical_numerators);
		reserve_array(buffer_pointer, buffer_byte_size, pixels, shadow_split_buffers.domain_confidence_sums);

		// Weights -> resolve
		reserve_array(buffer_pointer, buffer_byte_size, pixels, shadow_split_buffers.resolve_flags);
		reserve_array(buffer_pointer, buffer_byte_size, neighbor_slots, shadow_split_buffers.neighbor_pixels);
		reserve_array(buffer_pointer, buffer_byte_size, neighbor_slots, shadow_split_buffers.neighbor_targets);
		reserve_array(buffer_pointer, buffer_byte_size, neighbor_slots, shadow_split_buffers.neighbor_weights);
		reserve_array(buffer_pointer, buffer_byte_size, neighbor_slots, shadow_split_buffers.neighbor_weight_corrections);
		reserve_array(buffer_pointer, buffer_byte_size, neighbor_slots * 4, shadow_split_buffers.reverse_rays);
		reserve_array(buffer_pointer, buffer_byte_size, neighbor_slots, shadow_split_buffers.reverse_viewpoints);
		reserve_array(buffer_pointer, buffer_byte_size, pixels, shadow_split_buffers.canonical_weights);

		shadow_split_buffers.pixel_count = buffer_pointer == nullptr ? 0 : pixel_count;

		return buffer_byte_size;
	}

private:
	/**
	 * Reserves 'element_count' elements of type T at a 256 byte aligned offset of the buffer whose first 'buffer_byte_size' bytes are already reserved,
	 * and moves 'buffer_byte_size' past them. 'out_array' points to the reserved elements in the buffer 'buffer_pointer', or is nullptr when the buffer
	 * pointer is nullptr (only the size of the buffer is computed)
	 */
	template <typename T>
	static void reserve_array(unsigned char* buffer_pointer, size_t& buffer_byte_size, size_t element_count, T*& out_array)
	{
		size_t offset	 = (buffer_byte_size + 255) / 256 * 256;
		buffer_byte_size = offset + element_count * sizeof(T);

		out_array = buffer_pointer == nullptr ? nullptr : reinterpret_cast<T*>(buffer_pointer + offset);
	}
#endif // #ifndef __KERNELCC__
};

#endif // #ifndef HOST_DEVICE_RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_BUFFERS_H
