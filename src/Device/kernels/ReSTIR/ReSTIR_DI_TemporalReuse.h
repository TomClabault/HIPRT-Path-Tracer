/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_TEMPORAL_REUSE_H
#define DEVICE_RESTIR_DI_TEMPORAL_REUSE_H 

#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/ReSTIR/DI/TemporalMISWeight.h"
#include "Device/includes/ReSTIR/DI/TemporalNormalizationWeight.h"
#include "Device/includes/ReSTIR/DI/Surface.h"
#include "Device/includes/ReSTIR/DI/Utils.h"
#include "Device/includes/Sampling.h"

#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

 /** References:
 *
 * [1] [Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting] https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal/
 * [2] [A Gentle Introduction to ReSTIR: Path Reuse in Real-time] https://intro-to-restir.cwyman.org/
 * [3] [A Gentle Introduction to ReSTIR: Path Reuse in Real-time - SIGGRAPH 2023 Presentation Video] https://dl.acm.org/doi/10.1145/3587423.3595511#sec-supp
 * [4] [NVIDIA RTX DI SDK - Github] https://github.com/NVIDIAGameWorks/RTXDI
 * [5] [Generalized Resampled Importance Sampling Foundations of ReSTIR] https://research.nvidia.com/publication/2022-07_generalized-resampled-importance-sampling-foundations-restir
 * [6] [Uniform disk sampling] https://rh8liuqy.github.io/Uniform_Disk.html
 * [7] [Reddit Post for the Jacobian Term needed] https://www.reddit.com/r/GraphicsProgramming/comments/1eo5hqr/restir_di_light_sample_pdf_confusion/
 * [8] [Rearchitecting Spatiotemporal Resampling for Production] https://research.nvidia.com/publication/2021-07_rearchitecting-spatiotemporal-resampling-production
 * [9] [Adventures in Hybrid Rendering] https://diharaw.github.io/post/adventures_in_hybrid_rendering/
 * [10] [NVIDIA ReBLUR - Fast Denoising with Self Stabilizing Recurrent Blurs] https://developer.nvidia.com/gtc/2020/video/s22699-vid
 */

// By convention, the temporal neighbor is the first one to be resampled in for loops 
// (for looping over the neighbors when resampling / computing MIS weights)
// So instead of hardcoding 0 everywhere in the code, we just basically give it a name
// with a #define
#define TEMPORAL_NEIGHBOR_ID 0
// Same when resampling the initial candidates
#define INITIAL_CANDIDATES_ID 1

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReSTIR_DI_TemporalReuse(HIPRTRenderData render_data, int2 res)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_DI_TemporalReuse(HIPRTRenderData render_data, int2 res, int x, int y)
#endif
{
#ifdef __KERNELCC__
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
	uint32_t center_pixel_index = (x + y * res.x);
	if (center_pixel_index >= res.x * res.y)
		return;

	if (!render_data.aux_buffers.pixel_active[center_pixel_index])
		// Pixel inactive because of adaptive sampling, returning
		return;

	// Initializing the random generator
	unsigned int seed;
	if (render_data.render_settings.freeze_random)
		seed = wang_hash(center_pixel_index + 1);
	else
		seed = wang_hash((center_pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_seed);
	Xorshift32Generator random_number_generator(seed);

	if (render_data.render_settings.restir_di_settings.temporal_pass.temporal_buffer_clear_requested)
		// We requested a temporal buffer clear for ReSTIR DI
		render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs[center_pixel_index] = ReSTIRDIReservoir();

	// Surface data of the center pixel
	ReSTIRDISurface center_pixel_surface = get_pixel_surface(render_data, center_pixel_index);

	int temporal_neighbor_pixel_index = find_temporal_neighbor_index(render_data, center_pixel_surface.shading_point, center_pixel_surface.shading_normal, res, make_int2(x, y), center_pixel_index, center_pixel_surface.material.roughness, random_number_generator);
	if (temporal_neighbor_pixel_index == -1)
	{
		// Temporal occlusion / disoccusion, temporal neighbor is invalid,
		// we're only going to resample the initial candidates so let's set that as
		// the output right away

		// The output of this temporal pass is just the initial candidates reservoir
		render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs[center_pixel_index] = render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs[center_pixel_index];

		return;
	}

	// Resampling the initial candidates
	ReSTIRDIReservoir new_reservoir;
	ReSTIRDIReservoir initial_candidates_reservoir = render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs[center_pixel_index];
	ReSTIRDIReservoir temporal_neighbor_reservoir = render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs[temporal_neighbor_pixel_index];
	// M-capping the temporal neighbor
	if (render_data.render_settings.restir_di_settings.m_cap > 0)
		temporal_neighbor_reservoir.M = hippt::min(temporal_neighbor_reservoir.M, render_data.render_settings.restir_di_settings.m_cap);

	ReSTIRDISurface temporal_neighbor_surface;
	if (render_data.render_settings.use_prev_frame_g_buffer())
		// If we're allowing the use of last frame's g-buffer (which is required for unbiasedness in motion),
		// then we're reading from that g-buffer
		temporal_neighbor_surface = get_pixel_surface_previous_frame(render_data, temporal_neighbor_pixel_index);
	else
		// Reading from the current frame's g-buffer otherwise
		temporal_neighbor_surface = get_pixel_surface(render_data, temporal_neighbor_pixel_index);
	
	ReSTIRDITemporalResamplingMISWeight<ReSTIR_DI_BiasCorrectionWeights> resampling_mis_weight;
	// Will keep the index of the neighbor that has been selected by resampling. 
	// Either 0 or 1 for the temporal resampling pass
	int selected_neighbor = 0;
	float init_cand_mis_weight = resampling_mis_weight.get_resampling_MIS_weight(render_data, 
		initial_candidates_reservoir, 
		temporal_neighbor_surface, center_pixel_surface,
		initial_candidates_reservoir.M, temporal_neighbor_reservoir.M,
		/* indicating that we're currently resampling the initial candidates */ INITIAL_CANDIDATES_ID,
		random_number_generator);

	if (new_reservoir.combine_with(initial_candidates_reservoir, init_cand_mis_weight, initial_candidates_reservoir.sample.target_function, /* jacobian is 1 when reusing at the exact same spot */ 1.0f, random_number_generator))
		selected_neighbor = INITIAL_CANDIDATES_ID;
	new_reservoir.sanity_check(make_int2(x, y));


	// ---
	// The rest of the code resamples the temporal neighbor
	// ---

	if (temporal_neighbor_reservoir.M > 0)
	{
		float target_function_at_center = 0.0f;
		if (temporal_neighbor_reservoir.UCW != 0.0f)
			// Only resampling if the temporal neighbor isn't empty
			//
			// If the temporal neiughor's reservoir is empty, then we do not get
			// inside that if() and the target function stays at 0.0f which eliminates
			// most of the computations afterwards
			target_function_at_center = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_TargetFunctionVisibility>(render_data, temporal_neighbor_reservoir.sample, center_pixel_surface);

		float jacobian_determinant = 1.0f;
		// If the neighbor reservoir is invalid, do not compute the jacobian
		if (target_function_at_center > 0.0f && temporal_neighbor_reservoir.UCW != 0.0f)
		{
			// The reconnection shift is what is implicitely used in ReSTIR DI. We need this because
			// the initial light sample candidates that we generate on the area of the lights have an
			// area measure PDF. This area measure PDF is converted to solid angle in the initial candidates
			// sampling routine by multiplying by the distance squared and dividing by the cosine
			// angle at the light source. However, a PDF in solid angle measure is only viable at a
			// given point. We say "solid angle with respect to the shading point". This means that
			// reusing a light sample with PDF (the UCW of the neighbor reservoir) in solid angle
			// from a neighbor is invalid since that PDF is only valid at the neighbor point, not
			// at the point we're resampling from (the center pixel). We thus need to convert from the
			// "solid angle PDF at the neighbor" to the solid angle at the center pixel and we do
			// that by multiplying by the jacobian determinant of the reconnection shift in solid
			// angle, Eq. 52 of 2022, "Generalized Resampled Importance Sampling".
			jacobian_determinant = get_jacobian_determinant_reconnection_shift(render_data, 
				temporal_neighbor_reservoir, 
				center_pixel_surface.shading_point,
				/* recomputing the point without the normal offset */ temporal_neighbor_surface.shading_point - temporal_neighbor_surface.shading_normal * 1.0e-4f);

			if (jacobian_determinant == -1.0f)
				// Sample too dissimilar, not going to resample it so setting
				// the jacobian to 0.0f so that the reservoir combination fails
				// for this sample
				jacobian_determinant = 0.0f;
		}

		float temporal_neighbor_resampling_mis_weight = 1.0f;
		if (target_function_at_center > 0.0f)
			// No need to compute the MIS weight if the target function is 0.0f because we're never going to pick
			// that sample anyway when combining the reservoir since the resampling weight will be 0.0f because of
			// the multiplication by the target function that is 0.0f
			temporal_neighbor_resampling_mis_weight = resampling_mis_weight.get_resampling_MIS_weight(render_data,
				temporal_neighbor_reservoir, 
				temporal_neighbor_surface, center_pixel_surface,
				initial_candidates_reservoir.M, temporal_neighbor_reservoir.M,
				/* indicating that we're currently resampling the temporal neighbor */ TEMPORAL_NEIGHBOR_ID,
				random_number_generator);

		// Combining as in Alg. 6 of the paper
		if (new_reservoir.combine_with(temporal_neighbor_reservoir, temporal_neighbor_resampling_mis_weight, target_function_at_center, jacobian_determinant, random_number_generator))
			selected_neighbor = TEMPORAL_NEIGHBOR_ID;
		new_reservoir.sanity_check(make_int2(x, y));
	}

	float normalization_numerator = 1.0f;
	float normalization_denominator = 1.0f;

	ReSTIRDITemporalNormalizationWeight<ReSTIR_DI_BiasCorrectionWeights> normalization_weight;
	normalization_weight.get_normalization(render_data,
		new_reservoir, 
		initial_candidates_reservoir.M, temporal_neighbor_reservoir.M,
		center_pixel_surface, temporal_neighbor_surface, 
		selected_neighbor,
		center_pixel_index, temporal_neighbor_pixel_index,
		random_number_generator, normalization_numerator, normalization_denominator);

	new_reservoir.end_with_normalization(normalization_numerator, normalization_denominator);
	new_reservoir.sanity_check(make_int2(x, y));

#if ReSTIR_DI_DoVisibilityReuse && ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
	ReSTIR_DI_visibility_reuse(render_data, new_reservoir, center_pixel_surface.shading_point);
#endif

	render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs[center_pixel_index] = new_reservoir;
}

#endif
