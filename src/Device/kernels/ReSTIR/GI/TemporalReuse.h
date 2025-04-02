/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_GI_SPATIAL_REUSE_H
#define DEVICE_RESTIR_GI_SPATIAL_REUSE_H 

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/PathTracing.h"
#include "Device/includes/ReSTIR/Surface.h"
#include "Device/includes/ReSTIR/TemporalMISWeight.h"
#include "Device/includes/ReSTIR/TemporalNormalizationWeight.h"
#include "Device/includes/ReSTIR/Utils.h"
#include "Device/includes/ReSTIR/UtilsTemporal.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"

#include "HostDeviceCommon/RenderData.h"

 /** References:
  *
  * [1] [ReSTIR GI: Path Resampling for Real-Time Path Tracing] https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing
  * [2] [A Gentle Introduction to ReSTIR: Path Reuse in Real-time] https://intro-to-restir.cwyman.org/
  * [3] [A Gentle Introduction to ReSTIR: Path Reuse in Real-time - SIGGRAPH 2023 Presentation Video] https://dl.acm.org/doi/10.1145/3587423.3595511#sec-supp
  */

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_GI_TemporalReuse(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_GI_TemporalReuse(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		return;

	int DEBUG_X = 317;
	int DEBUG_Y = 344;

	uint32_t center_pixel_index = (x + y * render_data.render_settings.render_resolution.x);

	if (!render_data.aux_buffers.pixel_active[center_pixel_index] || render_data.g_buffer.first_hit_prim_index[center_pixel_index] == -1)
		// Pixel inactive because of adaptive sampling, returning
		// Or also we don't have a primary hit
		return;

	if (render_data.render_settings.restir_gi_settings.common_temporal_pass.temporal_buffer_clear_requested)
		// We requested a temporal buffer clear for ReSTIR GI
		render_data.render_settings.restir_gi_settings.temporal_pass.input_reservoirs[center_pixel_index] = ReSTIRGIReservoir();

	if (render_data.render_settings.sample_number == 0 && render_data.render_settings.accumulate)
		// First frame of accumulation, no temporal history, just outputting the initial candidates
		render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs[center_pixel_index] = render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer[center_pixel_index];

	// Initializing the random generator
	unsigned int seed;
	if (render_data.render_settings.freeze_random)
		seed = wang_hash(center_pixel_index + 1);
	else
		seed = wang_hash((center_pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);
	Xorshift32Generator random_number_generator(seed);


	// Surface data of the center pixel
	ReSTIRSurface center_pixel_surface = get_pixel_surface(render_data, center_pixel_index, random_number_generator);
	int temporal_neighbor_pixel_index = find_temporal_neighbor_index<true>(render_data, center_pixel_surface.shading_point, ReSTIRSettingsHelper::get_normal_for_rejection_heuristic<true>(render_data, center_pixel_surface), center_pixel_index, random_number_generator).x;
	if (temporal_neighbor_pixel_index == -1 || render_data.render_settings.freeze_random)
	{
		// Temporal occlusion / disoccusion, temporal neighbor is invalid,
		// we're only going to resample the initial candidates so let's set that as
		// the output right away
		//
		// We're also 'disabling' temporal accumulation if the random is frozen otherwise
		// very strong correlations will creep up, corrupt the render and potentially invalidate
		// performance measurements (which we're probably trying to measure since we froze the random)

		// The output of this temporal pass is just the initial candidates reservoir
		render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs[center_pixel_index] = render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer[center_pixel_index];

		return;
	}


	ReSTIRGIReservoir temporal_neighbor_reservoir = render_data.render_settings.restir_gi_settings.temporal_pass.input_reservoirs[temporal_neighbor_pixel_index];
	if (temporal_neighbor_reservoir.M == 0)
	{
		// No temporal neighbor, the output of this temporal pass is just the initial candidates reservoir
		render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs[center_pixel_index] = render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer[center_pixel_index];

		return;
	}

	ReSTIRGIReservoir temporal_reuse_output_reservoir;
	ReSTIRSurface temporal_neighbor_surface = get_pixel_surface(render_data, temporal_neighbor_pixel_index, render_data.render_settings.use_prev_frame_g_buffer(), random_number_generator);

	ReSTIRTemporalResamplingMISWeight<ReSTIR_GI_BiasCorrectionWeights, true> mis_weight_function;

	// /* ------------------------------- */
	// Resampling the temporal neighbor
	// /* ------------------------------- */

	ReSTIRGIReservoir initial_candidates_reservoir = render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer[center_pixel_index];
	if (temporal_neighbor_reservoir.M > 0)
	{
		float target_function_at_center = 0.0f;
		if (temporal_neighbor_reservoir.UCW > 0.0f)
			// Only resampling if the temporal neighbor isn't empty
			//
			// If the temporal neighbor's reservoir is empty, then we do not get
			// inside that if() and the target function stays at 0.0f which eliminates
			// most of the computations afterwards
			//
			// Matching the visibility used here with the bias correction mode for ease 
			// of use (and because manually handling the visibility in the target 
			// function of the temporal reuse is tricky for the user to use in 
			// combination with other parameters
			target_function_at_center = ReSTIR_GI_evaluate_target_function<ReSTIR_GI_BiasCorrectionUseVisibility>(render_data, temporal_neighbor_reservoir.sample, center_pixel_surface, random_number_generator);

		float shift_mapping_jacobian = 1.0f;
		if (temporal_neighbor_reservoir.UCW > 0.0f && !temporal_neighbor_reservoir.sample.is_envmap_path())
		{
			// Only attempting the shift if the neighbor reservoir is valid
			// 
			// Also, if this is the last neighbor resample (meaning that it is the center pixel), 
			// the shift mapping is going to be an identity shift with a jacobian of 1 so we don't need to do it
			shift_mapping_jacobian = get_jacobian_determinant_reconnection_shift(temporal_neighbor_reservoir.sample.sample_point, temporal_neighbor_reservoir.sample.sample_point_geometric_normal, center_pixel_surface.shading_point, render_data.g_buffer_prev_frame.primary_hit_position[temporal_neighbor_pixel_index], render_data.render_settings.restir_gi_settings.get_jacobian_heuristic_threshold());

#if 0
			// TODO below is a test of BSDF ratio jacobian for unbiased ReSTIR GI but this doesn't seem to work
			if (render_data.render_settings.DEBUG_DO_BSDF_RATIO)
			{
				float new_pdf;
				BSDFContext new_pdf_context(hippt::normalize(center_pixel_surface.shading_point - temporal_neighbor_reservoir.sample.sample_point), temporal_neighbor_reservoir.sample.sample_point_shading_normal, temporal_neighbor_reservoir.sample.sample_point_geometric_normal, temporal_neighbor_reservoir.sample.incident_light_direction_at_sample_point, temporal_neighbor_reservoir.sample.incident_light_info_at_sample_point, temporal_neighbor_reservoir.sample.sample_point_volume_state, false, temporal_neighbor_reservoir.sample.sample_point_material.unpack(), 1, 0.0f);
				bsdf_dispatcher_eval(render_data, new_pdf_context, new_pdf, random_number_generator);

				float old_pdf;
				BSDFContext old_pdf_context(hippt::normalize(render_data.g_buffer.primary_hit_position[temporal_neighbor_pixel_index] - temporal_neighbor_reservoir.sample.sample_point), temporal_neighbor_reservoir.sample.sample_point_shading_normal, temporal_neighbor_reservoir.sample.sample_point_geometric_normal, temporal_neighbor_reservoir.sample.incident_light_direction_at_sample_point, temporal_neighbor_reservoir.sample.incident_light_info_at_sample_point, temporal_neighbor_reservoir.sample.sample_point_volume_state, false, temporal_neighbor_reservoir.sample.sample_point_material.unpack(), 1, 0.0f);
				bsdf_dispatcher_eval(render_data, old_pdf_context, old_pdf, random_number_generator);

				float bsdf_pdf_ratio = new_pdf / old_pdf;
				shift_mapping_jacobian *= bsdf_pdf_ratio;
			}
#endif
		}

#if ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_1_OVER_M
		float temporal_neighbor_resampling_mis_weight = mis_weight_function.get_resampling_MIS_weight(temporal_neighbor_reservoir);
#elif ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_1_OVER_Z
		float temporal_neighbor_resampling_mis_weight = mis_weight_function.get_resampling_MIS_weight(temporal_neighbor_reservoir);
#elif ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_MIS_LIKE
		float temporal_neighbor_resampling_mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, temporal_neighbor_reservoir);
#elif ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_MIS_GBH
		float temporal_neighbor_resampling_mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, temporal_neighbor_reservoir,
			initial_candidates_reservoir, temporal_neighbor_surface, center_pixel_surface,
			temporal_neighbor_reservoir.M, TEMPORAL_NEIGHBOR_ID, random_number_generator);
#elif ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_PAIRWISE_MIS
		float temporal_neighbor_resampling_mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data,
			temporal_neighbor_reservoir, initial_candidates_reservoir,
			center_pixel_surface, temporal_neighbor_surface,
			target_function_at_center * shift_mapping_jacobian, TEMPORAL_NEIGHBOR_ID,
			random_number_generator);
#elif ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE
		float temporal_neighbor_resampling_mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data,
			temporal_neighbor_reservoir, initial_candidates_reservoir,
			center_pixel_surface, temporal_neighbor_surface,

			target_function_at_center * shift_mapping_jacobian, TEMPORAL_NEIGHBOR_ID,
			random_number_generator);
#else
#error "Unsupported bias correction mode"
#endif

		//if (x == DEBUG_X && y == DEBUG_Y)
		//{
		//	hippt::atomic_fetch_add(&render_data.render_settings.DEBUG_SUM_COUNT[0], 1ull);
		//	printf("2. sample %d --> [%d, %d] ; temporal cand = [targ: %f, MIS: %f, shift: %f, UCW: %f]\n", render_data.render_settings.sample_number, x, y, temporal_neighbor_reservoir.sample.target_function, temporal_neighbor_resampling_mis_weight, shift_mapping_jacobian, temporal_neighbor_reservoir.UCW);
		//	// printf("2. sample %d --> [%d, %d] ; init cand = [targ: %f, UCW: %f, MIS: %f]\n", render_data.render_settings.sample_number, x, y, initial_candidates_reservoir.sample.target_function, initial_candidates_mis_weight, initial_candidates_reservoir.UCW);
		//	// printf("2. sample %d --> [%d, %d], targ: %f, UCW: %f, weight_sum: %f, nume: %f, denom: %f\n", render_data.render_settings.sample_number, x, y, temporal_neighbor_reservoir.sample.target_function, temporal_reuse_output_reservoir.UCW, temporal_reuse_output_reservoir.weight_sum, normalization_numerator, normalization_denominator);
		//}

		// Combining as in Alg. 6 of the paper
		temporal_reuse_output_reservoir.combine_with(temporal_neighbor_reservoir, temporal_neighbor_resampling_mis_weight, target_function_at_center, shift_mapping_jacobian, random_number_generator);
		temporal_reuse_output_reservoir.sanity_check(make_int2(x, y));
	}

	// /* ------------------------------- */
	// Resampling the initial candidates
	// /* ------------------------------- */

#if ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_1_OVER_M
	float initial_candidates_mis_weight = mis_weight_function.get_resampling_MIS_weight(initial_candidates_reservoir);
#elif ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_1_OVER_Z
	float initial_candidates_mis_weight = mis_weight_function.get_resampling_MIS_weight(initial_candidates_reservoir);
#elif ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_MIS_LIKE
	float initial_candidates_mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, initial_candidates_reservoir);
#elif ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_MIS_GBH
	float initial_candidates_mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, initial_candidates_reservoir,
		initial_candidates_reservoir, temporal_neighbor_surface, center_pixel_surface,
		temporal_neighbor_reservoir.M, INITIAL_CANDIDATES_ID, random_number_generator);
#elif ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_PAIRWISE_MIS
	float initial_candidates_mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data,
		temporal_neighbor_reservoir, initial_candidates_reservoir,
		center_pixel_surface, temporal_neighbor_surface, 

		/* unused */ 0.0f, INITIAL_CANDIDATES_ID, 
		random_number_generator);
#elif ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE
	float initial_candidates_mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, 
		temporal_neighbor_reservoir, initial_candidates_reservoir,
		center_pixel_surface, temporal_neighbor_surface, 

		/* unused */ 0.0f, INITIAL_CANDIDATES_ID, random_number_generator);
#else
#error "Unsupported bias correction mode"
#endif

	temporal_reuse_output_reservoir.combine_with(initial_candidates_reservoir, initial_candidates_mis_weight, initial_candidates_reservoir.sample.target_function, /* jacobian is 1 when reusing at the exact same spot */ 1.0f, random_number_generator);
	temporal_reuse_output_reservoir.sanity_check(make_int2(x, y));

	float normalization_numerator = 1.0f;
	float normalization_denominator = 1.0f;

	ReSTIRTemporalNormalizationWeight<ReSTIR_GI_BiasCorrectionWeights, true> normalization_function;
#if ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_MIS_GBH
	normalization_function.get_normalization(normalization_numerator, normalization_denominator);
#elif ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_PAIRWISE_MIS
	normalization_function.get_normalization(normalization_numerator, normalization_denominator);
#elif ReSTIR_GI_BiasCorrectionWeights == RESTIR_GI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE
	normalization_function.get_normalization(normalization_numerator, normalization_denominator);
#else
#error "Unsupported bias correction mode"
#endif

	temporal_reuse_output_reservoir.end_with_normalization(normalization_numerator, normalization_denominator);
	temporal_reuse_output_reservoir.sanity_check(make_int2(x, y));

	// M-capping so that we don't have to M-cap when reading reservoirs on the next frame
	if (render_data.render_settings.restir_gi_settings.m_cap > 0)
		// M-capping the temporal neighbor if an M-cap has been given
		temporal_reuse_output_reservoir.M = hippt::min(temporal_reuse_output_reservoir.M, render_data.render_settings.restir_gi_settings.m_cap);

	render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs[center_pixel_index] = temporal_reuse_output_reservoir;

	static int counter = 0;
	/*if (counter++ % 10 == 0 && hippt::is_nan(render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs[center_pixel_index].UCW) || hippt::is_inf(render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs[center_pixel_index].UCW))
		printf("HERE %d: [%d, %d], %f/%f, %f/%f\n", render_data.render_settings.sample_number, x, y, temporal_reuse_output_reservoir.UCW, render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs[center_pixel_index].UCW, temporal_reuse_output_reservoir.weight_sum, render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs[center_pixel_index].weight_sum);*/
}

#endif
