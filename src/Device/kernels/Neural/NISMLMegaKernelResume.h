/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_NISML_MEGA_KERNEL_RESUME_H
#define KERNELS_NISML_MEGA_KERNEL_RESUME_H

#include "Device/includes/NISMLMegaKernelCommon.h"

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded pointer as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char NISML_MEGAKERNEL_RESUME_RENDER_DATA[sizeof(HIPRTRenderData*)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) NISMLMegaKernelResume(unsigned int stage_index)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline NISMLMegaKernelResume(HIPRTRenderData render_data, unsigned int stage_index, int x, int y)
#endif
{
#ifdef __KERNELCC__
	HIPRTRenderData* render_data_pointer = *reinterpret_cast<HIPRTRenderData**>(NISML_MEGAKERNEL_RESUME_RENDER_DATA);
	HIPRTRenderData& render_data		 = *render_data_pointer;
	unsigned int x						 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y						 = blockIdx.y * blockDim.y + threadIdx.y;
#endif
	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		return;

	unsigned int pixel_index			= x + y * render_data.render_settings.render_resolution.x;
	NISMLMegaKernelPathState path_state = render_data.nisml_mega_kernel.path_states[pixel_index];
	if (path_state == NISMLMegaKernelPathState::FINISHED || path_state == NISMLMegaKernelPathState::UNINITIALIZED)
		return;

	RayPayload ray_payload;
	hiprtRay ray;
	HitInfo closest_hit_info;
	bool intersection_found;
	nisml_megakernel_load_path(render_data, pixel_index, ray_payload, ray, closest_hit_info, intersection_found);

	Xorshift32Generator random_number_generator(render_data.nisml_mega_kernel.path_data[pixel_index].rng_state);
	NEEDeferredMISContext nee_deferred_MIS_context;

	NISResult nis_result;
	if (path_state == NISMLMegaKernelPathState::WAITING_FOR_NIS)
	{
		NISQuery query	 = render_data.nisml_mega_kernel.queries[render_data.nisml_mega_kernel.path_data[pixel_index].query_index];
		float* residuals = render_data.nisml_mega_kernel.residuals +
						   render_data.nisml_mega_kernel.path_data[pixel_index].query_index * render_data.nisml_mega_kernel.residual_stride;

		NISMLLightSample nisml_sample;
		ColorRGB32F emissive_geometry_direct_contribution =
			sample_one_light_no_MIS_neural_many_lights_from_residuals(render_data, ray_payload, closest_hit_info, -ray.direction, query.sg_specular_weight,
																	  query.alpha_x, query.alpha_y, random_number_generator, residuals, nisml_sample);

		nis_result.emissive_triangle_global_index = nisml_sample.emissive_triangle_global_index;
		nis_result.cluster_index				  = nisml_sample.cluster_index;
		nis_result.cluster_probability			  = nisml_sample.cluster_probability;
		nis_result.conditional_light_probability  = nisml_sample.conditional_light_probability;
		nis_result.emissive_triangle_pdf		  = nisml_sample.emissive_triangle_pdf;

		if (ray_payload.bounce > 0 || render_data.render_settings.enable_direct_lighting)
		{
			ColorRGB32F unclamped_direct_lighting = estimate_direct_lighting_from_emissive_contribution<true>(
				render_data, ray_payload, emissive_geometry_direct_contribution, closest_hit_info, -ray.direction, random_number_generator);

			ray_payload.ray_color +=
				clamp_direct_lighting_estimation(unclamped_direct_lighting, render_data.render_settings.indirect_contribution_clamp, ray_payload.bounce);
		}
	}
	/*else if (ray_payload.next_ray_state != RayState::MISSED && intersection_found &&
			 (ray_payload.bounce > 0 || render_data.render_settings.enable_direct_lighting))
		ray_payload.ray_color +=
			estimate_direct_lighting(render_data, ray_payload, closest_hit_info, -ray.direction, x, y, nee_deferred_MIS_context, random_number_generator);*/

	if (ray_payload.next_ray_state != RayState::MISSED && intersection_found && (ray_payload.bounce > 0 || render_data.render_settings.enable_direct_lighting))
		sanity_check<true>(render_data, ray_payload.ray_color, x, y);

	if (path_state == NISMLMegaKernelPathState::WAITING_FOR_NIS)
	{
		nis_result.rng_state							   = random_number_generator.m_state.seed;
		render_data.nisml_mega_kernel.results[pixel_index] = nis_result;
	}

	if (ray_payload.next_ray_state == RayState::MISSED)
	{
		nisml_megakernel_finalize_path_with_context(render_data, pixel_index, x, y, ray_payload, ray, closest_hit_info, random_number_generator,
													nee_deferred_MIS_context, false);
		return;
	}

	if (!intersection_found)
	{
		ray_payload.ray_color += path_tracing_miss_gather_envmap(render_data, ray_payload, ray.direction, pixel_index);
		ray_payload.next_ray_state = RayState::MISSED;
		nisml_megakernel_finalize_path_with_context(render_data, pixel_index, x, y, ray_payload, ray, closest_hit_info, random_number_generator,
													nee_deferred_MIS_context, true);
		return;
	}

	BSDFIncidentLightInfo sampled_light_info = BSDFIncidentLightInfo::NO_INFO; // This variable is never used, this is just for debugging on the CPU
																			   // so that we know what the BSDF sampled
	bool valid_indirect_bounce = path_tracing_compute_next_indirect_bounce(render_data, ray_payload, closest_hit_info, -ray.direction, ray,
																		   random_number_generator, sampled_light_info, nee_deferred_MIS_context);

	if (!valid_indirect_bounce)
	{
		nisml_megakernel_finalize_path_with_context(render_data, pixel_index, x, y, ray_payload, ray, closest_hit_info, random_number_generator,
													nee_deferred_MIS_context, true);
		return;
	}

	if (ray_payload.bounce >= render_data.render_settings.nb_bounces)
	{
		// The original megakernel increments the loop counter before the final deferred NEE pass.
		ray_payload.bounce++;
		nisml_megakernel_finalize_path_with_context(render_data, pixel_index, x, y, ray_payload, ray, closest_hit_info, random_number_generator,
													nee_deferred_MIS_context, false);
		return;
	}

	ray_payload.bounce++;
	intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, ray, ray_payload, closest_hit_info, random_number_generator);
	ray_payload.ray_color +=
		do_deferred_NEE_MIS(render_data, intersection_found, ray_payload, closest_hit_info, nee_deferred_MIS_context, random_number_generator);

	nisml_megakernel_store_path(render_data, pixel_index, ray_payload, ray, closest_hit_info, intersection_found);
	render_data.nisml_mega_kernel.path_data[pixel_index].rng_state = random_number_generator.m_state.seed;
	render_data.nisml_mega_kernel.path_states[pixel_index]		   = NISMLMegaKernelPathState::READY_TO_RESUME;
}

#endif
