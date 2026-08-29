/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_NISML_MEGA_KERNEL_GENERATE_QUERIES_H
#define KERNELS_NISML_MEGA_KERNEL_GENERATE_QUERIES_H

#include "Device/includes/NISMLMegaKernelCommon.h"

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char NISML_MEGAKERNEL_GENERATE_QUERIES_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) NISMLMegaKernelGenerateQueries(unsigned int stage_index)
#else // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline NISMLMegaKernelGenerateQueries(HIPRTRenderData render_data, unsigned int stage_index, int x, int y)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(NISML_MEGAKERNEL_GENERATE_QUERIES_RENDER_DATA);
	unsigned int x				 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y				 = blockIdx.y * blockDim.y + threadIdx.y;
#endif
	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		return;

	unsigned int pixel_index = x + y * render_data.render_settings.render_resolution.x;
	if (stage_index == 0)
	{
		nisml_megakernel_initialize_path(render_data, pixel_index);
		if (render_data.nisml_mega_kernel.path_states[pixel_index] == NISMLMegaKernelPathState::FINISHED)
			return;
	}

	if (render_data.nisml_mega_kernel.path_states[pixel_index] != NISMLMegaKernelPathState::READY_TO_RESUME)
		return;

	RayPayload ray_payload;
	hiprtRay ray;
	HitInfo closest_hit_info;
	bool intersection_found;
	nisml_megakernel_load_path(render_data, pixel_index, ray_payload, ray, closest_hit_info, intersection_found);

	if (ray_payload.next_ray_state == RayState::MISSED || !intersection_found)
	{
		render_data.nisml_mega_kernel.path_states[pixel_index] = NISMLMegaKernelPathState::READY_TO_RESUME;
		return;
	}

	if (ray_payload.bounce == 0)
		store_denoiser_AOVs(render_data, pixel_index, closest_hit_info.shading_normal, ray_payload.material.base_color);
	else
	{
		bool ReGIR_primary_hit = render_data.render_settings.regir_settings.compute_is_primary_hit(ray_payload);

		// Storing data for ReGIR representative points
		ReGIR_update_representative_data(render_data, closest_hit_info.inter_point, closest_hit_info.geometric_normal, render_data.current_camera,
										 closest_hit_info.primitive_index, ReGIR_primary_hit, ray_payload.material);
	}

	bool direct_lighting_enabled = ray_payload.bounce > 0 || render_data.render_settings.enable_direct_lighting;
	Xorshift32Generator random_number_generator(render_data.nisml_mega_kernel.path_data[pixel_index].rng_state);
	unsigned int query_index = 0;
	bool query_enqueued		 = direct_lighting_enabled &&
						  append_nisml_query(render_data, pixel_index, ray_payload, closest_hit_info, -ray.direction, random_number_generator, query_index);

	if (query_enqueued)
	{
		render_data.nisml_mega_kernel.path_data[pixel_index].query_index = query_index;
		render_data.nisml_mega_kernel.path_states[pixel_index]			 = NISMLMegaKernelPathState::WAITING_FOR_NIS;
	}
	else
		render_data.nisml_mega_kernel.path_states[pixel_index] = NISMLMegaKernelPathState::READY_TO_RESUME;
}

#endif // #ifndef KERNELS_NISML_MEGA_KERNEL_GENERATE_QUERIES_H
