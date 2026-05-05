/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_PT_SHADING_H
#define KERNELS_RESTIR_PT_SHADING_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/LightSampling/LightClamping.h"
#include "Device/includes/LightSampling/NEEEstimators.h"
#include "Device/includes/PathTracing.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/SanityCheck.h"

#include "HostDeviceCommon/KernelOptions/ReSTIRPTOptions.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRSettingsHelper.h"
#include "HostDeviceCommon/Xorshift.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_PT_Shading(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PT_Shading(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif

	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		return;

	uint32_t pixel_index = x + y * render_data.render_settings.render_resolution.x;

	if (!render_data.aux_buffers.pixel_active[pixel_index])
		return;

	Xorshift32Generator random_number_generator(render_data.get_updated_random_seed(pixel_index));

	hiprtRay ray;
	ray.direction = -render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

	HitInfo closest_hit_info;
	closest_hit_info.primitive_index = render_data.g_buffer.first_hit_prim_index[pixel_index];
	if (closest_hit_info.primitive_index == -1)
	{
		// Geometry miss, directly into the envmap
		ColorRGB32F envmap_radiance = path_tracing_miss_gather_envmap(render_data, ColorRGB32F(1.0f), ray.direction, 0, pixel_index);

		path_tracing_accumulate_color(render_data, pixel_index, envmap_radiance);

		return;
	}

	closest_hit_info.inter_point	  = render_data.g_buffer.primary_hit_position[pixel_index];
	closest_hit_info.shading_normal	  = render_data.g_buffer.shading_normals[pixel_index].unpack();
	closest_hit_info.geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();

	// Initializing the ray with the information from the camera ray pass
	RayPayload ray_payload;
	ray_payload.next_ray_state = RayState::BOUNCE;
	// Loading the first hit in the ray payload
	ray_payload.material = render_data.g_buffer.materials[pixel_index].unpack();
	ray_payload.volume_state.reconstruct_first_hit(ray_payload.material, render_data.buffers.material_indices, closest_hit_info.primitive_index,
												   random_number_generator);

	float3_t view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

	ColorRGB32F camera_outgoing_radiance;
	if (render_data.render_settings.enable_direct_lighting)
		// Adding the directly visible emission from an emissive surface
		camera_outgoing_radiance += ray_payload.material.emission;

	ReSTIRPTReservoir resampling_reservoir = render_data.render_settings.restir_pt_settings.restir_output_reservoirs[pixel_index];
	if (resampling_reservoir.UCW > 0.0f && (!resampling_reservoir.sample.di_sample || render_data.render_settings.enable_direct_lighting))
	{
		// Only doing the shading if we do actually have a sample

		float3_t to_light_direction_visible_point;
		if (resampling_reservoir.sample.is_envmap_path())
			to_light_direction_visible_point = resampling_reservoir.sample.rc_vertex;
		else
			to_light_direction_visible_point = hippt::normalize(resampling_reservoir.sample.rc_vertex - closest_hit_info.inter_point);

		// Computing the BSDF throughput at the first hit
		//  - view direction: towards the camera
		//  - incident light direction: towards the sample point
		float bsdf_pdf_first_hit;
		BSDFContext bsdf_first_hit_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, to_light_direction_visible_point,
										   resampling_reservoir.sample.incident_light_info_at_visible_point, ray_payload.volume_state, false,
										   ray_payload.material, 0.0f);
		ColorRGB32F bsdf_first_hit = bsdf_dispatcher_eval(render_data, bsdf_first_hit_context, bsdf_pdf_first_hit, random_number_generator);

		ColorRGB32F first_hit_throughput(0.0f);
		if (bsdf_pdf_first_hit > 0.0f)
			first_hit_throughput = bsdf_first_hit * hippt::abs(hippt::dot(to_light_direction_visible_point, closest_hit_info.shading_normal));

		ColorRGB32F secondary_hit_throughput = ColorRGB32F(1.0f);
		if (resampling_reservoir.sample.is_envmap_path())
			camera_outgoing_radiance +=
				path_tracing_miss_gather_envmap(render_data, first_hit_throughput, to_light_direction_visible_point, 1, pixel_index) * resampling_reservoir.UCW;
		else if (!resampling_reservoir.sample.di_sample)
		{
			float3_t view_direction					 = hippt::normalize(closest_hit_info.inter_point - resampling_reservoir.sample.rc_vertex);
			float3_t to_light_direction_sample_point = resampling_reservoir.sample.rc_vertex_incident_light_direction;
			float3_t shading_normal_sample_point	 = resampling_reservoir.sample.rc_vertex_shading_normal.unpack();
			float3_t geometric_normal_sample_point	 = resampling_reservoir.sample.rc_vertex_geometric_normal.unpack();

			// Reproducing roughness accumulation
			ray_payload.accumulate_roughness(resampling_reservoir.sample.incident_light_info_at_visible_point);
			// TODO the ray volume state should be advanced/updated/pushed into here to reproduce the state that it's in at the sample point
			BSDFContext secondary_hit_eval_context(view_direction, shading_normal_sample_point, geometric_normal_sample_point, to_light_direction_sample_point,
												   resampling_reservoir.sample.incident_light_info_at_sample_point, ray_payload.volume_state, false,
												   resampling_reservoir.sample.rc_vertex_material, 0.0f);

			float trash_pdf;
			ColorRGB32F bsdf_secondary_hit = bsdf_dispatcher_eval(render_data, secondary_hit_eval_context, trash_pdf, random_number_generator);
			secondary_hit_throughput	   = bsdf_secondary_hit * hippt::abs(hippt::dot(to_light_direction_sample_point, shading_normal_sample_point));
		}

		camera_outgoing_radiance += first_hit_throughput * secondary_hit_throughput * resampling_reservoir.sample.rc_vertex_incident_radiance * resampling_reservoir.UCW;
	}

	render_data.store_updated_random_seed(pixel_index, random_number_generator.m_state.seed);

	// Setting the 'camera_outgoing_radiance' into the ray color just for the call to 'sanity_check'
	ray_payload.ray_color = camera_outgoing_radiance;
	if (!sanity_check(render_data, ray_payload.ray_color, x, y))
		return;

	if (render_data.render_settings.restir_pt_settings.debug_view == ReSTIRPTDebugView::PT_FINAL_RESERVOIR_UCW)
		path_tracing_accumulate_color(render_data, pixel_index,
									  ColorRGB32F(resampling_reservoir.UCW) * render_data.render_settings.restir_pt_settings.debug_view_scale_factor);
	else if (render_data.render_settings.restir_pt_settings.debug_view == ReSTIRPTDebugView::PT_TARGET_FUNCTION)
		path_tracing_accumulate_color(render_data, pixel_index,
									  ColorRGB32F(resampling_reservoir.sample.target_function) *
										  render_data.render_settings.restir_pt_settings.debug_view_scale_factor);
	else if (render_data.render_settings.restir_pt_settings.debug_view == ReSTIRPTDebugView::PT_WEIGHT_SUM)
		path_tracing_accumulate_color(render_data, pixel_index,
									  ColorRGB32F(resampling_reservoir.weight_sum) * render_data.render_settings.restir_pt_settings.debug_view_scale_factor);
	else if (render_data.render_settings.restir_pt_settings.debug_view == ReSTIRPTDebugView::PT_M_COUNT)
		path_tracing_accumulate_color(render_data, pixel_index,
									  ColorRGB32F(resampling_reservoir.M) * render_data.render_settings.restir_pt_settings.debug_view_scale_factor);
	else if (render_data.render_settings.restir_pt_settings.debug_view == ReSTIRPTDebugView::PT_PER_PIXEL_REUSE_RADIUS &&
			 render_data.render_settings.restir_pt_settings.common_spatial_pass.per_pixel_spatial_reuse_radius != nullptr)
	{
		float radius_percentage = (render_data.render_settings.restir_pt_settings.common_spatial_pass.per_pixel_spatial_reuse_radius[pixel_index] /
								   (float)render_data.render_settings.restir_pt_settings.common_spatial_pass.reuse_radius);
		ColorRGB32F debug_color = hippt::lerp(ColorRGB32F(2.0f, 0.0f, 0.0f), ColorRGB32F(0.0f, 2.0f, 0.0f), radius_percentage);

		debug_set_final_color(render_data, x, y, debug_color);
	}
	else if (render_data.render_settings.restir_pt_settings.debug_view == ReSTIRPTDebugView::PT_PER_PIXEL_VALID_DIRECTIONS_PERCENTAGE &&
			 render_data.render_settings.restir_pt_settings.common_spatial_pass.per_pixel_spatial_reuse_radius != nullptr)
	{
		unsigned char accepted_directions =
			hippt::popc(ReSTIRSettingsHelper::get_spatial_reuse_direction_mask_ull<ReSTIR_VARIANT_PT>(render_data, pixel_index));
		float accepted_percentage = accepted_directions / 32.0f;
		ColorRGB32F debug_color	  = hippt::lerp(ColorRGB32F(2.0f, 0.0f, 0.0f), ColorRGB32F(0.0f, 2.0f, 0.0f), accepted_percentage);

		debug_set_final_color(render_data, x, y, debug_color);
	}
	else if (render_data.render_settings.restir_pt_settings.debug_view == ReSTIRPTDebugView::PT_SHADE_ONLY_INITIAL_CANDIDATES &&
			 ReSTIR_PT_DebugViewShadeOnlyInitialCandidatesEnabled)
	{
		// The initial candidate's color is set into accumulated_ray_colors by the InitialCandidatesPass for this debug view so we just reuse that color
		ray_payload.ray_color = render_data.buffers.accumulated_ray_colors[pixel_index];
#if DisplayOnlySampleN == KERNEL_OPTION_FALSE
		// If not only displaying sample n, we need to scale that, otherwise it's the debug code of display only sample N that does the scaling
		ray_payload.ray_color *= render_data.render_settings.sample_number + 1;
#endif

		ColorRGB32F debug_color;
		path_tracing_compute_debug_view_debug_color(render_data, ray_payload, pixel_index, random_number_generator, debug_color);

		// Regular output
		path_tracing_accumulate_color(render_data, pixel_index, ray_payload.ray_color, debug_color);
	}
	else
	{
		ColorRGB32F debug_color;
		path_tracing_compute_debug_view_debug_color(render_data, ray_payload, pixel_index, random_number_generator, debug_color);

		// Regular output
		path_tracing_accumulate_color(render_data, pixel_index, ray_payload.ray_color, debug_color);
	}
}

#endif
