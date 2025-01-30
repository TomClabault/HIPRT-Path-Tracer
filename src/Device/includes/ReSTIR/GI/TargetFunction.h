/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_GI_TARGET_FUNCTION_H
#define DEVICE_RESTIR_GI_TARGET_FUNCTION_H

#include "Device/includes/ReSTIR/Surface.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "Device/includes/ReSTIR/GI/Utils.h"
#include "HostDeviceCommon/RenderData.h"

template <bool withVisiblity>
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_GI_evaluate_target_function(const HIPRTRenderData& render_data, const ReSTIRGISample& sample, ReSTIRDISurface& surface, Xorshift32Generator& random_number_generator)
{
#ifndef __KERNELCC__
	std::cerr << "ReSTIR_GI_evaluate_target_function() wrong specialization called: " << withVisiblity << std::endl;
	Utils::debugbreak();
#endif

	return -1.0f;
}

template <>
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_GI_evaluate_target_function<KERNEL_OPTION_FALSE>(const HIPRTRenderData& render_data, const ReSTIRGISample& sample, ReSTIRDISurface& surface, Xorshift32Generator& random_number_generator)
{
	float3 incident_light_direction;
	if (ReSTIR_GI_is_envmap_path(sample.second_hit_normal))
		// For envmap path, the direction is stored in the 'second_hit_point' value
		incident_light_direction = sample.second_hit_point;
	else
		// Not an envmap path, the direction is the difference between the current shading
		// point and the reconnection point
		incident_light_direction = sample.second_hit_point - surface.shading_point;

	float bsdf_pdf;
	float bsdf_luminance = bsdf_dispatcher_eval(render_data, surface.material, surface.ray_volume_state, false, surface.view_direction, surface.shading_normal, surface.geometric_normal, incident_light_direction, bsdf_pdf, random_number_generator, 0, sample.incident_light_info).luminance();
	if (bsdf_pdf > 0.0f)
	{
		bsdf_luminance /= bsdf_pdf;
		bsdf_luminance *= hippt::abs(hippt::dot(surface.shading_normal, incident_light_direction));
	}

	return bsdf_luminance * sample.outgoing_radiance_to_first_hit.luminance();
}

template <>
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_GI_evaluate_target_function<KERNEL_OPTION_TRUE>(const HIPRTRenderData& render_data, const ReSTIRGISample& sample, ReSTIRDISurface& surface, Xorshift32Generator& random_number_generator)
{
	float distance_to_sample_point;
	float3 visibility_ray_direction;
	if (ReSTIR_GI_is_envmap_path(sample.second_hit_normal))
	{
		// For envmap path, the direction is stored in the 'second_hit_point' value
		visibility_ray_direction = sample.second_hit_point;
		distance_to_sample_point = 1.0e35f;
	}
	else
	{
		// Not an envmap path, the direction is the difference between the current shading
		// point and the reconnection point
		visibility_ray_direction = sample.second_hit_point - surface.shading_point;
		distance_to_sample_point = hippt::length(visibility_ray_direction);
	}

	hiprtRay visibility_ray;
	visibility_ray.origin = surface.shading_point;
	visibility_ray.direction = visibility_ray_direction / distance_to_sample_point;

	bool sample_point_occluded = evaluate_shadow_ray(render_data, visibility_ray, distance_to_sample_point, surface.last_hit_primitive_index, 0, random_number_generator);

	float bsdf_pdf;
	float bsdf_luminance = bsdf_dispatcher_eval(render_data, surface.material, surface.ray_volume_state, false, surface.view_direction, surface.shading_normal, surface.geometric_normal, visibility_ray_direction, bsdf_pdf, random_number_generator, 0, sample.incident_light_info).luminance();
	if (bsdf_pdf > 0.0f)
	{
		bsdf_luminance /= bsdf_pdf;
		bsdf_luminance *= hippt::abs(hippt::dot(surface.shading_normal, visibility_ray_direction));
	}

	return !sample_point_occluded * bsdf_luminance * sample.outgoing_radiance_to_first_hit.luminance();
}

#endif
