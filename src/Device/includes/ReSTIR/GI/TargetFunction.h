/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_GI_TARGET_FUNCTION_H
#define DEVICE_RESTIR_GI_TARGET_FUNCTION_H

#include "Device/includes/ReSTIR/Surface.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "HostDeviceCommon/RenderData.h"

template <bool withVisiblity>
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_GI_evaluate_target_function(const HIPRTRenderData& render_data, const ReSTIRGISample& sample, ReSTIRSurface& surface, Xorshift32Generator& random_number_generator, bool debug=false, int x=-1, int y=-1)
{
#ifndef __KERNELCC__
	std::cerr << "ReSTIR_GI_evaluate_target_function() wrong specialization called: " << withVisiblity << std::endl;
	Utils::debugbreak();
#endif

	return -1.0f;
}

template <>
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_GI_evaluate_target_function<KERNEL_OPTION_FALSE>(const HIPRTRenderData& render_data, const ReSTIRGISample& sample, ReSTIRSurface& surface, Xorshift32Generator& random_number_generator, bool debug, int x, int y)
{

	float3 incident_light_direction;
	if (sample.is_envmap_path())
		// For envmap path, the direction is stored in the 'sample_point' value
		incident_light_direction = sample.sample_point;
	else
		// Not an envmap path, the direction is the difference between the current shading
		// point and the reconnection point
		incident_light_direction = hippt::normalize(sample.sample_point - surface.shading_point);

	if (hippt::dot(incident_light_direction, surface.shading_normal) <= 0.001f)
		return 0.0f;

	//return sample.outgoing_radiance_to_visible_point.luminance();

	float bsdf_pdf;
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, surface.material, surface.ray_volume_state, false, surface.view_direction, surface.shading_normal, surface.geometric_normal, incident_light_direction, bsdf_pdf, random_number_generator, 0, sample.incident_light_info);
	if (bsdf_pdf > 0.0f)
	{
		//bsdf_color /= bsdf_pdf;
		bsdf_color *= hippt::abs(hippt::dot(surface.shading_normal, incident_light_direction));
	}

	return (bsdf_color * sample.outgoing_radiance_to_visible_point).luminance();
}

template <>
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_GI_evaluate_target_function<KERNEL_OPTION_TRUE>(const HIPRTRenderData& render_data, const ReSTIRGISample& sample, ReSTIRSurface& surface, Xorshift32Generator& random_number_generator, bool debug, int x, int y)
{
	float distance_to_sample_point;
	float3 incident_light_direction;
	if (sample.is_envmap_path())
	{
		// For envmap path, the direction is stored in the 'sample_point' value
		incident_light_direction = sample.sample_point;
		distance_to_sample_point = 1.0e35f;
	}
	else
	{
		// Not an envmap path, the direction is the difference between the current shading
		// point and the reconnection point
		incident_light_direction = sample.sample_point - surface.shading_point;
		distance_to_sample_point = hippt::length(incident_light_direction);
		incident_light_direction /= distance_to_sample_point;
	}

	hiprtRay visibility_ray;
	visibility_ray.origin = surface.shading_point;
	visibility_ray.direction = incident_light_direction;

	bool sample_point_occluded = evaluate_shadow_ray(render_data, visibility_ray, distance_to_sample_point, surface.last_hit_primitive_index, 0, random_number_generator);
	if (sample_point_occluded)
		return 0.0f;
	else if (hippt::dot(incident_light_direction, surface.shading_normal) <= 0.001f)
		return 0.0f;

	float bsdf_pdf;
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, surface.material, surface.ray_volume_state, false, surface.view_direction, surface.shading_normal, surface.geometric_normal, incident_light_direction, bsdf_pdf, random_number_generator, 0, sample.incident_light_info);
	if (bsdf_pdf > 0.0f)
	{
		//bsdf_color /= bsdf_pdf;
		bsdf_color *= hippt::abs(hippt::dot(surface.shading_normal, incident_light_direction));
	}

	return (bsdf_color * sample.outgoing_radiance_to_visible_point).luminance();
}

#endif
