/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_TARGET_FUNCTION_H
#define DEVICE_RESTIR_DI_TARGET_FUNCTION_H

#include "Device/includes/ReSTIR/Utils.h"
#include "HostDeviceCommon/RenderData.h"

template <bool withVisiblity>
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_DI_evaluate_target_function(const HIPRTRenderData& render_data, const ReSTIRDISample& sample, ReSTIRSurface& surface, Xorshift32Generator& random_number_generator)
{
#ifndef __KERNELCC__
	std::cerr << "ReSTIR_DI_evaluate_target_function() wrong specialization called: " << withVisiblity << std::endl;
	Utils::debugbreak();
#endif

	return -1.0f;
}

template <>
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_DI_evaluate_target_function<KERNEL_OPTION_FALSE>(const HIPRTRenderData& render_data, const ReSTIRDISample& sample, ReSTIRSurface& surface, Xorshift32Generator& random_number_generator)
{
	if (sample.emissive_triangle_index == -1 && !(sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE))
		// Not an envmap sample and no emissive triangle sampled
		return 0.0f;

	float bsdf_pdf;
	float3 sample_direction;

	if (sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE)
		sample_direction = matrix_X_vec(render_data.world_settings.envmap_to_world_matrix, sample.point_on_light_source);
	else
		sample_direction = hippt::normalize(sample.point_on_light_source - surface.shading_point);

	float cosine_term = hippt::max(0.0f, hippt::dot(surface.shading_normal, sample_direction));
	if (cosine_term == 0.0f)
		// If the cosine term is 0.0f, the rest is going to be multiplied by that zero-cosine-term
		// and everything is going to be 0.0f anyway so we can return already
		return 0.0f;

	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, surface.material, surface.ray_volume_state, false,
		surface.view_direction, surface.shading_normal, surface.geometric_normal, sample_direction,
		bsdf_pdf, random_number_generator, /* current bounce, always for ReSTIR */ 0, sample.flags_to_BSDF_incident_light_info());

	ColorRGB32F sample_emission;
	if (sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE)
	{
		float envmap_pdf;
		sample_emission = envmap_eval(render_data, sample_direction, envmap_pdf);
	}
	else
	{
		int material_index = render_data.buffers.material_indices[sample.emissive_triangle_index];
		sample_emission = render_data.buffers.materials_buffer.get_emission(material_index);
	}

	float target_function = (bsdf_color * sample_emission * cosine_term).luminance();
	if (target_function == 0.0f)
		// Quick exit because computing the visiblity that follows isn't going
		// to change anything to the fact that we have 0.0f target function here
		return 0.0f;

	return target_function;
}

template <>
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_DI_evaluate_target_function<KERNEL_OPTION_TRUE>(const HIPRTRenderData& render_data, const ReSTIRDISample& sample, ReSTIRSurface& surface, Xorshift32Generator& random_number_generator)
{
	if (sample.emissive_triangle_index == -1 && !(sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE))
		// No sample
		return 0.0f;

	float bsdf_pdf;
	float distance_to_light;
	float3 sample_direction;
	if (sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE)
	{
		sample_direction = matrix_X_vec(render_data.world_settings.envmap_to_world_matrix, sample.point_on_light_source);
		distance_to_light = 1.0e35f;
	}
	else
	{
		sample_direction = sample.point_on_light_source - surface.shading_point;
		sample_direction = sample_direction / (distance_to_light = hippt::length(sample_direction));
	}

	float cosine_term = hippt::max(0.0f, hippt::dot(surface.shading_normal, sample_direction));
	if (cosine_term == 0.0f)
		// If the cosine term is 0.0f, the rest is going to be multiplied by that zero-cosine-term
		// and everything is going to be 0.0f anyway so we can return already
		return 0.0f;

	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, surface.material, surface.ray_volume_state, false,
		surface.view_direction, surface.shading_normal, surface.geometric_normal, sample_direction,
		bsdf_pdf, random_number_generator, /* current bounce, always for ReSTIR */ 0, sample.flags_to_BSDF_incident_light_info());

	ColorRGB32F sample_emission;
	if (sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE)
	{
		float envmap_pdf;
		sample_emission = envmap_eval(render_data, sample_direction, envmap_pdf);
	}
	else
	{
		int material_index = render_data.buffers.material_indices[sample.emissive_triangle_index];
		sample_emission = render_data.buffers.materials_buffer.get_emission(material_index);
	}

	float target_function = (bsdf_color * sample_emission * cosine_term).luminance();
	if (target_function == 0.0f)
		// Quick exit because computing the visiblity that follows isn't going
		// to change anything to the fact that we have 0.0f target function here
		return 0.0f;

	hiprtRay shadow_ray;
	shadow_ray.origin = surface.shading_point;
	shadow_ray.direction = sample_direction;

	bool visible = !evaluate_shadow_ray(render_data, shadow_ray, distance_to_light, surface.last_hit_primitive_index, /* bounce. Always 0 for ReSTIR DI*/ 0, random_number_generator);

	target_function *= visible;

	return target_function;
}

#endif
