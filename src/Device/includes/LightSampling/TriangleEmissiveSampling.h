/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_EMISSIVE_SAMPLING_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_EMISSIVE_SAMPLING_H
 
#include "Device/includes/LightSampling/LightTree/LightTreeATSSampling.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSampling.h"
#include "Device/includes/LightSampling/TriangleSampling.h"

#include "HostDeviceCommon/KernelOptions/ReGIROptions.h"

 /**
 * The PDF is computed in area measure
 */
HIPRT_DEVICE LightSampleInformation sample_one_light_uniform(const HIPRTRenderData& render_data,
    float3 shading_point, float3 view_direction, float3 shading_normal, 
	const DeviceUnpackedEffectiveMaterial& material,
    Xorshift32Generator& random_number_generator)
{
    if (render_data.buffers.emissive_triangles_count == 0)
        return LightSampleInformation();

    int random_emissive_triangle_index = random_number_generator.random_index(render_data.buffers.emissive_triangles_count);
    int triangle_index = render_data.buffers.emissive_triangles_primitive_indices[random_emissive_triangle_index];

    LightSampleInformation triangle_sample_information;
	triangle_sample_information.emissive_triangle_global_index = triangle_index;
	triangle_sample_information.pdf = 1.0f / render_data.buffers.emissive_triangles_count;

    return triangle_sample_information;
}

HIPRT_DEVICE LightSamplePointInformation sample_one_point_on_light_uniform(const HIPRTRenderData& render_data,
    float3 shading_point, float3 view_direction, float3 shading_normal,
    const DeviceUnpackedEffectiveMaterial& material,
    Xorshift32Generator& random_number_generator)
{
    if (render_data.buffers.emissive_triangles_count == 0)
        return LightSamplePointInformation();

    int random_emissive_triangle_index = random_number_generator.random_index(render_data.buffers.emissive_triangles_count);
    int triangle_index = render_data.buffers.emissive_triangles_primitive_indices[random_emissive_triangle_index];

    LightSamplePointInformation light_sample = sample_point_on_light_and_fill_light_sample_information(render_data,
        shading_point, view_direction, shading_normal,
        material,
        triangle_index, random_number_generator);

    // PDF of that triangle sampled uniformly amongst all emissive triangles
    light_sample.area_measure_pdf /= render_data.buffers.emissive_triangles_count;

    return light_sample;
}

HIPRT_DEVICE LightSampleInformation sample_one_light_power(const HIPRTRenderData& render_data,
    float3 shading_point, float3 view_direction, float3 shading_normal, 
    const DeviceUnpackedEffectiveMaterial& material,
    Xorshift32Generator& random_number_generator)
{
    if (render_data.buffers.emissive_triangles_count == 0)
        return LightSampleInformation();

    int random_emissive_triangle_index = render_data.buffers.emissive_triangles_power_alias_table.sample(random_number_generator);
    int triangle_index = render_data.buffers.emissive_triangles_primitive_indices[random_emissive_triangle_index];

	ColorRGB32F emission = triangle_load_emission(render_data, triangle_index);
	float triangle_area = triangle_load_area(render_data, triangle_index);

    LightSampleInformation triangle_sample_information;
    triangle_sample_information.emissive_triangle_global_index = triangle_index;
    triangle_sample_information.pdf = (emission.luminance() * triangle_area) / render_data.buffers.emissive_triangles_power_alias_table.sum_elements;

    return triangle_sample_information;
}

/**
 * This function directly returns the sampled point data on the light itself sampled by power
 * 
 * This function has been faster than sampling the triangle first and then sampling the point on it so
 * that's why it's there
 */
HIPRT_DEVICE LightSamplePointInformation sample_one_point_on_light_power(const HIPRTRenderData& render_data,
    float3 shading_point, float3 view_direction, float3 shading_normal,
    const DeviceUnpackedEffectiveMaterial& material,
    Xorshift32Generator& random_number_generator)
{
    if (render_data.buffers.emissive_triangles_count == 0)
        return LightSamplePointInformation();

    int random_emissive_triangle_index = render_data.buffers.emissive_triangles_power_alias_table.sample(random_number_generator);
    int triangle_index = render_data.buffers.emissive_triangles_primitive_indices[random_emissive_triangle_index];

    LightSamplePointInformation light_sample = sample_point_on_light_and_fill_light_sample_information(render_data,
        shading_point, view_direction, shading_normal,
        material,
        triangle_index, random_number_generator);

    // PDF of sampling that triangle according to its power
    light_sample.area_measure_pdf *= (light_sample.emission.luminance() * light_sample.light_area) / render_data.buffers.emissive_triangles_power_alias_table.sum_elements;

    return light_sample;
}

template <int samplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE LightSampleArray<DirectLightSampleCount<samplingStrategy>()> sample_one_light(const HIPRTRenderData& render_data,
    const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
    int last_hit_primitive_index, RayPayload& ray_payload,
    Xorshift32Generator& random_number_generator)
{
    LightSampleArray<DirectLightSampleCount<samplingStrategy>()> light_samples;

    if constexpr (samplingStrategy == LSS_BASE_UNIFORM)
    {
        light_samples[0] = sample_one_light_uniform(render_data,
            shading_point, view_direction, shading_normal,
            ray_payload.material,
            random_number_generator);
    }
    else if constexpr (samplingStrategy == LSS_BASE_POWER)
    {
        light_samples[0] = sample_one_light_power(render_data,
            shading_point, view_direction, shading_normal,
            ray_payload.material,
            random_number_generator);

        if (DirectLightSampleCount<LSS_BASE_POWER>() > 1)
        {
            light_samples[1] = sample_one_light_power(render_data,
                shading_point, view_direction, shading_normal,
                ray_payload.material,
                random_number_generator);
        }
    }
    else if constexpr (samplingStrategy == LSS_BASE_LIGHT_TREE_ATS)
    {
        light_samples = sample_one_emissive_triangle_light_tree_ats(render_data,
            shading_point, view_direction, shading_normal, geometric_normal,
            last_hit_primitive_index, ray_payload, random_number_generator);
    }
    else if constexpr (samplingStrategy == LSS_BASE_LIGHT_TREE_SG)
    {
        light_samples[0] = sample_one_emissive_triangle_light_tree_sg(render_data,
            shading_point, view_direction, shading_normal, geometric_normal,
            ray_payload.material, last_hit_primitive_index, random_number_generator);
    }

	return light_samples;
}

HIPRT_DEVICE LightSamplePointInformation sample_one_point_on_light_regir(const HIPRTRenderData& render_data,
    const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
    int last_hit_primitive_index, RayPayload& ray_payload,
    bool& out_need_fallback_sampling,
    Xorshift32Generator& random_number_generator);

template <int samplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE LightSamplePointArray<DirectLightSampleCount<samplingStrategy>()> sample_one_point_on_light(const HIPRTRenderData& render_data,
    const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
    int last_hit_primitive_index, RayPayload& ray_payload,
    Xorshift32Generator& random_number_generator)
{
    LightSamplePointArray<DirectLightSampleCount<samplingStrategy>()> light_point_samples;

    if constexpr (samplingStrategy == LSS_BASE_UNIFORM)
    {
        light_point_samples[0] = sample_one_point_on_light_uniform(render_data,
            shading_point, view_direction, shading_normal,
            ray_payload.material,
            random_number_generator);
    }
    else if constexpr (samplingStrategy == LSS_BASE_POWER)
    {
        light_point_samples[0] = sample_one_point_on_light_power(render_data,
            shading_point, view_direction, shading_normal,
            ray_payload.material,
            random_number_generator);

        // TODO THIS IS DEBUG REMOVE THIS
        if (DirectLightSampleCount<LSS_BASE_POWER>() > 1)
        {
            light_point_samples[1] = sample_one_point_on_light_power(render_data,
                shading_point, view_direction, shading_normal,
                ray_payload.material,
                random_number_generator);
        }
    }
    else if constexpr (samplingStrategy == LSS_BASE_LIGHT_TREE_ATS)
    {
        unsigned int seed_before = random_number_generator.m_state.seed;
        random_number_generator.m_state.seed = seed_before;

        LightSampleArray<DirectLightSampleCount<LSS_BASE_LIGHT_TREE_ATS>()> light_samples = sample_one_emissive_triangle_light_tree_ats(render_data,
            shading_point, view_direction, shading_normal, geometric_normal, 
            last_hit_primitive_index, ray_payload, random_number_generator);

		for (int i = 0; i < DirectLightSampleCount<LSS_BASE_LIGHT_TREE_ATS>(); i++)
        {
            light_point_samples[i] = sample_point_on_light_and_fill_light_sample_information(render_data,
                shading_point, view_direction, shading_normal,
                ray_payload.material,
                light_samples[i].emissive_triangle_global_index, random_number_generator);
            light_point_samples[i].area_measure_pdf *= light_samples[i].pdf;
        }
    }
    else if constexpr (samplingStrategy == LSS_BASE_LIGHT_TREE_SG)
    {
        LightSampleInformation light_sample = sample_one_emissive_triangle_light_tree_sg(render_data, 
            shading_point, view_direction, shading_normal, geometric_normal, 
            ray_payload.material, last_hit_primitive_index, random_number_generator);

        light_point_samples[0] = sample_point_on_light_and_fill_light_sample_information(render_data,
            shading_point, view_direction, shading_normal,
            ray_payload.material,
            light_sample.emissive_triangle_global_index, random_number_generator);

        light_point_samples[0].area_measure_pdf *= light_sample.pdf;
    }
    else if constexpr (samplingStrategy == LSS_BASE_REGIR)
    {
        bool point_outside_grid = false;

        light_point_samples[0] = sample_one_point_on_light_regir(render_data,
            shading_point, view_direction, shading_normal, geometric_normal,
            last_hit_primitive_index, ray_payload,
            point_outside_grid,
            random_number_generator);

        if (!point_outside_grid)
            return light_point_samples;
        else
        {
#if ReGIR_FallbackLightSamplingStrategy == LSS_BASE_REGIR
            // Invalid fallback strategy
            invalid ReGIR light sampling fallback strategy
#endif

            // Fallback method as the point was outside of the ReGIR grid
            light_point_samples = sample_one_point_on_light<ReGIR_FallbackLightSamplingStrategy>(render_data,
                shading_point, view_direction, shading_normal, geometric_normal,
                last_hit_primitive_index, ray_payload,
                random_number_generator);
        }
    }

    return light_point_samples;
}

#endif
