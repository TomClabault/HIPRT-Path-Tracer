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
HIPRT_DEVICE LightSampleInformation sample_one_emissive_triangle_uniform(const HIPRTRenderData& render_data, float3 shading_point, Xorshift32Generator& random_number_generator)
{
    if (render_data.buffers.emissive_triangles_count == 0)
        return LightSampleInformation();

    int random_emissive_triangle_index = random_number_generator.random_index(render_data.buffers.emissive_triangles_count);
    int triangle_index = render_data.buffers.emissive_triangles_primitive_indices[random_emissive_triangle_index];

    LightSampleInformation light_sample = sample_point_on_generic_triangle_and_fill_light_sample_information(render_data, shading_point, triangle_index, random_number_generator);

    // PDF of that triangle sampled uniformly amongst all emissive triangles
    light_sample.area_measure_pdf /= render_data.buffers.emissive_triangles_count;

    return light_sample;
}

HIPRT_DEVICE LightSampleInformation sample_one_emissive_triangle_power(const HIPRTRenderData& render_data, float3 shading_point, Xorshift32Generator& random_number_generator)
{
    if (render_data.buffers.emissive_triangles_count == 0)
        return LightSampleInformation();

    int random_emissive_triangle_index = render_data.buffers.emissive_triangles_power_alias_table.sample(random_number_generator);
    int triangle_index = render_data.buffers.emissive_triangles_primitive_indices[random_emissive_triangle_index];

    LightSampleInformation light_sample = sample_point_on_generic_triangle_and_fill_light_sample_information(render_data, shading_point, triangle_index, random_number_generator);

    // PDF of sampling that triangle according to its power
    light_sample.area_measure_pdf *= (light_sample.emission.luminance() * light_sample.light_area) / render_data.buffers.emissive_triangles_power_alias_table.sum_elements;

    return light_sample;
}

HIPRT_DEVICE LightSampleInformation sample_one_emissive_triangle_regir(
    const HIPRTRenderData& render_data,
    const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
    int last_hit_primitive_index, RayPayload& ray_payload,
    bool& out_need_fallback_sampling,
    Xorshift32Generator& random_number_generator);

template <int samplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE LightSampleInformation sample_one_emissive_triangle(const HIPRTRenderData& render_data,
    const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
    int last_hit_primitive_index, RayPayload& ray_payload,
    Xorshift32Generator& random_number_generator)
{
    if constexpr (samplingStrategy == LSS_BASE_UNIFORM)
    {
        return sample_one_emissive_triangle_uniform(render_data, shading_point, random_number_generator);
    }
    else if constexpr (samplingStrategy == LSS_BASE_POWER)
    {
        return sample_one_emissive_triangle_power(render_data, shading_point, random_number_generator);
    }
    else if constexpr (samplingStrategy == LSS_BASE_LIGHT_TREE_ATS)
    {
        return sample_one_emissive_triangle_light_tree_ats(render_data, shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index, ray_payload, random_number_generator);
    }
    else if constexpr (samplingStrategy == LSS_BASE_LIGHT_TREE_SG)
    {
        return sample_one_emissive_triangle_light_tree_sg(render_data, shading_point, view_direction, shading_normal, geometric_normal, ray_payload.material,
            last_hit_primitive_index, random_number_generator);
    }
    else if constexpr (samplingStrategy == LSS_BASE_REGIR)
    {
        bool point_outside_grid = false;

        LightSampleInformation light_sample = sample_one_emissive_triangle_regir(render_data,
            shading_point, view_direction, shading_normal, geometric_normal,
            last_hit_primitive_index, ray_payload,
            point_outside_grid,
            random_number_generator);

        if (!point_outside_grid)
            return light_sample;
        else
        {
#if ReGIR_FallbackLightSamplingStrategy == LSS_BASE_REGIR
            // Invalid fallback strategy
            invalid ReGIR light sampling fallback strategy
#endif

                // Fallback method as the point was outside of the ReGIR grid
                return sample_one_emissive_triangle<ReGIR_FallbackLightSamplingStrategy>(render_data,
                    shading_point, view_direction, shading_normal, geometric_normal,
                    last_hit_primitive_index, ray_payload,
                    random_number_generator);
        }
    }
}

#endif
