/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_DI_LIGHTS_PRESAMPLING_H
#define KERNELS_RESTIR_DI_LIGHTS_PRESAMPLING_H

#include "Device/includes/Envmap.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/LightUtils.h"
#include "Device/kernel_parameters/ReSTIR/DI/LightPresamplingParameters.h"

#include "HostDeviceCommon/RenderData.h"

 /** References:
 *
 * [1] [Rearchitecting Spatiotemporal Resampling for Production] https://research.nvidia.com/publication/2021-07_rearchitecting-spatiotemporal-resampling-production
 */

HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRDIPresampledLight presample_envmap(const WorldSettings& world_settings, float envmap_sampling_probability, Xorshift32Generator& random_number_generator)
{
    ReSTIRDIPresampledLight presampled_envmap;
    presampled_envmap.flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE;

    ColorRGB32F radiance = envmap_sample(world_settings, presampled_envmap.point_on_light_source, presampled_envmap.pdf, random_number_generator);

    // Moving the direction to envmap space because that's what we use for ReSTIR DI
    presampled_envmap.point_on_light_source = matrix_X_vec(world_settings.world_to_envmap_matrix, presampled_envmap.point_on_light_source);
    presampled_envmap.radiance = radiance;
    presampled_envmap.pdf *= envmap_sampling_probability;

    return presampled_envmap;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRDIPresampledLight presample_emissive_triangle(const LightPresamplingParameters& parameters, float light_sampling_probability, Xorshift32Generator& random_number_generator)
{
    ReSTIRDIPresampledLight presampled_light;

    int random_index = random_number_generator.random_index(parameters.emissive_triangles_count);
    int triangle_index = parameters.emissive_triangles_indices[random_index];

    float3 vertex_A = parameters.vertices_positions[parameters.triangles_indices[triangle_index * 3 + 0]];
    float3 vertex_B = parameters.vertices_positions[parameters.triangles_indices[triangle_index * 3 + 1]];
    float3 vertex_C = parameters.vertices_positions[parameters.triangles_indices[triangle_index * 3 + 2]];

    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();

    float sqrt_r1 = sqrt(rand_1);
    float u = 1.0f - sqrt_r1;
    float v = (1.0f - rand_2) * sqrt_r1;

    float3 AB = vertex_B - vertex_A;
    float3 AC = vertex_C - vertex_A;
    float3 random_point_on_triangle = vertex_A + AB * u + AC * v;

    float3 normal = hippt::cross(AB, AC);
    float length_normal = hippt::length(normal);
    if (length_normal > 1.0e-6f)
    {
        // Avoiding very small triangles

        float triangle_area = length_normal * 0.5f;

        presampled_light.point_on_light_source = random_point_on_triangle;
        presampled_light.light_source_normal = normal / length_normal;
        presampled_light.emissive_triangle_index = triangle_index;

        // PDF in area measure
        presampled_light.pdf = 1.0f / triangle_area;
        presampled_light.pdf /= parameters.emissive_triangles_count;
        presampled_light.pdf *= light_sampling_probability;
        presampled_light.radiance = parameters.materials.get_emission(parameters.material_indices[triangle_index]);
    }

    return presampled_light;
}

// TODO try just passing LightPresamplingParameters in there instead of everything individually
HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRDIPresampledLight ReSTIR_DI_presample_one_light(const LightPresamplingParameters& parameters, float envmap_sampling_probability, Xorshift32Generator& random_number_generator)
{
    ReSTIRDIPresampledLight presampled_light;
    if (random_number_generator() < envmap_sampling_probability)
        presampled_light = presample_envmap(parameters.world_settings, envmap_sampling_probability, random_number_generator);
    else
        presampled_light = presample_emissive_triangle(parameters, 1.0f - envmap_sampling_probability, random_number_generator);

    return presampled_light;
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_DI_LightsPresampling(LightPresamplingParameters presampling_parameters)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_DI_LightsPresampling(LightPresamplingParameters presampling_parameters, int x)
#endif
{
    if (presampling_parameters.emissive_triangles_count == 0 && presampling_parameters.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
        // No initial candidates to sample since no lights
        return;

#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    if (x >= presampling_parameters.subset_size * presampling_parameters.number_of_subsets)
        return;

    uint32_t thread_index = x;

    unsigned int seed;
    if (presampling_parameters.freeze_random)
        seed = wang_hash(thread_index + 1);
    else
        seed = wang_hash((thread_index + 1) * (presampling_parameters.sample_number + 1) * presampling_parameters.random_number);

    Xorshift32Generator random_number_generator(seed);

    float envmap_candidate_probability = 0.0f;
    if (presampling_parameters.world_settings.ambient_light_type == AmbientLightType::ENVMAP)
    {
        if (presampling_parameters.emissive_triangles_count == 0)
            // Only the envmap to sample
            envmap_candidate_probability = 1.0f;
        else
            envmap_candidate_probability = presampling_parameters.envmap_sampling_probability;
    }

    presampling_parameters.out_light_samples[x] = ReSTIR_DI_presample_one_light(presampling_parameters, envmap_candidate_probability, random_number_generator);
}

#endif
