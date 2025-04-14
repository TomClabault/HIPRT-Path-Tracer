/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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

HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRDIPresampledLight presample_emissive_triangle(const HIPRTRenderData& render_data, float light_sampling_probability, Xorshift32Generator& random_number_generator)
{
    ReSTIRDIPresampledLight presampled_light;

    // We're passing 0.0f as the position here because we do not have a position when presampling lights in screen space
    // The position is only used for "spatial sampling" schemes such as ReGIR or light trees for example and such schemes are not
    // compatible with ReSTIR light presampling anyways
    LightSampleInformation light_sample = sample_one_emissive_triangle(render_data, make_float3(0.0f, 0.0f, 0.f), random_number_generator);

    if (light_sample.area_measure_pdf > 0.0f)
    {
        presampled_light.point_on_light_source = light_sample.point_on_light;
        presampled_light.light_source_normal = light_sample.light_source_normal;
        presampled_light.emissive_triangle_index = light_sample.emissive_triangle_index;

        // PDF in area measure
        presampled_light.pdf = light_sample.area_measure_pdf;
        presampled_light.pdf *= light_sampling_probability;
        presampled_light.radiance = light_sample.emission;
    }

    return presampled_light;
}

// TODO try just passing LightPresamplingParameters in there instead of everything individually
HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRDIPresampledLight ReSTIR_DI_presample_one_light(const HIPRTRenderData& render_data, const LightPresamplingParameters& parameters, float envmap_sampling_probability, Xorshift32Generator& random_number_generator)
{
    ReSTIRDIPresampledLight presampled_light;
    if (random_number_generator() < envmap_sampling_probability)
        presampled_light = presample_envmap(render_data.world_settings, envmap_sampling_probability, random_number_generator);
    else
        presampled_light = presample_emissive_triangle(render_data, 1.0f - envmap_sampling_probability, random_number_generator);

    return presampled_light;
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_DI_LightsPresampling(LightPresamplingParameters presampling_parameters, HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_DI_LightsPresampling(LightPresamplingParameters presampling_parameters, HIPRTRenderData render_data, int x)
#endif
{
    if (render_data.buffers.emissive_triangles_count == 0 && render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
        // No initial candidates to sample since no lights
        return;

#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    if (x >= presampling_parameters.subset_size * presampling_parameters.number_of_subsets)
        return;

    uint32_t thread_index = x;

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(thread_index + 1);
    else
        seed = wang_hash((thread_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);

    Xorshift32Generator random_number_generator(seed);

    float envmap_candidate_probability = 0.0f;
    if (render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP)
    {
        if (render_data.buffers.emissive_triangles_count == 0)
            // Only the envmap to sample
            envmap_candidate_probability = 1.0f;
        else
            envmap_candidate_probability = presampling_parameters.envmap_sampling_probability;
    }

    presampling_parameters.out_light_samples[x] = ReSTIR_DI_presample_one_light(render_data, presampling_parameters, envmap_candidate_probability, random_number_generator);
}

#endif
