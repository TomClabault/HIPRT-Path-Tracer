/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/Baker/GGXConductorDirectionalAlbedoSettings.h"

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/BSDFs/Microfacet.h"

#include "HostDeviceCommon/RenderData.h"

 /* References:
 * [1][Practical multiple scattering compensation for microfacet models, Turquin, 2019]
 * [2][Revisiting Physically Based Shading at Imageworks, Kulla & Conty, SIGGRAPH 2017]
 * [3][Dassault Enterprise PBR 2025 Specification]
 * [4][Google - Physically Based Rendering in Filament]
 * [5][MaterialX codebase on Github]
 * [6][Blender's Cycles codebase on Github]
 *
 * This kernel computes the directional albedo of a conductor BRDF for use
 * in energy compensation code (MicrofacetEnergyCompensation.h) as proposed in
 * [Practical multiple scattering compensation for microfacet models, Turquin, 2019]
 *
 * The kernel outputs its results in one buffer (which is then written to disk as a texture).
 * The texture is parameterized by cos_theta_o (cosine view direction) and the roughness of
 * the BRDF
 */

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline GGXConductorDirectionalAlbedoBake(int kernel_iterations, int current_iteration, GGXConductorDirectionalAlbedoSettings bake_settings, float* out_buffer)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline GGXConductorDirectionalAlbedoBake(int kernel_iterations, int current_iteration, GGXConductorDirectionalAlbedoSettings bake_settings, float* out_buffer, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif

    const uint32_t pixel_index = (x + y * bake_settings.texture_size_cos_theta);

    if (x >= bake_settings.texture_size_cos_theta || y >= bake_settings.texture_size_roughness)
        return;

    Xorshift32Generator random_number_generator(wang_hash(pixel_index + 1) * current_iteration);

    float roughness = 1.0f / (bake_settings.texture_size_roughness - 1) * y;
    roughness = hippt::max(roughness, 1.0e-4f);

    float cos_theta_o = 1.0f / (bake_settings.texture_size_cos_theta - 1) * x;
    cos_theta_o = hippt::max(GGX_DOT_PRODUCTS_CLAMP, cos_theta_o);
    float sin_theta_o = sin(acos(cos_theta_o));

    float3 local_view_direction = hippt::normalize(make_float3(cos(0.0f) * sin_theta_o, sin(0.0f) * sin_theta_o, cos_theta_o));

    int iterations_per_kernel = floor(hippt::max(1.0f, (float)GPUBakerConstants::COMPUTE_ELEMENT_PER_BAKE_KERNEL_LAUNCH / (bake_settings.texture_size_cos_theta * bake_settings.texture_size_roughness)));
    int nb_kernel_launch = ceil(bake_settings.integration_sample_count / (float)iterations_per_kernel);
    int nb_samples = nb_kernel_launch * iterations_per_kernel;

    for (int sample = 0; sample < kernel_iterations; sample++)
    {
        float3 sampled_local_to_light_direction = microfacet_GGX_sample_reflection(roughness, 0.0f, local_view_direction, random_number_generator);
        if (sampled_local_to_light_direction.z < 0)
            // Sampled direction below surface
            continue;

        float eval_pdf;
        float directional_albedo = torrance_sparrow_GGX_eval<0>(HIPRTRenderData(), roughness, 0.0f, /* fresnel */ ColorRGB32F(1.0f), 
                                                                 local_view_direction, sampled_local_to_light_direction, hippt::normalize(local_view_direction + sampled_local_to_light_direction), eval_pdf).r;
        directional_albedo /= eval_pdf;
        directional_albedo *= sampled_local_to_light_direction.z;

        out_buffer[pixel_index] += directional_albedo / nb_samples;
    }
}
