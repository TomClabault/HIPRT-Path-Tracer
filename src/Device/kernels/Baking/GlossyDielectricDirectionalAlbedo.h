/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */


#include "Device/includes/BSDFs/Lambertian.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Sampling.h"

#include "HostDeviceCommon/RenderData.h"

#include "Renderer/Baker/GlossyDielectricDirectionalAlbedoSettings.h"

 /* References:
 * [1][Practical multiple scattering compensation for microfacet models, Turquin, 2019]
 * [2][Revisiting Physically Based Shading at Imageworks, Kulla & Conty, SIGGRAPH 2017]
 * [3][Dassault Enterprise PBR 2025 Specification]
 * [4][Google - Physically Based Rendering in Filament]
 * [5][MaterialX codebase on Github]
 * [6][Blender's Cycles codebase on Github]
 *
 * The kernel outputs its results in one buffer (which is then written to disk as a texture).
 * The texture is parameterized by cos_theta_o (cosine view direction), the roughness of
 * the specular GGX layer and its IOR
 */

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline GlossyDielectricDirectionalAlbedoBake(HIPRTRenderData render_data, GlossyDielectricDirectionalAlbedoSettings bake_settings, float* out_buffer)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline GlossyDielectricDirectionalAlbedoBake(HIPRTRenderData render_data, GlossyDielectricDirectionalAlbedoSettings bake_settings, float* out_buffer, int x, int y, int z)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
#endif

    const uint32_t pixel_index = (x + y * bake_settings.texture_size_cos_theta_o + z * bake_settings.texture_size_cos_theta_o * bake_settings.texture_size_roughness);

    if (x >= bake_settings.texture_size_cos_theta_o || y >= bake_settings.texture_size_roughness || z >= bake_settings.texture_size_ior)
        return;

    Xorshift32Generator random_number_generator(wang_hash(pixel_index + 1));

    out_buffer[pixel_index] = 0.0f;

    float cos_theta_o = 1.0f / (bake_settings.texture_size_cos_theta_o - 1) * x;
    cos_theta_o = hippt::max(GTR2_DOT_PRODUCTS_CLAMP, cos_theta_o);
    cos_theta_o = powf(cos_theta_o, 2.5f);
    float sin_theta_o = sin(acos(cos_theta_o));

    float roughness = 1.0f / (bake_settings.texture_size_roughness - 1) * y;
    roughness = hippt::max(roughness, 1.0e-4f);

    // Integrates for interface reflectivities of IORs between 1.0f and 3.0f
    float F0 = 1.0f / (bake_settings.texture_size_ior - 1) * z;
    // Relative eta (eta_t / eta_i) from F0
    // Using F0^4 to get more precision near 0
    F0 *= F0; // F0^2
    F0 *= F0; // F0^4
    float sqrt_F0 = sqrtf(hippt::clamp(0.0f, 0.99f, F0));
    float relative_ior = (1.0f + sqrt_F0) / (1.0f - sqrt_F0);

    float3 local_view_direction = hippt::normalize(make_float3(cos(0.0f) * sin_theta_o, sin(0.0f) * sin_theta_o, cos_theta_o));

    for (int sample = 0; sample < bake_settings.integration_sample_count; sample++)
    {
        // Sampling the specular GGX lobe or diffuse lobe
        float rand_lobe = random_number_generator();
        float3 sampled_local_to_light_direction;
        if (rand_lobe < 0.5f)
        {
            // Sampling the specular lobe
            sampled_local_to_light_direction = microfacet_GTR2_sample_reflection(roughness, /* anisotropy */ 0.0f, local_view_direction, random_number_generator);

            if (sampled_local_to_light_direction.z < 0)
                // Sampled direction below surface, this can happen with microfacet
                // sampling
                continue;
        }
        else
            // Sampling the diffuse lobe
            sampled_local_to_light_direction = cosine_weighted_sample_z_up_frame(random_number_generator);

        float3 microfacet_normal = hippt::normalize(local_view_direction + sampled_local_to_light_direction);
        float total_pdf = 0.0f;

        float F = full_fresnel_dielectric(hippt::dot(microfacet_normal, sampled_local_to_light_direction), relative_ior);
        float eval_pdf_specular;
        float directional_albedo_specular = torrance_sparrow_GTR2_eval<0>(HIPRTRenderData(), ColorRGB32F(F0), roughness, /* aniso */ 0.0f, ColorRGB32F(F),
                                                                          local_view_direction, sampled_local_to_light_direction, microfacet_normal, eval_pdf_specular).r;
        // Multiplying the PDF by 0.5f because we have a 50% chance to sample the specular lobe
        total_pdf += eval_pdf_specular * 0.5f;
        float specular_layer_throughput = 1.0f;
        specular_layer_throughput *= 1.0f - full_fresnel_dielectric(sampled_local_to_light_direction.z, relative_ior);
        specular_layer_throughput *= 1.0f - full_fresnel_dielectric(local_view_direction.z, relative_ior);

        // A material with the base color defined is the only thing needed for
        // lambertian_brdf_eval()
        SimplifiedRendererMaterial mat;
        mat.base_color = ColorRGB32F(1.0f);
        float eval_pdf_diffuse;
        float directional_albedo_diffuse = lambertian_brdf_eval(mat, sampled_local_to_light_direction.z, eval_pdf_diffuse).r;
        // Multiplying the PDF by 0.5f because we have a 50% chance to sample the diffuse lobe
        total_pdf += eval_pdf_diffuse * 0.5f;
        // Only the fraction of light that got through the specular layer
        // and that can get back to the viewer contributes to the illumination
        // we get from the diffuse layer
        directional_albedo_diffuse *= specular_layer_throughput;

        float final_albedo = directional_albedo_specular + directional_albedo_diffuse;
        final_albedo *= sampled_local_to_light_direction.z;
        final_albedo /= total_pdf;

        out_buffer[pixel_index] += final_albedo / bake_settings.integration_sample_count;
    }

#ifndef __KERNELCC__
    // Some sanity checks on the CPU
    float threshold = 1.1f;
    if (out_buffer[pixel_index] > threshold || out_buffer[pixel_index] < 0 || std::isinf(out_buffer[pixel_index]) || std::isnan(out_buffer[pixel_index]))
        std::cout << "Error at x, y, z = [" << x << ", " << y << ", " << z << "]. Value = " << out_buffer[pixel_index] << std::endl;
#endif
}
