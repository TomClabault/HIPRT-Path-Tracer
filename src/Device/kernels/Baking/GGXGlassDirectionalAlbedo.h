/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */


#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/BSDFs/Principled.h"

#include "HostDeviceCommon/RenderData.h"

#include "Renderer/Baker/GGXGlassDirectionalAlbedoSettings.h"

/* References:
* [1][Practical multiple scattering compensation for microfacet models, Turquin, 2019]
* [2][Revisiting Physically Based Shading at Imageworks, Kulla & Conty, SIGGRAPH 2017]
* [3][Dassault Enterprise PBR 2025 Specification]
* [4][Google - Physically Based Rendering in Filament]
* [5][MaterialX codebase on Github]
* [6][Blender's Cycles codebase on Github]
* 
* This kernel computes the directional albedo of a glass BSDF for use
* in energy compensation code (MicrofacetEnergyCompensation.h) as proposed in
* [Practical multiple scattering compensation for microfacet models, Turquin, 2019]
* 
* The kernel outputs its results in two buffers (which are then written to disk as textures).
* The two textures are parameterized by cos_theta_o (cosine view direction), the roughness of
* the BSDF and the reflectance at normal incidence F0 which relates to the relative IOR at
* the interface of the BSDF
* 
* The first texture is the directional albedo precomputation when hitting the object
* from the outside
* The second texture is used when inside the object: its IOR is simply inversed
*/

HIPRT_HOST_DEVICE HIPRT_INLINE float GGX_glass_E_eval(float relative_ior, float roughness, const float3& local_view_direction, const float3& local_to_light_direction, float& pdf)
{
    pdf = 0.0f;

    float NoV = local_view_direction.z;
    float NoL = local_to_light_direction.z;

    if (hippt::abs(NoL) < 1.0e-8f)
        // Check to avoid dividing by 0 later on
        return 0.0f;

    // We're in the case of reflection if the view direction and the bounced ray (light direction) are in the same hemisphere
    bool reflecting = NoL * NoV > 0;

    if (hippt::abs(relative_ior - 1.0f) < 1.0e-5f)
        relative_ior = 1.0f + 1.0e-5f;

    // Computing the generalized (that takes refraction into account) half vector
    float3 local_half_vector;
    if (reflecting)
        local_half_vector = local_to_light_direction + local_view_direction;
    else
        // We need to take the relative_ior into account when refracting to compute
        // the half vector (this is the "generalized" part of the half vector computation)
        local_half_vector = local_to_light_direction * relative_ior + local_view_direction;

    local_half_vector = hippt::normalize(local_half_vector);
    if (local_half_vector.z < 0.0f)
        // Because the rest of the function we're going to compute here assume
        // that the microfacet normal is in the same hemisphere as the surface
        // normal, we're going to flip it if needed
        local_half_vector = -local_half_vector;

    float HoL = hippt::dot(local_to_light_direction, local_half_vector);
    float HoV = hippt::dot(local_view_direction, local_half_vector);

    if (HoL * NoL < 0.0f || HoV * NoV < 0.0f)
        // Backfacing microfacets when the microfacet normal isn't in the same
        // hemisphere as the view dir or light dir
        return 0.0f;

    float albedo;
    float F = full_fresnel_dielectric(hippt::dot(local_view_direction, local_half_vector), relative_ior);
    if (reflecting)
    {
        HIPRTRenderData render_data;
        albedo = torrance_sparrow_GGX_eval_reflect<0>(render_data, roughness, 0.0f, ColorRGB32F(F),
                                               local_view_direction, local_to_light_direction, local_half_vector, pdf).r;

        // Scaling the PDF by the probability of being here (reflection of the ray and not transmission)
        pdf *= F;
    }
    else
    {
        float dot_prod = HoL + HoV / relative_ior;
        float dot_prod2 = dot_prod * dot_prod;
        float denom = dot_prod2 * NoL * NoV;

        float alpha_x;
        float alpha_y;
        MaterialUtils::get_alphas(roughness, 0.0f, alpha_x, alpha_y);

        float D = GGX_anisotropic(alpha_x, alpha_y, local_half_vector);
        float G1_V = G1_Smith(alpha_x, alpha_y, local_view_direction);
        float G1_L = G1_Smith(alpha_x, alpha_y, local_to_light_direction);
        float G2 = G1_V * G1_L;

        float dwm_dwi = hippt::abs(HoL) / dot_prod2;
        float D_pdf = G1_V / hippt::abs(NoV) * D * hippt::abs(HoV);
        pdf = dwm_dwi * D_pdf;
        // Taking refraction probability into account
        pdf *= (1.0f - F);

        // We added a check a few lines above to "avoid dividing by 0 later on". This is where.
        // When NoL is 0, denom is 0 too and we're dividing by 0. 
        // The PDF of this case is as low as 1.0e-9 (light direction sampled perpendicularly to the normal)
        // so this is an extremely rare case.
        // The PDF being non-zero, we could actualy compute it, it's valid but not with floats :D
        albedo = D * (1 - F) * G2 * hippt::abs(HoL * HoV / denom);
    }

    return albedo;
}

/**
 * The sampled direction is returned in the local shading frame of the basis used for 'local_view_direction'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 GGX_glass_E_sample(float relative_ior, float roughness, const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
    if (hippt::abs(relative_ior - 1.0f) < 1.0e-5f)
        relative_ior = 1.0f + 1.0e-5f;

    float alpha_x;
    float alpha_y;
    MaterialUtils::get_alphas(roughness, /* ignoring anisotropy */ 0.0f, alpha_x, alpha_y);
    float3 microfacet_normal = GGX_anisotropic_sample_microfacet(local_view_direction, alpha_x, alpha_y, random_number_generator);

    float F = full_fresnel_dielectric(hippt::dot(local_view_direction, microfacet_normal), relative_ior);
    float rand_1 = random_number_generator();

    float3 sampled_direction;
    if (rand_1 < F)
        // Reflection
        sampled_direction = reflect_ray(local_view_direction, microfacet_normal);
    else
    {
        // Refraction

        if (hippt::dot(microfacet_normal, local_view_direction) < 0.0f)
            // For the refraction operation that follows, we want the direction to refract (the view
            // direction here) to be in the same hemisphere as the normal (the microfacet normal here)
            // so we're flipping the microfacet normal in case it wasn't in the same hemisphere as
            // the view direction
            // Relative_eta as already been flipped above in the code
            microfacet_normal = -microfacet_normal;

        refract_ray(local_view_direction, microfacet_normal, sampled_direction, relative_ior);
    }

    return sampled_direction;
}

HIPRT_HOST_DEVICE HIPRT_INLINE void glass_directional_albedo_integration(int kernel_iterations, int current_iteration, uint32_t x, uint32_t y, uint32_t z, uint32_t pixel_index, GGXGlassDirectionalAlbedoSettings bake_settings, float* out_buffer, bool exiting_surface)
{
    Xorshift32Generator random_number_generator(wang_hash(pixel_index + 1) * current_iteration);

    float cos_theta_o = 1.0f / (bake_settings.texture_size_cos_theta_o - 1.0f) * x;
    cos_theta_o = hippt::max(GGX_DOT_PRODUCTS_CLAMP, cos_theta_o);
    cos_theta_o = powf(cos_theta_o, 2.5f);
    float sin_theta_o = sin(acos(cos_theta_o));

    float roughness = 1.0f / (bake_settings.texture_size_roughness - 1.0f) * y;
    roughness = hippt::max(roughness, 1.0e-4f);

    // Integrates for interface reflectivities of IORs between 1.0f and 3.0f
    float F0 = 1.0f / (bake_settings.texture_size_ior - 1.0f) * z;
    // Relative eta (eta_t / eta_i) from F0
    // Using F0^4 to get more precision near 0
    F0 *= F0; // F0^2
    F0 *= F0; // F0^4
    float sqrt_F0 = sqrtf(hippt::clamp(0.0f, 0.99f, F0));
    float relative_ior = (1.0f + sqrt_F0) / (1.0f - sqrt_F0);

    float3 local_view_direction = hippt::normalize(make_float3(cos(0.0f) * sin_theta_o, sin(0.0f) * sin_theta_o, cos_theta_o));

    if (exiting_surface)
        // Inverting the relative IOR in case we're inside the surface
        relative_ior = 1.0f / relative_ior;

    int iterations_per_kernel = floor(hippt::max(1.0f, GPUBakerConstants::COMPUTE_ELEMENT_PER_BAKE_KERNEL_LAUNCH / static_cast<float>(bake_settings.texture_size_cos_theta_o * bake_settings.texture_size_roughness * bake_settings.texture_size_ior)));
    int nb_kernel_launch = ceil(bake_settings.integration_sample_count / static_cast<float>(iterations_per_kernel));
    int nb_samples = nb_kernel_launch * iterations_per_kernel;

    for (int sample = 0; sample < kernel_iterations; sample++)
    {
        float3 sampled_local_to_light_direction = GGX_glass_E_sample(relative_ior, roughness, local_view_direction, random_number_generator);

        float eval_pdf;
        float directional_albedo = GGX_glass_E_eval(relative_ior, roughness, local_view_direction, sampled_local_to_light_direction, eval_pdf);
        if (eval_pdf == 0.0f)
            continue;

        directional_albedo /= eval_pdf;
        directional_albedo *= hippt::abs(sampled_local_to_light_direction.z);

        out_buffer[pixel_index] += directional_albedo / nb_samples;
    }
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline GGXGlassDirectionalAlbedoBakeEntering(int kernel_iterations, int current_iteration, GGXGlassDirectionalAlbedoSettings bake_settings, float* out_buffer)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline GGXGlassDirectionalAlbedoBakeEntering(int kernel_iterations, int current_iteration, GGXGlassDirectionalAlbedoSettings bake_settings, float* out_buffer, int x, int y, int z)
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

    glass_directional_albedo_integration(kernel_iterations, current_iteration, x, y, z, pixel_index, bake_settings, out_buffer, false);
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline GGXGlassDirectionalAlbedoBakeExiting(int kernel_iterations, int current_iteration, GGXGlassDirectionalAlbedoSettings bake_settings, float* out_buffer)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline GGXGlassDirectionalAlbedoBakeExiting(int kernel_iterations, int current_iteration, GGXGlassDirectionalAlbedoSettings bake_settings, float* out_buffer, int x, int y, int z)
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

    glass_directional_albedo_integration(kernel_iterations, current_iteration, x, y, z, pixel_index, bake_settings, out_buffer, true);
}
