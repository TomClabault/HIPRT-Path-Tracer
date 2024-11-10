/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_PRINCIPLED_H
#define DEVICE_PRINCIPLED_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ONB.h"
#include "Device/includes/BSDFs/OrenNayar.h"
#include "Device/includes/RayPayload.h"
#include "Device/includes/Sampling.h"
#include "Device/includes/BSDFs/SheenLTC.h"

#include "HostDeviceCommon/Material.h"
#include "HostDeviceCommon/Xorshift.h"

 /** References:
  *
  * [1] [CSE 272 University of California San Diego - Disney BSDF Homework] https://cseweb.ucsd.edu/~tzli/cse272/wi2024/homework1.pdf
  * [2] [GLSL Path Tracer implementation by knightcrawler25] https://github.com/knightcrawler25/GLSL-PathTracer
  * [3] [SIGGRAPH 2012 Course] https://blog.selfshadow.com/publications/s2012-shading-course/#course_content
  * [4] [SIGGRAPH 2015 Course] https://blog.selfshadow.com/publications/s2015-shading-course/#course_content
  * [5] [Burley 2015 Course Notes - Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering] https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf
  * [6] [PBRT v3 Source Code] https://github.com/mmp/pbrt-v3
  * [7] [PBRT v4 Source Code] https://github.com/mmp/pbrt-v4
  * [8] [Blender's Cycles Source Code] https://github.com/blender/cycles
  * [9] [Autodesk Standard Surface] https://autodesk.github.io/standard-surface/
  * [10] [Blender Principled BSDF] https://docs.blender.org/manual/fr/dev/render/shader_nodes/shader/principled.html
  * [11] [Open PBR Specification] https://academysoftwarefoundation.github.io/OpenPBR/#formalism/layering
  * [12] [Enterprise PBR Specification] https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec-2025x.md.html
  * 
  * Important note: none of the lobes of this implementation includes the cosine term.
  * The cosine term NoL needs to be taken into account outside of the BSDF
  */

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_coat_eval(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_halfway_vector, float incident_ior, float& out_pdf)
{
    ColorRGB32F F0 = ColorRGB32F(F0_from_eta(material.coat_ior, incident_ior));
    return torrance_sparrow_GTR2_eval(render_data, F0, material.coat_roughness, material.coat_anisotropy, material.coat_ior, incident_ior, local_view_direction, local_to_light_direction, local_halfway_vector, out_pdf);

    //// The coat lobe is just a microfacet lobe
    //return torrance_sparrow_GTR2_eval(render_data, material.base_color, material.coat_roughness, material.coat_anisotropy, material.coat_ior, incident_ior, local_view_direction, local_to_light_direction, local_halfway_vector, out_pdf);
}

/**
 * The sampled direction is returned in the local shading frame of the basis used for 'local_view_direction'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 principled_coat_sample(const SimplifiedRendererMaterial& material, const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
    return microfacet_GTR2_sample_reflection(material.coat_roughness, material.coat_anisotropy, local_view_direction, random_number_generator);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_sheen_eval(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, const float3& local_view_direction, const float3& local_to_light_direction, float& pdf, float& out_sheen_reflectance)
{
    return sheen_ltc_eval(render_data, material, local_to_light_direction, local_view_direction, pdf, out_sheen_reflectance);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 principled_sheen_sample(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, const float3& local_view_direction, const float3& shading_normal, Xorshift32Generator& random_number_generator)
{
    return sheen_ltc_sample(render_data, material, local_view_direction, shading_normal, random_number_generator);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_metallic_eval(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, float incident_ior, const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_half_vector, float& pdf)
{
    float HoL = hippt::clamp(1.0e-8f, 1.0f, hippt::dot(local_half_vector, local_to_light_direction));

    ColorRGB32F F = adobe_f82_tint_fresnel(material.base_color, material.metallic_F82, material.metallic_F90, material.metallic_F90_falloff_exponent, HoL);

    return torrance_sparrow_GTR2_eval<PrincipledBSDFGGXUseMultipleScattering>(render_data, material.base_color, material.roughness, material.anisotropy, F, local_view_direction, local_to_light_direction, local_half_vector, pdf);
}

/**
 * The sampled direction is returned in the local shading frame of the basis used for 'local_view_direction'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 principled_metallic_sample(const SimplifiedRendererMaterial& material, const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
    return microfacet_GTR2_sample_reflection(material.roughness, material.anisotropy, local_view_direction, random_number_generator);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_diffuse_eval(const SimplifiedRendererMaterial& material, const float3& local_view_direction, const float3& local_to_light_direction, float& pdf)
{
    // The diffuse lobe is a simple Oren Nayar lobe
#if PrincipledBSDFDiffuseLobe == PRINCIPLED_DIFFUSE_LOBE_LAMBERTIAN
    return lambertian_brdf_eval(material, local_to_light_direction.z, pdf);
#elif PrincipledBSDFDiffuseLobe == PRINCIPLED_DIFFUSE_LOBE_OREN_NAYAR
    return oren_nayar_brdf_eval(material, local_view_direction, local_to_light_direction, pdf);
#endif
}

/**
 * The sampled direction is returned in world space
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 principled_diffuse_sample(const float3& surface_normal, Xorshift32Generator& random_number_generator)
{
    // Our Oren-Nayar diffuse lobe is sampled by a cosine weighted distribution
    return cosine_weighted_sample_around_normal(surface_normal, random_number_generator);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_specular_eval(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, float incident_ior, const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_half_vector, float& pdf)
{
    // The specular lobe is just another GGX (GTR2) lobe
    return torrance_sparrow_GTR2_eval(render_data, material.base_color, material.roughness, material.anisotropy, material.ior, incident_ior, local_view_direction, local_to_light_direction, local_half_vector, pdf);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 principled_specular_sample(const SimplifiedRendererMaterial& material, const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
    return microfacet_GTR2_sample_reflection(material.roughness, material.anisotropy, local_view_direction, random_number_generator);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_glass_eval(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& local_view_direction, const float3& local_to_light_direction, float& pdf)
{
    pdf = 0.0f;

    float NoV = local_view_direction.z;
    float NoL = local_to_light_direction.z;

    if (hippt::abs(NoL) < 1.0e-8f)
        // Check to avoid dividing by 0 later on
        return ColorRGB32F(0.0f);

    // We're in the case of reflection if the view direction and the bounced ray (light direction) are in the same hemisphere
    bool reflecting = NoL * NoV > 0;

    // Relative eta = eta_t / eta_i
    float eta_t = ray_volume_state.outgoing_mat_index == InteriorStackImpl<InteriorStackStrategy>::MAX_MATERIAL_INDEX ? 1.0 : render_data.buffers.materials_buffer[ray_volume_state.outgoing_mat_index].ior;
    float eta_i = ray_volume_state.incident_mat_index == InteriorStackImpl<InteriorStackStrategy>::MAX_MATERIAL_INDEX ? 1.0 : render_data.buffers.materials_buffer[ray_volume_state.incident_mat_index].ior;
    float relative_eta = eta_t / eta_i;

    // relative_eta can be 1 when refracting from a volume into another volume of the same IOR.
    // This in conjunction with the view direction and the light direction being the negative of
    // one another will lead the microfacet normal to be the null vector which then causes
    // NaNs.
    // 
    // Example:
    // The view and light direction can be the negative of one another when looking straight at a
    // flat window for example. The view direction is aligned with the normal of the window
    // in this configuration whereas the refracting light direction (and it is very likely to refract
    // in this configuration) is going to point exactly away from the view direction and the normal.
    // 
    // We then have
    // 
    // half_vector  = light_dir + relative_eta * view_dir
    //              = light_dir + 1.0f * view_dir
    //              = light_dir + view_dir = (0, 0, 0)
    //
    // Normalizing this null vector then leads to a NaNs because of the zero-length.
    //
    // We're settings relative_eta to 1.00001f to avoid this issue
    if (hippt::abs(relative_eta - 1.0f) < 1.0e-5f)
        relative_eta = 1.0f + 1.0e-5f;

    // Computing the generalized (that takes refraction into account) half vector
    float3 local_half_vector;
    if (reflecting)
        local_half_vector = local_to_light_direction + local_view_direction;
    else
        // We need to take the relative_eta into account when refracting to compute
        // the half vector (this is the "generalized" part of the half vector computation)
        local_half_vector = local_to_light_direction * relative_eta + local_view_direction;

    local_half_vector = hippt::normalize(local_half_vector);
    if (local_half_vector.z < 0.0f)
        // Because the rest of the function we're going to compute here assume
        // that the microfacet normal is in the same hemisphere as the surface
        // normal, we're going to flip it if needed
        local_half_vector = -local_half_vector;

    float HoL = hippt::dot(local_to_light_direction, local_half_vector);
    float HoV = hippt::dot(local_view_direction, local_half_vector);

    float F = full_fresnel_dielectric(HoV, relative_eta);
    if (HoL * NoL < 0.0f || HoV * NoV < 0.0f)
        // Backfacing microfacets when the microfacet normal isn't in the same
        // hemisphere as the view dir or light dir
        return ColorRGB32F(0.0f);


#if PrincipledBSDFGGXUseMultipleScattering == KERNEL_OPTION_TRUE
    bool inside_object = ray_volume_state.inside_material;
    float relative_eta_for_correction = inside_object ? 1.0f / relative_eta : relative_eta;
    float exponent_correction = GGX_glass_energy_conservation_get_correction_exponent(material.roughness, relative_eta_for_correction);

    // We're storing cos_theta_o^2.5 in the LUT so we're retrieving it with pow(1.0f / 2.5f) i.e.
    // sqrt 2.5
    //
    // We're using a "correction exponent" to forcefully get rid of energy gains at grazing angles due
    // to float precision issues: storing in the LUT with cos_theta^2.5 but fetching with pow(1.0f / 2.6f)
    // for example darkens to overall appearance and helps remove energy gains
    float view_direction_tex_fetch = powf(hippt::max(0.0f, local_view_direction.z), 1.0f / exponent_correction);

    float F0 = F0_from_eta(eta_t, eta_i);
    // sqrt(sqrt()) of F0 here because we're storing F0^4 in the LUT
    float F0_remapped = sqrt(sqrt(F0));

    float3 uvw = make_float3(view_direction_tex_fetch, material.roughness, F0_remapped);

    void* texture = inside_object ? render_data.brdfs_data.GGX_Ess_glass_inverse : render_data.brdfs_data.GGX_Ess_glass;
    int3 dims = make_int3(GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_COS_THETA_O, GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_ROUGHNESS, GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_IOR);
    float compensation_term = sample_texture_3D_rgb_32bits(texture, dims, uvw, render_data.brdfs_data.use_hardware_tex_interpolation).r;
#endif

    ColorRGB32F color;
    if (reflecting)
    {
        color = torrance_sparrow_GTR2_eval<0>(render_data, material.base_color, material.roughness, material.anisotropy, ColorRGB32F(F), local_view_direction, local_to_light_direction, local_half_vector, pdf);

#if PrincipledBSDFGGXUseMultipleScattering == KERNEL_OPTION_TRUE
        // [Turquin, 2019] Eq. 18
        color /= compensation_term;
#endif

        // Scaling the PDF by the probability of being here (reflection of the ray and not transmission)
        pdf *= F;

        // Popping the ray volume stack is done in the glass_sample function for
        // reflections so we're not doing it here
    }
    else
    {
        float dot_prod = HoL + HoV / relative_eta;
        float dot_prod2 = dot_prod * dot_prod;
        float denom = dot_prod2 * NoL * NoV;

        float alpha_x;
        float alpha_y;
        SimplifiedRendererMaterial::get_alphas(material.roughness, material.anisotropy, alpha_x, alpha_y);

        float D = GTR2_anisotropic(alpha_x, alpha_y, local_half_vector);
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
        color = sqrt(material.base_color) * D * (1 - F) * G2 * hippt::abs(HoL * HoV / denom);

#if PrincipledBSDFGGXUseMultipleScattering == KERNEL_OPTION_TRUE
        // [Turquin, 2019] Eq. 18
        color /= compensation_term;
#endif

        if (ray_volume_state.incident_mat_index != InteriorStackImpl<InteriorStackStrategy>::MAX_MATERIAL_INDEX)
        {
            // If we're not coming from the air, this means that we were in a volume and we're currently
            // refracting out of the volume or into another volume.
            // This is where we take the absorption of our travel into account using Beer-Lambert's law.
            // Note that we want to use the absorption of the material we finished traveling in.
            // The BSDF we're evaluating right now is using the new material we're refracting in, this is not
            // by this material that the ray has been absorbed. The ray has been absorded by the volume
            // it was in before refracting here, so it's the incident mat index

            const SimplifiedRendererMaterial& incident_material = render_data.buffers.materials_buffer[ray_volume_state.incident_mat_index];
            // Remapping the absorption coefficient so that it is more intuitive to manipulate
            // according to Burley, 2015 [5].
            // This effectively gives us a "at distance" absorption coefficient.
            ColorRGB32F absorption_coefficient = log(incident_material.absorption_color) / incident_material.absorption_at_distance;
            color = color * exp(absorption_coefficient * ray_volume_state.distance_in_volume);

            // We changed volume so we're resetting the distance
            ray_volume_state.distance_in_volume = 0.0f;
            if (ray_volume_state.inside_material)
                // We refracting out of a volume so we're poping the stack
                ray_volume_state.interior_stack.pop(ray_volume_state.inside_material);
        }
    }

    return color;
}

/**
 * The sampled direction is returned in the local shading frame of the basis used for 'local_view_direction'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 principled_glass_sample(const RendererMaterial* materials_buffer, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
    float eta_t = ray_volume_state.outgoing_mat_index == InteriorStackImpl<InteriorStackStrategy>::MAX_MATERIAL_INDEX ? 1.0f : materials_buffer[ray_volume_state.outgoing_mat_index].ior;
    float eta_i = ray_volume_state.incident_mat_index == InteriorStackImpl<InteriorStackStrategy>::MAX_MATERIAL_INDEX ? 1.0f : materials_buffer[ray_volume_state.incident_mat_index].ior;
    float relative_eta = eta_t / eta_i;
    // To avoid sampling directions that would lead to a null half_vector. 
    // Explained in more details in principled_glass_eval.
    if (hippt::abs(relative_eta - 1.0f) < 1.0e-5f)
        relative_eta = 1.0f + 1.0e-5f;

    float alpha_x;
    float alpha_y;
    SimplifiedRendererMaterial::get_alphas(material.roughness, material.anisotropy, alpha_x, alpha_y);

    float3 microfacet_normal = GTR2_anisotropic_sample_microfacet(local_view_direction, alpha_x, alpha_y, random_number_generator);

    float F = full_fresnel_dielectric(hippt::dot(local_view_direction, microfacet_normal), relative_eta);
    float rand_1 = random_number_generator();

    float3 sampled_direction;
    if (rand_1 < F)
    {
        // Reflection
        sampled_direction = reflect_ray(local_view_direction, microfacet_normal);

        // This is a reflection, we're poping the stack
        ray_volume_state.interior_stack.pop(false);
    }
    else
    {
        // Refraction

        if (hippt::dot(microfacet_normal, local_view_direction) < 0.0f)
            // For the refraction operation that follows, we want the direction to refract (the view
            // direction here) to be in the same hemisphere as the normal (the microfacet normal here)
            // so we're flipping the microfacet normal in case it wasn't in the same hemisphere as
            // the view direction
            microfacet_normal = -microfacet_normal;

        // refract_ray() takes relative_eta = eta_i / eta_t so that's why we're inverting eta
        // here
        refract_ray(local_view_direction, microfacet_normal, sampled_direction, relative_eta);
    }

    return sampled_direction;
}

/**
 * 'internal' functions are just so that 'principled_bsdf_eval' looks nicer
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_coat_layer(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, const float3& local_view_direction, const float3 local_to_light_direction, const float3& local_half_vector, const float3& shading_normal, float incident_ior, float coat_weight, float coat_proba, ColorRGB32F& layers_throughput, float& out_cumulative_pdf)
{
    if (coat_weight > 0.0f)
    {
        float coat_pdf;
        ColorRGB32F contribution = principled_coat_eval(render_data, material, local_view_direction, local_to_light_direction, local_half_vector, incident_ior, coat_pdf);
        contribution *= coat_weight;
        contribution *= layers_throughput;

        out_cumulative_pdf += coat_pdf * coat_proba;

        //ColorRGB32F transmitted_light = ColorRGB32F(1.0f);
        //// Energy loss
        ///*transmitted_light *= 1.0f - full_fresnel_dielectric(hippt::dot(local_half_vector, local_to_light_direction), incident_ior, material.coat_ior);
        //transmitted_light *= 1.0f - full_fresnel_dielectric(hippt::dot(local_half_vector, local_view_direction), material.coat_ior, incident_ior);*/

        //transmitted_light *= 1.0f - hippt::max(full_fresnel_dielectric(hippt::dot(local_half_vector, local_to_light_direction), incident_ior, material.coat_ior), full_fresnel_dielectric(hippt::dot(local_half_vector, local_view_direction), material.coat_ior, incident_ior));

        //// Taking the color of the absorbing coat medium into account when the light that got transmitted
        //// travels through it
        //transmitted_light *= material.coat_medium_absorption;
        //// When light hits the coat layer, only the refracted part gets transmitted to the layer below.
        //// Because the coat layer is a dielectric layer, that refracted part is given by the Fresnel laws
        //// 
        //// Also, the coat layer color absorbs the light so we're taking that color into account as well
        ////
        //// Why we're using the shading normal here and not the microfacet normal is explained
        //// in the internal_eval_specular_layer()

        //// Blending the coat layer between completely transparent (no effect on the transmitted light
        //// throughput when material.coat is 0.0f) and full absorption contribution.
        //transmitted_light = (1.0f - coat_weight) * ColorRGB32F(1.0f) + coat_weight * transmitted_light;

        
        float HoV = hippt::clamp(1.0e-8f, 1.0f, hippt::dot(local_half_vector, local_view_direction));
        float HoL = hippt::clamp(1.0e-8f, 1.0f, hippt::dot(local_half_vector, local_to_light_direction));

        float transmitted_light = 1.0f;
        transmitted_light *= 1.0f - hippt::max(full_fresnel_dielectric(HoL, incident_ior, material.coat_ior), full_fresnel_dielectric(HoV, incident_ior, material.coat_ior));

        layers_throughput *= transmitted_light;

        return contribution;
    }

    return ColorRGB32F(0.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_sheen_layer(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, const float3& local_view_direction, const float3& local_to_light_direction, const float3& world_space_to_light_direction, const float3& shading_normal, float incident_ior, float sheen_weight, float sheen_proba, ColorRGB32F& layers_throughput, float& out_cumulative_pdf)
{
    if (sheen_weight > 0.0f)
    {
        float sheen_reflectance;
        float sheen_pdf;
        ColorRGB32F contribution = principled_sheen_eval(render_data, material, local_view_direction, local_to_light_direction, sheen_pdf, sheen_reflectance);
        contribution *= sheen_weight;
        contribution *= layers_throughput;

        out_cumulative_pdf += sheen_pdf * sheen_proba;

        // Same as the coat layer for the sheen: only the refracted light goes into the layer below
        // 
        // The proportion of light that is reflected is given by the Ri component of AiBiRi
        // (see 'sheen_ltc_eval') which is returned by 'principled_sheen_eval' in 'sheen_reflectance'
        layers_throughput *= 1.0f - material.sheen * sheen_reflectance;

        return contribution;
    }
    
    return ColorRGB32F(0.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_metal_layer(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, const float3& local_view_direction, const float3 local_to_light_direction, const float3& local_half_vector, float incident_ior, float metal_weight, float metal_proba, ColorRGB32F& layers_throughput, float& out_cumulative_pdf)
{
    if (metal_weight > 0.0f)
    {
        float metal_pdf;
        ColorRGB32F contribution = principled_metallic_eval(render_data, material, incident_ior, local_view_direction, local_to_light_direction, local_half_vector, metal_pdf);
        contribution *= metal_weight;
        contribution *= layers_throughput;

        out_cumulative_pdf += metal_pdf * metal_proba;

        // There is nothing below the metal layer so we don't have a layer_throughput absorption here
        // ...

        return contribution;
    }

    return ColorRGB32F(0.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_specular_layer(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, const float3& local_view_direction, const float3 local_to_light_direction, const float3& local_half_vector, const float3& shading_normal, float incident_ior, float specular_weight, float specular_proba, ColorRGB32F& layers_throughput, float& out_cumulative_pdf)
{
    if (specular_weight > 0.0f)
    {
        float specular_pdf;
        ColorRGB32F contribution = principled_specular_eval(render_data, material, incident_ior, local_view_direction, local_to_light_direction, local_half_vector, specular_pdf);
        contribution *= (1.0f - material.specular_tint) * ColorRGB32F(1.0f) + material.specular_tint * material.specular_color;
        contribution *= specular_weight;
        contribution *= layers_throughput;

        // Only the transmitted portion of the light goes to the layer below
        // We're using the shading normal here and not the microfacet normal because:
        // We want the proportion of light that reaches the layer below. That's given
        // by 1.0f - fresnelReflection.
        // 
        // However, that's quite incorrect.
        // 
        // By computing 1.0f - fresnelReflectionMicrofacetNormal, we're computing the light
        // that goes through only that one microfacet with the microfacet normal. But light
        // reaches the layer below through all microfacets, not just the one with our current
        // normal here (local_half_vector). To compute this correctly, we would actually need
        // to integrate over the microfacet normals and compute the fresnel refraction
        // (1.0f - fresnelReflection) for each of them and weight that contribution by the
        // probability given by the normal distribution function for the microfacet normal.
        // 
        // We can't do that integration online so we're instead using the "average normal" of
        // the microfacets to compute the fresnel.
        // And the "average normal" is actually the surface normal, shading_normal, because that's
        // how the NDF is defined
        layers_throughput *= 1.0f - material.specular * hippt::max(full_fresnel_dielectric(local_to_light_direction.z, incident_ior, material.ior), full_fresnel_dielectric(local_view_direction.z, incident_ior, material.ior));

        out_cumulative_pdf += specular_pdf * specular_proba;

        return contribution;
    }

    return ColorRGB32F(0.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_diffuse_layer(const SimplifiedRendererMaterial& material, const float3& local_view_direction, const float3 local_to_light_direction, float diffuse_weight, float diffuse_proba, ColorRGB32F& layers_throughput, float& out_cumulative_pdf)
{
    if (diffuse_weight > 0.0f)
    {
        float diffuse_pdf;
        ColorRGB32F contribution = principled_diffuse_eval(material, local_view_direction, local_to_light_direction, diffuse_pdf);
        contribution *= diffuse_weight;
        contribution *= layers_throughput;

        out_cumulative_pdf += diffuse_pdf * diffuse_proba;

        return contribution;
    }

    return ColorRGB32F(0.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_glass_layer(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& local_view_direction, const float3 local_to_light_direction, float glass_weight, float glass_proba, ColorRGB32F& layers_throughput, float& out_cumulative_pdf)
{
    if (glass_weight > 0.0f)
    {
        float glass_pdf;
        ColorRGB32F contribution = principled_glass_eval(render_data, material, ray_volume_state, local_view_direction, local_to_light_direction, glass_pdf);
        contribution *= glass_weight;
        contribution *= layers_throughput;

        // There is nothing below the glass layer so we don't have a layer_throughput absorption here
        // ...

        out_cumulative_pdf += glass_pdf * glass_proba;

        return contribution;
    }

    return ColorRGB32F(0.0f);
}

#define DISABLE_DIFFUSE_LOBE 0

// TODO total internal reflection fresnel on the way out of a layer
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_bsdf_eval(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& view_direction, float3 shading_normal, const float3& to_light_direction, float& pdf)
{
    pdf = 0.0f;

    // Only the glass lobe is considered when evaluating
    // the BSDF from inside the object so we're going to use that
    // 'outside_object' flag to nullify the other lobes if we're
    // inside the object
    bool outside_object = hippt::dot(view_direction, shading_normal) > 0;
    bool refracting = hippt::dot(shading_normal, to_light_direction) < 0.0f || hippt::dot(shading_normal, view_direction) < 0.0f;
    if (!outside_object)
        // For the rest of the computations to be correct, we want the normal
        // in the same hemisphere as the view direction
        shading_normal = -shading_normal;

    float3 T, B;
    build_ONB(shading_normal, T, B);
    float3 local_view_direction = world_to_local_frame(T, B, shading_normal, view_direction);
    float3 local_to_light_direction = world_to_local_frame(T, B, shading_normal, to_light_direction);
    float3 local_half_vector = hippt::normalize(local_view_direction + local_to_light_direction);

    // Rotated ONB for the anisotropic GTR2 evaluation (metallic/glass lobes for example)
    float3 TR, BR;
    build_rotated_ONB(shading_normal, TR, BR, material.anisotropy_rotation * M_PI);
    float3 local_view_direction_rotated = world_to_local_frame(TR, BR, shading_normal, view_direction);
    float3 local_to_light_direction_rotated = world_to_local_frame(TR, BR, shading_normal, to_light_direction);
    float3 local_half_vector_rotated = hippt::normalize(local_view_direction_rotated + local_to_light_direction_rotated);

    // Linear blending weights for the lobes
    // 
    // Everytime we multiply by "outside_object" is because we want to disable
    // the lobe if we're inside the object
    //
    // The layering follows the one of the principled BSDF of blender:
    // [10] https://docs.blender.org/manual/fr/dev/render/shader_nodes/shader/principled.html
    float coat_weight = material.coat * outside_object;
    float sheen_weight = material.sheen * outside_object;
    float metal_weight = material.metallic * outside_object;
    float specular_weight = (1.0f - material.metallic) * (1.0f - material.specular_transmission) * material.specular * outside_object;
    float diffuse_weight = (1.0f - material.metallic) * (1.0f - material.specular_transmission) * outside_object;
    // If inside the object, the glass lobe is the only existing lobe so it has weight
    // 1.0f
    float glass_weight = !outside_object ? 1.0f :  (1.0f - material.metallic)* material.specular_transmission;
#if DISABLE_DIFFUSE_LOBE == 1
    diffuse_weight = 0.0f;
#endif

    float coat_sample_proba = coat_weight;
    float sheen_sample_proba = sheen_weight;
    float metal_sample_proba = metal_weight;
    float specular_sample_proba = specular_weight;
    float diffuse_sample_proba = diffuse_weight;
    float glass_sample_proba = glass_weight;
    float probability_normalize_factor = coat_sample_proba + sheen_sample_proba + metal_sample_proba + specular_sample_proba + diffuse_sample_proba + glass_sample_proba;

    // For the given to_light_direction, normal, view_direction etc..., what's the probability
    // that the 'principled_bsdf_sample()' function would have sampled the lobe?
    float coat_proba_norm = coat_sample_proba / probability_normalize_factor;
    float sheen_proba_norm = sheen_sample_proba / probability_normalize_factor;
    float metal_proba_norm = metal_sample_proba / probability_normalize_factor;
    float specular_proba_norm = specular_sample_proba / probability_normalize_factor;
    float diffuse_proba_norm = diffuse_sample_proba / probability_normalize_factor;
    float glass_proba_norm = glass_sample_proba / probability_normalize_factor;

    // Keeps track of the remaining light's energy as we traverse layers
    ColorRGB32F layers_throughput = ColorRGB32F(1.0f);
    ColorRGB32F final_color = ColorRGB32F(0.0f);

    // In the 'internal_eval_coat_layer' function calls below, we're passing
    // 'weight * !refracting' so that lobes that do not allow refractions
    // (which is pretty much all of them except glass) do no get evaluated
    // (because their weight becomes 0)
    float incident_ior = ray_volume_state.incident_mat_index == /* air */ InteriorStackImpl<InteriorStackStrategy>::MAX_MATERIAL_INDEX ? 1.0f : render_data.buffers.materials_buffer[ray_volume_state.incident_mat_index].ior;
    final_color += internal_eval_coat_layer(render_data, material, local_view_direction, local_to_light_direction, local_half_vector, shading_normal, incident_ior, coat_weight * !refracting, coat_proba_norm, layers_throughput, pdf);
    final_color += internal_eval_sheen_layer(render_data, material, local_view_direction, local_to_light_direction, to_light_direction, shading_normal, incident_ior, sheen_weight, sheen_proba_norm, layers_throughput, pdf);
    final_color += internal_eval_metal_layer(render_data, material, local_view_direction_rotated, local_to_light_direction_rotated, local_half_vector_rotated, incident_ior, metal_weight * !refracting, metal_proba_norm, layers_throughput, pdf);
    // Careful here to evaluate the glass layer before the specular
    // layer otherwise, layers_throughput is going to be modified
    // by the specular layer evaluation to take the fresnel of the
    // specular layer into account. But we don't want that for the
    // glass layer. The glass layer isn't below the specular layer 
    // so we don't want to take that into account
    final_color += internal_eval_glass_layer(render_data, material, ray_volume_state, local_view_direction_rotated, local_to_light_direction_rotated, glass_weight, glass_proba_norm, layers_throughput, pdf);
    final_color += internal_eval_specular_layer(render_data, material, local_view_direction_rotated, local_to_light_direction_rotated, local_half_vector_rotated, shading_normal, incident_ior, specular_weight * !refracting, specular_proba_norm, layers_throughput, pdf);
    final_color += internal_eval_diffuse_layer(material, local_view_direction, local_to_light_direction, diffuse_weight * !refracting, diffuse_proba_norm, layers_throughput, pdf);

    return final_color;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_bsdf_sample(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal, float3& output_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    pdf = 0.0f;

    float3 normal = shading_normal;

    float glass_sampling_weight = (1.0f - material.metallic) * material.specular_transmission;
    bool outside_object = hippt::dot(view_direction, normal) > 0;
    if (hippt::is_zero(glass_sampling_weight) && !outside_object)
    {
        // If we're not sampling the glass lobe so we're checking
        // whether the view direction is below the upper hemisphere around the shading
        // normal or not. This may be the case mainly due to normal mapping / smooth vertex normals. 
        // 
        // See Microfacet-based Normal Mapping for Robust Monte Carlo Path Tracing, Eric Heitz, 2017
        // for some illustrations of the problem and a solution (not implemented here because
        // it requires quite a bit of code and overhead). 
        // 
        // We're flipping the normal instead which is a quick dirty fix solution mentioned
        // in the above mentioned paper.
        // 
        // The Position-free Multiple-bounce Computations for Smith Microfacet BSDFs by 
        // Wang et al. 2022 proposes an alternative position-free solution that even solves
        // the multi-scattering issue of microfacet BRDFs on top of the dark fringes issue we're
        // having here

        normal = reflect_ray(shading_normal, geometric_normal);
        outside_object = true;
    }

    // Computing the weights for sampling the lobes
    float coat_sampling_weight = material.coat * outside_object;
    float sheen_sampling_weight = material.sheen * outside_object;
    float metal_sampling_weight = material.metallic * outside_object;
    float specular_sampling_weight = (1.0f - material.metallic) * (1.0f - material.specular_transmission) * material.specular * outside_object;
    // The diffuse lobe is below the specular lobe so it 
    // has the same probability of being sampled
    float diffuse_sampling_weight = (1.0f - material.metallic) * (1.0f - material.specular_transmission) * outside_object;
#if DISABLE_DIFFUSE_LOBE == 1
    diffuse_sampling_weight = 0.0f;
#endif

    float normalize_factor = 1.0f / (coat_sampling_weight + sheen_sampling_weight + metal_sampling_weight + specular_sampling_weight + diffuse_sampling_weight + glass_sampling_weight);
    coat_sampling_weight *= normalize_factor;
    sheen_sampling_weight *= normalize_factor;
    metal_sampling_weight *= normalize_factor;
    specular_sampling_weight *= normalize_factor;
    diffuse_sampling_weight *= normalize_factor;
    glass_sampling_weight *= normalize_factor;

    float cdf[5];
    cdf[0] = coat_sampling_weight;
    cdf[1] = cdf[0] + sheen_sampling_weight;
    cdf[2] = cdf[1] + metal_sampling_weight;
    cdf[3] = cdf[2] + specular_sampling_weight;
    cdf[4] = cdf[3] + diffuse_sampling_weight;
    // The last cdf[] is implicitely 1.0f so don't need to include it

    float rand_1 = random_number_generator();
    bool sampling_glass_lobe = rand_1 > cdf[4];
    if (sampling_glass_lobe)
    {
        // We're going to sample the glass lobe

        float dot_shading = hippt::dot(view_direction, shading_normal);
        float dot_geometric = hippt::dot(view_direction, geometric_normal);
        if (dot_shading * dot_geometric < 0)
        {
            // The view direction is below the surface normal (probably because of normal mapping / smooth normals).
            // 
            // We're going to flip the normal for the same reason as explained above to avoid black fringes
            // the reason we're also checking for the dot product with the geometric normal here
            // is because in the case of the glass lobe of the BRDF, we could be legitimately having
            // the dot product between the shading normal and the view direction be negative when we're
            // currently travelling inside the surface. To make sure that we're in the case of the black fringes
            // caused by normal mapping and microfacet BRDFs, we're also checking with the geometric normal.
            // 
            // If the view direction isn't below the geometric normal but is below the shading normal, this
            // indicates that we're in the case of the black fringes and we can flip the normal
            // 
            // If both dot products are negative, this means that we're travelling inside the surface
            // and we shouldn't flip the normal
            normal = reflect_ray(shading_normal, geometric_normal);
        }
    }

    if (!sampling_glass_lobe)
        // We're going to sample a reflective lobe so we're poping the stack
        ray_volume_state.interior_stack.pop(false);

    if (hippt::dot(view_direction, normal) < 0)
        // We want the normal in the same hemisphere as the view direction
        // for the rest of the calculations
        normal = -normal;
        
    // Rotated ONB for the anisotropic GTR2 evaluation
    float3 TR, BR;
    build_rotated_ONB(normal, TR, BR, material.anisotropy_rotation * M_PI);
    float3 local_view_direction_rotated = world_to_local_frame(TR, BR, normal, view_direction);

    if (rand_1 < cdf[0])
    {
        float3 TR_coat, BR_coat;
        build_rotated_ONB(normal, TR_coat, BR_coat, material.coat_anisotropy_rotation * M_PI);
        float3 local_view_direction_rotated_coat = world_to_local_frame(TR_coat, BR_coat, normal, view_direction);

        output_direction = local_to_world_frame(TR_coat, BR_coat, normal, principled_coat_sample(material, local_view_direction_rotated_coat, random_number_generator));
    }
    else if (rand_1 < cdf[1])
    {
        float3 T, B;
        build_ONB(normal, T, B);
        float3 local_view_direction = world_to_local_frame(T, B, normal, view_direction);

        output_direction = local_to_world_frame(T, B, normal, principled_sheen_sample(render_data, material, local_view_direction, normal, random_number_generator));
    }
    else if (rand_1 < cdf[2])
        output_direction = local_to_world_frame(TR, BR, normal, principled_metallic_sample(material, local_view_direction_rotated, random_number_generator));
    else if (rand_1 < cdf[3])
        output_direction = local_to_world_frame(TR, BR, normal, principled_specular_sample(material, local_view_direction_rotated, random_number_generator));
    else if (rand_1 < cdf[4])
        // No call to local_to_world_frame() since the sample diffuse functions
        // already returns in world space around the given normal
        output_direction = principled_diffuse_sample(normal, random_number_generator);
    else
        // When sampling the glass lobe, if we're reflecting off the glass, we're going to have to pop the stack.
        // This is handled inside glass_sample because we cannot know from here if we refracted or reflected
        output_direction = local_to_world_frame(TR, BR, normal, principled_glass_sample(render_data.buffers.materials_buffer, material, ray_volume_state, local_view_direction_rotated, random_number_generator));

    if (hippt::dot(output_direction, shading_normal) < 0 && !sampling_glass_lobe)
        // It can happen that the light direction sampled is below the surface. 
        // We return 0.0 in this case if we didn't sample the glass lobe
        // because no lobe other than the glass lobe allows refractions
        return ColorRGB32F(0.0f);

    // Not using 'normal' here because eval() needs to know whether or not we're inside the surface.
    // This is because is we're inside the surface, we're only going to evaluate the glass lobe.
    // If we were using 'normal', we would always be outside the surface because 'normal' is flipped
    // (a few lines above in the code) so that it is in the same hemisphere as the view direction and
    // eval() will then think that we're always outside the surface even though that's not the case
    return principled_bsdf_eval(render_data, material, ray_volume_state, view_direction, shading_normal, output_direction, pdf);
}

#endif
