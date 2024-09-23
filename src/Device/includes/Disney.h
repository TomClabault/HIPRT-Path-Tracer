/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_DISNEY_H
#define DEVICE_DISNEY_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ONB.h"
#include "Device/includes/OrenNayar.h"
#include "Device/includes/RayPayload.h"
#include "Device/includes/Sampling.h"
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
 * [9] [CS184 Adaptive sampling] https://cs184.eecs.berkeley.edu/sp24/docs/hw3-1-part-5
 * 
 * Important note: none of the lobes of this implementation includes the cosine term.
 * The cosine term NoL needs to be taken into account outside of the BSDF
 */

HIPRT_HOST_DEVICE HIPRT_INLINE float disney_schlick_weight(float f0, float abs_cos_angle)
{
    return 1.0f + (f0 - 1.0f) * pow(1.0f - abs_cos_angle, 5.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F disney_diffuse_eval(const SimplifiedRendererMaterial& material, const float3& view_direction, const float3& surface_normal, const float3& to_light_direction, float& pdf)
{
    float3 half_vector = hippt::normalize(to_light_direction + view_direction);

    float LoH = hippt::clamp(0.0f, 1.0f, hippt::abs(hippt::dot(to_light_direction, half_vector)));
    float NoL = hippt::clamp(0.0f, 1.0f, hippt::abs(hippt::dot(surface_normal, to_light_direction)));
    float NoV = hippt::clamp(0.0f, 1.0f, hippt::abs(hippt::dot(surface_normal, view_direction)));

    pdf = NoL / M_PI;

    ColorRGB32F diffuse_part;
    float diffuse_90 = 0.5f + 2.0f * material.roughness * LoH * LoH;
    // Lambertian base_color
    //diffuse_part = material.base_color / M_PI;
    // Disney base_color
    diffuse_part = material.base_color / M_PI * disney_schlick_weight(diffuse_90, NoL) * disney_schlick_weight(diffuse_90, NoV);
    // Oren nayar base_color
    //diffuse_part = oren_nayar_eval(material, view_direction, surface_normal, to_light_direction);

    ColorRGB32F fake_subsurface_part = ColorRGB32F(0.0f);
    if (material.subsurface > 0.0f)
    {
        float subsurface_90 = material.roughness * LoH * LoH;
        fake_subsurface_part = 1.25f * material.base_color / M_PI *
            (disney_schlick_weight(subsurface_90, NoL) * disney_schlick_weight(subsurface_90, NoV) * (1.0f / (NoL + NoV) - 0.5f) + 0.5f);
    }

    return (1.0f - material.subsurface) * diffuse_part + material.subsurface * fake_subsurface_part;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 disney_diffuse_sample(const float3& surface_normal, Xorshift32Generator& random_number_generator)
{
    return cosine_weighted_sample(surface_normal, random_number_generator);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F disney_metallic_fresnel(const SimplifiedRendererMaterial& material, const float3& local_half_vector, const float3& local_to_light_direction)
{
    // The summary of what is below is the following:
    //
    // If the material is 100% metallic, then the fresnel term color is going to be 
    // the base_color of the material i.e. typical conductor response.
    // 
    // If the material is 0% metallic, then the fresnel term color is going to be
    // material.specular_color modulated by the material.specular_tint coefficient (which blends 
    // between white and material.specular_color) and the material.specular coefficient which
    // dictates whether we have a specular at all
    ColorRGB32F Ks = ColorRGB32F(1.0f - material.specular_tint) + material.specular_tint * material.specular_color;
    float R0 = ((material.ior - 1.0f) * (material.ior - 1.0f)) / ((material.ior + 1.0f) * (material.ior + 1.0f));
    ColorRGB32F C0 = material.specular * R0 * (1.0f - material.metallic) * Ks + material.metallic * material.base_color;

    return C0 + (ColorRGB32F(1.0f) - C0) * pow(hippt::clamp(0.0f, 1.0f, 1.0f - hippt::dot(local_half_vector, local_to_light_direction)), 5.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F disney_metallic_eval(const SimplifiedRendererMaterial& material, const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_half_vector, ColorRGB32F F, float& pdf)
{
    // Maxing 1.0e-8f here to avoid zeros
    float NoV = hippt::max(1.0e-8f, hippt::abs(local_view_direction.z));
    float NoL = hippt::max(1.0e-8f, hippt::abs(local_to_light_direction.z));

    float D = GTR2_anisotropic(material, local_half_vector);
    float G1_V = G1(material.alpha_x, material.alpha_y, local_view_direction);
    float G1_L = G1(material.alpha_x, material.alpha_y, local_to_light_direction);
    float G = G1_V * G1_L;

    pdf = D * G1_V / (4.0f * NoV);
    return F * D * G / (4.0f * NoL * NoV);
}

/**
 * The sampled direction is returned in the local shading frame of the basis used for 'local_view_direction'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 disney_metallic_sample(const SimplifiedRendererMaterial& material, const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
	// The view direction can sometimes be below the shading normal hemisphere
	// because of normal mapping
    int below_normal = (local_view_direction.z < 0) ? -1 : 1;
	float3 microfacet_normal = GGX_sample(local_view_direction * below_normal, material.alpha_x, material.alpha_y, random_number_generator);
	float3 sampled_direction = reflect_ray(local_view_direction, microfacet_normal * below_normal);

    // Should already be normalized but float imprecisions...
    return hippt::normalize(sampled_direction);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F disney_clearcoat_eval(const SimplifiedRendererMaterial& material, const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_halfway_vector, float& pdf)
{
    if (local_view_direction.z * local_to_light_direction.z < 0)
        return ColorRGB32F(0.0f);

    float num = material.clearcoat_ior - 1.0f;
    float denom = material.clearcoat_ior + 1.0f;
    ColorRGB32F R0 = ColorRGB32F((num * num) / (denom * denom));

    float HoL = hippt::clamp(1.0e-8f, 1.0f, hippt::dot(local_halfway_vector, local_to_light_direction));
    float clearcoat_gloss = 1.0f - material.clearcoat_roughness;
    float alpha_g = (1.0f - clearcoat_gloss) * 0.1f + clearcoat_gloss * 0.001f;

    ColorRGB32F F_clearcoat = fresnel_schlick(R0, HoL);
    float D_clearcoat = GTR1(alpha_g, hippt::abs(local_halfway_vector.z));
    float G_clearcoat = disney_clearcoat_masking_shadowing(local_view_direction) * disney_clearcoat_masking_shadowing(local_to_light_direction);

    pdf = D_clearcoat * hippt::abs(local_halfway_vector.z) / (4.0f * HoL);
    return F_clearcoat * D_clearcoat * G_clearcoat / (4.0f * local_view_direction.z);
}

/**
 * The sampled direction is returned in the local shading frame of the basis used for 'local_view_direction'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 disney_clearcoat_sample(const SimplifiedRendererMaterial& material, const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
    float clearcoat_gloss = 1.0f - material.clearcoat_roughness;
    float alpha_g = (1.0f - clearcoat_gloss) * 0.1f + clearcoat_gloss * 0.001f;
    float alpha_g_2 = alpha_g * alpha_g;

    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();

    float cos_theta = sqrt((1.0f - pow(alpha_g_2, 1.0f - rand_1)) / (1.0f - alpha_g_2));
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    float phi = 2.0f * M_PI * rand_2;
    float cos_phi = cos(phi);
    float sin_phi = sqrt(1.0f - cos_phi * cos_phi);

    float3 microfacet_normal = hippt::normalize(make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta));
    float3 sampled_direction = reflect_ray(local_view_direction, microfacet_normal);

    // Should already be normalized but float precision issues...
    return hippt::normalize(sampled_direction);
}

// TODO have materials_buffer as a global variable to avoid having to pass it around like that?
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F disney_glass_eval(const RendererMaterial* materials_buffer, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& local_view_direction, const float3& local_to_light_direction, float& pdf)
{
    float NoV = local_view_direction.z;
    float NoL = local_to_light_direction.z;

    if (hippt::abs(NoL) < 1.0e-8f)
        // Check to avoid dividing by 0 later on
        return ColorRGB32F(0.0f);

    // We're in the case of reflection if the view direction and the bounced ray (light direction) are in the same hemisphere
    bool reflecting = NoL * NoV > 0;

    // Relative eta = eta_t / eta_i
    float eta_t = ray_volume_state.outgoing_mat_index == InteriorStackImpl<InteriorStackStrategy>::MAX_MATERIAL_INDEX ? 1.0 : materials_buffer[ray_volume_state.outgoing_mat_index].ior;
    float eta_i = ray_volume_state.incident_mat_index == InteriorStackImpl<InteriorStackStrategy>::MAX_MATERIAL_INDEX ? 1.0 : materials_buffer[ray_volume_state.incident_mat_index].ior;
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

    if (HoL * NoL < 0.0f || HoV * NoV < 0.0f)
        // Backfacing microfacets when the microfacet normal isn't in the same
        // hemisphere as the view dir or light dir
        return ColorRGB32F(0.0f);

    ColorRGB32F color;
    float F = fresnel_dielectric(hippt::dot(local_view_direction, local_half_vector), relative_eta);
    if (reflecting)
    {
        color = disney_metallic_eval(material, local_view_direction, local_to_light_direction, local_half_vector, ColorRGB32F(F), pdf);

        // Scaling the PDF by the probability of being here (reflection of the ray and not transmission)
        pdf *= F;
    }
    else
    {
        float dot_prod = HoL + HoV / relative_eta;
        float dot_prod2 = dot_prod * dot_prod;
        float denom = dot_prod2 * NoL * NoV;

        float D = GTR2_anisotropic(material, local_half_vector);
        float G1_V = G1(material.alpha_x, material.alpha_y, local_view_direction);
        float G = G1_V * G1(material.alpha_x, material.alpha_y, local_to_light_direction);

        float dwm_dwi = hippt::abs(HoL) / dot_prod2;
        float D_pdf = G1_V / hippt::abs(NoV) * D * hippt::abs(HoV);
        pdf = dwm_dwi * D_pdf * (1.0f - F);

        // We added a check a few lines above to "avoid dividing by 0 later on". This is where.
        // When NoL is 0, denom is 0 too and we're dividing by 0. 
        // The PDF of this case is as low as 1.0e-9 (light direction sampled perpendicularly to the normal)
        // so this is an extremely rare case.
        // The PDF being non-zero, we could actualy compute it, it's valid but absolutely not with floats :D
        color = sqrt(material.base_color) * D * (1 - F) * G * hippt::abs(HoL * HoV / denom);

        if (ray_volume_state.incident_mat_index != InteriorStackImpl<InteriorStackStrategy>::MAX_MATERIAL_INDEX)
        {
            // If we're not coming from the air, this means that we were in a volume and we're currently
            // refracting out of the volume or into another volume.
            // This is where we take the absorption of our travel into account using Beer-Lambert's law.
            // Note that we want to use the absorption of the material we finished traveling in.
            // The BSDF we're evaluating right now is using the new material we're refracting in, this is not
            // by this material that the ray has been absorbed. The ray has been absorded by the volume
            // it was in before refracting here, so it's the incident mat index

            const SimplifiedRendererMaterial& incident_material = materials_buffer[ray_volume_state.incident_mat_index];
            // Remapping the absorption coefficient so that it is more intuitive to manipulate
            // according to Burley, 2015 [5].
            // This effectively gives us a "at distance" absorption coefficient.
            ColorRGB32F absorption_coefficient = log(incident_material.absorption_color) / incident_material.absorption_at_distance;
            color = color * exp(absorption_coefficient * ray_volume_state.distance_in_volume);

            // We changed volume so we're resetting the distance
            ray_volume_state.distance_in_volume = 0.0f;
            if (ray_volume_state.leaving_mat)
                // We refracting out of a volume so we're poping the stack
                ray_volume_state.interior_stack.pop(ray_volume_state.leaving_mat);
        }
    }

    return color;
}

/**
 * The sampled direction is returned in the local shading frame of the basis used for 'local_view_direction'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 disney_glass_sample(const RendererMaterial* materials_buffer, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
    // Relative eta = eta_t / eta_i
    float eta_t = ray_volume_state.outgoing_mat_index == InteriorStackImpl<InteriorStackStrategy>::MAX_MATERIAL_INDEX ? 1.0 : materials_buffer[ray_volume_state.outgoing_mat_index].ior;
    float eta_i = ray_volume_state.incident_mat_index == InteriorStackImpl<InteriorStackStrategy>::MAX_MATERIAL_INDEX ? 1.0 : materials_buffer[ray_volume_state.incident_mat_index].ior;
    float relative_eta = eta_t / eta_i;
    // To avoid sampling directions that would lead to a null half_vector. 
    // Explained in more details in glass_eval.
    if (hippt::abs(relative_eta - 1.0f) < 1.0e-5f)
        relative_eta = 1.0f + 1.0e-5f;

    float3 microfacet_normal = GGX_sample(local_view_direction, material.alpha_x, material.alpha_y, random_number_generator);

    float F = fresnel_dielectric(hippt::dot(local_view_direction, microfacet_normal), relative_eta);
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
            // Relative_eta as already been flipped above in the code
            microfacet_normal = -microfacet_normal;

        refract_ray(local_view_direction, microfacet_normal, sampled_direction, relative_eta);
    }

    return sampled_direction;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F disney_sheen_eval(const SimplifiedRendererMaterial& material, const float3& local_to_light_direction, const float3& local_half_vector, float& pdf)
{
    ColorRGB32F sheen_color = ColorRGB32F(1.0f - material.sheen_tint) + material.sheen_color * material.sheen_tint;

    float NoL = hippt::abs(local_to_light_direction.z);
    // Clamping here because floating point errors can give us a dot > 1 sometimes
    // leading to 1.0f - dot being negative and the BRDF returns a negative color
    float HoL = hippt::clamp(0.0f, 1.0f, hippt::dot(local_half_vector, local_to_light_direction));

    pdf = NoL / M_PI;

    return sheen_color * pow(1.0f - HoL, 5.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F disney_bsdf_eval(const RendererMaterial* materials_buffer, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& view_direction, float3 shading_normal, const float3& to_light_direction, float& pdf)
{
    pdf = 0.0f;

    // We're only going to compute the diffuse, metallic, clearcoat and sheen lobes if we're 
    // outside of the object. Said otherwise, only the glass lobe is considered while traveling 
    // inside the object
    bool outside_object = hippt::dot(view_direction, shading_normal) > 0;
    if (!outside_object)
        // For the rest of the computations to be correct, we want the normal
        // in the same hemisphere as the view direction
        shading_normal = -shading_normal;

    float3 T, B;
    build_ONB(shading_normal, T, B);
    float3 local_view_direction = world_to_local_frame(T, B, shading_normal, view_direction);
    float3 local_to_light_direction = world_to_local_frame(T, B, shading_normal, to_light_direction);
    float3 local_half_vector = hippt::normalize(local_view_direction + local_to_light_direction);

    // Rotated ONB for the anisotropic GTR2 evaluation (metallic and glass only)
    float3 TR, BR;
    build_rotated_ONB(shading_normal, TR, BR, material.anisotropic_rotation * M_PI);
    float3 local_view_direction_rotated = world_to_local_frame(TR, BR, shading_normal, view_direction);
    float3 local_to_light_direction_rotated = world_to_local_frame(TR, BR, shading_normal, to_light_direction);
    float3 local_half_vector_rotated = hippt::normalize(local_view_direction_rotated + local_to_light_direction_rotated);

    float glass_weight = (1.0f - material.metallic) * material.specular_transmission;
    float diffuse_weight = (1.0f - material.metallic) * (1.0f - material.specular_transmission) * outside_object;
    float metal_weight = (1.0f - material.specular_transmission * (1.0f - material.metallic)) * outside_object;
    float clearcoat_weight = 0.25f * material.clearcoat * outside_object;
    float sheen_weight = (1.0f - material.metallic) * material.sheen * outside_object;

    float weight_sum = (diffuse_weight + metal_weight + clearcoat_weight + glass_weight + sheen_weight);

    float normalize_factor = 1.0f / weight_sum;
    float diffuse_proba = diffuse_weight * normalize_factor;
    float metal_proba = metal_weight * normalize_factor;
    float clearcoat_proba = clearcoat_weight * normalize_factor;
    float glass_proba = glass_weight * normalize_factor;
    float sheen_proba = sheen_weight * normalize_factor;

    ColorRGB32F final_color = ColorRGB32F(0.0f);
    float tmp_pdf = 0.0f;

    // Diffuse
    final_color += diffuse_weight > 0 && outside_object ? diffuse_weight * disney_diffuse_eval(material, view_direction, shading_normal, to_light_direction, tmp_pdf) : ColorRGB32F(0.0f);
    pdf += tmp_pdf * diffuse_proba;
    tmp_pdf = 0.0f;

    // Metallic
    // Computing a custom fresnel term based on the material specular, specular tint, ... coefficients
    ColorRGB32F metallic_fresnel = disney_metallic_fresnel(material, local_half_vector, local_to_light_direction);
    metal_weight = (1.0f - material.specular_transmission * (1.0f - material.metallic));
    final_color += metal_weight > 0 && outside_object ? metal_weight * disney_metallic_eval(material, local_view_direction_rotated, local_to_light_direction_rotated, local_half_vector_rotated, metallic_fresnel, tmp_pdf) : ColorRGB32F(0.0f);
    pdf += tmp_pdf * metal_proba;
    tmp_pdf = 0.0f;

    // Clearcoat
    final_color += clearcoat_weight > 0 && outside_object ? clearcoat_weight * disney_clearcoat_eval(material, local_view_direction_rotated, local_to_light_direction_rotated, local_half_vector_rotated, tmp_pdf) : ColorRGB32F(0.0f);
    pdf += tmp_pdf * clearcoat_proba;
    tmp_pdf = 0.0f;

    // Glass
    final_color += glass_weight > 0 ? glass_weight * disney_glass_eval(materials_buffer, material, ray_volume_state, local_view_direction_rotated, local_to_light_direction_rotated, tmp_pdf) : ColorRGB32F(0.0f);
    pdf += tmp_pdf * glass_proba;
    tmp_pdf = 0.0f;

    // Sheen
    final_color += sheen_weight > 0 && outside_object ? sheen_weight * disney_sheen_eval(material, local_to_light_direction, local_half_vector, tmp_pdf) : ColorRGB32F(0.0f);
    pdf += tmp_pdf * sheen_proba;

    return final_color;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F disney_bsdf_sample(const RendererMaterial* materials_buffer, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal, float3& output_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    pdf = 0.0f;

    float3 normal = shading_normal;

    float glass_weight = (1.0f - material.metallic) * material.specular_transmission;
    bool outside_object = hippt::dot(view_direction, normal) > 0;
    if (glass_weight == 0.0f && !outside_object)
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

    float diffuse_weight = (1.0f - material.metallic) * (1.0f - material.specular_transmission) * outside_object;
    float metal_weight = (1.0f - material.specular_transmission * (1.0f - material.metallic)) * outside_object;
    float clearcoat_weight = 0.25f * material.clearcoat * outside_object;

    // No sheen weight here because we're not importance sampling the sheen weight,
    // it's pretty weak of a lobe so there's no need to bother
    float normalize_factor = 1.0f / (diffuse_weight + metal_weight + clearcoat_weight + glass_weight);
    diffuse_weight *= normalize_factor;
    metal_weight *= normalize_factor;
    clearcoat_weight *= normalize_factor;
    glass_weight *= normalize_factor;

    float cdf[4];
    cdf[0] = diffuse_weight;
    cdf[1] = cdf[0] + metal_weight;
    cdf[2] = cdf[1] + clearcoat_weight;
    cdf[3] = cdf[2] + glass_weight;

    float rand_1 = random_number_generator();
    if (rand_1 > cdf[2])
    {
        // We're going to sample the glass lobe

        float dot_shading = hippt::dot(view_direction, shading_normal);
        float dot_geometric = hippt::dot(view_direction, geometric_normal);
        if (dot_shading * dot_geometric < 0)
        {
            // The view direction is below the surface normal because of normal mapping / smooth normals.
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

    if (rand_1 < cdf[2])
        // This means that we're going to sample the diffuse, metallic or clearcoat lobe which are all
        // reflective so we're poping the stack
        ray_volume_state.interior_stack.pop(false);

    if (hippt::dot(view_direction, normal) < 0)
        // We want the normal in the same hemisphere as the view direction
        // for the rest of the calculations
        normal = -normal;

    // Rotated ONB for the anisotropic GTR2 evaluation (metallic and glass only)
    float3 local_view_direction_rotated;
    float3 TR, BR;
    build_rotated_ONB(normal, TR, BR, material.anisotropic_rotation * M_PI);
    local_view_direction_rotated = world_to_local_frame(TR, BR, normal, view_direction);

    if (rand_1 < cdf[0])
        output_direction = disney_diffuse_sample(normal, random_number_generator);
    else if (rand_1 < cdf[1])
        output_direction = local_to_world_frame(TR, BR, normal, disney_metallic_sample(material, local_view_direction_rotated, random_number_generator));
    else if (rand_1 < cdf[2])
        output_direction = local_to_world_frame(TR, BR, normal, disney_clearcoat_sample(material, local_view_direction_rotated, random_number_generator));
    else
        // When sampling the glass lobe, if we're reflecting off the glass, we're going to have to pop the stack.
        // This is handled inside glass_sample because we cannot know from here if we refracted or reflected
        output_direction = local_to_world_frame(TR, BR, normal, disney_glass_sample(materials_buffer, material, ray_volume_state, local_view_direction_rotated, random_number_generator));

    if (hippt::dot(output_direction, shading_normal) < 0 && !(rand_1 > cdf[2]))
        // It can happen that the light direction sampled is below the surface. 
        // We return 0.0 in this case because the glass lobe wasn't sampled
        // so we can't have a bounce direction below the surface
        // 
        // We're also checking that we're not sampling the glass lobe because this
        // is a valid configuration for the glass lobe
        return ColorRGB32F(0.0f);

    // Not using 'normal' here because eval() needs to know whether or not we're inside the surface.
    // This is because is we're inside the surface, we're only going to evaluate the glass lobe.
    // If we were using 'normal', we would always be outside the surface because 'normal' is flipped
    // (a few lines above in the code) so that it is in the same hemisphere as the view direction and
    // eval() will then think that we're always outside the surface even though that's not the case
    return disney_bsdf_eval(materials_buffer, material, ray_volume_state, view_direction, shading_normal, output_direction, pdf);
}

#endif