/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_PRINCIPLED_H
#define DEVICE_PRINCIPLED_H

#include "Device/includes/Dispersion.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ONB.h"
#include "Device/includes/BSDFs/Lambertian.h"
#include "Device/includes/BSDFs/Microfacet.h"
#include "Device/includes/BSDFs/OrenNayar.h"
#include "Device/includes/BSDFs/PrincipledEnergyCompensation.h"
#include "Device/includes/BSDFs/ThinFilm.h"
#include "Device/includes/RayPayload.h"
#include "Device/includes/Sampling.h"
#include "Device/includes/BSDFs/SheenLTC.h"

#include "HostDeviceCommon/Material/MaterialUnpacked.h"
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
  * [11] [Open PBR Specification] https://academysoftwarefoundation.github.io/OpenPBR/
  * [12] [Enterprise PBR Specification] https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec-2025x.md.html
  * [13] [Arbitrarily Layered Micro-Facet Surfaces, Weidlich, Wilkie] https://www.cg.tuwien.ac.at/research/publications/2007/weidlich_2007_almfs/weidlich_2007_almfs-paper.pdf
  * [14] [A Practical Extension to Microfacet Theory for the Modeling of Varying Iridescence, Belcour, Barla, 2017] https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
  * [15] [MaterialX Implementation Code] https://github.com/AcademySoftwareFoundation/MaterialX
  * [16] [Khronos GLTF 2.0 KHR_materials_iridescence Implementation Notes] https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_iridescence/README.md
  * [17] [Khronos GLTF 2.0 KHR_materials_diffuse_transmission Implementation Notes] https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_diffuse_transmission/README.md
  * 
  * Important note: none of the lobes of this implementation includes the cosine term.
  * The cosine term NoL needs to be taken into account outside of the BSDF
  */

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_coat_eval(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, 
                                                                const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_halfway_vector, 
                                                                float incident_medium_ior, 
                                                                float& out_pdf, BSDFIncidentLightInfo incident_light_info,
                                                                int current_bounce)
{
    // The coat lobe is just a microfacet lobe
    float HoL = hippt::clamp(1.0e-8f, 1.0f, hippt::dot(local_halfway_vector, local_to_light_direction));

    // We're only evaluating the coat lobe if, either:
    // - The incident light direction was sampled from the clearcoat lobe
    // - The coat is not a delta distribution (not perfectly smooth)
    // - The incident light direction was sampled from another perfectly specular lobe
    //
    // Because if none of these two conditions are true, the evaluation of the coat will
    // yield 0.0f anyways
    //
    // All the conditions are handled in 'is_specular_delta_reflection_sampled'
    MaterialUtils::SpecularDeltaReflectionSampled coat_delta_direction_sampled = MaterialUtils::is_specular_delta_reflection_sampled(material, material.coat_roughness, material.coat_anisotropy, incident_light_info);

    ColorRGB32F F = ColorRGB32F(full_fresnel_dielectric(HoL, incident_medium_ior, material.coat_ior));

    return torrance_sparrow_GGX_eval_reflect<0>(render_data, material.coat_roughness, material.coat_anisotropy, false, F, 
        local_view_direction, local_to_light_direction, local_halfway_vector, 
        out_pdf, coat_delta_direction_sampled, current_bounce);
}

/**
 * The sampled direction is returned in the local shading frame of the basis used for 'local_view_direction'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 principled_coat_sample(const DeviceUnpackedEffectiveMaterial& material, 
    const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
    return microfacet_GGX_sample_reflection(material.coat_roughness, material.coat_anisotropy, local_view_direction, random_number_generator);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_sheen_eval(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, 
    const float3& local_view_direction, const float3& local_to_light_direction, float& pdf, float& out_sheen_reflectance)
{
    return sheen_ltc_eval(render_data, material, local_to_light_direction, local_view_direction, pdf, out_sheen_reflectance);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 principled_sheen_sample(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, 
    const float3& local_view_direction, const float3& shading_normal, Xorshift32Generator& random_number_generator)
{
    return sheen_ltc_sample(render_data, material, local_view_direction, shading_normal, random_number_generator);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_metallic_eval(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, 
                                                                    float roughness, float anisotropy, float incident_ior, 
                                                                    const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_half_vector, 
                                                                    float& pdf, BSDFIncidentLightInfo incident_light_info,
                                                                    int current_bounce)
{
    MaterialUtils::SpecularDeltaReflectionSampled metal_delta_direction_sampled = MaterialUtils::is_specular_delta_reflection_sampled(material, roughness, anisotropy, incident_light_info);
    if (metal_delta_direction_sampled == MaterialUtils::SpecularDeltaReflectionSampled::SPECULAR_PEAK_NOT_SAMPLED)
    {
        // The distribution isn't worth evaluating because it's specular but we the incident
        // light direction wasn't sampled from a specular distribution

        pdf = 0.0f;
        return ColorRGB32F(0.0f);
    }

    float HoL = hippt::clamp(1.0e-8f, 1.0f, hippt::dot(local_half_vector, local_to_light_direction));

    ColorRGB32F F_metal = adobe_f82_tint_fresnel(material.base_color, material.metallic_F82, material.metallic_F90, material.metallic_F90_falloff_exponent, HoL);
    ColorRGB32F F_thin_film = thin_film_fresnel(material, incident_ior, HoL);
    ColorRGB32F F = hippt::lerp(F_metal, F_thin_film, material.thin_film);

    return torrance_sparrow_GGX_eval_reflect<PrincipledBSDFDoEnergyCompensation && PrincipledBSDFDoMetallicEnergyCompensation>(render_data, 
        roughness, anisotropy, material.do_metallic_energy_compensation, F, 
        local_view_direction, local_to_light_direction, local_half_vector, 
        pdf, metal_delta_direction_sampled,
        current_bounce);
}

/**
 * The sampled direction is returned in the local shading frame of the basis used for 'local_view_direction'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 principled_metallic_sample(float roughness, float anisotropy, 
    const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
    return microfacet_GGX_sample_reflection(roughness, anisotropy, local_view_direction, random_number_generator);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_diffuse_eval(const DeviceUnpackedEffectiveMaterial& material, 
    const float3& local_view_direction, const float3& local_to_light_direction, float& pdf)
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
    return cosine_weighted_sample_around_normal_world_space(surface_normal, random_number_generator);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_specular_fresnel(const DeviceUnpackedEffectiveMaterial& material, float relative_specular_ior, float cos_theta_i)
{
    // We want the IOR of the layer we're coming from for the thin-film fresnel
    // 
    // 'relative_specular_IOR' is "A / B"
    // with A the IOR of the specular layer
    // and B the IOR of the layer (or medium) above the specular layer
    //
    // so the IOR of the layer above is 1.0f / (relative_IOR / specular_ior) = specular_IOR / relative_IOR
    float layer_above_IOR = material.ior / relative_specular_ior;

    // Computing the fresnel term
    // It's either the thin film fresnel for thin film interference or the usual
    // non colored dielectric/dielectric fresnel.
    //
    // We're lerping between the two based on material.thin_film
    float material_thin_film = material.thin_film;
    ColorRGB32F F_specular;
    if (material_thin_film < 1.0f)
        F_specular = ColorRGB32F(full_fresnel_dielectric(cos_theta_i, relative_specular_ior));

    ColorRGB32F F_thin_film;
    if (material_thin_film > 0.0f) 
        F_thin_film = thin_film_fresnel(material, layer_above_IOR, cos_theta_i);

    ColorRGB32F F = hippt::lerp(F_specular, F_thin_film, material_thin_film);

    return F;
}

/**
 * Returns the relative IOR as "A /B"
 * with A the IOR of the specular layer
 * and B the IOR of the layer (or medium) above the specular layer
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float principled_specular_relative_ior(const DeviceUnpackedEffectiveMaterial& material, float incident_medium_ior)
{
    if (material.coat == 0.0f)
        return material.ior;

    // When computing the specular layer, the incident IOR actually isn't always
    // that of the incident medium because we may have the coat layer above us instead of the medium
    // so the "proper" IOR to use here is actually the lerp between the medium and the coat
    // IOR depending on the coat factor
    float incident_layer_ior = hippt::lerp(incident_medium_ior, material.coat_ior, material.coat);
    float relative_ior = material.ior / incident_layer_ior;
    if (relative_ior < 1.0f)
        // If the coat IOR (which we're coming from) is greater than the IOR
        // of the base layer (which is the specular layer with IOR material.ior)
        // then we may hit total internal reflection when entering the specular layer from
        // the coat layer above. This manifests as a weird ring near grazing angles.
        //
        // This weird ring should not happen in reality. It only happens because we're
        // not bending the rays when refracting into the coat layer: we compute the
        // fresnel at the specular/coat interface as if the light direction just went
        // straight through the coat layer without refraction. There will always be
        // some refraction at the air/coat interface if the coat layer IOR is > 1.0f.
        //
        // The proper solution would be to actually bend the ray after it hits the coat layer.
        // We would then be evaluating the fresnel at the coat/specular interface with a
        // incident light cosine angle that is different and we wouldn't get total internal reflection.
        //
        // This is explained in the [OpenPBR Spec 2024]
        // https://academysoftwarefoundation.github.io/OpenPBR/#model/coat/totalinternalreflection
        // 
        // A more computationally efficient solution is to simply invert the IOR as done here.
        // This is also explained in the OpenPBR spec as well as in 
        // [Novel aspects of the Adobe Standard Material, Kutz, Hasan, Edmondson, 2023]
        // https://helpx.adobe.com/content/dam/substance-3d/general-knowledge/asm/Adobe%20Standard%20Material%20-%20Technical%20Documentation%20-%20May2023.pdf
        relative_ior = 1.0f / relative_ior;

    return relative_ior;
}

/**
 * 'relative_ior' is eta_t / eta_i with 'eta_t' the IOR of the glossy layer and
 * 'eta_i' the IOR of 
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_specular_eval(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, float relative_ior, 
                                                                    const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_half_vector, 
                                                                    float& pdf, BSDFIncidentLightInfo incident_light_info,
                                                                    int current_bounce)
{
    ColorRGB32F F = principled_specular_fresnel(material, relative_ior, hippt::dot(local_to_light_direction, local_half_vector));

    MaterialUtils::SpecularDeltaReflectionSampled is_specular_delta_reflection_sampled = MaterialUtils::is_specular_delta_reflection_sampled(material, material.roughness, material.anisotropy, incident_light_info);

    // The specular lobe is just another GGX lobe
    // 
    // We actually don't want energy compensation here for the specular layer
    // (hence the torrance_sparrow_GGX_eval_reflect<0>) because energy compensation
    // for the specular layer is handled for the glossy based (specular + diffuse lobe)
    // as a whole, not just in the specular layer 
    ColorRGB32F specular = torrance_sparrow_GGX_eval_reflect<0>(render_data, material.roughness, material.anisotropy, /* do_energy_compensation */ false, F, 
                                                                local_view_direction, local_to_light_direction, local_half_vector, 
                                                                pdf, is_specular_delta_reflection_sampled,
                                                                current_bounce);

    return specular;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 principled_specular_sample(float roughness, float anisotropy, const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
    return microfacet_GGX_sample_reflection(roughness, anisotropy, local_view_direction, random_number_generator);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_beer_absorption(const HIPRTRenderData& render_data, RayVolumeState& ray_volume_state)
{
    // Note that we want to use the absorption of the material we finished traveling in.
    // The BSDF we're evaluating right now is using the new material we're refracting in, this is not
    // by this material that the ray has been absorbed. The ray has been absorded by the volume
    // it was in before refracting here, so it's the incident mat index
    ColorRGB32F absorption_color;
    if (render_data.bsdfs_data.white_furnace_mode)
        absorption_color = ColorRGB32F(1.0f);
    else
        absorption_color = render_data.buffers.materials_buffer.get_absorption_color(ray_volume_state.incident_mat_index);
    if (!absorption_color.is_white())
    {
        // Capping the distance to avoid numerical issues at 0 distance
        // (can happen depending on the geometry of the scene if a ray exits a volume very quickly after entering it)
        ray_volume_state.distance_in_volume = hippt::max(ray_volume_state.distance_in_volume, 1.0e-6f);

        // Remapping the absorption coefficient so that it is more intuitive to manipulate
        // according to Burley, 2015 [5].
        // This effectively gives us a "at distance" absorption coefficient.
        ColorRGB32F absorption_coefficient = log(absorption_color) / render_data.buffers.materials_buffer.get_absorption_at_distance(ray_volume_state.incident_mat_index);
        return exp(absorption_coefficient * ray_volume_state.distance_in_volume);
    }

    return ColorRGB32F(1.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_glass_eval(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, 
                                                                 RayVolumeState& ray_volume_state, bool update_ray_volume_state,
                                                                 const float3& local_view_direction, const float3& local_to_light_direction, float& pdf,
                                                                 BSDFIncidentLightInfo incident_light_info, 
                                                                 int current_bounce)
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
    float eta_i = ray_volume_state.incident_mat_index == NestedDielectricsInteriorStack::MAX_MATERIAL_INDEX ? 1.0f : render_data.buffers.materials_buffer.get_ior(ray_volume_state.incident_mat_index);
    float eta_t = ray_volume_state.outgoing_mat_index == NestedDielectricsInteriorStack::MAX_MATERIAL_INDEX ? 1.0f : render_data.buffers.materials_buffer.get_ior(ray_volume_state.outgoing_mat_index);

    float dispersion_abbe_number = material.dispersion_abbe_number;
    float dispersion_scale = material.dispersion_scale;
    eta_i = compute_dispersion_ior(dispersion_abbe_number, dispersion_scale, eta_i, hippt::abs(ray_volume_state.sampled_wavelength));
    eta_t = compute_dispersion_ior(dispersion_abbe_number, dispersion_scale, eta_t, hippt::abs(ray_volume_state.sampled_wavelength));

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

    bool thin_walled = material.thin_walled;
    float scaled_roughness = MaterialUtils::get_thin_walled_roughness(thin_walled, material.roughness, relative_eta);

    float3 local_half_vector;
    if (scaled_roughness <= MaterialUtils::ROUGHNESS_CLAMP)
        // Fast path for specular glass
        local_half_vector = make_float3(0.0f, 0.0f, 1.0f);
    else
    {
        // Computing the generalized (that takes refraction into account) half vector
        if (reflecting)
            local_half_vector = local_to_light_direction + local_view_direction;
        else
        {
            if (thin_walled)
                // Thin walled material refract without light bending (because both refractions interfaces are simulated in one layer of material)
                // just refract straight through i.e. light_direction = -view_direction
                // It can be as si
                local_half_vector = local_to_light_direction * make_float3(1.0f, 1.0f, -1.0f) + local_view_direction;
            else
                // We need to take the relative_eta into account when refracting to compute
                // the half vector (this is the "generalized" half vector)
                local_half_vector = local_to_light_direction * relative_eta + local_view_direction;
        }

        local_half_vector = hippt::normalize(local_half_vector);
    }

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

    // Note: for specular glass, the compensation term will never be evaluated as there is no energy loss.
    // The function will return very quickly and will return 1.0f
    float compensation_term = get_GGX_energy_compensation_dielectrics(render_data, material, ray_volume_state.inside_material, eta_t, eta_i, relative_eta, local_view_direction.z, current_bounce);

    float thin_film = material.thin_film;
    ColorRGB32F F_thin_film;
    if (thin_film > 0.0f)
        F_thin_film = thin_film_fresnel(material, eta_i, HoV);

    ColorRGB32F F_no_thin_film;
    if (thin_film < 1.0f)
        F_no_thin_film = ColorRGB32F(full_fresnel_dielectric(HoV, relative_eta));

    ColorRGB32F F = hippt::lerp(F_no_thin_film, F_thin_film, thin_film);
    float f_reflect_proba = F.luminance();
    if (thin_walled && f_reflect_proba < 1.0f && thin_film == 0.0f && scaled_roughness < 0.1f)
        // If this is not total reflection, adjusting the fresnel term to account for inter-reflections within the thin interface
        // Not doing this if thin-film is present because that would not be accurate at all. Thin-film
        // effect require phase shift computations and that's expensive so we're just not doing it here
        // instead
        //
        // Reference: Dielectric BSDF, PBR Book 4ed: https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
        //
        // If there is no thin-film, the fresnel reflectance is non-colored and is the same
        // value for all RGB wavelengths. This means that f_reflect_proba is actually just the fresnel reflection factor
        //
        // This fresnel scaling only works at roughness 0 but still using below 0.1f for a close enough approximation
        f_reflect_proba += hippt::square(1.0f - f_reflect_proba) * f_reflect_proba / (1.0f - hippt::square(f_reflect_proba));

    ColorRGB32F color;
    if (reflecting)
    {
        MaterialUtils::SpecularDeltaReflectionSampled delta_glass_direction_sampled = MaterialUtils::is_specular_delta_reflection_sampled(material, material.roughness, material.anisotropy, incident_light_info);

        color = torrance_sparrow_GGX_eval_reflect<0>(render_data, scaled_roughness, material.anisotropy, false, F, 
                                                        local_view_direction, local_to_light_direction, local_half_vector,
                                                        pdf, delta_glass_direction_sampled, current_bounce);
        // [Turquin, 2019] Eq. 18 for dielectric microfacet energy compensation
        color /= compensation_term;

        // Scaling the PDF by the probability of being here (reflection of the ray and not transmission)
        pdf *= f_reflect_proba;
    }
    else
    {
        color = torrance_sparrow_GGX_eval_refract(material, scaled_roughness, relative_eta, F, 
            local_view_direction, local_to_light_direction, local_half_vector, 
            pdf, incident_light_info);
        // Taking refraction russian roulette probability into account
        pdf *= 1.0f - f_reflect_proba;

        // [Turquin, 2019] Eq. 18 for dielectric microfacet energy compensation
        color /= compensation_term;

        if (thin_walled)
            // Thin materials use the base color squared to represent both the entry and the exit
            // simultaneously
            color *= material.base_color;

        if (thin_walled && update_ray_volume_state)
            // For thin materials, refracting in equals refracting out so we're poping the stack
            ray_volume_state.interior_stack.pop(ray_volume_state.inside_material);
        else if (ray_volume_state.incident_mat_index != NestedDielectricsInteriorStack::MAX_MATERIAL_INDEX)
        {
            // If we're not coming from the air, this means that we were in a volume and we're currently
            // refracting out of the volume or into another volume.
            // This is where we take the absorption of our travel into account using Beer-Lambert's law.
            color *= principled_beer_absorption(render_data, ray_volume_state);

            if (update_ray_volume_state)
            {
                // We changed volume so we're resetting the distance
                ray_volume_state.distance_in_volume = 0.0f;
                if (ray_volume_state.inside_material)
                    // We refracting out of a volume so we're poping the stack
                    ray_volume_state.interior_stack.pop(ray_volume_state.inside_material);
            }
        }
    }

    return color;
}

/**
 * The sampled direction is returned in the local shading frame of the basis used for 'local_view_direction'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 principled_glass_sample(const DevicePackedTexturedMaterialSoA& materials_buffer, const DeviceUnpackedEffectiveMaterial& material,
                                                              RayVolumeState& ray_volume_state, bool update_ray_volume_state,
                                                              const float3& local_view_direction, Xorshift32Generator& random_number_generator,
                                                              BSDFIncidentLightInfo& out_sampled_light_info)
{
    float eta_i = ray_volume_state.incident_mat_index == NestedDielectricsInteriorStack::MAX_MATERIAL_INDEX ? 1.0f : materials_buffer.get_ior(ray_volume_state.incident_mat_index);
    float eta_t = ray_volume_state.outgoing_mat_index == NestedDielectricsInteriorStack::MAX_MATERIAL_INDEX ? 1.0f : materials_buffer.get_ior(ray_volume_state.outgoing_mat_index);

    float dispersion_abbe_number = material.dispersion_abbe_number;
    float dispersion_scale = material.dispersion_scale;
    eta_i = compute_dispersion_ior(dispersion_abbe_number, dispersion_scale, eta_i, hippt::abs(ray_volume_state.sampled_wavelength));
    eta_t = compute_dispersion_ior(dispersion_abbe_number, dispersion_scale, eta_t, hippt::abs(ray_volume_state.sampled_wavelength));

    float relative_eta = eta_t / eta_i;
    // To avoid sampling directions that would lead to a null half_vector.
    // Explained in more details in principled_glass_eval.
    if (hippt::abs(relative_eta - 1.0f) < 1.0e-5f)
        relative_eta = 1.0f + 1.0e-5f;

    bool thin_walled = material.thin_walled;
    float scaled_roughness = MaterialUtils::get_thin_walled_roughness(thin_walled, material.roughness, relative_eta);
    float alpha_x, alpha_y;
    MaterialUtils::get_alphas(scaled_roughness, material.anisotropy, alpha_x, alpha_y);

    float3 microfacet_normal = GGX_anisotropic_sample_microfacet(local_view_direction, alpha_x, alpha_y, random_number_generator);
    float HoV = hippt::dot(local_view_direction, microfacet_normal);

    float thin_film = material.thin_film;
    ColorRGB32F F_thin_film;
    if (thin_film > 0.0f)
        F_thin_film = thin_film_fresnel(material, eta_i, HoV);

    ColorRGB32F F_no_thin_film;
    if (thin_film < 1.0f)
        F_no_thin_film = ColorRGB32F(full_fresnel_dielectric(HoV, relative_eta));

    ColorRGB32F F = hippt::lerp(F_no_thin_film, F_thin_film, thin_film);
    float f_reflect_proba = F.luminance();
    if (f_reflect_proba < 1.0f && thin_film == 0.0f && thin_walled && scaled_roughness < 0.1f)
        // If this is not total reflection, adjusting the fresnel term to account for inter-reflections within the thin interface
        // Not doing this if thin-film is present because that would not be accurate at all. Thin-film
        // effect require phase shift computations and that's very expensive so we're just not doing it here
        // instead
        //
        // Reference: Dielectric BSDF, PBR Book 4ed: https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
        //
        // If there is no thin-film, the fresnel reflectance is non-colored and is the same
        // value for all RGB wavelengths. This means that f_reflect_proba is actually just the fresnel reflection factor
        //
        // This fresnel scaling only works at roughness 0 but still using below 0.1f for a close enough approximation
        f_reflect_proba += hippt::square(1.0f - f_reflect_proba) * f_reflect_proba / (1.0f - hippt::square(f_reflect_proba));

    float rand_1 = random_number_generator();

    float3 sampled_direction;
    if (rand_1 < f_reflect_proba)
    {
        // Reflection
        sampled_direction = reflect_ray(local_view_direction, microfacet_normal);
        out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFLECT_LOBE;

        // This is a reflection, we're poping the stack
        if (update_ray_volume_state)
            ray_volume_state.interior_stack.pop(false);
    }
    else
    {
        // Refraction
        out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFRACT_LOBE;

        if (hippt::dot(microfacet_normal, local_view_direction) < 0.0f)
            // For the refraction operation that follows, we want the direction to refract (the view
            // direction here) to be in the same hemisphere as the normal (the microfacet normal here)
            // so we're flipping the microfacet normal in case it wasn't in the same hemisphere as
            // the view direction
            microfacet_normal = -microfacet_normal;

        if (thin_walled)
        {
            // Because the interface is thin (and so we refract twice, "cancelling" the bending the light),
            // the refraction direction is just the incoming (view direction) reflected
            // and flipped about the normal plane

            float3 reflected = reflect_ray(local_view_direction, microfacet_normal);
            // Now flipping
            reflected.z *= -1.0f;

            // Refraction through the thin walled material. 
            // We're poping the stack because we're not inside the material even
            // though this is a refraction. A thin material has no inside
            if (update_ray_volume_state)
                ray_volume_state.interior_stack.pop(false);

            return reflected;
        }
        else
            sampled_direction = refract_ray(local_view_direction, microfacet_normal, relative_eta);
    }

    return sampled_direction;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_diffuse_transmission_eval(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, 
                                                                                RayVolumeState& ray_volume_state, bool update_ray_volume_state, 
                                                                                const float3& local_view_direction, float3 local_to_light_direction, 
                                                                                float& diffuse_transmission_pdf)
{
    diffuse_transmission_pdf = 0.0f;

    if (local_view_direction.z * local_to_light_direction.z > 0.0f)
        // Both are in the same hemisphere, incorrect for a transmission only lobe
        return ColorRGB32F(0.0f);

    ColorRGB32F color = material.base_color * M_INV_PI;
    if (ray_volume_state.incident_mat_index != NestedDielectricsInteriorStack::MAX_MATERIAL_INDEX)
    {
        // If we're not coming from the air, this means that we were in a volume and we're currently
        // refracting out of the volume or into another volume.
        // This is where we take the absorption of our travel into account using Beer-Lambert's law.
        color *= principled_beer_absorption(render_data, ray_volume_state);

        if (update_ray_volume_state)
        {
            // We changed volume so we're resetting the distance
            ray_volume_state.distance_in_volume = 0.0f;
            if (ray_volume_state.inside_material)
                // We refracting out of a volume so we're poping the stack
                ray_volume_state.interior_stack.pop(ray_volume_state.inside_material);
        }
    }

    diffuse_transmission_pdf = hippt::abs(local_to_light_direction.z * M_INV_PI);

    return color;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 principled_diffuse_transmission_sample(float3 surface_normal, Xorshift32Generator& random_number_generator)
{
    // Negating the normal here because by convention the surface normal given
    // to this function is in the same hemisphere as the view direction but we
    // want to sample a refraction, on the other side of the normal
    return cosine_weighted_sample_around_normal_world_space(-surface_normal, random_number_generator);
}

/**
 * Reference:
 * 
 * [1] [Open PBR Specification - Coat Darkening] https://academysoftwarefoundation.github.io/OpenPBR/#model/coat/darkening
 * 
 * 'relative_eta' must be coat_ior / incident_medium_ior
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_coat_compute_darkening(const DeviceUnpackedEffectiveMaterial& material, float relative_eta, float view_dir_fresnel)
{
    if (material.coat_darkening == 0.0f)
        return ColorRGB32F(1.0f);
    
    // Fraction of light that exhibits total internal reflection inside the clearcoat layer,
    // assuming a perfectly diffuse base
    float Kr = 1.0f - (1.0f - fresnel_hemispherical_albedo_fit(relative_eta)) / (relative_eta * relative_eta); // Eq. 66

    // Fraction of light that exhibits total internal reflection inside the clearcoat layer,
    // assuming a perfectly smooth base
    float Ks = view_dir_fresnel; // Eq. 67

    // The roughness of the base layer isn't just material.roughness: 
    // 
    // What if material.roughness is 0.0f but there is no specular, or metallic or glass layer.
    // This means that there is just the diffuse lobe below the clearcoat layer. So even if
    // material.roughness is 0.0f, because the coat layer is directly on top of the diffuse layer,
    // the roughness of the base layer is 1.0f
    //
    // Now what if we have 0 specular but 1 metallic? Then we must use the roughness of the metallic layer
    // (which is actually just material.roughness).
    //
    // Same for the glass lobe (and specular lobe actually)
    //
    // So that's why we have these max() calls below
    //
    // The TL;DR is that we must use material.roughness is one of the base layer lobes (metallic/specular/glass) is 1.0f
    // Otherwise, is all the base layer lobes are 0.0f, then the roughness is 1.0f because this is just the diffuse lobe
    // And we lerp for the intermediate cases
    float base_roughness = hippt::lerp(1.0f, material.roughness, hippt::max(material.specular_transmission, hippt::max(material.metallic, material.specular)));
    // Now because our base, in the general case, isn't perfectly diffuse or perfectly smooth
    // we're lerping between the two values based on the roughness of the based layer and this gives us a good
    // approximation of how much total internal reflection we have inside the coat layer
    float K = hippt::lerp(Ks, Kr, base_roughness); // Eq. 68

    // The base albedo is the albedo of the BSDF below the clearcoat.
    // Because the BSDF below the clearcoat may be composed of many layers,
    // we're approximating the overall as the blending of the albedos of the individual
    // lobes.
    //
    // Only the base substrate of the BSDF and the sheen layer have albedos so we only
    // have to mix those two
    float sheen = material.sheen;
    ColorRGB32F base_albedo = (material.base_color + material.sheen_color * sheen) / (1.0f + sheen);
    // This approximation of the amount of total internal reflection can then be used to
    // compute the darkening of the base caused by the clearcoating
    ColorRGB32F darkening = (1.0f - K) / (ColorRGB32F(1.0f) - base_albedo * K);

    // Disabling more or less the darkening based on:
    //  - whether or not we have a coat layer at all
    //  - whether or not we have coat darkening enabled at all or not
    //  - whether or not we have a diffuse transmission lobe below the coat
    //      layer, in which case there is no TIR between the diffuse
    //      transmission lobe and the coat layer because the diffuse
    //      transmission lobe is a BTDF only, it doesn't
    //      reflect light --> no TIR --> no darkening
    darkening = hippt::lerp(ColorRGB32F(1.0f), darkening, material.coat * material.coat_darkening * (1.0f - material.diffuse_transmission));

    return darkening;
}

/**
 * 'internal' functions are just so that 'principled_bsdf_eval' looks nicer
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_coat_layer(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, 
                                                                    const float3& local_view_direction, const float3 local_to_light_direction, const float3& local_half_vector, const float3& shading_normal, 
                                                                    float incident_ior, float coat_weight, bool refracting, float coat_proba, ColorRGB32F& layers_throughput, float& out_cumulative_pdf,
                                                                    BSDFIncidentLightInfo incident_light_info,
                                                                    int current_bounce)
{
    // '|| refracting' here is needed because if we have our coat
    // lobe on top of the glass lobe, we want to still compute the portion
    // of light that is left for the glass lobe after going through the coat lobe
    // so that's why we get into to if() block that does the computation but
    // we're only going to compute the absorption of the coat layer
    float coat_ior = material.coat_ior;
    if (coat_weight > 0.0f || refracting)
    {
        float coat_pdf = 0.0f;
        ColorRGB32F contribution;
        if (!refracting)
        {
            // The coat layer only contribtues for light direction in the same
            // hemisphere as the view direction (so reflections only, not refractions)
            contribution = principled_coat_eval(render_data, material, local_view_direction, local_to_light_direction, local_half_vector, incident_ior, coat_pdf, incident_light_info, current_bounce);
            contribution *= coat_weight;
            contribution *= layers_throughput;
        }        

        out_cumulative_pdf += coat_pdf * coat_proba;

        // We're using hippt::abs() in the fresnel computation that follow because
        // we may compute these fresnels with incident light directions that are below
        // the hemisphere (for refractions for example) so that's where we want
        // the cosine angle not to be negative

        ColorRGB32F layer_below_attenuation = ColorRGB32F(1.0f);
        // Only the transmitted portion of the light goes to the layer below
        // We're using the shading normal here and not the microfacet normal because:
        // We want the proportion of light that reaches the layer below.
        // That's given by 1.0f - fresnelReflection.
        // 
        // But '1.0f - fresnelReflection' needs to be computed with the shading normal, 
        // not the microfacet normal i.e. it needs to be 1.0f - Fresnel(dot(N, L)), 
        // not 1.0f - Fresnel(dot(H, L))
        // 
        // By computing 1.0f - Fresnel(dot(H, L)), we're computing the light
        // that goes through only that one microfacet with the microfacet normal. But light
        // reaches the layer below through many other microfacets, not just the one with our current
        // micronormal here (local_half_vector). To compute this correctly, we would actually need
        // to integrate over the microfacet normals and compute the fresnel transmission portion
        // (1.0f - Fresnel(dot(H, L))) for each of them and weight that contribution by the
        // probability given by the normal distribution function for the microfacet normal.
        // 
        // We can't do that integration online so we're instead using the shading normal to compute
        // the transmitted portion of light. That's actually either a good approximation or the
        // exact solution. That was shown in GDC 2017 [PBR Diffuse Lighting for GGX + Smith Microsurfaces]
        layer_below_attenuation *= 1.0f - full_fresnel_dielectric(hippt::abs(local_to_light_direction.z), incident_ior, coat_ior);

        // Also, when light reflects off of the layer below the coat layer, some of that reflected light
        // will hit total internal reflection against the coat/air interface. This means that only
        // the part of light that does not hit total internal reflection actually reaches the viewer.
        // 
        // That's why we're computing another fresnel term here to account for that. And additional note:
        // computing that fresnel with the direction reflected from the base layer or with the viewer direction
        // is the same, Fresnel is symmetrical. But because we don't have the exact direction reflected from the
        // base layer, we're using the view direction instead
        float view_dir_fresnel = full_fresnel_dielectric(hippt::abs(local_view_direction.z), incident_ior, coat_ior);
        layer_below_attenuation *= 1.0f - view_dir_fresnel;

        if (!material.coat_medium_absorption.is_white())
        {
            // Only computing the medium absorption if there is actually
            // some absorption

            // Taking the color of the absorbing coat medium into account when the light that got transmitted
            // travels through it
            //
            // The distance traveled into the coat depends on the angle at which we're looking
            // at it and the angle in which light goes: the grazier the angles, the more the
            // absorption since we're traveling further in the coat before leaving
            //
            // Reference: [11], [13]
            // 
            // It can happen that 'incident_refracted_angle' or 'outgoing_refracted_angle'
            // are 0.0f 
            float incident_refracted_angle = hippt::max(1.0e-6f, sqrtf(1.0f - (1.0f - local_to_light_direction.z * local_to_light_direction.z) / (coat_ior * coat_ior)));
            float outgoing_refracted_angle = hippt::max(1.0e-6f, sqrtf(1.0f - (1.0f - local_view_direction.z * local_view_direction.z) / (coat_ior * coat_ior)));

            // Reference: [11], [13]
            float traveled_distance_angle = 1.0f / incident_refracted_angle + 1.0f / outgoing_refracted_angle;
            ColorRGB32F coat_absorption = exp(-(ColorRGB32F(1.0f) - pow(sqrt(material.coat_medium_absorption), traveled_distance_angle)) * material.coat_medium_thickness);
            layer_below_attenuation *= coat_absorption;
        }

        layer_below_attenuation *= principled_coat_compute_darkening(material, coat_ior / incident_ior, view_dir_fresnel);

        // If the coat layer has 0 weight, we should not get any light attenuation.
        // But if the coat layer has 1 weight, we should get the full attenuation that we
        // computed in 'layer_below_attenuation' so we're lerping between no attenuation
        // and full attenuation based on the material coat weight.
        layer_below_attenuation = hippt::lerp(ColorRGB32F(1.0f), layer_below_attenuation, material.coat);

        layers_throughput *= layer_below_attenuation;

        return contribution;
    }

    return ColorRGB32F(0.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_sheen_layer(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, 
                                                                     const float3& local_view_direction, const float3& local_to_light_direction, 
                                                                     const float3& world_space_to_light_direction, const float3& shading_normal,
                                                                     float incident_ior, float sheen_weight, float sheen_proba, 
                                                                     ColorRGB32F& layers_throughput, float& out_cumulative_pdf)
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

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_metal_layer(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, 
                                                                     float roughness, float anisotropy,
                                                                     const float3& local_view_direction, const float3 local_to_light_direction, const float3& local_half_vector, 
                                                                     float incident_ior, 
                                                                     float metal_weight, float metal_proba, 
                                                                     const ColorRGB32F& layers_throughput, float& out_cumulative_pdf,
                                                                     BSDFIncidentLightInfo incident_light_info, bool evaluating_first_metal_lobe,
                                                                     int current_bounce)
{
    if (metal_weight > 0.0f)
    {
        float metal_pdf = 0.0f;
        ColorRGB32F contribution;
        
        contribution = principled_metallic_eval(render_data, material,
            roughness, anisotropy, incident_ior,
            local_view_direction, local_to_light_direction, local_half_vector, metal_pdf, incident_light_info, current_bounce);
        contribution *= metal_weight;
        contribution *= layers_throughput;

        out_cumulative_pdf += metal_pdf * metal_proba;

        // There is nothing below the metal layer so we don't have a
        // layer_throughput attenuation here
        // ...

        return contribution;
    }

    return ColorRGB32F(0.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_glass_layer(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, 
                                                                     RayVolumeState& ray_volume_state, bool update_ray_volume_state,
                                                                     const float3& local_view_direction, const float3 local_to_light_direction,
                                                                     float glass_weight, float glass_proba, 
                                                                     const ColorRGB32F& layers_throughput, float& out_cumulative_pdf,
                                                                     BSDFIncidentLightInfo incident_light_info, 
                                                                     int current_bounce)
{
    if (glass_weight > 0.0f)
    {
        float glass_pdf = 0.0f;
        ColorRGB32F contribution;
        
        contribution = principled_glass_eval(render_data, material, ray_volume_state, update_ray_volume_state, 
                                             local_view_direction, local_to_light_direction, 
                                             glass_pdf, incident_light_info,
                                             current_bounce);
        contribution *= glass_weight;
        contribution *= layers_throughput;

        // There is nothing below the glass layer so we don't have a layer_throughput absorption here
        // ...

        out_cumulative_pdf += glass_pdf * glass_proba;

        return contribution;
    }

    return ColorRGB32F(0.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_diffuse_transmission_layer(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material,    
                                                                                    RayVolumeState& ray_volume_state, bool update_ray_volume_state,
                                                                                    const float3& local_view_direction, const float3 local_to_light_direction,
                                                                                    float diffuse_transmission_weight, float diffuse_transmission_proba,
                                                                                    const ColorRGB32F& layers_throughput, float& out_cumulative_pdf)
{
    if (diffuse_transmission_weight > 0.0f)
    {
        float diffuse_transmission_pdf = 0.0f;
        ColorRGB32F contribution = principled_diffuse_transmission_eval(render_data, material, ray_volume_state, update_ray_volume_state, local_view_direction, local_to_light_direction, diffuse_transmission_pdf);
        contribution *= diffuse_transmission_weight;
        contribution *= layers_throughput;

        // There is nothing below the glass layer so we don't have a layer_throughput absorption here
        // ...

        out_cumulative_pdf += diffuse_transmission_pdf * diffuse_transmission_proba;

        return contribution;
    }

    return ColorRGB32F(0.0f);
}

/**
 * Reference:
 *
 * [1] [Open PBR Specification - Coat Darkening] https://academysoftwarefoundation.github.io/OpenPBR/#model/coat/darkening
 *
 * 'relative_eta' must be coat_ior / incident_medium_ior
 *
 * This function computes the darkening/increase in saturation that happens
 * as light is trapped in the specular layer and bounces on the diffuse base.
 * 
 * This is essentially the same function as 'principled_coat_compute_darkening'
 * but simplified since we know that only a diffuse base can be below the specular layer
 * 
 * 'relative_eta' should be specular_ior / coat_ior (or divided by the incident
 * medium ior if there is no coating)
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_specular_compute_darkening(const DeviceUnpackedEffectiveMaterial& material, float relative_eta, float view_dir_fresnel)
{
    if (material.specular_darkening == 0.0f)
        return ColorRGB32F(1.0f);

    // Fraction of light that exhibits total internal reflection inside the clearcoat layer,
    // assuming a perfectly diffuse base
    float Kr = 1.0f - (1.0f - fresnel_hemispherical_albedo_fit(relative_eta)) / (relative_eta * relative_eta); // Eq. 66 of OpenPBR

    // For the specular layer total internal reflection, we know that the base below is diffuse
    // so K is just Kr
    float K = Kr;

    // The base albedo is the albedo of the BSDF below the specular layer.
    // That's just the diffuse lobe so the base albedo is simple here.
    ColorRGB32F base_albedo = material.base_color;
    // This approximation of the amount of total internal reflection can then be used to
    // compute the darkening of the base caused by the clearcoating
    ColorRGB32F darkening = (1.0f - K) / (ColorRGB32F(1.0f) - base_albedo * K);

    // Disabling more or less the darkening based on:
    //  - whether or not we have a specular layer at all
    //  - whether or not we have specular darkening enabled at all or not
    //  - whether or not we have a diffuse transmission lobe below the specular
    //      layer, in which case there is no TIR between the diffuse
    //      transmission lobe and the specular layer because the diffuse
    //      transmission lobe is a BTDF only, it doesn't
    //      reflect light --> no TIR --> no darkening
    darkening = hippt::lerp(ColorRGB32F(1.0f), darkening, material.specular * material.specular_darkening * (1.0f - material.diffuse_transmission));

    return darkening;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_specular_layer(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material,
                                                                        const float3& local_view_direction, const float3 local_to_light_direction, 
                                                                        const float3& local_half_vector, const float3& shading_normal, 
                                                                        float incident_medium_ior, float specular_weight, bool refracting, float specular_proba, 
                                                                        ColorRGB32F& layers_throughput, float& out_cumulative_pdf,
                                                                        BSDFIncidentLightInfo incident_light_info,
                                                                        int current_bounce)
{
    if (specular_weight > 0.0f || refracting)
    {
        float relative_ior = principled_specular_relative_ior(material, incident_medium_ior);

        float specular_pdf = 0.0f;
        ColorRGB32F contribution;
        
        contribution = principled_specular_eval(render_data, material,
                                                relative_ior, local_view_direction, local_to_light_direction, local_half_vector, 
                                                specular_pdf, incident_light_info,
                                                current_bounce);

        // Tinting the specular reflection color
        contribution *= hippt::lerp(ColorRGB32F(1.0f), material.specular_tint * material.specular_color, material.specular);
        contribution *= specular_weight;
        contribution *= layers_throughput;

        ColorRGB32F layer_below_attenuation(1.0f);
        // Only the transmitted portion of the light goes to the layer below
        // We're using the shading normal here and not the microfacet normal because:
        // We want the proportion of light that reaches the layer below.
        // That's given by 1.0f - fresnelReflection.
        // 
        // But '1.0f - fresnelReflection' needs to be computed with the shading normal, 
        // not the microfacet normal i.e. it needs to be 1.0f - Fresnel(dot(N, L)), 
        // not 1.0f - Fresnel(dot(H, L))
        // 
        // By computing 1.0f - Fresnel(dot(H, L)), we're computing the light
        // that goes through only that one microfacet with the microfacet normal. But light
        // reaches the layer below through many other microfacets, not just the one with our current
        // micronormal here (local_half_vector). To compute this correctly, we would actually need
        // to integrate over the microfacet normals and compute the fresnel transmission portion
        // (1.0f - Fresnel(dot(H, L))) for each of them and weight that contribution by the
        // probability given by the normal distribution function for the microfacet normal.
        // 
        // We can't do that integration online so we're instead using the shading normal to compute
        // the transmitted portion of light. That's actually either a good approximation or the
        // exact solution. That was shown in GDC 2017 [PBR Diffuse Lighting for GGX + Smith Microsurfaces]
        //
        // We need the hippt::abs() here because we may be evaluating the fresnel terms with light directions/view 
        // directions that are below the surface because we're evaluating the specular lobe for a refracted direction
        ColorRGB32F light_dir_fresnel = principled_specular_fresnel(material, relative_ior, hippt::abs(local_to_light_direction.z));
        // If we have a diffuse transmission lobe below the specular instead of the diffuse lobe, then we cannot
        // have TIR in between the diffuse lobe and the specular lobe (inside the specular layer) because the diffuse
        // transmission lobe is a BTDF only, it doesn't reflect any light --> no TIR
        //
        // We're cancelling the light_dir_fresnel instead of the view_dir_fresnel (which is the one that models the TIR)
        // though because otherwise it seems to break, not sure why. The handling of fresnel effects when light is coming
        // from below the specular (or coat lobe) lobe isn't perfect yet
        light_dir_fresnel *= (1.0f - material.diffuse_transmission);
        layer_below_attenuation *= ColorRGB32F(1.0f) - light_dir_fresnel;

        // Also, when light reflects off of the layer below the specular layer, some of that reflected light
        // will hit total internal reflection against the specular/[coat or air] interface. This means that only
        // the part of light that does not hit total internal reflection actually reaches the viewer.
        // 
        // That's why we're computing another fresnel term here to account for that. And additional note:
        // computing that fresnel with the direction reflected from the base layer or with the viewer direction
        // is the same, Fresnel is symmetrical. But because we don't have the exact direction reflected from the
        // base layer, we're using the view direction instead
        ColorRGB32F view_dir_fresnel = principled_specular_fresnel(material, relative_ior, hippt::abs(local_view_direction.z));
        layer_below_attenuation *= ColorRGB32F(1.0f) - view_dir_fresnel;

        // Taking into account the total internal reflection inside the specular layer 
        // (bouncing on the base diffuse layer). We're using the luminance of the fresnel here because
        // the specular layer may have thin film interference which colors the fresnel but
        // we're going to assume that the fresnel is non-colored and thus we just take the luminance
        layer_below_attenuation *= principled_specular_compute_darkening(material, relative_ior, view_dir_fresnel.luminance());

        // If the specular layer has 0 weight, we should not get any light absorption.
        // But if the specular layer has 1 weight, we should get the full absorption that we
        // computed in 'layer_below_attenuation' so we're lerping between no absorption
        // and full absorption based on the material specular weight.
        layer_below_attenuation = hippt::lerp(ColorRGB32F(1.0f), layer_below_attenuation, material.specular);

        layers_throughput *= layer_below_attenuation;

        out_cumulative_pdf += specular_pdf * specular_proba;

        return contribution;
    }

    return ColorRGB32F(0.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_diffuse_layer(const HIPRTRenderData& render_data, float incident_ior, const DeviceUnpackedEffectiveMaterial& material, 
    const float3& local_view_direction, const float3 local_to_light_direction, float diffuse_weight, float diffuse_proba, ColorRGB32F& layers_throughput, float& out_cumulative_pdf)
{
    if (diffuse_weight > 0.0f)
    {
        float diffuse_pdf;
        ColorRGB32F contribution = principled_diffuse_eval(material, local_view_direction, local_to_light_direction, diffuse_pdf);
        contribution *= diffuse_weight;
        contribution *= layers_throughput;

        // Nothing below the diffuse layer so we don't have a layer throughput
        // attenuation here

        out_cumulative_pdf += diffuse_pdf * diffuse_proba;

        return contribution;
    }

    return ColorRGB32F(0.0f);
}

/**
 * The "glossy base" is the combination of a specular GGX layer
 * on top of a diffuse BRDF.
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F internal_eval_glossy_base(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material,
                                                                     const float3& local_view_direction, const float3 local_to_light_direction, const float3& local_half_vector, 
                                                                     const float3& local_view_direction_rotated, const float3 local_to_light_direction_rotated, const float3& local_half_vector_rotated,
                                                                     const float3& shading_normal, 
                                                                     float incident_medium_ior, float diffuse_weight, float specular_weight, bool refracting,
                                                                     float diffuse_proba_norm, float specular_proba_norm, 
                                                                     ColorRGB32F& layers_throughput, float& out_cumulative_pdf,
                                                                     BSDFIncidentLightInfo incident_light_info, int current_bounce)
{
    ColorRGB32F glossy_base_contribution = ColorRGB32F(0.0f);

    // Evaluating the two components of the glossy base
    glossy_base_contribution += internal_eval_specular_layer(render_data, material,
        local_view_direction_rotated, local_to_light_direction_rotated, local_half_vector_rotated, shading_normal, 
        incident_medium_ior, specular_weight, refracting, specular_proba_norm, layers_throughput, out_cumulative_pdf, incident_light_info, current_bounce);
    glossy_base_contribution += internal_eval_diffuse_layer(render_data, incident_medium_ior, material, local_view_direction, local_to_light_direction, diffuse_weight, diffuse_proba_norm, layers_throughput, out_cumulative_pdf);

    float glossy_base_energy_compensation = get_principled_energy_compensation_glossy_base(render_data, material, incident_medium_ior, local_view_direction.z, current_bounce);
    return glossy_base_contribution / glossy_base_energy_compensation;
}

/**
 * Computes the lobes weights for the principled BSDF
 */
HIPRT_HOST_DEVICE HIPRT_INLINE void principled_bsdf_get_lobes_weights(const DeviceUnpackedEffectiveMaterial& material,
                                                                      bool outside_object,
                                                                      float& out_coat_weight, float& out_sheen_weight,
                                                                      float& out_metal_1_weight, float& out_metal_2_weight,
                                                                      float& out_specular_weight,
                                                                      float& out_diffuse_weight, 
                                                                      float& out_glass_weight, float& out_diffuse_transmission_weight)
{
    // Linear blending weights for the lobes
    // 
    // Everytime we multiply by "outside_object" is because we want to disable
    // the lobe if we're inside the object
    //
    // The layering follows the one of the principled BSDF of blender:
    // [10] https://docs.blender.org/manual/fr/dev/render/shader_nodes/shader/principled.html

    out_coat_weight = material.coat * outside_object;
    out_sheen_weight = material.sheen * outside_object;
    // Metal 1 and metal 2 are the two metallic lobes for the two roughnesses.
    // Having 2 roughnesses (linearly blended together) can enable interesting effects
    // that cannot be achieved with a single GGX metal lobe.
    // 
    // See [Revisiting Physically Based Shading at Imageworks, Kulla & Conty, SIGGRAPH 2017],
    // "Double Specular" for more details
    float metallic = material.metallic;
    out_metal_1_weight = metallic * outside_object;
    out_metal_2_weight = metallic * outside_object;

    float second_roughness_weight = material.second_roughness_weight;
    out_metal_1_weight = hippt::lerp(out_metal_1_weight, 0.0f, second_roughness_weight);
    out_metal_2_weight = hippt::lerp(0.0f, out_metal_2_weight, second_roughness_weight);

    float specular_transmission = material.specular_transmission;
    float diffuse_transmission = material.diffuse_transmission;
    out_glass_weight = !outside_object ? (1.0f - diffuse_transmission) : (1.0f - metallic) * (1.0f - diffuse_transmission) * specular_transmission;
    out_diffuse_transmission_weight = !outside_object ? diffuse_transmission : (1.0f - metallic) * diffuse_transmission;

    out_specular_weight = (1.0f - metallic) * (1.0f - specular_transmission * (1.0f - diffuse_transmission)) * material.specular * outside_object;
    out_diffuse_weight = (1.0f - metallic) * (1.0f - specular_transmission) * (1.0f - diffuse_transmission) * outside_object;
}

/**
 * Computes the lobes weights for the principled BSDF and also reflects
 * the shading normal about the geometric normal if the view direction
 * is below the shading normal (probably due to normal mapping / smooth
 * vertex normals)
 */
HIPRT_HOST_DEVICE HIPRT_INLINE void principled_bsdf_get_lobes_weights_fringe_fix(const DeviceUnpackedEffectiveMaterial& material, 
                                                                      const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
                                                                      float3& in_out_normal,
                                                                      bool& out_outside_object, 
                                                                      float& out_coat_weight, float& out_sheen_weight,
                                                                      float& out_metal_1_weight, float& out_metal_2_weight,
                                                                      float& out_specular_weight,
                                                                      float& out_diffuse_weight, 
                                                                      float& out_glass_weight, float& out_diffuse_transmission_weight)
{
    out_outside_object = hippt::dot(view_direction, in_out_normal) > 0 || (material.thin_walled && material.specular_transmission > 0.0f);
    out_glass_weight = (1.0f - material.metallic) * (1.0f - material.diffuse_transmission) * material.specular_transmission;
    out_diffuse_transmission_weight = (1.0f - material.metallic) * material.diffuse_transmission;
    if (hippt::is_zero(out_glass_weight) && hippt::is_zero(out_diffuse_transmission_weight) && !out_outside_object)
    {
        // We're not sampling the glass lobe so we're checking
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
        in_out_normal = reflect_ray(shading_normal, geometric_normal);

        // We we're "inside" of the object because of normal mapping / smooth vertex normals
        // getting the view direction below the surface but now that we've flipped the normal,
        // we're outside the object
        out_outside_object = true;
    }

    principled_bsdf_get_lobes_weights(material, out_outside_object, 
                                      out_coat_weight, out_sheen_weight, 
                                      out_metal_1_weight, out_metal_2_weight, 
                                      out_specular_weight, out_diffuse_weight, 
                                      out_glass_weight, out_diffuse_transmission_weight);

    // TODO necessary?
    //if (!out_outside_object)
    //    // If inside the object, the glass lobe is the only existing lobe so it has weight
    //    // 1.0f
    //    out_glass_weight = 1.0f;
}

HIPRT_HOST_DEVICE HIPRT_INLINE void principled_bsdf_get_lobes_sampling_proba(
    float coat_weight, float sheen_weight, float metal_1_weight, float metal_2_weight,
    float specular_weight, float diffuse_weight, float glass_weight, float diffuse_transmission_weight,

    float& out_coat_sampling_proba, float& out_sheen_sampling_proba,
    float& out_metal_1_sampling_proba, float& out_metal_2_sampling_proba,
    float& out_specular_sampling_proba, float& out_diffuse_sampling_proba,
    float& out_glass_sampling_proba, float& out_diffuse_transmission_sampling_proba)
{
    float normalize_factor = 1.0f / (coat_weight + sheen_weight
                                     + metal_1_weight + metal_2_weight
                                     + specular_weight + diffuse_weight
                                     + glass_weight + diffuse_transmission_weight);

    out_coat_sampling_proba = coat_weight * normalize_factor;
    out_sheen_sampling_proba = sheen_weight * normalize_factor;
    out_metal_1_sampling_proba = metal_1_weight * normalize_factor;
    out_metal_2_sampling_proba = metal_2_weight * normalize_factor;
    out_specular_sampling_proba = specular_weight * normalize_factor;
    out_diffuse_sampling_proba = diffuse_weight * normalize_factor;
    out_glass_sampling_proba = glass_weight * normalize_factor;
    out_diffuse_transmission_sampling_proba = diffuse_transmission_weight * normalize_factor;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_bsdf_eval(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, 
                                                                RayVolumeState& ray_volume_state, bool update_ray_volume_state,
                                                                const float3& view_direction, float3 shading_normal, const float3& to_light_direction, 
                                                                float& pdf,
                                                                int current_bounce, BSDFIncidentLightInfo incident_light_info)
{
    pdf = 0.0f;

    // Only the glass lobe is considered when evaluating
    // the BSDF from inside the object so we're going to use that
    // 'outside_object' flag to nullify the other lobes if we're
    // inside the object
    //
    // Note that we're always outside of thin materials, they have no volume interior
    bool outside_object = hippt::dot(view_direction, shading_normal) > 0 || material.thin_walled;
    bool refracting = hippt::dot(shading_normal, to_light_direction) < 0.0f && outside_object;
    if (hippt::dot(view_direction, shading_normal) < 0.0f)
        // For the rest of the computations to be correct, we want the normal
        // in the same hemisphere as the view direction
        shading_normal = -shading_normal;

    float3 T, B;
    build_ONB(shading_normal, T, B);
    float3 local_view_direction = world_to_local_frame(T, B, shading_normal, view_direction);
    float3 local_to_light_direction = world_to_local_frame(T, B, shading_normal, to_light_direction);
    float3 local_half_vector = hippt::normalize(local_view_direction + local_to_light_direction);

    // Rotated ONB for the anisotropic GGX evaluation (metallic/glass lobes for example)
    float3 TR, BR;
    build_rotated_ONB(shading_normal, TR, BR, material.anisotropy_rotation * M_PI);
    float3 local_view_direction_rotated = world_to_local_frame(TR, BR, shading_normal, view_direction);
    float3 local_to_light_direction_rotated = world_to_local_frame(TR, BR, shading_normal, to_light_direction);
    float3 local_half_vector_rotated = hippt::normalize(local_view_direction_rotated + local_to_light_direction_rotated);

    float coat_weight, sheen_weight, metal_1_weight, metal_2_weight;
    float specular_weight, diffuse_weight, glass_weight, diffuse_transmission_weight;
    principled_bsdf_get_lobes_weights(material, outside_object,
                                      coat_weight, sheen_weight, metal_1_weight, metal_2_weight,
                                      specular_weight, diffuse_weight, glass_weight, diffuse_transmission_weight);

    float incident_medium_ior = ray_volume_state.incident_mat_index == /* air */ NestedDielectricsInteriorStack::MAX_MATERIAL_INDEX ? 1.0f : render_data.buffers.materials_buffer.get_ior(ray_volume_state.incident_mat_index);
    // For the given to_light_direction, normal, view_direction etc..., what's the probability
    // that the 'principled_bsdf_sample()' function would have sampled the lobe?
    float coat_proba, sheen_proba, metal_1_proba, metal_2_proba;
    float specular_proba, diffuse_proba, glass_proba, diffuse_transmission_proba;
    principled_bsdf_get_lobes_sampling_proba(coat_weight, sheen_weight, metal_1_weight, metal_2_weight,
                                             specular_weight, diffuse_weight, glass_weight, diffuse_transmission_weight,

                                             coat_proba, sheen_proba, metal_1_proba, metal_2_proba,
                                             specular_proba, diffuse_proba, glass_proba, diffuse_transmission_proba);

    // Keeps track of the remaining light's energy as we traverse layers
    ColorRGB32F layers_throughput = ColorRGB32F(1.0f);
    ColorRGB32F final_color = ColorRGB32F(0.0f);

    // In the 'internal_eval_coat_layer' function calls below, we're passing
    // 'weight * !refracting' so that lobes that do not allow refractions
    // (which is pretty much all of them except glass) do no get evaluated
    // (because their weight becomes 0)
    final_color += internal_eval_coat_layer(render_data, material,
                                            local_view_direction, local_to_light_direction, local_half_vector, shading_normal, 
                                            incident_medium_ior, coat_weight, refracting, coat_proba, layers_throughput, pdf, incident_light_info, current_bounce);
    final_color += internal_eval_sheen_layer(render_data, material, 
                                             local_view_direction, local_to_light_direction, to_light_direction, shading_normal, 
                                             incident_medium_ior, sheen_weight, sheen_proba, layers_throughput, pdf);
    final_color += internal_eval_metal_layer(render_data, material, material.roughness, material.anisotropy, 
                                             local_view_direction_rotated, local_to_light_direction_rotated, local_half_vector_rotated, incident_medium_ior, 
                                             metal_1_weight * !refracting, metal_1_proba, layers_throughput, pdf, incident_light_info, true, current_bounce);
    final_color += internal_eval_metal_layer(render_data, material, material.second_roughness, material.anisotropy, 
                                             local_view_direction_rotated, local_to_light_direction_rotated, local_half_vector_rotated, incident_medium_ior, 
                                             metal_2_weight * !refracting, metal_2_proba, layers_throughput, pdf, incident_light_info, false, current_bounce);
    // Careful here to evaluate the glass layer before the glossy
    // base otherwise, layers_throughput is going to be modified
    // by the specular layer evaluation (in the glossy base) to 
    // take the fresnel of the specular layer into account. 
    // But we don't want that for the glass layer. 
    // The glass layer isn't below the specular layer , it's "next to"
    // the specular layer so we don't want the specular-layer-fresnel-attenuation
    // there
    final_color += internal_eval_glass_layer(render_data, material,
                                             ray_volume_state, update_ray_volume_state, local_view_direction_rotated, local_to_light_direction_rotated,
                                             glass_weight, glass_proba, layers_throughput, pdf, incident_light_info, current_bounce);
    final_color += internal_eval_glossy_base(render_data, material,
                                             local_view_direction, local_to_light_direction, local_half_vector, 
                                             local_view_direction_rotated, local_to_light_direction_rotated, local_half_vector_rotated, shading_normal, 
                                             incident_medium_ior, diffuse_weight * !refracting, specular_weight, refracting, 
                                             diffuse_proba, specular_proba, 
                                             layers_throughput, pdf, 
                                             incident_light_info, current_bounce);
    final_color += internal_eval_diffuse_transmission_layer(render_data, material,
                                                            ray_volume_state, update_ray_volume_state,
                                                            local_view_direction, local_to_light_direction, diffuse_transmission_weight, diffuse_transmission_proba, layers_throughput, pdf);

    // The clearcoat compensation is done here and not in the clearcoat function
    // because the clearcoat sits on top of everything else. This means that the clearcoat
    // closure contains the full BSDF below. So the full BSDF below + the clearcoat (= the whole BSDF actually)
    // should be compensated, not just the clearcoat lobe. So that's why we're doing
    // it here,  after the full BSDF evaluation so that everything gets compensated
    final_color /= get_principled_energy_compensation_clearcoat_lobe(render_data, material, incident_medium_ior, local_view_direction.z, current_bounce);

    return final_color;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_bsdf_sample(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material,
                                                                  RayVolumeState& ray_volume_state, bool update_ray_volume_state,
                                                                  const float3& view_direction, const float3& shading_normal, const float3& geometric_normal, float3& output_direction,
                                                                  float& pdf, Xorshift32Generator& random_number_generator, 
                                                                  int current_bounce, BSDFIncidentLightInfo* light_sample_info_out)
{
    pdf = 0.0f;

    float3 normal = shading_normal;

    // Computing the weights for sampling the lobes
    bool outside_object;
    float coat_sampling_weight;
    float sheen_sampling_weight;
    float metal_1_sampling_weight;
    float metal_2_sampling_weight;
    float specular_sampling_weight;
    float diffuse_sampling_weight;
    float glass_sampling_weight;
    float diffuse_transmission_weight;
    principled_bsdf_get_lobes_weights_fringe_fix(material, view_direction,
        shading_normal, geometric_normal, normal,
        outside_object,
        coat_sampling_weight, sheen_sampling_weight,
        metal_1_sampling_weight, metal_2_sampling_weight,
        specular_sampling_weight, diffuse_sampling_weight, glass_sampling_weight, diffuse_transmission_weight);

    float coat_sampling_proba, sheen_sampling_proba, metal_1_sampling_proba;
    float metal_2_sampling_proba, specular_sampling_proba, diffuse_sampling_proba;
    float glass_sampling_proba, diffuse_transmission_sampling_proba;
    principled_bsdf_get_lobes_sampling_proba(
        coat_sampling_weight, sheen_sampling_weight, metal_1_sampling_weight, metal_2_sampling_weight,
        specular_sampling_weight, diffuse_sampling_weight, glass_sampling_weight, diffuse_transmission_weight,

        coat_sampling_proba, sheen_sampling_proba, metal_1_sampling_proba, metal_2_sampling_proba,
        specular_sampling_proba, diffuse_sampling_proba, glass_sampling_proba, diffuse_transmission_sampling_proba);

    // Not using a float[] array here because array[] are super poorly handled 
    // in general by the HIP compiler on AMD
    float cdf0 = coat_sampling_proba;
    float cdf1 = cdf0 + sheen_sampling_proba;
    float cdf2 = cdf1 + metal_1_sampling_proba;
    float cdf3 = cdf2 + metal_2_sampling_proba;
    float cdf4 = cdf3 + specular_sampling_proba;
    float cdf5 = cdf4 + diffuse_sampling_proba;
    float cdf6 = cdf5 + diffuse_transmission_sampling_proba;
    // The last cdf[] is implicitely 1.0f so don't need to include it

    float rand_1 = random_number_generator();
    bool sampling_diffuse_transmission_lobe = rand_1 > cdf5 && rand_1 < cdf6;
    bool sampling_glass_lobe = rand_1 > cdf6;
    if (sampling_glass_lobe || sampling_diffuse_transmission_lobe)
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

    if (update_ray_volume_state)
        if (!sampling_glass_lobe && !sampling_diffuse_transmission_lobe)
            // We're going to sample a reflective lobe so we're poping the stack
            ray_volume_state.interior_stack.pop(false);

    if (hippt::dot(view_direction, normal) < 0)
        // We want the normal in the same hemisphere as the view direction
        // for the rest of the calculations
        normal = -normal;

    // Rotated ONB for the anisotropic GGX evaluation
    float3 TR, BR;
    build_rotated_ONB(normal, TR, BR, material.anisotropy_rotation * M_PI);
    float3 local_view_direction_rotated = world_to_local_frame(TR, BR, normal, view_direction);

    BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
    if (rand_1 < cdf0)
    {
        // Sampling the coat lobe

        float3 TR_coat, BR_coat;
        build_rotated_ONB(normal, TR_coat, BR_coat, material.coat_anisotropy_rotation * M_PI);
        float3 local_view_direction_rotated_coat = world_to_local_frame(TR_coat, BR_coat, normal, view_direction);

        incident_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_COAT_LOBE;
        output_direction = local_to_world_frame(TR_coat, BR_coat, normal, principled_coat_sample(material, local_view_direction_rotated_coat, random_number_generator));
    }
    else if (rand_1 < cdf1)
    {
        // Sampling the sheen lobe

        float3 T, B;
        build_ONB(normal, T, B);
        float3 local_view_direction = world_to_local_frame(T, B, normal, view_direction);

        output_direction = local_to_world_frame(T, B, normal, principled_sheen_sample(render_data, material, local_view_direction, normal, random_number_generator));
    }
    else if (rand_1 < cdf2)
    {
        // First metallic lobe sample
        incident_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_FIRST_METAL_LOBE;
        output_direction = local_to_world_frame(TR, BR, normal, principled_metallic_sample(material.roughness, material.anisotropy, local_view_direction_rotated, random_number_generator));
    }
    else if (rand_1 < cdf3)
    {
        // Second metallic lobe sample
        incident_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_SECOND_METAL_LOBE;
        output_direction = local_to_world_frame(TR, BR, normal, principled_metallic_sample(material.second_roughness, material.anisotropy, local_view_direction_rotated, random_number_generator));
    }
    else if (rand_1 < cdf4)
    {
        // Sampling the specular lobe
        incident_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_SPECULAR_LOBE;
        output_direction = local_to_world_frame(TR, BR, normal, principled_specular_sample(material.roughness, material.anisotropy, local_view_direction_rotated, random_number_generator));
    }
    else if (rand_1 < cdf5)
        // No call to local_to_world_frame() since the sample diffuse functions
        // already returns in world space around the given normal
        output_direction = principled_diffuse_sample(normal, random_number_generator);
    else if (rand_1 < cdf6)
        // Diffuse transmission lobe
        output_direction = principled_diffuse_transmission_sample(normal, random_number_generator);
    else
    {
        // When sampling the glass lobe, if we're reflecting off the glass, we're going to have to pop the stack.
        // This is handled inside glass_sample because we cannot know from here if we refracted or reflected
        output_direction = local_to_world_frame(TR, BR, normal, principled_glass_sample(render_data.buffers.materials_buffer, material,
            ray_volume_state, update_ray_volume_state, local_view_direction_rotated, random_number_generator, incident_light_info));
    }

    if (hippt::dot(output_direction, shading_normal) < 0 && !sampling_glass_lobe && !sampling_diffuse_transmission_lobe)
        // It can happen that the light direction sampled is below the surface. 
        // We return 0.0 in this case if we didn't sample the glass lobe
        // because no lobe other than the glass lobe allows refractions
        return ColorRGB32F(0.0f);

    // Giving some information about what the BSDF sampled to the caller
    if (light_sample_info_out != nullptr)
        *light_sample_info_out = incident_light_info;

    // Not using 'normal' here because eval() needs to know whether or not we're inside the surface.
    // This is because is we're inside the surface, we're only going to evaluate the glass lobe.
    // If we were using 'normal', we would always be outside the surface because 'normal' is flipped
    // (a few lines above in the code) so that it is in the same hemisphere as the view direction and
    // eval() will then think that we're always outside the surface even though that's not the case
    return principled_bsdf_eval(render_data, material, ray_volume_state, update_ray_volume_state, view_direction, normal, output_direction, pdf, current_bounce, incident_light_info);
}

#endif
