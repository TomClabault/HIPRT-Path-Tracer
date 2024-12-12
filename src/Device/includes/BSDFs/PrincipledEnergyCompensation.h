/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_PRINCIPLED_ENERGY_COMPENSATION_H
#define DEVICE_PRINCIPLED_ENERGY_COMPENSATION_H

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_bsdf_eval(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& view_direction, float3 shading_normal, const float3& to_light_direction, float& pdf);
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_bsdf_sample(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal, float3& output_direction, float& pdf, Xorshift32Generator& random_number_generator);
HIPRT_HOST_DEVICE HIPRT_INLINE float principled_specular_relative_ior(const SimplifiedRendererMaterial& material, float incident_medium_ior);

HIPRT_HOST_DEVICE HIPRT_INLINE float get_principled_energy_compensation_glossy_base(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, float incident_medium_ior, float NoV)
{
    float ms_compensation = 1.0f;

#if PrincipledBSDFGGXUseMultipleScattering == KERNEL_OPTION_TRUE
    int3 texture_dims = make_int3(GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_COS_THETA_O, GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_ROUGHNESS, GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR);

    float relative_ior = principled_specular_relative_ior(material, incident_medium_ior);
    if (hippt::abs(relative_ior - 1.0f) < 1.0e-3f)
        // If the relative ior is very close to 1.0f,
        // adding some offset to avoid singularities at 1.0f which cause
        // fireflies
        relative_ior += 1.0e-3f;

    // We're storing cos_theta_o^2.5 in the LUT so we're retrieving with
    // root 2.5
    float view_dir_remapped = pow(NoV, 1.0f / 2.5f);
    // sqrt(sqrt(F0)) here because we're storing F0^4 in the LUT
    float F0_remapped = sqrt(sqrt(F0_from_eta_t_and_relative(material.ior, relative_ior)));

    float3 uvw = make_float3(view_dir_remapped, material.roughness, F0_remapped);
    float multiple_scattering_compensation = sample_texture_3D_rgb_32bits(render_data.bsdfs_data.glossy_dielectric_Ess, texture_dims, uvw, render_data.bsdfs_data.use_hardware_tex_interpolation).r;

    // Applying the compensation term for energy preservation
    // If material.specular == 1, then we want the full energy compensation
    // If material.specular == 0, then we only have the diffuse lobe and so we
    // need no energy compensation at all and so we just divide by 1 to basically do nothing
    ms_compensation = hippt::lerp(1.0f, multiple_scattering_compensation, material.specular);
    // Multi scatter compensation is not tabulated to take thin film interference into account.
    // That's because thin film interference completely modifies the fresnel term and the
    // tabulated multi scatter compensation only accounts for the usual dielectric fresnel
    // 
    // So we're progressively disabling ms compensation on the glossy base as the thin-film 
    // is more and more pronounced
    ms_compensation = hippt::lerp(ms_compensation, 1.0f, material.thin_film);
#endif

    return ms_compensation;
}

/**
 * This function gives an approximation of the energy lost by the clearcoat layer
 * by assuming that whatever is under the clearcoat is lambertian (which *may*
 * obviously be a very rough approximation, depending on what's the BSDF below the clearcoat)
 * 
 * This basically treats the clearcoat layer exactly the same as a specular/diffuse
 * (just like the "glossy base" of the principled BSDF) and so that's why we're using the LUTs
 * of the glossy base
 * 
 * The approximation can be harsh but in reasonable scenarios (where we're clearcoating something 
 * quite diffuse: which is usually the case because WHO CLEARCOATS A MIRROR?), it's actually quite good and 
 * it's way better than nothing and cheap compared to the full on-the-fly integration that we
 * would have to do otherwise (or full interlayer-multiple-scattering simulation)
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float get_principled_energy_compensation_clearcoat_lobe(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, float incident_medium_ior, float NoV)
{
    float ms_compensation = 1.0f;

#if PrincipledBSDFGGXUseMultipleScattering == KERNEL_OPTION_TRUE
    int3 texture_dims = make_int3(GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_COS_THETA_O, GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_ROUGHNESS, GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR);

    if (hippt::abs(material.coat_ior / incident_medium_ior - 1.0f) < 1.0e-3f)
        // If the relative ior is very close to 1.0f,
        // adding some offset to avoid singularities which cause
        // fireflies
        incident_medium_ior += 1.0e-3f;

    // We're storing cos_theta_o^2.5 in the LUT so we're retrieving with
    // root 2.5
    float view_dir_remapped = pow(NoV, 1.0f / 2.5f);
    // sqrt(sqrt(F0)) here because we're storing F0^4 in the LUT
    float F0_remapped = sqrt(sqrt(F0_from_eta(material.coat_ior, incident_medium_ior)));

    float3 uvw = make_float3(view_dir_remapped, material.coat_roughness, F0_remapped);
    float multiple_scattering_compensation = sample_texture_3D_rgb_32bits(render_data.bsdfs_data.glossy_dielectric_Ess, texture_dims, uvw, render_data.bsdfs_data.use_hardware_tex_interpolation).r;

    // Applying the compensation term for energy preservation
    // If material.specular == 1, then we want the full energy compensation
    // If material.specular == 0, then we only have the diffuse lobe and so we
    // need no energy compensation at all and so we just divide by 1 to basically do nothing
    ms_compensation = hippt::lerp(1.0f, multiple_scattering_compensation, material.coat);
    // Multi scatter compensation is not tabulated to take thin film interference into account.
    // That's because thin film interference completely modifies the fresnel term and the
    // tabulated multi scatter compensation only accounts for the usual dielectric fresnel
    // 
    // So we're progressively disabling ms compensation on the glossy base as the thin-film 
    // is more and more pronounced
    ms_compensation = hippt::lerp(ms_compensation, 1.0f, material.thin_film);
#endif

    return ms_compensation;
}

 /**
  * On-the-fly integration of the directional albedo of the clearcoat layer
  * (which is basically the whole BSDF since the clearcoat lobe is the topmost lobe of the BSDF)
  *
  * The directional albedo for the given view direction is returned
  *
  * This returned directional albedo can then be used to ensure energy conservation & preservation
  * of the BSDF
  */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_monte_carlo_directional_albedo(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& view_direction, float3 shading_normal, float3 geometric_normal, Xorshift32Generator& random_number_generator)
{
    ColorRGB32F directional_albedo = ColorRGB32F(0.0f);

    // TODO use fresnel simplifications in there to accelerate integration?
    // TODO disable some lobes in the integration for faster integration at the cost of error?
    // TODO only integrate on one color channel instead of 3 since directional albedo is just a float

    SimplifiedRendererMaterial white_material = material;
    white_material.base_color = ColorRGB32F(1.0f);
    white_material.absorption_color = ColorRGB32F(1.0f);
    white_material.coat_medium_absorption = ColorRGB32F(1.0f);
    white_material.metallic_F82 = ColorRGB32F(1.0f);
    white_material.metallic_F90 = ColorRGB32F(1.0f);
    white_material.sheen_color = ColorRGB32F(1.0f);
    white_material.specular_color = ColorRGB32F(1.0f);

    for (int i = 0; i < material.energy_preservation_monte_carlo_samples; i++)
    {
        float pdf;
        float3 sampled_direction;
        RayVolumeState ray_volume_state_copy = ray_volume_state;
        ColorRGB32F bsdf_directional_albedo_sample = principled_bsdf_sample(render_data, white_material, ray_volume_state_copy, view_direction, shading_normal, geometric_normal, sampled_direction, pdf, random_number_generator);
        if (pdf != 0.0f)
            // Correct sampled direction
            // 
            // Incorrect sampled direction can happen when the GGX sampled direction is below the surface
            // (can happen because GGX sampling is not 100% perfect) 

            // abs() of the cosine term here because we may be sampling refractions
            directional_albedo += bsdf_directional_albedo_sample / pdf * hippt::abs(hippt::dot(sampled_direction, shading_normal));
    }

    directional_albedo /= material.energy_preservation_monte_carlo_samples;
    if (directional_albedo.is_black())
        // No valid samples were found, no compensation could be computed,
        // returning 1.0f such that dividing by that compensation term does nothing
        directional_albedo = ColorRGB32F(1.0f);

    return directional_albedo;
}

/**
 * Evaluates the BSDF with strong energy conservation & preservation
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_bsdf_eval_energy_compensated(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& view_direction, float3 shading_normal, float3 geometric_normal, const float3& to_light_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    ColorRGB32F final_color = principled_bsdf_eval(render_data, material, ray_volume_state, view_direction, shading_normal, to_light_direction, pdf);

    ColorRGB32F principled_directional_albedo(1.0f);
    if (material.enforce_strong_energy_conservation && material.thin_film == 0.0f)
        // Only computing the compensation if we actually want it for this material
        principled_directional_albedo = principled_monte_carlo_directional_albedo(render_data, material, ray_volume_state, view_direction, shading_normal, geometric_normal, random_number_generator);

    return final_color / principled_directional_albedo;
}

/**
 * Samples the BSDF with energy conservation & preservation
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F principled_bsdf_sample_energy_compensated(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, RayVolumeState& ray_volume_state, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal, float3& output_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    ColorRGB32F color = principled_bsdf_sample(render_data, material, ray_volume_state, view_direction, shading_normal, geometric_normal, output_direction, pdf, random_number_generator);

    ColorRGB32F clearcoat_directional_albedo(1.0f);
    if (material.enforce_strong_energy_conservation && material.thin_film == 0.0f)
        // Only computing the compensation if we actually want it for this material
        clearcoat_directional_albedo = principled_monte_carlo_directional_albedo(render_data, material, ray_volume_state, view_direction, shading_normal, geometric_normal, random_number_generator);

    return color / clearcoat_directional_albedo;
}

#endif
