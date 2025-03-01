/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_BSDF_MICROFACET_H
#define DEVICE_BSDF_MICROFACET_H

#include "Device/includes/Sampling.h"
#include "Device/includes/BSDFs/MicrofacetEnergyCompensation.h"

 // Clamping value for dot products when evaluating the GGX distribution
 // This helps with fireflies due to numerical imprecisions
 //
 // 1.0e-3f seems indistinguishable from 1.0e-8f (which is closer to
 // "ground truth" since we're not clamping as hard) except that 1.0e-8f
 // has a bunch of fireflies / is not very stable at all.
 //
 // So even though 1.0e-3f may seem a bit harsh, it's actually fine
#define GGX_DOT_PRODUCTS_CLAMP 1.0e-3f

/**
 * Evaluates the GGX anisotropic normal distribution function
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float GGX_anisotropic(float alpha_x, float alpha_y, const float3& local_half_vector)
{
    float denom = (local_half_vector.x * local_half_vector.x) / (alpha_x * alpha_x) +
        (local_half_vector.y * local_half_vector.y) / (alpha_y * alpha_y) +
        (local_half_vector.z * local_half_vector.z);

    return 1.0f / (M_PI * alpha_x * alpha_y * denom * denom);
}

/**
 * Evaluates the visible normal distribution function with GGX as
 * the normal disitrbution function
 *
 * Reference: [Sampling the GGX Distribution of Visible Normals, Heitz, 2018]
 * Equation 3
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float GGX_anisotropic_vndf(float D, float G1V, const float3& local_view_direction, const float3& local_halfway_vector)
{
    float HoL = hippt::max(GGX_DOT_PRODUCTS_CLAMP, hippt::dot(local_view_direction, local_halfway_vector));
    return G1V * D * HoL / local_view_direction.z;
}

/**
 * Lambda function for the denominator of the G1 Smith masking/shadowing functions
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float G1_Smith_lambda(float alpha_x, float alpha_y, const float3& local_direction)
{
    float ax = local_direction.x * alpha_x;
    float ay = local_direction.y * alpha_y;

    return (-1.0f + sqrt(1.0f + (ax * ax + ay * ay) / (local_direction.z * local_direction.z))) * 0.5f;
}

/**
 * G1 Smith masking/shadowing (depending on whether local_direction is wo or wi) function
 *
 * Reference: [Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz, 2014]
 * Equation 43
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float G1_Smith(float alpha_x, float alpha_y, const float3& local_direction)
{
    float lambda = G1_Smith_lambda(alpha_x, alpha_y, local_direction);

    return 1.0f / (1.0f + lambda);
}

/**
 * 'incident_light_direction_is_from_GGX_sample' should be true if the 'local_to_light_direction' given comes from
 * sampling the microfacet distribution that is being evaluated by this function call
 * 
 * false otherwise (if 'local_to_light_direction' comes from light sampling NEE, or sampling another lobe of the BSDF, ...).
 * This parameter only matters if the BRDF is perfectly smooth: roughness < MaterialConstants::ROUGHNESS_CLAMP
 */
template <bool useMultipleScatteringEnergyCompensation>
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F torrance_sparrow_GGX_eval_reflect(const HIPRTRenderData& render_data, float material_roughness, float material_anisotropy, bool material_do_energy_compensation, const ColorRGB32F& F, 
                                                                             const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_halfway_vector, 
                                                                             float& out_pdf, MaterialUtils::SpecularDeltaReflectionSampled incident_light_direction_is_from_GGX_sample,
                                                                             int current_bounce)
{
    out_pdf = -1.0f;
    return ColorRGB32F(-1.0f);
}

/**
 * Evaluates the Torrance Sparrow BRDF 'FDG / 4.NoL.NoV' with the
 * GGX as the microfacet distribution
 * function with single scattering (no energy compensation)
 *
 * 'incident_light_direction_is_from_GGX_sample' should be true if the 'local_to_light_direction' given comes from
 * sampling the microfacet distribution that is being evaluated by this function call
 * 
 * false otherwise (if 'local_to_light_direction' comes from light sampling NEE, or sampling another lobe of the BSDF, ...).
 * This parameter only matters if the BRDF is perfectly smooth: roughness < MaterialConstants::ROUGHNESS_CLAMP
 * 
 * Reference: [Sampling the GGX Distribution of Visible Normals, Heitz, 2018]
 * Equation 15
 */
template <>
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F torrance_sparrow_GGX_eval_reflect<0>(const HIPRTRenderData& render_data, float material_roughness, float material_anisotropy, bool material_do_energy_compensation, const ColorRGB32F& F,
                                                                                const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_halfway_vector, 
                                                                                float& out_pdf, MaterialUtils::SpecularDeltaReflectionSampled incident_light_direction_is_from_GGX_sample,
                                                                                int current_bounce)
{
    out_pdf = 0.0f;
    if (MaterialUtils::is_perfectly_smooth(material_roughness) && PrincipledBSDFDeltaDistributionEvaluationOptimization == KERNEL_OPTION_TRUE)
    {
        // Fast path for perfectly specular BRDF
        if (incident_light_direction_is_from_GGX_sample == MaterialUtils::SpecularDeltaReflectionSampled::SPECULAR_PEAK_NOT_SAMPLED)
            // For a perfectly smooth GGX distribution (a delta distribution), anything other than a
            // perfectly sampled reflection direction is going to yield 0 contribution
            return ColorRGB32F(0.0f);
        else
        {
            if (hippt::dot(reflect_ray(local_view_direction, make_float3(0.0f, 0.0f, 1.0f)), local_to_light_direction) < MaterialConstants::DELTA_DISTRIBUTION_ALIGNEMENT_THRESHOLD)
            {
                // Just an additional check that we indeed have the incident light
                // direction aligned with the perfect reflection direction
                //
                // This additional check is mainly useful for ReSTIR where we need
                // to evaluate the BRDF with a sample that may have been sampled from
                // a delta distribution at a neighbor (so it checks all the boxes for the shortcut
                // and we could just quickly return MaterialConstants::DELTA_DISTRIBUTION_HIGH_VALUE
                // but because that sample wasn't sampled at the current pixel, there
                // is a good chance that it doesn't actually align with the perfect
                // reflection direction = it doesn't align with the specular peak = 0 contribution

                out_pdf = 0.0f;
                return ColorRGB32F(0.0f);
            }

            out_pdf = MaterialConstants::DELTA_DISTRIBUTION_HIGH_VALUE;
            return ColorRGB32F(MaterialConstants::DELTA_DISTRIBUTION_HIGH_VALUE) * F / hippt::abs(local_to_light_direction.z);
        }
    }

    float alpha_x;
    float alpha_y;
    MaterialUtils::get_alphas(material_roughness, material_anisotropy, alpha_x, alpha_y);

    // GGX normal distribution
    float D = GGX_anisotropic(alpha_x, alpha_y, local_halfway_vector);

    // GGX visible normal distribution for evaluating the PDF
    float lambda_V = G1_Smith_lambda(alpha_x, alpha_y, local_view_direction);
    float G1V = 1.0f / (1.0f + lambda_V);
    float Dvisible = GGX_anisotropic_vndf(D, G1V, local_view_direction, local_halfway_vector);

    // Maxing 1.0e-8f here to avoid zeros and numerical imprecisions
    float NoV = hippt::max(GGX_DOT_PRODUCTS_CLAMP, hippt::abs(local_view_direction.z));
    float NoL = hippt::max(GGX_DOT_PRODUCTS_CLAMP, hippt::abs(local_to_light_direction.z));

    // Because we're exactly sampling the visible normals distribution function,
    // that's exactly our PDF.
    // 
    // Additionally, because we need to take into account the reflection operator
    // that we're going to apply to get our final 'to light direction' and so the
    // jacobian determinant of that reflection operator is the (4.0f * NoV) in the
    // denominator
    out_pdf = Dvisible / (4.0f * hippt::dot(local_view_direction, local_halfway_vector));
    if (out_pdf == 0.0f)
        return ColorRGB32F(0.0f);
    else
    {
        float lambda_L = G1_Smith_lambda(alpha_x, alpha_y, local_to_light_direction);

        if (render_data.bsdfs_data.GGX_masking_shadowing == GGXMaskingShadowingFlavor::HeightUncorrelated)
        {
            float G1L = 1.0f / (1.0f + lambda_L);
            float G2 = G1V * G1L;

            return F * D * G2 / (4.0f * NoL * NoV);
        }
        else // Default to GGXMaskingShadowingFlavor::HeightCorrelated
        {
            float G2HeightCorrelated = 1.0f / (1.0f + lambda_V + lambda_L);

            return F * D * G2HeightCorrelated / (4.0f * NoL * NoV);
        }
    }
}

/**
 * 'incident_light_direction_is_from_GGX_sample' should be true if the 'local_to_light_direction' given comes from
 * sampling the microfacet distribution that is being evaluated by this function call
 *
 * false otherwise(if 'local_to_light_direction' comes from light sampling NEE, or sampling another lobe of the BSDF, ...).
 * This parameter only matters if the BRDF is perfectly smooth: roughness < MaterialConstants::ROUGHNESS_CLAMP
 */
template <>
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F torrance_sparrow_GGX_eval_reflect<1>(const HIPRTRenderData& render_data, float material_roughness, float material_anisotropy, bool material_do_energy_compensation, const ColorRGB32F& F, 
                                                                                const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_halfway_vector, 
                                                                                float& out_pdf, MaterialUtils::SpecularDeltaReflectionSampled incident_light_direction_is_from_GGX_sample,
                                                                                int current_bounce)
{
    ColorRGB32F ms_compensation_term = get_GGX_energy_compensation_conductors(render_data, F, material_roughness, material_do_energy_compensation, local_view_direction, current_bounce);
    ColorRGB32F single_scattering = torrance_sparrow_GGX_eval_reflect<0>(render_data, material_roughness, material_anisotropy, false, F, 
        local_view_direction, local_to_light_direction, local_halfway_vector, 
        out_pdf, incident_light_direction_is_from_GGX_sample,
        current_bounce);

    return single_scattering * ms_compensation_term;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F torrance_sparrow_GGX_eval_refract(const DeviceUnpackedEffectiveMaterial& material, float roughness, float relative_eta, ColorRGB32F fresnel_reflectance,
                                                                             const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_halfway_vector, 
                                                                             float& out_pdf, BSDFIncidentLightInfo incident_light_info)
{
    float NoL = local_to_light_direction.z;
    float NoV = local_view_direction.z;
    float HoL = hippt::dot(local_to_light_direction, local_halfway_vector);
    float HoV = hippt::dot(local_view_direction, local_halfway_vector);

    ColorRGB32F color;
    if (MaterialUtils::is_perfectly_smooth(roughness) && PrincipledBSDFDeltaDistributionEvaluationOptimization == KERNEL_OPTION_TRUE)
    {
        // Fast path for specular glass
        bool incident_direction_is_perfect_refraction = hippt::dot(refract_ray(local_view_direction, make_float3(0.0f, 0.0f, 1.0f), relative_eta), local_to_light_direction) > MaterialConstants::DELTA_DISTRIBUTION_ALIGNEMENT_THRESHOLD;
        if (incident_light_info == BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFRACT_LOBE && incident_direction_is_perfect_refraction)
        {
            // When the glass is perfectly smooth i.e. delta distribution, our only hope is to sample
            // directly from the glass lobe. If we didn't sample from the glass lobe, this is going to be 0
            // contribution

            // Just some high value because this is a delta distribution
            // And also, take fresnel into account
            color = ColorRGB32F(MaterialConstants::DELTA_DISTRIBUTION_HIGH_VALUE) * (ColorRGB32F(1.0f) - fresnel_reflectance) * material.base_color;
            color /= hippt::abs(NoL);
            out_pdf = MaterialConstants::DELTA_DISTRIBUTION_HIGH_VALUE;
        }
        else
        {
            color = ColorRGB32F(0.0f);
            out_pdf = 0.0f;
        }
    }
    else
    {
        float dot_prod = HoL + HoV / relative_eta;
        float dot_prod2 = dot_prod * dot_prod;
        float denom = dot_prod2 * NoL * NoV;

        float alpha_x;
        float alpha_y;
        MaterialUtils::get_alphas(roughness, material.anisotropy, alpha_x, alpha_y);

        float D = GGX_anisotropic(alpha_x, alpha_y, local_halfway_vector);
        float G1_V = G1_Smith(alpha_x, alpha_y, local_view_direction);
        float G1_L = G1_Smith(alpha_x, alpha_y, local_to_light_direction);
        float G2 = G1_V * G1_L;

        float dwm_dwi = hippt::abs(HoL) / dot_prod2;
        float D_pdf = G1_V / hippt::abs(NoV) * D * hippt::abs(HoV);
        out_pdf = dwm_dwi * D_pdf;

        // We added a check a few lines above to "avoid dividing by 0 later on". This is where.
        // When NoL is 0, denom is 0 too and we're dividing by 0. 
        // The PDF of this case is as low as 1.0e-9 (light direction sampled perpendicularly to the normal)
        // so this is an extremely rare case.
        // The PDF being non-zero, we could actualy compute it, it's valid but not with floats :D
        color = material.base_color * D * (ColorRGB32F(1.0f) - fresnel_reflectance) * G2 * hippt::abs(HoL * HoV / denom);
    }

    return color;
}

/**
 * Reference: [Sampling the GGX Distribution of Visible Normals, Unity: Heitz ; 2018]
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 GGX_VNDF_sample(const float3 local_view_direction, float alpha_x, float alpha_y, Xorshift32Generator& random_number_generator)
{
    float r1 = random_number_generator();
    float r2 = random_number_generator();

    // Stretching the ellipsoid to the hemisphere configuration
    float3 Vh = hippt::normalize(float3{ alpha_x * local_view_direction.x, alpha_y * local_view_direction.y, local_view_direction.z });

    // Orthonormal basis construction
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0f ? float3{ -Vh.y, Vh.x, 0 } / sqrt(lensq) : float3{ 1.0f, 0.0f, 0.0f };
    float3 T2 = hippt::cross(Vh, T1);

    // Parametrization of the projected area of the hemisphere
    float r = sqrt(r1);
    float phi = M_TWO_PI * r2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;

    // Sampling the hemisphere
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(hippt::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // Un-stretching back to our ellipsoid
    return hippt::normalize(float3{ alpha_x * Nh.x, alpha_y * Nh.y, hippt::max(0.0f, Nh.z) });
}

/**
 * Sample the distribution anisotropic GGX of visible normals using
 * the spherical caps formulation which is slightly faster than the traditional
 * VNDF sampling by Heitz 2018.
 *
 * Reference: [Sampling Visible GGX Normals with Spherical Caps, Dupuy, Benyoub, 2023]
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 GGX_VNDF_spherical_caps_sample(const float3 local_view_direction, float alpha_x, float alpha_y, Xorshift32Generator& random_number_generator)
{
    float r1 = random_number_generator();
    float r2 = random_number_generator();

    // Stretching the ellipsoid to the hemisphere configuration
    float3 Vh = hippt::normalize(make_float3(alpha_x * local_view_direction.x, alpha_y * local_view_direction.y, local_view_direction.z));

    // Sample a spherical cap in (-wi.z, 1]
    float phi = M_TWO_PI * r1;
    float z = (1.0f - r2) * (1.0f + Vh.z) - Vh.z;
    float sinTheta = sqrtf(hippt::clamp(0.0f, 1.0f, 1.0f - z * z));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    float3 c = make_float3(x, y, z);

    // Compute microfacet normal
    float3 Nh = c + Vh;

    // Un-stretching back to our ellipsoid
    return hippt::normalize(make_float3(alpha_x * Nh.x, alpha_y * Nh.y, Nh.z));
}

/**
 * Samples a microfacet normal from the distribution of visible normals of
 * the GGX normal function distribution
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 GGX_anisotropic_sample_microfacet(const float3& local_view_direction, float alpha_x, float alpha_y, Xorshift32Generator& random_number_generator)
{
    if (alpha_x <= MaterialConstants::ROUGHNESS_CLAMP && alpha_y <= MaterialConstants::ROUGHNESS_CLAMP)
        // For delta GGX distribution, the sampled normal is always the same as the surface normal
        // (so (0, 0, 1) in local space
        //
        // This is basically a small optimization to avoid to whole sampling routine
        return make_float3(0.0f, 0.0f, 1.0f);

#if PrincipledBSDFAnisotropicGGXSampleFunction == GGX_VNDF_SAMPLING
    return GGX_VNDF_sample(local_view_direction, alpha_x, alpha_y, random_number_generator);
#elif PrincipledBSDFAnisotropicGGXSampleFunction == GGX_VNDF_SPHERICAL_CAPS
    return GGX_VNDF_spherical_caps_sample(local_view_direction, alpha_x, alpha_y, random_number_generator);
#elif PrincipledBSDFAnisotropicGGXSampleFunction == GGX_VNDF_BOUNDED
#else
#endif
}

/*
 * Samples a microfacet normal from the distribution of visible normals of
 * the GGX normal function distribution and reflects the given view direction
 * about that microfacet normal to produce a 'to_light_direction' in local
 * shading space that is then returned by that function
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 microfacet_GGX_sample_reflection(float roughness, float anisotropy, const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
    // The view direction can sometimes be below the shading normal hemisphere
    // because of normal mapping / smooth normals
    int below_normal = (local_view_direction.z < 0) ? -1 : 1;
    float alpha_x, alpha_y;
    MaterialUtils::get_alphas(roughness, anisotropy, alpha_x, alpha_y);

    float3 microfacet_normal = GGX_anisotropic_sample_microfacet(local_view_direction * below_normal, alpha_x, alpha_y, random_number_generator);
    float3 sampled_direction = reflect_ray(local_view_direction, microfacet_normal * below_normal);

    // Should already be normalized but float imprecisions...
    return hippt::normalize(sampled_direction);
}

#endif
