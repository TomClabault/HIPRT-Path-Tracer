/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_SAMPLING_H
#define DEVICE_SAMPLING_H

#include "Device/includes/ONB.h"
#include "Device/includes/Texture.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/KernelOptions.h"
#include "HostDeviceCommon/Material.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

/**
 * Returns the radical inverse base 2 of a given number.
 * Used for generating 2D points following the Hammersley point set
 * 
 * Reference: [Holger Dammertz, Hammersley Points on the Hemisphere] http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float radical_inverse_base_2(unsigned int index) {
    index = (index << 16u) | (index >> 16u);
    index = ((index & 0x55555555u) << 1u) | ((index & 0xAAAAAAAAu) >> 1u);
    index = ((index & 0x33333333u) << 2u) | ((index & 0xCCCCCCCCu) >> 2u);
    index = ((index & 0x0F0F0F0Fu) << 4u) | ((index & 0xF0F0F0F0u) >> 4u);
    index = ((index & 0x00FF00FFu) << 8u) | ((index & 0xFF00FF00u) >> 8u);
    return float(index) * 2.3283064365386963e-10f; // / 0x100000000
}

/**
 * Generates a 2D point of the Hammersley point set given the total number
 * of points that are going to be sampled and the index of the point
 * (in [0, number_of_points -1]) that we're sampling right now
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float2 sample_hammersley_2D(unsigned int number_of_points, unsigned int point_index)
{
    return make_float2(static_cast<float>(point_index) / static_cast<float>(number_of_points), radical_inverse_base_2(point_index));
}

/**
 * Returns integer pixel coordinates offset from the center of the disk
 * given the radius of the disk and two random numbers in [0, 1] u and v
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float2 sample_in_disk_uv(float radius, float2 uv)
{
    float r_sqrt_v = radius * sqrtf(uv.y);
    float x = r_sqrt_v * cos(2.0f * M_PI * uv.x);
    float y = r_sqrt_v* sin(2.0f * M_PI * uv.x);

    return make_float2(x, y);
}

/**
 * Returns integer pixel coordinates offset from the center of the disk of radius 'radius'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float2 sample_in_disk(float radius, Xorshift32Generator& random_number_generator)
{
    float u1 = random_number_generator();
    float u2 = random_number_generator();

    return sample_in_disk_uv(radius, make_float2(u1, u2));
}

/**
 * Power heuristic with a hardcoded Beta exponent of 2 and two sampling strategies only
 *
 * This implementation already contains the 1/nb_pdf_a fraction of the MIS estimator. This means
 * that you should not divide by 1/nb_pdf_a in the evaluation of your function where you use
 * the MIS weight
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float power_heuristic(float pdf_a, int nb_pdf_a, float pdf_b, int nb_pdf_b)
{
    float p_a_sqr = (nb_pdf_a * pdf_a) * (nb_pdf_a * pdf_a);
    float p_b_sqr = (nb_pdf_b * pdf_b) * (nb_pdf_b * pdf_b);

    // Note that we should have a multiplication by nb_pdf_a^2 in the
    // numerator but because we're going to divide by nb_pdf_a in the
    // function evaluation that use this MIS weight according to the
    // MIS estimator, we're only multiplying by nb_pdf_a (not squared)
    // since the squared nb_pdf_a would be cancelled by the division by
    // nb_pdf_a
    return nb_pdf_a * pdf_a * pdf_a / (p_a_sqr + p_b_sqr);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float power_heuristic(float pdf_a, float pdf_b)
{
    return power_heuristic(pdf_a, 1, pdf_b, 1);
}

/**
 * Balance heuristic for MIS weights computation
 *
 * This implementation already contains the 1/nb_pdf_a fraction of the MIS estimator. This means
 * that you should not divide by 1/nb_pdf_a in the evaluation of your function where you use
 * the MIS weight
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float balance_heuristic(float pdf_a, float nb_pdf_a, float pdf_b, float nb_pdf_b)
{
    // Note that we should have a multiplication by nb_pdf_a in the
    // numerator but because we're going to divide by nb_pdf_a in the
    // function evaluation that use this MIS weight according to the
    // MIS estimator, this multiplication in the numerator that we
    // would have here would be canceled and that would be basically
    // wasted maths so we're not doing it and we should not do it
    // in the function evaluation either.
    return pdf_a / (nb_pdf_a * pdf_a + nb_pdf_b * pdf_b);
}

/**
 * Balance heuristic for 3 strategies
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float balance_heuristic(float pdf_a, int nb_pdf_a, float pdf_b, int nb_pdf_b, int pdf_c, int nb_pdf_c)
{
    // Note that we should have a multiplication by nb_pdf_a in the
    // numerator but because we're going to divide by nb_pdf_a in the
    // function evaluation that use this MIS weight according to the
    // MIS estimator, this multiplication in the numerator that we
    // would have here would be canceled and that would be basically
    // wasted maths so we're not doing it and we should not do it
    // in the function evaluation either.
    return pdf_a / (nb_pdf_a * pdf_a + nb_pdf_b * pdf_b + nb_pdf_c * pdf_c);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float balance_heuristic(float pdf_a, float pdf_b)
{
    return balance_heuristic(pdf_a, 1, pdf_b, 1);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float balance_heuristic(float pdf_a, float pdf_b, float pdf_c)
{
    return balance_heuristic(pdf_a, 1, pdf_b, 1, pdf_c, 1);
}

/**
 * Reflects a ray about a normal. This function requires that dot(ray_direction, surface_normal) > 0 i.e.
 * ray_direction and surface_normal are in the same hemisphere
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 reflect_ray(const float3& ray_direction, const float3& surface_normal)
{
    return -ray_direction + 2.0f * hippt::dot(ray_direction, surface_normal) * surface_normal;
}

/**
 * Refracts a ray about a normal. This function requires that dot(ray_direction, surface_normal) > 0 i.e.
 * ray_direction and surface_normal are in the same hemisphere
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool refract_ray(const float3& ray_direction, const float3& surface_normal, float3& refract_direction, float relative_eta)
{
    float NoI = hippt::dot(ray_direction, surface_normal);

    float sin_theta_i_2 = 1.0f - NoI * NoI;
    float root_term = 1.0f - sin_theta_i_2 / (relative_eta * relative_eta);
    if (root_term < 0.0f)
        return false;

    float cos_theta_t = sqrt(root_term);
    refract_direction = -ray_direction / relative_eta + (NoI / relative_eta - cos_theta_t) * surface_normal;

    return true;
}

/** 
 * Reference:
 * 
 * [1] [Lambertian Reflection Without Tangents], Edd Biddulph https://fizzer.neocities.org/lambertnotangent
 * 
 * The sampled direction is returned in world space
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 cosine_weighted_sample_around_normal(const float3& normal, Xorshift32Generator& random_number_generator)
{
    float rand_1 = random_number_generator();
    float rand_2 = 2.0f * random_number_generator() - 1.0f;
    if (rand_1 < 1.0e-8f && rand_2 < -0.999999f && normal.z > 0.999999f)
    {
        // Slight perturbation when this would result in a singularity:
        // When rand_1 is 0.0f and rand_2 is -1.0f, this results in a theta
        // of 0.0f which then gives sphere_point = {0.0f, 0.0f, -1.0f}. In
        // conjunction with a normal of {0.0f, 0.0f, 1.0f}, we get a null vector
        // at the return statement that is then normalized --> NaN
        rand_1 += 1.0e-7f;
        rand_2 += 1.0e-7f;
    }

    float theta = 2.0f * M_PI * rand_1;

    float2 xy = sqrt(1.0f - rand_2 * rand_2) * make_float2(cos(theta), sin(theta));
    float3 sphere_point = make_float3(xy.x, xy.y, rand_2);

    return hippt::normalize(normal + sphere_point);
}

/**
 * Reference:
 *
 * [1] [Global Illumination Compendium], https://people.cs.kuleuven.be/~philip.dutre/GI/TotalCompendium.pdf
 *
 * The sampled direction is returned in a local frame with Z as the up axis
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 cosine_weighted_sample_z_up_frame(Xorshift32Generator& random_number_generator)
{
    float r1 = random_number_generator();
    float r2 = random_number_generator();

    float phi = 2.0f * M_PI * r1;
    float cos_theta = sqrt(r2);
    float sin_theta = sqrt(1 - cos_theta * cos_theta);

    return hippt::normalize(make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta));
}

/**
 * Schlick's approximation for dielectric fresnel reflectance
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F fresnel_schlick(ColorRGB32F R0, float angle)
{
    return R0 + (ColorRGB32F(1.0f) - R0) * pow((1.0f - angle), 5.0f);
}

/**
 * Full reflectance fresnel dielectric formula
 * 
 * 'relative_eta' is eta_t / eta_i = transmitted media IOR / incident media IOR
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float full_fresnel_dielectric(float cos_theta_i, float relative_eta)
{
    // Computing cos_theta_t
    float sin_theta_i2 = 1.0f - cos_theta_i * cos_theta_i;
    float sin_theta_t2 = sin_theta_i2 / (relative_eta * relative_eta);

    if (sin_theta_t2 >= 1.0f)
        // Total internal reflection, 0% refraction, all reflection
        return 1.0f;

    float cos_theta_t = sqrt(1.0f - sin_theta_t2);
    float r_parallel = (relative_eta * cos_theta_i - cos_theta_t) / (relative_eta * cos_theta_i + cos_theta_t);
    float r_perpendicular = (cos_theta_i - relative_eta * cos_theta_t) / (cos_theta_i + relative_eta * cos_theta_t);
    return (r_parallel * r_parallel + r_perpendicular * r_perpendicular) / 2;
}

/**
 * Override of full_fresnel_dielectric with two separate eta
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float full_fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t)
{
    return full_fresnel_dielectric(cos_theta_i, eta_t / eta_i);
}

/**
 * Computes the reflectance at normal incidence from the two
 * given eta and uses that reflectance to compute the dielectric
 * fresnel reflectance using schlick's approximation
 * 
 * This function is basically a shorthand for:
 *      ColorRGB32F R0 = <compute R0 from etas>
 *      return schlick(R0, NoL)
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F fresnel_schlick_from_ior(float eta_i, float eta_t, float cos_theta_i)
{
    float R0_nume = eta_t - eta_i;
    float R0_denom = eta_t + eta_i;
    float R0 = (R0_nume * R0_nume) / (R0_denom * R0_denom);

    return fresnel_schlick(ColorRGB32F(R0), cos_theta_i);
}

/**
 * Overload with normal and light direction
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F fresnel_schlick_from_ior(float eta_i, float eta_t, const float3& normal, const float3& local_to_light_direction)
{
    float NoL = hippt::clamp(1.0e-8f, 1.0f, hippt::dot(normal, local_to_light_direction));
 
    return fresnel_schlick_from_ior(eta_i, eta_t, NoL);
}

/**
 * Implementation of [Artist Friendly Metallic Fresnel, Gulbrandsen, 2014] for
 * computing the complex index of refraction of metals from two intuitive color parameters
 * 'reflectivity' and 'edge_tint'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F gulbrandsen_metallic_complex_fresnel(const ColorRGB32F& reflectivity, const ColorRGB32F& edge_tint, float cos_theta_i)
{
    // TODO we should precompute k and n on the CPU from 'reflectivity' and 'edge_tint'

    
    // Computing n and k from the 'reflectivity' and 'edge_tint' artist parameters
    ColorRGB32F one = ColorRGB32F(1.0f);
    ColorRGB32F sqrt_r = sqrt(reflectivity);
    ColorRGB32F left_n = edge_tint * ((one - reflectivity) / (one + reflectivity));
    ColorRGB32F right_n = (one - edge_tint) * ((one + sqrt_r) / (one - sqrt_r));
    ColorRGB32F n = left_n + right_n;

    ColorRGB32F k_left = n + one;
    k_left *= k_left;
    k_left *= reflectivity;
    ColorRGB32F k_right = n - one;
    k_right *= k_right;
    ColorRGB32F k_sqr = (k_left - k_right) / (one - reflectivity);

    // Computing the approximation for non polarized light based on Rs and Rp
    // for the perpendicular and parallel components of the light
    ColorRGB32F Rs_nume = n * n + k_sqr - 2.0f * n * cos_theta_i + ColorRGB32F(cos_theta_i * cos_theta_i);
    ColorRGB32F Rs_denom = n * n + k_sqr + 2.0f * n * cos_theta_i + ColorRGB32F(cos_theta_i * cos_theta_i);
    ColorRGB32F Rs = Rs_nume / Rs_denom;

    ColorRGB32F Rp_nume = (n * n + k_sqr) * cos_theta_i * cos_theta_i - 2.0f * n * cos_theta_i + one;
    ColorRGB32F Rp_denom = (n * n + k_sqr) * cos_theta_i * cos_theta_i + 2.0f * n * cos_theta_i + one;
    ColorRGB32F Rp = Rp_nume / Rp_denom;

    return 0.5f * (Rs + Rp);
}

/**
 * Evaluates GTR2 anisotropic normal distribution function
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float GTR2_anisotropic(float alpha_x, float alpha_y, const float3& local_half_vector)
{
    float denom = (local_half_vector.x * local_half_vector.x) / (alpha_x * alpha_x) +
        (local_half_vector.y * local_half_vector.y) / (alpha_y * alpha_y) +
        (local_half_vector.z * local_half_vector.z);

    return 1.0f / (M_PI * alpha_x * alpha_y * denom * denom);
}

// Clamping value for dot products when evaluating the GTR2 distribution
// This helps with fireflies due to numerical imprecisions
//
// 1.0e-3f seems indistinguishable from 1.0e-8f (which is closer to
// "ground truth" since we're not clamping as hard) except that 1.0e-8f
// has a bunch of fireflies / is not very stable at all.
//
// So even though 1.0e-3f may seem a bit harsh, it's actually fine
#define GTR2_DOT_PRODUCTS_CLAMP 1.0e-3f

/**
 * Evaluates the visible normal distribution function with GTR2 as
 * the normal disitrbution function
 *
 * Reference: [Sampling the GGX Distribution of Visible Normals, Heitz, 2018]
 * Equation 3
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float GTR2_anisotropic_vndf(float alpha_x, float alpha_y, float D, float G1V, const float3& local_view_direction, const float3& local_halfway_vector)
{
    float HoL = hippt::max(GTR2_DOT_PRODUCTS_CLAMP, hippt::dot(local_view_direction, local_halfway_vector));
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

template <bool useMultipleScatteringEnergyCompensation>
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F torrance_sparrow_GTR2_eval(const HIPRTRenderData& render_data, const ColorRGB32F& F0, float material_roughness, float material_anisotropy, const ColorRGB32F& F, const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_halfway_vector, float& out_pdf)
{
    out_pdf = -1.0f;
    return ColorRGB32F(-1.0f);
}

/**
 * Evaluates the Torrance Sparrow BRDF 'FDG / 4.NoL.NoV' with the
 * Generalized Trowbridge-Reitz 2 (GTR2) as the microfacet distribution
 * function with single scattering (no energy compensation)
 * 
 * Reference: [Sampling the GGX Distribution of Visible Normals, Heitz, 2018]
 * Equation 15
 */
template <>
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F torrance_sparrow_GTR2_eval<0>(const HIPRTRenderData& render_data, const ColorRGB32F& F0, float material_roughness, float material_anisotropy, const ColorRGB32F& F, const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_halfway_vector, float& out_pdf)
{
    out_pdf = 0.0f;

    float alpha_x;
    float alpha_y;
    SimplifiedRendererMaterial::get_alphas(material_roughness, material_anisotropy, alpha_x, alpha_y);

    // GTR2 normal distribution
    float D = GTR2_anisotropic(alpha_x, alpha_y, local_halfway_vector);

    // GTR2 visible normal distribution for evaluating the PDF
    float lambda_V2 = G1_Smith_lambda(alpha_x, alpha_y, local_view_direction);
    float G1V = 1.0f / (1.0f + lambda_V2);
    float Dvisible = GTR2_anisotropic_vndf(alpha_x, alpha_y, D, G1V, local_view_direction, local_halfway_vector);

    // Maxing 1.0e-8f here to avoid zeros and numerical imprecisions
    float NoV = hippt::max(GTR2_DOT_PRODUCTS_CLAMP, hippt::abs(local_view_direction.z));
    float NoL = hippt::max(GTR2_DOT_PRODUCTS_CLAMP, hippt::abs(local_to_light_direction.z));

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
        float G2HeightCorrelated = 1.0f / (1.0f + lambda_V2 + lambda_L);

        return F * D * G2HeightCorrelated / (4.0f * NoL * NoV);
    }
}

template <>
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F torrance_sparrow_GTR2_eval<1>(const HIPRTRenderData& render_data, const ColorRGB32F& F0, float material_roughness, float material_anisotropy, const ColorRGB32F& F, const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_halfway_vector, float& out_pdf)
{
    const void* GGX_Ess_texture_pointer = nullptr;
#ifdef __KERNELCC__
    GGX_Ess_texture_pointer = &render_data.brdfs_data.GGX_Ess;
#else
    GGX_Ess_texture_pointer = render_data.brdfs_data.GGX_Ess;
#endif

    // Reading the precomputed hemispherical directional albedo from the texture
    float Ess = sample_texture_rgb_32bits(GGX_Ess_texture_pointer, 0, make_int2(96, 96), false, make_float2(hippt::max(0.0f, local_view_direction.z), material_roughness)).r;

    // Computing kms, [Practical multiple scattering compensation for microfacet models, Turquin, 2017], Eq. 10
    float kms = (1.0f - Ess) / Ess;

    ColorRGB32F fresnel_compensation_term;
#if PrincipledBSDFGGXUseMultipleScatteringDoFresnel == KERNEL_OPTION_TRUE
    // [Practical multiple scattering compensation for microfacet models, Turquin, 2017], Eq. 15
    fresnel_compensation_term = F0;
#else
    // 1.0f F so that the fresnel compensation has no effect
    fresnel_compensation_term = ColorRGB32F(1.0f);
#endif
    // Computing the compensation term and multiplying by the single scattering non-energy conserving base GGX BRDF,
    // Eq. 9
    ColorRGB32F ms_compensation_term = ColorRGB32F(1.0f) + fresnel_compensation_term * kms;

    ColorRGB32F single_scattering = torrance_sparrow_GTR2_eval<0>(render_data, F0, material_roughness, material_anisotropy, F, local_view_direction, local_to_light_direction, local_halfway_vector, out_pdf);
    return single_scattering * ms_compensation_term;
}

/**
 * Evaluation of a torrance-sparrow model with the GTR2 normal distribution
 * and G1 masking-shadowing
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F torrance_sparrow_GTR2_eval(const HIPRTRenderData& render_data, const ColorRGB32F& F0, float material_roughness, float material_anisotropy, float material_ior, float incident_ior, const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_halfway_vector, float& out_pdf)
{
    float HoL = hippt::clamp(1.0e-8f, 1.0f, hippt::dot(local_halfway_vector, local_to_light_direction));

    ColorRGB32F F = ColorRGB32F(full_fresnel_dielectric(HoL, incident_ior, material_ior));

    return torrance_sparrow_GTR2_eval<PrincipledBSDFGGXUseMultipleScattering>(render_data, F0, material_roughness, material_anisotropy, F, local_view_direction, local_to_light_direction, local_halfway_vector, out_pdf);
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
    float phi = 2.0f * M_PI * r2;
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
 * Sample the distribution (anisotropic GGX/GTR2) of visible normals using
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
    float phi = 2.0f * M_PI * r1;
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
 * the GTR2 normal function distribution
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 GTR2_anisotropic_sample_microfacet(const float3& local_view_direction, float alpha_x, float alpha_y, Xorshift32Generator& random_number_generator)
{
#if PrincipledBSDFAnisotropicGGXSampleFunction == GGX_NO_VNDF
#elif PrincipledBSDFAnisotropicGGXSampleFunction == GGX_VNDF_SAMPLING
    return GGX_VNDF_sample(local_view_direction, alpha_x, alpha_y, random_number_generator);
#elif PrincipledBSDFAnisotropicGGXSampleFunction == GGX_VNDF_SPHERICAL_CAPS
    return GGX_VNDF_spherical_caps_sample(local_view_direction, alpha_x, alpha_y, random_number_generator);
#elif PrincipledBSDFAnisotropicGGXSampleFunction == GGX_VNDF_BOUNDED
#else
#endif
}

/*
 * Samples a microfacet normal from the distribution of visible normals of
 * the GTR2 normal function distribution and reflects the given view direction
 * about that microfacet normal to produce a 'to_light_direction' in local
 * shading space that is then returned by that function
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 microfacet_GTR2_sample_reflection(float roughness, float anisotropy, const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
    // The view direction can sometimes be below the shading normal hemisphere
    // because of normal mapping / smooth normals
    int below_normal = (local_view_direction.z < 0) ? -1 : 1;
    float alpha_x, alpha_y;
    SimplifiedRendererMaterial::get_alphas(roughness, anisotropy, alpha_x, alpha_y);

    float3 microfacet_normal = GTR2_anisotropic_sample_microfacet(local_view_direction * below_normal, alpha_x, alpha_y, random_number_generator);
    float3 sampled_direction = reflect_ray(local_view_direction, microfacet_normal * below_normal);

    // Should already be normalized but float imprecisions...
    return hippt::normalize(sampled_direction);
}

#endif
