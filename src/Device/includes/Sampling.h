/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_SAMPLING_H
#define DEVICE_SAMPLING_H

#include "Device/includes/ONB.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material.h"
#include "HostDeviceCommon/Xorshift.h"

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
HIPRT_HOST_DEVICE HIPRT_INLINE float balance_heuristic(float pdf_a, int nb_pdf_a, float pdf_b, int nb_pdf_b)
{
    // Note that we should have a multiplication by nb_pdf_a in the
    // numerator but because we're going to divide by nb_pdf_a in the
    // function evaluation that use this MIS weight according to the
    // MIS estimator, this multiplication in the numerator that we
    // would have here would be canceled at that would be basically
    // wasted maths so we're not doing it and we should not do it
    // in the function evaluation either.
    return pdf_a / (nb_pdf_a * pdf_a + nb_pdf_b * pdf_b);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float balance_heuristic(float pdf_a, float pdf_b)
{
    return power_heuristic(pdf_a, 1, pdf_b, 1);
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
HIPRT_HOST_DEVICE HIPRT_INLINE float3 cosine_weighted_sample(const float3& normal, Xorshift32Generator& random_number_generator)
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

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F fresnel_schlick(ColorRGB32F F0, float angle)
{
    return F0 + (ColorRGB32F(1.0f) - F0) * pow((1.0f - angle), 5.0f);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float fresnel_dielectric(float cos_theta_i, float relative_eta)
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

HIPRT_HOST_DEVICE HIPRT_INLINE float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t)
{
    return fresnel_dielectric(cos_theta_i, eta_t / eta_i);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float GGX_normal_distribution(float alpha, float NoH)
{
    //To avoid numerical instability when NoH basically == 1, i.e when the
    //material is a perfect mirror and the normal distribution function is a Dirac

    NoH = hippt::min(NoH, 0.999999f);
    float alpha2 = alpha * alpha;
    float NoH2 = NoH * NoH;
    float b = (NoH2 * (alpha2 - 1.0f) + 1.0f);
    return alpha2 / M_PI / (b * b);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float GTR2_anisotropic(const RendererMaterial& material, const float3& local_half_vector)
{
    float denom = (local_half_vector.x * local_half_vector.x) / (material.alpha_x * material.alpha_x) +
        (local_half_vector.y * local_half_vector.y) / (material.alpha_y * material.alpha_y) +
        (local_half_vector.z * local_half_vector.z);

    return 1.0f / (M_PI * material.alpha_x * material.alpha_y * denom * denom);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float G1_schlick_ggx(float k, float dot_prod)
{
    return dot_prod / (dot_prod * (1.0f - k) + k);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float GGX_smith_masking_shadowing(float roughness_squared, float NoV, float NoL)
{
    float k = roughness_squared / 2.0f;

    return G1_schlick_ggx(k, NoL) * G1_schlick_ggx(k, NoV);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float G1(float alpha_x, float alpha_y, const float3& local_direction)
{
    float ax = local_direction.x * alpha_x;
    float ay = local_direction.y * alpha_y;

    float lambda = (sqrt(1.0f + (ax * ax + ay * ay) / (local_direction.z * local_direction.z)) - 1.0f) * 0.5f;

    return 1.0f / (1.0f + lambda);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 GGXVNDF_sample(const float3& local_view_direction, float alpha_x, float alpha_y, Xorshift32Generator& random_number_generator)
{
    float r1 = random_number_generator();
    float r2 = random_number_generator();

    float3 Vh = hippt::normalize(float3{ alpha_x * local_view_direction.x, alpha_y * local_view_direction.y, local_view_direction.z });

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0f ? float3{-Vh.y, Vh.x, 0} / sqrt(lensq) : float3{ 1.0f, 0.0f, 0.0f };
    float3 T2 = hippt::cross(Vh, T1);

    float r = sqrt(r1);
    float phi = 2.0f * M_PI * r2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;

    float3 Nh = t1 * T1 + t2 * T2 + sqrt(hippt::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    return hippt::normalize(float3{alpha_x * Nh.x, alpha_y * Nh.y, hippt::max(0.0f, Nh.z)});
}

HIPRT_HOST_DEVICE HIPRT_INLINE float GTR1(float alpha_g, float local_halfway_z)
{
    float alpha_g_2 = alpha_g * alpha_g;

    float num = alpha_g_2 - 1.0f;
    float denom = M_PI * log(alpha_g_2) * (1.0f + (alpha_g_2 - 1.0f) * local_halfway_z * local_halfway_z);

    return num / denom;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float disney_clearcoat_masking_shadowing(const float3& direction)
{
    // The clearcoat geometry term always uses 0.25f roughness
    return G1(0.25f, 0.25f, direction);
}

#endif
