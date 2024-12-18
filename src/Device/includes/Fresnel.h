/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_FRESNEL_H
#define DEVICE_FRESNEL_H

#include "HostDeviceCommon/Color.h"

HIPRT_HOST_DEVICE HIPRT_INLINE float F0_from_eta(float eta_t, float eta_i)
{
    float nume_F0 = (eta_t - eta_i);
    float denom_F0 = (eta_t + eta_i);
    float F0 = (nume_F0 * nume_F0) / (denom_F0 * denom_F0);

    return F0;
}

/**
 * relative_eta here is eta_t / eta_i
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float F0_from_eta_t_and_relative_ior(float eta_t, float relative_eta)
{
    return F0_from_eta(eta_t, /* eta_i */ eta_t / relative_eta);
}

/**
 * Schlick's approximation for dielectric fresnel reflectance
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F fresnel_schlick(ColorRGB32F F0, float angle)
{
    return F0 + (ColorRGB32F(1.0f) - F0) * hippt::pow_5(1.0f - angle);
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
 *      ColorRGB32F F0 = <compute F0 from etas>
 *      return schlick(F0, NoL)
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F fresnel_schlick_from_ior(float eta_i, float eta_t, float cos_theta_i)
{
    float F0 = F0_from_eta(eta_t, eta_i);

    return fresnel_schlick(ColorRGB32F(F0), cos_theta_i);
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
 * Reference:
 *
 * [1] [Generalization of Adobe’s Fresnel Model, Hoffman, 2023]
 * [2] [Adobe Standard Material, Technical Documentation, Kutz, Hasan, Edmondson]
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F adobe_f82_tint_fresnel(const ColorRGB32F& F0, const ColorRGB32F& F82, const ColorRGB32F& F90, float F90_falloff_exponent, float cos_theta)
{
    ColorRGB32F base_term = F0 + (F90 - F0) * pow(1.0f - cos_theta, F90_falloff_exponent);
    float lazanyi_correction = cos_theta * hippt::pow_6(1.0f - cos_theta);

    // cos_theta_max for beta exponent = 6 in the lazanyi correction term
    constexpr float cos_theta_max = 1.0f / 7.0f;
    constexpr float denom_a = cos_theta_max * hippt::pow_6(1.0f - cos_theta_max);

    ColorRGB32F nume_a = (F0 + (F90 - F0) * pow(1.0f - cos_theta_max, F90_falloff_exponent)) * (ColorRGB32F(1.0f) - F82);
    ColorRGB32F a = nume_a / denom_a;

    ColorRGB32F F = base_term - a * lazanyi_correction;
    F.clamp(0.0f, 1.0f);

    return F;
}

#endif
