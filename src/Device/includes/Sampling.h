/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_SAMPLING_H
#define DEVICE_SAMPLING_H

#include "Device/includes/Fresnel.h"
#include "Device/includes/ONB.h"
#include "Device/includes/Texture.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/Material/Material.h"
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
    float x = r_sqrt_v * cos(M_TWO_PI * uv.x);
    float y = r_sqrt_v* sin(M_TWO_PI * uv.x);

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
 * 
 * relative_eta here must be eta_t / eta_i
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

    float theta = M_TWO_PI * rand_1;

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

    float phi = M_TWO_PI * r1;
    float cos_theta = sqrt(r2);
    float sin_theta = sqrt(1 - cos_theta * cos_theta);

    return hippt::normalize(make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta));
}

#endif
