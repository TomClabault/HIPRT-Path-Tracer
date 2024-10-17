/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_SAMPLING_H
#define DEVICE_SAMPLING_H

#include "Device/includes/ONB.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/KernelOptions.h"
#include "HostDeviceCommon/Material.h"
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
 * 'relative_eta' is eta_t / eta_i: transmitted media IOR / incident media IOR
 */
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

/**
 * Override of fresnel_dielectric with two separate eta
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t)
{
    return fresnel_dielectric(cos_theta_i, eta_t / eta_i);
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
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F fresnel_reflectance_from_ior(float eta_i, float eta_t, const float3& normal, const float3& local_to_light_direction)
{
    float HoL = hippt::clamp(1.0e-8f, 1.0f, hippt::dot(normal, local_to_light_direction));

    float R0_nume = eta_t - eta_i;
    float R0_denom = eta_t + eta_i;
    ColorRGB32F R0 = ColorRGB32F((R0_nume * R0_nume) / (R0_denom * R0_denom));

    return fresnel_schlick(R0, HoL);
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

HIPRT_HOST_DEVICE HIPRT_INLINE float GTR2_anisotropic(float alpha_x, float alpha_y, const float3& local_half_vector)
{
    float denom = (local_half_vector.x * local_half_vector.x) / (alpha_x * alpha_x) +
        (local_half_vector.y * local_half_vector.y) / (alpha_y * alpha_y) +
        (local_half_vector.z * local_half_vector.z);

    return 1.0f / (M_PI * alpha_x * alpha_y * denom * denom);
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
 * [Sampling Visible GGX Normals with Spherical Caps, Intel: Dupuy, Benyoub ; 2023]
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

HIPRT_HOST_DEVICE HIPRT_INLINE float3 GGX_aniso_sample(const float3& local_view_direction, float alpha_x, float alpha_y, Xorshift32Generator& random_number_generator)
{
#if GGXAnisotropicSampleFunction == GGX_NO_VNDF
#elif GGXAnisotropicSampleFunction == GGX_VNDF_SAMPLING
    return GGX_VNDF_sample(local_view_direction, alpha_x, alpha_y, random_number_generator);
#elif GGXAnisotropicSampleFunction == GGX_VNDF_SPHERICAL_CAPS
    return GGX_VNDF_spherical_caps_sample(local_view_direction, alpha_x, alpha_y, random_number_generator);
#elif GGXAnisotropicSampleFunction == GGX_VNDF_BOUNDED
#else
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE float GTR1(float alpha_g, float local_halfway_z)
{
    float alpha_g_2 = alpha_g * alpha_g;

    float num = alpha_g_2 - 1.0f;
    float denom = M_PI * log(alpha_g_2) * (1.0f + (alpha_g_2 - 1.0f) * local_halfway_z * local_halfway_z);

    return num / denom;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F microfacet_GTR2_eval(float material_roughness, const ColorRGB32F& F, const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_halfway_vector, float& out_pdf)
{
    float alpha_x;
    float alpha_y;
    SimplifiedRendererMaterial::get_alphas(material_roughness, 0.0f, alpha_x, alpha_y);

    float D = GTR2_anisotropic(alpha_x, alpha_y, local_halfway_vector);

    float G1V = G1(alpha_x, alpha_y, local_view_direction);
    float G1L = G1(alpha_x, alpha_y, local_to_light_direction);
    float G2 = G1V * G1L;

    // Maxing 1.0e-8f here to avoid zeros
    float NoV = hippt::max(1.0e-8f, hippt::abs(local_view_direction.z));
    float NoL = hippt::max(1.0e-8f, hippt::abs(local_to_light_direction.z));

    out_pdf = D * G1V / (4.0f * NoV);
    return F * D * G2 / (4.0f * NoL * NoV);
}

/**
 * Evaluation of a torrance-sparrow model with the GTR2 normal distribution
 * and G1 masking-shadowing
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F microfacet_GTR2_eval(float material_roughness, float material_ior, float incident_ior, const float3& local_view_direction, const float3& local_to_light_direction, const float3& local_halfway_vector, float& out_pdf)
{
    float HoL = hippt::clamp(1.0e-8f, 1.0f, hippt::dot(local_halfway_vector, local_to_light_direction));

    float R0_nume = material_ior - incident_ior;
    float R0_denom = material_ior + incident_ior;
    ColorRGB32F R0 = ColorRGB32F((R0_nume * R0_nume) / (R0_denom * R0_denom));
    ColorRGB32F F = fresnel_schlick(R0, HoL);

    return microfacet_GTR2_eval(material_roughness, F, local_view_direction, local_to_light_direction, local_halfway_vector, out_pdf);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 microfacet_GTR2_sample(float roughness, float anisotropy, const float3& local_view_direction, Xorshift32Generator& random_number_generator)
{
    // The view direction can sometimes be below the shading normal hemisphere
    // because of normal mapping / smooth normals
    int below_normal = (local_view_direction.z < 0) ? -1 : 1;
    float alpha_x, alpha_y;
    SimplifiedRendererMaterial::get_alphas(roughness, anisotropy, alpha_x, alpha_y);

    float3 microfacet_normal = GGX_aniso_sample(local_view_direction * below_normal, alpha_x, alpha_y, random_number_generator);
    float3 sampled_direction = reflect_ray(local_view_direction, microfacet_normal * below_normal);

    // Should already be normalized but float imprecisions...
    return hippt::normalize(sampled_direction);
}

#endif
