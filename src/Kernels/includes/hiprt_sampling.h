#ifndef HIPRT_SAMPLING_H
#define HIPRT_SAMPLING_H

#include "Kernels/includes/HIPRT_common.h"

/**
 * Reflects a ray about a normal. This function requires that dot(ray_direction, surface_normal) > 0 i.e.
 * ray_direction and surface_normal are in the same hemisphere
 */
__device__ hiprtFloat3 reflect_ray(const hiprtFloat3& ray_direction, const hiprtFloat3& surface_normal)
{
    return -ray_direction + 2.0f * dot(ray_direction, surface_normal) * surface_normal;
}

/**
 * Reflects a ray about a normal. This function requires that dot(ray_direction, surface_normal) > 0 i.e.
 * ray_direction and surface_normal are in the same hemisphere
 */
__device__ bool refract_ray(const hiprtFloat3& ray_direction, const hiprtFloat3& surface_normal, hiprtFloat3& refract_direction, float relative_eta)
{
    float NoI = dot(ray_direction, surface_normal);

    float sin_theta_i_2 = 1.0f - NoI * NoI;
    float root_term = 1.0f - sin_theta_i_2 / (relative_eta * relative_eta);
    if (root_term < 0.0f)
        return false;

    float cos_theta_t = sqrt(root_term);
    refract_direction = -ray_direction / relative_eta + (NoI / relative_eta - cos_theta_t) * surface_normal;

    return true;
}

__device__ hiprtFloat3 cosine_weighted_sample(const hiprtFloat3& normal, float& pdf, Xorshift32Generator& random_number_generator)
{
    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();

    float sqrt_rand_2 = sqrt(rand_2);
    float phi = 2.0f * (float)M_PI * rand_1;
    float cos_theta = sqrt_rand_2;
    float sin_theta = sqrt(RT_MAX(0.0f, 1.0f - cos_theta * cos_theta));

    pdf = sqrt_rand_2 / (float)M_PI;

    //Generating a random direction in a local space with Z as the Up hiprtFloat3
    hiprtFloat3 random_dir_local_space = hiprtFloat3(cos(phi) * sin_theta, sin(phi) * sin_theta, sqrt_rand_2);
    return local_to_world_frame(normal, random_dir_local_space);
}

__device__ Color fresnel_schlick(Color F0, float NoV)
{
    return F0 + (Color(1.0f) - F0) * pow((1.0f - NoV), 5.0f);
}

__device__ float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t)
{
    // Computing cos_theta_t
    float sinThetaI = sqrt(1.0f - cos_theta_i * cos_theta_i);
    float sin_theta_t = eta_i / eta_t * sinThetaI;

    if (sin_theta_t >= 1.0f)
        // Total internal reflection, 0% refraction, all reflection
        return 1.0f;

    float cos_theta_t = sqrt(1.0f - sin_theta_t * sin_theta_t);
    float r_parallel = ((eta_t * cos_theta_i) - (eta_i * cos_theta_t)) / ((eta_t * cos_theta_i) + (eta_i * cos_theta_t));
    float r_perpendicular = ((eta_i * cos_theta_i) - (eta_t * cos_theta_t)) / ((eta_i * cos_theta_i) + (eta_t * cos_theta_t));
    return (r_parallel * r_parallel + r_perpendicular * r_perpendicular) / 2;
}

__device__ float GGX_normal_distribution(float alpha, float NoH)
{
    //To avoid numerical instability when NoH basically == 1, i.e when the
    //material is a perfect mirror and the normal distribution function is a Dirac

    NoH = RT_MIN(NoH, 0.999999f);
    float alpha2 = alpha * alpha;
    float NoH2 = NoH * NoH;
    float b = (NoH2 * (alpha2 - 1.0f) + 1.0f);
    return alpha2 * M_1_PI / (b * b);
}

__device__ float GGX_normal_distribution_anisotropic(const RendererMaterial& material, const hiprtFloat3& local_half_hiprtFloat3)
{
    float denom = (local_half_hiprtFloat3.x * local_half_hiprtFloat3.x) / (material.alpha_x * material.alpha_x) +
        (local_half_hiprtFloat3.y * local_half_hiprtFloat3.y) / (material.alpha_y * material.alpha_y) +
        (local_half_hiprtFloat3.z * local_half_hiprtFloat3.z);

    return 1.0f / (M_PI * material.alpha_x * material.alpha_y * denom * denom);
}

__device__ float G1_schlick_ggx(float k, float dot_prod)
{
    return dot_prod / (dot_prod * (1.0f - k) + k);
}

__device__ float GGX_smith_masking_shadowing(float roughness_squared, float NoV, float NoL)
{
    float k = roughness_squared / 2.0f;

    return G1_schlick_ggx(k, NoL) * G1_schlick_ggx(k, NoV);
}

__device__ float GGX_masking_shadowing_anisotropic_aux(const RendererMaterial& material, const hiprtFloat3& local_direction)
{
    float ax = local_direction.x * material.alpha_x;
    float ay = local_direction.y * material.alpha_y;

    float denom = (sqrt(1.0f + (ax * ax + ay * ay) / (local_direction.z * local_direction.z)) - 1.0f) * 0.5f;

    return 1.0f / (1.0f + denom);
}

__device__ float GGX_masking_shadowing_anisotropic(const RendererMaterial& material, const hiprtFloat3& local_view_direction, const hiprtFloat3& local_to_light_direction)
{
    return GGX_masking_shadowing_anisotropic_aux(material, local_view_direction) * GGX_masking_shadowing_anisotropic_aux(material, local_to_light_direction);
}

__device__ hiprtFloat3 GGXVNDF_sample(hiprtFloat3 local_view_direction, float alpha_x, float alpha_y, Xorshift32Generator& random_number_generator)
{
    float r1 = random_number_generator();
    float r2 = random_number_generator();

    hiprtFloat3 Vh = normalize(hiprtFloat3(alpha_x * local_view_direction.x, alpha_y * local_view_direction.y, local_view_direction.z));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    hiprtFloat3 T1 = lensq > 0.0f ? hiprtFloat3(-Vh.y, Vh.x, 0) * 1.0f / sqrt(lensq) : hiprtFloat3(1.0f, 0.0f, 0.0f);
    hiprtFloat3 T2 = cross(Vh, T1);

    float r = sqrt(r1);
    float phi = 2.0f * M_PI * r2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;

    hiprtFloat3 Nh = t1 * T1 + t2 * T2 + sqrt(RT_MAX(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    return normalize(hiprtFloat3(alpha_x * Nh.x, alpha_y * Nh.y, RT_MAX(0.0f, Nh.z)));
}

#endif
