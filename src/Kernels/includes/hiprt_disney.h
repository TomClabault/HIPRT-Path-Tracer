#ifndef HIPRT_DISNEY_H
#define HIPRT_DISNEY_H

#include "Kernels/includes/HIPRT_common.h"
#include "Kernels/includes/hiprt_onb.h"
#include "Kernels/includes/hiprt_sampling.h"

// TODO return float
__device__ Color disney_schlick_weight(float f0, float abs_cos_angle)
{
    return Color(1.0f + (f0 - 1.0f) * pow(1.0f - abs_cos_angle, 5.0f));
}

__device__ Color disney_diffuse_eval(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, const hiprtFloat3& to_light_direction, float& pdf)
{
    hiprtFloat3 half_vector = normalize(to_light_direction + view_direction);

    float LoH = abs(dot(to_light_direction, half_vector));
    float NoL = abs(dot(surface_normal, to_light_direction));
    float NoV = abs(dot(surface_normal, view_direction));

    pdf = NoL / M_PI;

    Color diffuse_part;
    float diffuse_90 = 0.5f + 2.0f * material.roughness * LoH * LoH;
    diffuse_part = material.diffuse / M_PI * disney_schlick_weight(diffuse_90, NoL) * disney_schlick_weight(diffuse_90, NoV) * NoL;

    Color fake_subsurface_part;
    float subsurface_90 = material.roughness * LoH * LoH;
    fake_subsurface_part = 1.25f * material.diffuse / M_PI *
        (disney_schlick_weight(subsurface_90, NoL) * disney_schlick_weight(subsurface_90, NoV) * (1.0f / (NoL + NoV) - 0.5f) + Color(0.5f)) * NoL;

    return (1.0f - material.subsurface) * diffuse_part + material.subsurface * fake_subsurface_part;
}

__device__ Color disney_diffuse_sample(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, hiprtFloat3& output_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    output_direction = cosine_weighted_sample(surface_normal, pdf, random_number_generator);

    return disney_diffuse_eval(material, view_direction, surface_normal, output_direction, pdf);
}

__device__ float GTR2Aniso(float NDotH, float HDotX, float HDotY, float ax, float ay)
{
    float a = HDotX / ax;
    float b = HDotY / ay;
    float c = a * a + b * b + NDotH * NDotH;
    return 1.0 / (M_PI * ax * ay * c * c);
}

__device__ hiprtFloat3 SampleGTR2Aniso(float ax, float ay, float r1, float r2)
{
    float phi = r1 * 2.0f * M_PI;

    float sinPhi = ay * sin(phi);
    float cosPhi = ax * cos(phi);
    float tanTheta = sqrt(r2 / (1 - r2));

    return hiprtFloat3(tanTheta * cosPhi, tanTheta * sinPhi, 1.0);
}

__device__ float SmithG(float NDotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NDotV * NDotV;
    return (2.0 * NDotV) / (NDotV + sqrt(a + b - a * b));
}

__device__ float SmithGAniso(float NDotV, float VDotX, float VDotY, float ax, float ay)
{
    float a = VDotX * ax;
    float b = VDotY * ay;
    float c = NDotV;
    return (2.0 * NDotV) / (NDotV + sqrt(a * a + b * b + c * c));
}

__device__ Color disney_metallic_eval(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, const hiprtFloat3& to_light_direction, float& pdf)
{
    hiprtFloat3 half_vector = to_light_direction + view_direction;
    float length_half_vector = length(half_vector);
    if (length_half_vector == 0.0f)
        half_vector = view_direction;
    else
        half_vector = half_vector / length_half_vector;

    // Building the local shading frame
    hiprtFloat3 T, B;
    build_rotated_ONB(surface_normal, T, B, material.anisotropic_rotation);

    hiprtFloat3 local_half_vector = world_to_local_frame(T, B, surface_normal, half_vector);
    hiprtFloat3 local_view_direction = world_to_local_frame(T, B, surface_normal, view_direction);
    hiprtFloat3 local_to_light_direction = world_to_local_frame(T, B, surface_normal, to_light_direction);

    float NoV = abs(local_view_direction.z);
    float NoL = abs(local_to_light_direction.z);
    float HoL = abs(dot(half_vector, to_light_direction));

    Color F = fresnel_schlick(material.diffuse, HoL);
    float D = GGX_normal_distribution_anisotropic(material, local_half_vector);
    float G = GGX_masking_shadowing_anisotropic(material, local_view_direction, local_to_light_direction);

    pdf = G * D / (4.0 * NoV);
    return F * D * G / (4.0 * NoL * NoV);
}

__device__ Color disney_metallic_sample(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, hiprtFloat3& output_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    hiprtFloat3 local_view_direction = world_to_local_frame(surface_normal, view_direction);
    hiprtFloat3 microfacet_normal = GGXVNDF_sample(local_view_direction, material.alpha_x, material.alpha_y, random_number_generator);
    output_direction = reflect_ray(view_direction, local_to_world_frame(surface_normal, microfacet_normal));

    return disney_metallic_eval(material, view_direction, surface_normal, output_direction, pdf);
}

__device__ Color disney_eval(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, const hiprtFloat3& to_light_direction, float& pdf)
{
    return disney_metallic_eval(material, view_direction, surface_normal, to_light_direction, pdf);
}

__device__ Color disney_sample(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, hiprtFloat3& output_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    return disney_metallic_sample(material, view_direction, surface_normal, output_direction, pdf, random_number_generator);
}

#endif