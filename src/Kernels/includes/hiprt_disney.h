#ifndef HIPRT_DISNEY_H
#define HIPRT_DISNEY_H

#include "Kernels/includes/HIPRT_common.h"
#include "Kernels/includes/hiprt_onb.h"
#include "Kernels/includes/hiprt_oren_nayar.h"
#include "Kernels/includes/hiprt_sampling.h"

/* References:
 * [1] [CSE 272 University of California San Diego - Disney BSDF Homework] https://cseweb.ucsd.edu/~tzli/cse272/wi2024/homework1.pdf
 * [2] [GLSL Path Tracer implementation by knightcrawler25] https://github.com/knightcrawler25/GLSL-PathTracer
 * [3] [SIGGRAPH 2012 Course] https://blog.selfshadow.com/publications/s2012-shading-course/#course_content
 * [4] [SIGGRAPH 2015 Course] https://blog.selfshadow.com/publications/s2015-shading-course/#course_content
 * [5] [PBRT v3 Source Code] https://github.com/mmp/pbrt-v3/tree/master
 * [6] [Blender's Cycles Source Code] https://github.com/blender/cycles
 */

__device__ float disney_schlick_weight(float f0, float abs_cos_angle)
{
    return 1.0f + (f0 - 1.0f) * pow(1.0f - abs_cos_angle, 5.0f);
}

__device__ Color disney_diffuse_eval(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, const hiprtFloat3& to_light_direction, float& pdf)
{
    hiprtFloat3 half_vector = normalize(to_light_direction + view_direction);

    float LoH = clamp(0.0f, 1.0f, abs(dot(to_light_direction, half_vector)));
    float NoL = clamp(0.0f, 1.0f, abs(dot(surface_normal, to_light_direction)));
    float NoV = clamp(0.0f, 1.0f, abs(dot(surface_normal, view_direction)));

    pdf = NoL / M_PI;

    Color diffuse_part;
    float diffuse_90 = 0.5f + 2.0f * material.roughness * LoH * LoH;
    // Lambertian diffuse
    //diffuse_part = material.diffuse / M_PI;
    // Disney diffuse
    //diffuse_part = material.diffuse / M_PI * disney_schlick_weight(diffuse_90, NoL) * disney_schlick_weight(diffuse_90, NoV) * NoL;
    // Oren nayar diffuse
    diffuse_part = oren_nayar_eval(material, view_direction, surface_normal, to_light_direction);

    Color fake_subsurface_part;
    float subsurface_90 = material.roughness * LoH * LoH;
    fake_subsurface_part = 1.25f * material.diffuse / M_PI *
        (disney_schlick_weight(subsurface_90, NoL) * disney_schlick_weight(subsurface_90, NoV) * (1.0f / (NoL + NoV) - 0.5f) + 0.5f) * NoL;

    return (1.0f - material.subsurface) * diffuse_part + material.subsurface * fake_subsurface_part;
}

__device__ hiprtFloat3 disney_diffuse_sample(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, Xorshift32Generator& random_number_generator)
{
    float trash_pdf;
    hiprtFloat3 sampled_direction = cosine_weighted_sample(surface_normal, trash_pdf, random_number_generator);

    return sampled_direction;
}

__device__ Color disney_metallic_eval(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, const hiprtFloat3& to_light_direction, float& pdf)
{
    // Building the local shading frame
    hiprtFloat3 T, B;
    build_rotated_ONB(surface_normal, T, B, material.anisotropic_rotation);

    hiprtFloat3 local_view_direction = world_to_local_frame(T, B, surface_normal, view_direction);
    hiprtFloat3 local_to_light_direction = world_to_local_frame(T, B, surface_normal, to_light_direction);
    hiprtFloat3 local_half_vector = normalize(local_to_light_direction + local_view_direction);

    float NoV = abs(local_view_direction.z);
    float NoL = abs(local_to_light_direction.z);
    float HoL = abs(dot(local_half_vector, local_to_light_direction));

    Color F = fresnel_schlick(material.diffuse, NoL);
    
    float D = GTR2_anisotropic(material, local_half_vector);
    float G1_V = G1(material.alpha_x, material.alpha_y, local_view_direction);
    float G1_L = G1(material.alpha_x, material.alpha_y, local_to_light_direction);
    float G = G1_V * G1_L;

    pdf = D * G1_V / (4.0f * NoV);
    return F * D * G / (4.0 * NoL * NoV);
}

__device__ hiprtFloat3 disney_metallic_sample(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, Xorshift32Generator& random_number_generator)
{
	hiprtFloat3 local_view_direction = world_to_local_frame(surface_normal, view_direction);

	// The view direction can sometimes be below the shading normal hemisphere
	// because of normal mapping
    int below_normal = (local_view_direction.z < 0) ? -1 : 1;
	hiprtFloat3 microfacet_normal = GGXVNDF_sample(local_view_direction * below_normal, material.alpha_x, material.alpha_y, random_number_generator);
	hiprtFloat3 sampled_direction = reflect_ray(view_direction, local_to_world_frame(surface_normal, microfacet_normal * below_normal));

    return sampled_direction;
}

__device__ Color disney_clearcoat_eval(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, const hiprtFloat3& to_light_direction, float& pdf)
{
    hiprtFloat3 T, B;
    build_ONB(surface_normal, T, B);

    hiprtFloat3 local_view_direction = world_to_local_frame(T, B, surface_normal, view_direction);
    hiprtFloat3 local_to_light_direction = world_to_local_frame(T, B, surface_normal, to_light_direction);
    hiprtFloat3 local_halfway_vector = normalize(local_view_direction + local_to_light_direction);

    if (local_view_direction.z * local_to_light_direction.z < 0)
        return Color(0.0f);

    float num = material.clearcoatIOR - 1.0f;
    float denom = material.clearcoatIOR + 1.0f;
    Color R0 = Color((num * num) / (denom * denom));

    float HoV = dot(local_halfway_vector, local_to_light_direction);
    float clearcoat_gloss = 1.0f - material.clearcoat_roughness;
    float alpha_g = (1.0f - clearcoat_gloss) * 0.1f + clearcoat_gloss * 0.001f;

    Color F_clearcoat = fresnel_schlick(R0, HoV);
    float D_clearcoat = GTR1(alpha_g, abs(local_halfway_vector.z));
    float G_clearcoat = disney_clearcoat_masking_shadowing(local_view_direction) * disney_clearcoat_masking_shadowing(local_to_light_direction);

    pdf = D_clearcoat * abs(local_halfway_vector.z) / (4.0f * dot(local_halfway_vector, local_to_light_direction));
    return F_clearcoat * D_clearcoat * G_clearcoat / (4.0f * local_view_direction.z);
}

__device__ hiprtFloat3 disney_clearcoat_sample(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, Xorshift32Generator& random_number_generator)
{
    float clearcoat_gloss = 1.0f - material.clearcoat_roughness;
    float alpha_g = (1.0f - clearcoat_gloss) * 0.1f + clearcoat_gloss * 0.001f;
    float alpha_g_2 = alpha_g * alpha_g;

    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();

    float cos_theta = sqrt((1.0f - pow(alpha_g_2, 1.0f - rand_1)) / (1.0f - alpha_g_2));
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    float phi = 2.0f * M_PI * rand_2;
    float cos_phi = cos(phi);
    float sin_phi = sqrt(1.0f - cos_phi * cos_phi);

    hiprtFloat3 microfacet_normal = normalize(hiprtFloat3{sin_theta * cos_phi, sin_theta * sin_phi, cos_theta});
    hiprtFloat3 sampled_direction = reflect_ray(view_direction, local_to_world_frame(surface_normal, microfacet_normal));

    return sampled_direction;
}

__device__ Color disney_eval(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& shading_normal, const hiprtFloat3& to_light_direction, float& pdf)
{
    pdf = 0.0f;

    //return disney_diffuse_eval(material, view_direction, shading_normal, to_light_direction, pdf);
    //return disney_metallic_eval(material, view_direction, shading_normal, to_light_direction, pdf);
    return disney_clearcoat_eval(material, view_direction, shading_normal, to_light_direction, pdf);
}

__device__ Color disney_sample(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& shading_normal, const hiprtFloat3& geometric_normal, hiprtFloat3& output_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    pdf = 0.0f;

    hiprtFloat3 normal = shading_normal;

    // Checking whether the view direction is below the upprt hemisphere around the shading
    // normal or not. This may be the case mainly due to normal mapping / smooth vertex normals. 
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
    if (dot(view_direction, shading_normal) < 0)
    {
        normal = reflect_ray(shading_normal, geometric_normal);

        // In some cases, flipping the normal isn't enough to bring
        // the view direction in the upper hemisphere around the shading normal
        // Giving up and returning 0.0f in this case
        if (dot(view_direction, normal) < 0)
            return Color(0.0f);
    }

    //output_direction = disney_diffuse_sample(material, view_direction, normal, random_number_generator);
    //output_direction = disney_metallic_sample(material, view_direction, normal, random_number_generator);
    output_direction = disney_clearcoat_sample(material, view_direction, normal, random_number_generator);

    if (dot(output_direction, shading_normal) < 0)
    {
        // It can happen that the light direction sampled is below the surface. 
        // We return 0.0 in this case
        return Color(0.0f);
    }
    
    //return disney_diffuse_eval(material, view_direction, normal, output_direction, pdf);
    //return disney_metallic_eval(material, view_direction, normal, output_direction, pdf);
    return disney_clearcoat_eval(material, view_direction, normal, output_direction, pdf);
}

#endif