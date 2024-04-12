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
 * [5] [PBRT v3 Source Code] https://github.com/mmp/pbrt-v3
 * [6] [PBRT v4 Source Code] https://github.com/mmp/pbrt-v4
 * [7] [Blender's Cycles Source Code] https://github.com/blender/cycles
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
    // Lambertian base_color
    //diffuse_part = material.base_color / M_PI;
    // Disney base_color
    diffuse_part = material.base_color / M_PI * disney_schlick_weight(diffuse_90, NoL) * disney_schlick_weight(diffuse_90, NoV) * NoL;
    // Oren nayar base_color
    //diffuse_part = oren_nayar_eval(material, view_direction, surface_normal, to_light_direction);

    Color fake_subsurface_part = Color(0.0f);
    if (material.subsurface > 0)
    {
        float subsurface_90 = material.roughness * LoH * LoH;
        fake_subsurface_part = 1.25f * material.base_color / M_PI *
            (disney_schlick_weight(subsurface_90, NoL) * disney_schlick_weight(subsurface_90, NoV) * (1.0f / (NoL + NoV) - 0.5f) + 0.5f) * NoL;
    }

    return (1.0f - material.subsurface) * diffuse_part + material.subsurface * fake_subsurface_part;
}

__device__ hiprtFloat3 disney_diffuse_sample(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, Xorshift32Generator& random_number_generator)
{
    float trash_pdf;
    hiprtFloat3 sampled_direction = cosine_weighted_sample(surface_normal, trash_pdf, random_number_generator);

    return sampled_direction;
}

__device__ Color disney_metallic_fresnel(const RendererMaterial& material, const hiprtFloat3& local_half_vector, const hiprtFloat3& local_to_light_direction)
{
    // The summary of what is below is the following:
    //
    // If the material is 100% metallic, then the fresnel term color is going to be 
    // the base_color of the material i.e. typical conductor response.
    // 
    // If the material is 0% metallic, then the fresnel term color is going to be
    // material.specular_color modulated by the material.specular_tint coefficient (which blends 
    // between white and material.specular_color) and the material.specular coefficient which
    // dictates whether we have a specular at all
    Color Ks = Color(1.0f - material.specular_tint) + material.specular_tint * material.specular_color;
    float R0 = ((material.ior - 1.0f) * (material.ior - 1.0f)) / ((material.ior + 1.0f) * (material.ior + 1.0f));
    Color C0 = material.specular * R0 * (1.0f - material.metallic) * Ks + material.metallic * material.base_color;

    return C0 + (Color(1.0f) - C0) * pow(1.0f - dot(local_half_vector, local_to_light_direction), 5.0f);
}

__device__ Color disney_metallic_eval(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, const hiprtFloat3& to_light_direction, Color F, float& pdf)
{
    // Building the local shading frame
    hiprtFloat3 T, B;
    build_rotated_ONB(surface_normal, T, B, material.anisotropic_rotation * M_PI);

    hiprtFloat3 local_view_direction = world_to_local_frame(T, B, surface_normal, view_direction);
    hiprtFloat3 local_to_light_direction = world_to_local_frame(T, B, surface_normal, to_light_direction);
    hiprtFloat3 local_half_vector = normalize(local_to_light_direction + local_view_direction);

    float NoV = abs(local_view_direction.z);
    float NoL = abs(local_to_light_direction.z);
    float HoL = abs(dot(local_half_vector, local_to_light_direction));

    // TODO remove
    //{
    //    // F = (-2.0f, -2.0f, -2.0f) is the default argument when the overload without the 'Color F' argument
    //    // of disney_metallic_eval() was called. Thus, if no F was passed, we're computing it here.
    //    // Otherwise, we're going to use the given one
    //    if (F.r == -2.0f)
    //        F = fresnel_schlick(material.base_color, NoL);
    //}
    
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

    float num = material.clearcoat_ior - 1.0f;
    float denom = material.clearcoat_ior + 1.0f;
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

// TOOD can use local_view dir and light_dir here
__device__ Color disney_glass_eval(const RendererMaterial& material, const hiprtFloat3& view_direction, hiprtFloat3 surface_normal, const hiprtFloat3& to_light_direction, float& pdf)
{
    float start_NoV = dot(surface_normal, view_direction);
    if (start_NoV < 0.0f)
        surface_normal = -surface_normal;

    hiprtFloat3 T, B;
    build_rotated_ONB(surface_normal, T, B, material.anisotropic_rotation * M_PI);

    hiprtFloat3 local_to_light_direction = world_to_local_frame(T, B, surface_normal, to_light_direction);
    hiprtFloat3 local_view_direction = world_to_local_frame(T, B, surface_normal, view_direction);

    float NoV = local_view_direction.z;
    float NoL = local_to_light_direction.z;

    // We're in the case of reflection if the view direction and the bounced ray (light direction) are in the same hemisphere
    bool reflecting = NoL * NoV > 0;

    // Relative eta = eta_t / eta_i and we're assuming here that the eta of the incident light is air, 1.0f
    float relative_eta = material.ior;

    // Computing the generalized (that takes refraction into account) half vector
    hiprtFloat3 local_half_vector;
    if (reflecting)
        local_half_vector = local_to_light_direction + local_view_direction;
    else
    {
        // We want relative eta to always be eta_transmitted / eta_incident
        // so if we're refracting OUT of the surface, we're transmitting into
        // the air which has an eta of 1.0f so transmitted / incident
        // = 1.0f / material.ior (which relative_eta is equal to here)
        relative_eta = start_NoV > 0 ? material.ior : (1.0f / relative_eta);

        // We need to take the relative_eta into account when refracting to compute
        // the half vector (this is the "generalized" part of the half vector computation)
        local_half_vector = local_to_light_direction * relative_eta + local_view_direction;
    }

    local_half_vector = normalize(local_half_vector);
    if (local_half_vector.z < 0.0f)
        // Because the rest of the function we're going to compute here assume
        // that the microfacet normal is in the same hemisphere as the surface
        // normal, we're going to flip it if needed
        local_half_vector = -local_half_vector;

    float HoL = dot(local_to_light_direction, local_half_vector);
    float HoV = dot(local_view_direction, local_half_vector);

    // TODO to test removing that
    if (HoL * NoL < 0.0f || HoV * NoV < 0.0f)
        // Backfacing microfacets
        return Color(0.0f);

    Color color;
    float F = fresnel_dielectric(dot(local_view_direction, local_half_vector), relative_eta);
    if (reflecting)
    {
        color = disney_metallic_eval(material, view_direction, surface_normal, to_light_direction, Color(F), pdf);

        // Scaling the PDF by the probability of being here (reflection of the ray and not transmission)
        pdf *= F;
    }
    else
    {
        float dot_prod = HoL + HoV / relative_eta;
        float dot_prod2 = dot_prod * dot_prod;
        float denom = dot_prod2 * NoL * NoV;
        float D = GTR2_anisotropic(material, local_half_vector);
        float G1_V = G1(material.alpha_x, material.alpha_y, local_view_direction);
        float G = G1_V * G1(material.alpha_x, material.alpha_y, local_to_light_direction);

        float dwm_dwi = abs(HoL) / dot_prod2;
        float D_pdf = G1_V / abs(NoV) * D * abs(HoV);
        pdf = dwm_dwi * D_pdf * (1.0f - F);

        color = sqrt(material.base_color) * D * (1 - F) * G * abs(HoL * HoV / denom);
    }

    return color;
}

__device__ hiprtFloat3 disney_glass_sample(const RendererMaterial& material, const hiprtFloat3& view_direction, hiprtFloat3 surface_normal, Xorshift32Generator& random_number_generator)
{
    hiprtFloat3 T, B;
    build_rotated_ONB(surface_normal, T, B, material.anisotropic_rotation * M_PI);

    float relative_eta = material.ior;
    if (dot(surface_normal, view_direction) < 0)
    {
        // We want the surface normal in the same hemisphere as 
        // the view direction for the rest of the calculations
        surface_normal = -surface_normal;
        relative_eta = 1.0f / relative_eta;
    }

    hiprtFloat3 local_view_direction = world_to_local_frame(T, B, surface_normal, view_direction);
    hiprtFloat3 microfacet_normal = GGXVNDF_sample(local_view_direction, material.alpha_x, material.alpha_y, random_number_generator);
    if (microfacet_normal.z < 0)
        microfacet_normal = -microfacet_normal;

    float F = fresnel_dielectric(dot(local_view_direction, microfacet_normal), relative_eta);
    float rand_1 = random_number_generator();

    hiprtFloat3 sampled_direction;
    if (rand_1 < F)
    {
        // Reflection
        sampled_direction = reflect_ray(local_view_direction, microfacet_normal);
    }
    else
    {
        // Refraction

        if (dot(microfacet_normal, local_view_direction) < 0.0f)
            // For the refraction operation that follows, we want the direction to refract (the view
            // direction here) to be in the same hemisphere as the normal (the microfacet normal here)
            // so we're flipping the microfacet normal in case it wasn't in the same hemisphere as
            // the view direction
            // Relative_eta as already been flipped above in the code
            microfacet_normal = -microfacet_normal;

        if (!refract_ray(local_view_direction, microfacet_normal, sampled_direction, relative_eta))
            return hiprtFloat3(-2.0f, -2.0f, -2.0f);
    }

    return local_to_world_frame(T, B, surface_normal, sampled_direction);
}

__device__ Color disney_sheen_eval(const RendererMaterial& material, const hiprtFloat3& view_direction, hiprtFloat3 surface_normal, const hiprtFloat3& to_light_direction, float& pdf)
{
    Color sheen_color = Color(1.0f - material.sheen_tint) + material.sheen_color * material.sheen_tint;

    float base_color_luminance = material.base_color.luminance();
    Color specular_color = base_color_luminance > 0 ? material.base_color / base_color_luminance : Color(1.0f);

    hiprtFloat3 half_vector = normalize(view_direction + to_light_direction);

    float NoL = dot(surface_normal, to_light_direction);
    pdf = NoL / M_PI;

    // clamping operation here on the dot product because it has happened in the past
    // that the dot product was 1.000012f due to floating point precision errors, leading
    // to 1.0f - dot being negative and boom, the BRDF returns a negative color
    return sheen_color * pow(clamp(0.0f, 1.0f, 1.0f - dot(half_vector, to_light_direction)), 5.0f) * NoL;
}

__device__ hiprtFloat3 disney_sheen_sample(const RendererMaterial& material, const hiprtFloat3& view_direction, hiprtFloat3 surface_normal, Xorshift32Generator& random_number_generator)
{
    float trash_pdf;
    return cosine_weighted_sample(surface_normal, trash_pdf, random_number_generator);
}

__device__ Color disney_eval(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& shading_normal, const hiprtFloat3& to_light_direction, float& pdf)
{
    pdf = 0.0f;

    hiprtFloat3 T, B;
    build_ONB(shading_normal, T, B);

    hiprtFloat3 local_view_direction = world_to_local_frame(T, B, shading_normal, view_direction);
    hiprtFloat3 local_to_light_direction = world_to_local_frame(T, B, shading_normal, to_light_direction);
    hiprtFloat3 local_half_vector = normalize(local_view_direction + local_to_light_direction);

    Color final_color = Color(0.0f);
    // We're only going to compute the diffuse, metallic, clearcoat and sheen lobes if we're 
    // outside of the object. Said otherwise, only the glass lobe is considered while traveling 
    // inside the object
    bool outside_object = dot(view_direction, shading_normal) > 0;
    float tmp_pdf = 0.0f, tmp_weight;

    //// Diffuse
    //tmp_weight = (1.0f - material.metallic) * (1.0f - material.specular_transmission);
    //final_color += tmp_weight > 0 && outside_object ? disney_diffuse_eval(material, view_direction, shading_normal, to_light_direction, tmp_pdf) : Color(0.0f);
    //pdf += tmp_pdf * tmp_weight;

    //// Metallic
    //// Computing a custom fresnel term based on the material specular, specular tint, ... coefficients
    //Color metallic_fresnel = disney_metallic_fresnel(material, local_half_vector, local_to_light_direction);
    //tmp_weight = (1.0f - material.specular_transmission * (1.0f - material.metallic));
    //final_color += tmp_weight > 0 && outside_object ? disney_metallic_eval(material, view_direction, shading_normal, to_light_direction, metallic_fresnel, tmp_pdf) : Color(0.0f);
    //pdf += tmp_pdf * tmp_weight;

    //// Clearcoat
    //tmp_weight = 0.25f * material.clearcoat;
    //final_color += tmp_weight > 0 && outside_object ? disney_clearcoat_eval(material, view_direction, shading_normal, to_light_direction, tmp_pdf) : Color(0.0f);
    //pdf += tmp_pdf * tmp_weight;

    // Glass
    tmp_weight = (1.0f - material.metallic);
    final_color += tmp_weight > 0 ? disney_glass_eval(material, view_direction, shading_normal, to_light_direction, tmp_pdf) : Color(0.0f);
    pdf += tmp_pdf * tmp_weight;

    //// Sheen
    //tmp_weight = (1.0f - material.metallic) * material.sheen;
    //final_color += tmp_weight > 0 && outside_object ? disney_sheen_eval(material, view_direction, shading_normal, to_light_direction, tmp_pdf) : Color(0.0f);
    //pdf += tmp_pdf * tmp_weight;

    return final_color;
}

__device__ Color disney_sample(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& shading_normal, const hiprtFloat3& geometric_normal, hiprtFloat3& output_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    pdf = 0.0f;

    hiprtFloat3 normal = shading_normal;

    float diffuse_weight = 0.0f;//(1.0f - material.metallic) * (1.0f - material.specular_transmission);
    float metal_weight = 0.0f;//(1.0f - material.specular_transmission * (1.0f - material.metallic));
    float clearcoat_weight = 0.0f;// 0.25f * material.clearcoat;
    float glass_weight = (1.0f - material.metallic)* material.specular_transmission;

    float normalize_factor = 1.0f / (diffuse_weight + metal_weight + clearcoat_weight + glass_weight);
    diffuse_weight *= normalize_factor;
    metal_weight *= normalize_factor;
    clearcoat_weight *= normalize_factor;
    glass_weight *= normalize_factor;

    float cdf[4];
    cdf[0] = diffuse_weight;
    cdf[1] = cdf[0] + metal_weight;
    cdf[2] = cdf[1] + clearcoat_weight;
    cdf[3] = cdf[2] + glass_weight;

    float rand_1 = random_number_generator();
    if (rand_1 < cdf[3])
    {
        // This means that we're going to importance sample a lobe that is not glass
        // so we're going to "pre process" the normal to avoid black fringes
        
        
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
            normal = reflect_ray(shading_normal, geometric_normal);

        if (rand_1 < cdf[0])
        {
            output_direction = disney_diffuse_sample(material, view_direction, normal, random_number_generator);
        }
        else if (rand_1 < cdf[1])
        {
            output_direction = disney_metallic_sample(material, view_direction, normal, random_number_generator);
        }
        else if (rand_1 < cdf[2])
        {
            output_direction = disney_clearcoat_sample(material, view_direction, normal, random_number_generator);
        }

        if (dot(output_direction, shading_normal) < 0)
        {
            // It can happen that the light direction sampled is below the surface. 
            // We return 0.0 in this case
            return Color(0.0f);
        }
    }
    else
    {
        // The last lobe is glass

        float dot_shading = dot(view_direction, shading_normal);
        float dot_geometric = dot(view_direction, geometric_normal);
        if (dot_shading * dot_geometric < 0)
        {
            // The view direction is below the surface normal because of normal mapping / smooth normals.
            // We're going to flip the normal for the same reason as explained above to avoid black fringes
            // the reason we're also checking for the dot product with the geometric normal here
            // is because in the case of the glass lobe of the BRDF, we could be legitimately having
            // the dot product between the shading normal and the view direction be negative when we're
            // currently travelling inside the surface. To make sure that we're in the case of the black fringes
            // caused by normal mapping and microfacet BRDFs, we're also checking with the geometric normal.
            // If the view direction isn't below the geometric normal but is below the shading normal, this
            // indicates that we're in the case of the black fringes and we can flip the normal
            // If both dot products are negative, this means that we're travelling inside the surface
            // and we shouldn't flip the normal
            normal = reflect_ray(shading_normal, geometric_normal);
        }

        output_direction = disney_glass_sample(material, view_direction, normal, random_number_generator);
    }

    return disney_eval(material, view_direction, normal, output_direction, pdf);
}

#endif