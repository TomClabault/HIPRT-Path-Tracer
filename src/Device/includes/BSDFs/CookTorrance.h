/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_COOK_TORRANCE_H
#define DEVICE_COOK_TORRANCE_H

#include "HostDeviceCommon/Math.h"
#include "HostDeviceCommon/Material.h"
#include "Device/includes/Sampling.h"

HIPRT_HOST_DEVICE HIPRT_INLINE float GGX_normal_distribution(float alpha, float NoH)
{
    //To avoid numerical instability when NoH basically == 1, i.e when the
    //material is a perfect mirror and the normal distribution function is a Dirac

    NoH = hippt::min(NoH, 0.999999f);
    float alpha2 = alpha * alpha;
    float NoH2 = NoH * NoH;
    float b = (NoH2 * (alpha2 - 1.0f) + 1.0f);
    return alpha2 * M_INV_PI / (b * b);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float G1_schlick_ggx(float k, float dot_prod)
{
    return dot_prod / (dot_prod * (1.0f - k) + k);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float GGX_smith_masking_shadowing(float roughness_squared, float NoV, float NoL)
{
    float k = roughness_squared * 0.5f;

    return G1_schlick_ggx(k, NoL) * G1_schlick_ggx(k, NoV);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float cook_torrance_brdf_pdf(const SimplifiedRendererMaterial& material, const float3& view_direction, const float3& to_light_direction, const float3& surface_normal)
{
    float3 microfacet_normal = hippt::normalize(view_direction + to_light_direction);

    float alpha = material.roughness * material.roughness;

    float VoH = hippt::max(0.0f, hippt::dot(view_direction, microfacet_normal));
    float NoH = hippt::max(0.0f, hippt::dot(surface_normal, microfacet_normal));
    float D = GGX_normal_distribution(alpha, NoH);

    return D * NoH / (4.0f * VoH);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F cook_torrance_brdf(const SimplifiedRendererMaterial& material, const float3& to_light_direction, const float3& view_direction, const float3& surface_normal)
{
    ColorRGB32F brdf_color = ColorRGB32F(0.0f, 0.0f, 0.0f);
    ColorRGB32F base_color = material.base_color;

    float3 halfway_vector = hippt::normalize(view_direction + to_light_direction);

    float NoV = hippt::max(0.0f, hippt::dot(surface_normal, view_direction));
    float NoL = hippt::max(0.0f, hippt::dot(surface_normal, to_light_direction));
    float NoH = hippt::max(0.0f, hippt::dot(surface_normal, halfway_vector));
    float VoH = hippt::max(0.0f, hippt::dot(halfway_vector, view_direction));

    if (NoV > 0.0f && NoL > 0.0f && NoH > 0.0f)
    {
        float metallic = material.metallic;
        float roughness = material.roughness;

        float alpha = roughness * roughness;

        ////////// Cook Torrance BRDF //////////
        ColorRGB32F F;
        float D, G;

        //F0 = 0.04 for dielectrics, 1.0 for metals (approximation)
        ColorRGB32F F0 = ColorRGB32F(0.04f * (1.0f - metallic)) + metallic * base_color;

        //GGX Distribution function
        F = fresnel_schlick(F0, VoH);
        D = GGX_normal_distribution(alpha, NoH);
        G = GGX_smith_masking_shadowing(alpha, NoV, NoL);

        ColorRGB32F kD = ColorRGB32F(1.0f - metallic); //Metals do not have a base_color part
        kD = kD * (ColorRGB32F(1.0f) - F);//Only the transmitted light is diffused

        ColorRGB32F diffuse_part = kD * base_color * M_INV_PI;
        ColorRGB32F specular_part = (F * D * G) / (4.0f * NoV * NoL);

        brdf_color = diffuse_part + specular_part;
    }

    return brdf_color;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F cook_torrance_brdf_importance_sample(const SimplifiedRendererMaterial& material, const float3& view_direction, const float3& surface_normal, float3& output_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    pdf = 0.0f;

    float metallic = material.metallic;
    float roughness = material.roughness;
    float alpha = roughness * roughness;

    float rand1 = random_number_generator();
    float rand2 = random_number_generator();

    float phi = M_TWO_PI * rand1;
    float theta = acos((1.0f - rand2) / (rand2 * (alpha * alpha - 1.0f) + 1.0f));
    float sin_theta = sin(theta);

    // The microfacet normal is sampled in its local space, we'll have to bring it to the space
    // around the surface normal
    float3 microfacet_normal_local_space = make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos(theta));
    float3 microfacet_normal = local_to_world_frame(surface_normal, microfacet_normal_local_space);
    if (hippt::dot(microfacet_normal, surface_normal) < 0.0f)
        //The microfacet normal that we sampled was under the surface, this can happen
        return ColorRGB32F(0.0f);
    float3 to_light_direction = hippt::normalize(2.0f * hippt::dot(microfacet_normal, view_direction) * microfacet_normal - view_direction);
    float3 halfway_vector = microfacet_normal;
    output_direction = to_light_direction;

    ColorRGB32F brdf_color = ColorRGB32F(0.0f, 0.0f, 0.0f);
    ColorRGB32F base_color = material.base_color;

    float NoV = hippt::max(0.0f, hippt::dot(surface_normal, view_direction));
    float NoL = hippt::max(0.0f, hippt::dot(surface_normal, to_light_direction));
    float NoH = hippt::max(0.0f, hippt::dot(surface_normal, halfway_vector));
    float VoH = hippt::max(0.0f, hippt::dot(halfway_vector, view_direction));

    if (NoV > 0.0f && NoL > 0.0f && NoH > 0.0f)
    {
        /////////// Cook Torrance BRDF //////////
        ColorRGB32F F;
        float D, G;

        //GGX Distribution function
        D = GGX_normal_distribution(alpha, NoH);

        //F0 = 0.04 for dielectrics, 1.0 for metals (approximation)
        ColorRGB32F F0 = ColorRGB32F(0.04f * (1.0f - metallic)) + metallic * base_color;
        F = fresnel_schlick(F0, VoH);
        G = GGX_smith_masking_shadowing(alpha, NoV, NoL);

        ColorRGB32F kD = ColorRGB32F(1.0f - metallic); //Metals do not have a base_color part
        kD = kD * (ColorRGB32F(1.0f) - F);//Only the transmitted light is diffused

        ColorRGB32F diffuse_part = kD * base_color * M_INV_PI;
        ColorRGB32F specular_part = (F * D * G) / (4.0f * NoV * NoL);

        pdf = D * NoH / (4.0f * VoH);

        brdf_color = diffuse_part + specular_part;
    }

    return brdf_color;
}

#endif