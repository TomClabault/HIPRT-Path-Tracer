/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_MATERIAL_H
#define DEVICE_MATERIAL_H

#include "Device/includes/Texture.h"

#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/Material/MaterialUtils.h"
#include "HostDeviceCommon/RenderData.h"

#ifndef __KERNELCC__
#include "Image/Image.h"
#endif

template <typename T>
HIPRT_HOST_DEVICE HIPRT_INLINE T get_material_property(const HIPRTRenderData& render_data, bool is_srgb, const float2& texcoords, int texture_index);
HIPRT_HOST_DEVICE HIPRT_INLINE float2 get_metallic_roughness(const HIPRTRenderData& render_data, const float2& texcoords, int metallic_texture_index, int roughness_texture_index, int metallic_roughness_texture_index);
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F get_base_color(const HIPRTRenderData& render_data, float& out_alpha, const float2& texcoords, int base_color_texture_index);

HIPRT_HOST_DEVICE HIPRT_INLINE float get_hit_base_color_alpha(const HIPRTRenderData& render_data, unsigned short int base_color_texture_index, hiprtHit hit)
{
    if (base_color_texture_index == MaterialConstants::NO_TEXTURE)
        // Quick exit if no texture
        return 1.0f;

    float2 texcoords = uv_interpolate(render_data.buffers.triangles_indices, hit.primID, render_data.buffers.texcoords, hit.uv);

    // Getting the alpha for transparency check to see if we need to pass the ray through or not
    float alpha;
    get_base_color(render_data, alpha, texcoords, base_color_texture_index);

    return alpha;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_hit_base_color_alpha(const HIPRTRenderData& render_data, const DevicePackedTexturedMaterial& material, hiprtHit hit)
{
    return get_hit_base_color_alpha(render_data, material.get_base_color_texture_index(), hit);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_hit_base_color_alpha(const HIPRTRenderData& render_data, hiprtHit hit)
{
    int material_index = render_data.buffers.material_indices[hit.primID];
    unsigned short int base_color_texture_index = render_data.buffers.materials_buffer.get_base_color_texture_index(material_index);

    return get_hit_base_color_alpha(render_data, base_color_texture_index, hit);
}

HIPRT_HOST_DEVICE HIPRT_INLINE DeviceUnpackedEffectiveMaterial get_intersection_material(const HIPRTRenderData& render_data, int material_index, float2 texcoords)
{
    DeviceUnpackedTexturedMaterial material = render_data.buffers.materials_buffer.read_partial_material(material_index).unpack();

    float trash_alpha;
    if (render_data.bsdfs_data.white_furnace_mode)
        material.base_color = ColorRGB32F(1.0f);
    else
    {
        if (material.base_color_texture_index != MaterialConstants::NO_TEXTURE)
            material.base_color = get_base_color(render_data, trash_alpha, texcoords, material.base_color_texture_index);
    }

    // Reading some parameters from the textures
    float2 roughness_metallic = get_metallic_roughness(render_data, texcoords, material.metallic_texture_index, material.roughness_texture_index, material.roughness_metallic_texture_index);
    // TODO DEBUGGING RESTIR GI UNCOMMENT THIS
    //if (material.roughness_metallic_texture_index != MaterialConstants::NO_TEXTURE)
    //{
    //    material.roughness = roughness_metallic.x;
    //    material.metallic = roughness_metallic.y;
    //}
    //else
    //{
    //    if (material.roughness_texture_index != MaterialConstants::NO_TEXTURE)
    //        material.roughness = roughness_metallic.x;

    //    if (material.metallic_texture_index != MaterialConstants::NO_TEXTURE)
    //        material.metallic = roughness_metallic.y;

    //    // If not reading from a texture, setting the roughness into the roughness_metallic
    //    // variable because the roughness is going to be used later
    //    roughness_metallic.x = material.roughness;
    //}

    roughness_metallic.x = material.roughness;
    roughness_metallic.y = material.metallic;

    float anisotropy = get_material_property<float>(render_data, false, texcoords, material.anisotropic_texture_index);
    if (material.anisotropic_texture_index != MaterialConstants::NO_TEXTURE)
        material.anisotropy = anisotropy;
    
    float specular = get_material_property<float>(render_data, false, texcoords, material.specular_texture_index);
    if (material.specular_texture_index != MaterialConstants::NO_TEXTURE)
        material.specular = specular;

    float coat = get_material_property<float>(render_data, false, texcoords, material.coat_texture_index);
    if (material.coat_texture_index != MaterialConstants::NO_TEXTURE)
        material.coat = coat;
    else
        coat = material.coat;

    float sheen = get_material_property<float>(render_data, false, texcoords, material.sheen_texture_index);
    if (material.sheen_texture_index != MaterialConstants::NO_TEXTURE)
        material.sheen = sheen;

    float specular_transmission = get_material_property<float>(render_data, false, texcoords, material.specular_transmission_texture_index);
    if (material.specular_transmission_texture_index != MaterialConstants::NO_TEXTURE)
        material.specular_transmission = specular_transmission;

    ColorRGB32F emission = get_material_property<ColorRGB32F>(render_data, false, texcoords, material.emission_texture_index);
    if (material.emission_texture_index == MaterialConstants::NO_TEXTURE || material.emission_texture_index == MaterialConstants::CONSTANT_EMISSIVE_TEXTURE)
        emission = material.emission;

    DeviceUnpackedEffectiveMaterial unpacked_material(material);
    // TODO DEBUGGING RESTIR GI REMOVE THESE TWO LINES METALLIC/ROUGHNESS
    unpacked_material.roughness = material.roughness;
    unpacked_material.metallic = material.metallic;

    unpacked_material.emissive_texture_used = material.emission_texture_index != MaterialConstants::NO_TEXTURE;
    unpacked_material.emission = emission;
    // Roughening of the base roughness and second metallic roughness based
    // on the coat roughness. This should be precomputed instead of being done here
    //
    // Reference: [OpenPBR Surface 2024 Specification] https://academysoftwarefoundation.github.io/OpenPBR/#model/coat/roughening
    float coat_roughening = unpacked_material.coat_roughening;
    if (coat > 0.0f && coat_roughening > 0.0f)
    {
        float base_roughness = roughness_metallic.x;
        float coat_roughness = unpacked_material.coat_roughness;

        // Roughening of the base roughness of the material based on the coat roughness
        float target_base_roughness = hippt::pow_1_4(hippt::min(1.0f, hippt::pow_4(base_roughness) + 2.0f * hippt::pow_4(coat_roughness)));
        float roughened_base_roughness = hippt::lerp(base_roughness, target_base_roughness, coat);
        unpacked_material.roughness = hippt::lerp(base_roughness, roughened_base_roughness, coat_roughening);

        if (unpacked_material.second_roughness_weight > 0.0f)
        {
            // Roughening of the second metallic roughness based on the coat roughness

            float second_roughness = unpacked_material.second_roughness;
            float target_second_metal_roughness = hippt::pow_1_4(hippt::min(1.0f, hippt::pow_4(second_roughness) + 2.0f * hippt::pow_4(coat_roughness)));
            float roughened_second_metal_roughness = hippt::lerp(second_roughness, target_second_metal_roughness, coat);
            unpacked_material.second_roughness = hippt::lerp(second_roughness, roughened_second_metal_roughness, coat_roughening);
        }
    }

    return unpacked_material;
}

/**
 * The float2 returned is (roughness, metallic)
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float2 get_metallic_roughness(const HIPRTRenderData& render_data, const float2& texcoords, int metallic_texture_index, int roughness_texture_index, int metallic_roughness_texture_index)
{
    float2 out;

    if (metallic_roughness_texture_index != MaterialConstants::NO_TEXTURE)
    {
        ColorRGB32F rgb = sample_texture_rgb_8bits(render_data.buffers.material_textures, metallic_roughness_texture_index, false, texcoords);

        // Not converting to linear here because material properties (roughness and metallic) here are assumed to be linear already
        out.x = rgb.g;
        out.y = rgb.b;
    }
    else
    {
        out.x = get_material_property<float>(render_data, false, texcoords, roughness_texture_index);
        out.y = get_material_property<float>(render_data, false, texcoords, metallic_texture_index);
    }

    return out;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F get_base_color(const HIPRTRenderData& render_data, float& out_alpha, const float2& texcoords, int base_color_texture_index)
{
    out_alpha = 1.0f;
    ColorRGBA32F rgba = get_material_property<ColorRGBA32F>(render_data, true, texcoords, base_color_texture_index);
    if (base_color_texture_index != MaterialConstants::NO_TEXTURE)
    {
        ColorRGB32F base_color = ColorRGB32F(rgba.r, rgba.g, rgba.b);
        out_alpha = rgba.a;

        return base_color;
    }

    return ColorRGB32F();
}

template <typename T>
HIPRT_HOST_DEVICE HIPRT_INLINE T read_data(const ColorRGBA32F& rgba) {}

template<>
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F read_data<ColorRGBA32F>(const ColorRGBA32F& rgba)
{
    return rgba;
}

template<>
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F read_data<ColorRGB32F>(const ColorRGBA32F& rgba)
{
    return ColorRGB32F(rgba.r, rgba.g, rgba.b);
}

template<>
HIPRT_HOST_DEVICE HIPRT_INLINE float read_data<float>(const ColorRGBA32F& rgba)
{
    return rgba.r;
}

template <typename T>
HIPRT_HOST_DEVICE HIPRT_INLINE T get_material_property(const HIPRTRenderData& render_data, bool is_srgb, const float2& texcoords, int texture_index)
{
    if (texture_index == MaterialConstants::NO_TEXTURE || texture_index == MaterialConstants::CONSTANT_EMISSIVE_TEXTURE)
        return T();

    ColorRGBA32F rgba = sample_texture_rgba(render_data.buffers.material_textures, texture_index, is_srgb, texcoords);
    return read_data<T>(rgba);
}

#endif
