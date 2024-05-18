/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_MATERIAL_H
#define DEVICE_MATERIAL_H

#include "Device/includes/Texture.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

#ifndef __KERNELCC__
#include "Image/Image.h"
#endif

template <typename T>
HIPRT_HOST_DEVICE HIPRT_INLINE void get_material_property(const HIPRTRenderData& render_data, T& output_data, const float2& texcoords, int texture_index);
HIPRT_HOST_DEVICE HIPRT_INLINE void get_metallic_roughness(const HIPRTRenderData& render_data, float& metallic, float& roughness, const float2& texcoords, int metallic_texture_index, int roughness_texture_index, int metallic_roughness_texture_index);

HIPRT_HOST_DEVICE HIPRT_INLINE RendererMaterial get_intersection_material(const HIPRTRenderData& render_data, int material_index, float2 texcoords)
{
	RendererMaterial material = render_data.buffers.materials_buffer[material_index];

    get_material_property(render_data, material.emission, texcoords, material.emission_texture_index);
    get_material_property(render_data, material.base_color, texcoords, material.base_color_texture_index);

    get_metallic_roughness(render_data, material.metallic, material.roughness, texcoords, material.metallic_texture_index, material.roughness_texture_index, material.roughness_metallic_texture_index);
    get_material_property(render_data, material.oren_nayar_sigma, texcoords, material.oren_sigma_texture_index);
    get_material_property(render_data, material.subsurface, texcoords, material.subsurface_texture_index);
    
    get_material_property(render_data, material.specular, texcoords, material.specular_texture_index);
    get_material_property(render_data, material.specular_tint, texcoords, material.specular_tint_texture_index);
    get_material_property(render_data, material.specular_color, texcoords, material.specular_color_texture_index);
    
    get_material_property(render_data, material.anisotropic, texcoords, material.anisotropic_texture_index);
    get_material_property(render_data, material.anisotropic_rotation, texcoords, material.anisotropic_rotation_texture_index);
    
    get_material_property(render_data, material.clearcoat, texcoords, material.clearcoat_texture_index);
    get_material_property(render_data, material.clearcoat_roughness, texcoords, material.clearcoat_roughness_texture_index);
    get_material_property(render_data, material.clearcoat_ior, texcoords, material.clearcoat_ior_texture_index);
    
    get_material_property(render_data, material.sheen, texcoords, material.sheen_texture_index);
    get_material_property(render_data, material.sheen_tint, texcoords, material.sheen_tint_color_texture_index);
    get_material_property(render_data, material.sheen_color, texcoords, material.sheen_color_texture_index);
    
    get_material_property(render_data, material.specular_transmission, texcoords, material.specular_transmission_texture_index);

    // If the oren nayar microfacet normal standard deviation is spatially varying on the
    // surface, we'll need to make sure that the A and B precomputed coefficient are actually
    // precomputed according to that standard deviation
    if (material.oren_sigma_texture_index != -1)
        material.precompute_oren_nayar();

    // Same for the anisotropic, recomputing the precomputed alpha_x and alpha_y if necessary
    if (material.roughness_texture_index != -1 || material.roughness_metallic_texture_index || material.anisotropic_texture_index != -1 && material.anisotropic > 0.0f)
        material.precompute_anisotropic();

	return material;
}

HIPRT_HOST_DEVICE HIPRT_INLINE void get_metallic_roughness(const HIPRTRenderData& render_data, float& metallic, float& roughness, const float2& texcoords, int metallic_texture_index, int roughness_texture_index, int metallic_roughness_texture_index)
{
    if (metallic_roughness_texture_index != -1)
    {
        ColorRGB rgb = sample_texture_rgb(render_data.buffers.material_textures, metallic_roughness_texture_index, render_data.buffers.textures_dims[metallic_roughness_texture_index], false, texcoords);

        // Not converting to linear here because material properties (roughness and metallic) here are assumed to be linear already
        roughness = rgb.g;
        metallic = rgb.b;
    }
    else
    {
        get_material_property(render_data, metallic, texcoords, metallic_texture_index);
        get_material_property(render_data, roughness, texcoords, roughness_texture_index);
    }
}

template <typename T>
HIPRT_HOST_DEVICE HIPRT_INLINE void read_data(const ColorRGBA& rgba, bool is_srgb, T& data) {}

template<>
HIPRT_HOST_DEVICE HIPRT_INLINE void read_data(const ColorRGBA& rgba, bool is_srgb, ColorRGB& data)
{
    data.r = rgba.r;
    data.g = rgba.g;
    data.b = rgba.b;

    // sRGB to linear conversion
    if (is_srgb)
        data = pow(data, 2.2f);
}

template<>
HIPRT_HOST_DEVICE HIPRT_INLINE void read_data(const ColorRGBA& rgba, bool is_srgb, float& data)
{
    data = rgba.r;

    // sRGB to linear conversion
    if (is_srgb)
        data = pow(data, 2.2f);
}

template <typename T>
HIPRT_HOST_DEVICE HIPRT_INLINE void get_material_property(const HIPRTRenderData& render_data, T& output_data, const float2& texcoords, int texture_index)
{
    if (texture_index == -1)
        return;

    ColorRGBA rgba = sample_texture_rgba(render_data.buffers.material_textures, texture_index, render_data.buffers.textures_dims[texture_index], false, texcoords);
    read_data(rgba, render_data.buffers.texture_is_srgb[texture_index] == 1, output_data);
}

#endif
