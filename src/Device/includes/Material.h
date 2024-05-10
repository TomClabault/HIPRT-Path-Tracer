#ifndef DEVICE_MATERIAL_H
#define DEVICE_MATERIAL_H

#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

#ifndef __KERNELCC__
#include "Image/Image.h"
#endif

// TODO try templated
HIPRT_HOST_DEVICE HIPRT_INLINE void get_material_property(HIPRTRenderData& render_data, ColorRGB& color, const float2& texcoords, int texture_index)
{
    if (texture_index == -1)
        return;

    ColorRGBA rgba;
#ifdef __KERNELCC__
    int texture_width = render_data.buffers.material_textures_widths[texture_index];
    int texture_height = render_data.buffers.material_textures_heights[texture_index];

    int x = texcoords.x * texture_width;
    int y = (1.0f - texcoords.y) * texture_height;

    oroTextureObject_t texture = ((oroTextureObject_t*)render_data.buffers.material_textures)[texture_index];
    rgba = ColorRGBA(tex2D<float4>(reinterpret_cast<oroTextureObject_t>(texture), x, y));
#else
    const ImageRGBA& texture = ((ImageRGBA*)render_data.buffers.material_textures)[texture_index];

    int y = texcoords.x * texture.width;
    int x = (1.0f - texcoords.y) * texture.height;

    x = hippt::clamp(0, texture.width - 1, x);
    y = hippt::clamp(0, texture.height - 1, y);

    rgba = texture[x * texture.width + y];
#endif

    color.r = rgba.r;
    color.g = rgba.g;
    color.b = rgba.b;
}

HIPRT_HOST_DEVICE HIPRT_INLINE void get_material_property(HIPRTRenderData& render_data, float& property, const float2& texcoords, int texture_index)
{
    if (texture_index == -1)
        return;


    ColorRGBA rgba;
#ifdef __KERNELCC__
    int texture_width = render_data.buffers.material_textures_widths[texture_index];
    int texture_height = render_data.buffers.material_textures_heights[texture_index];

    int x = texcoords.x * texture_width;
    int y = (1.0f - texcoords.y) * texture_height;

    oroTextureObject_t texture = ((oroTextureObject_t*)render_data.buffers.material_textures)[texture_index];
    rgba = ColorRGBA(tex2D<float4>(reinterpret_cast<oroTextureObject_t>(texture), x, y));
#else
    const ImageRGBA& texture = ((ImageRGBA*)render_data.buffers.material_textures)[texture_index];

    int y = texcoords.x * texture.width;
    int x = (1.0f - texcoords.y) * texture.height;

    x = hippt::clamp(0, texture.width - 1, x);
    y = hippt::clamp(0, texture.height - 1, y);

    rgba = texture[x * texture.width + y];
#endif

    property = rgba.r;
}

HIPRT_HOST_DEVICE HIPRT_INLINE void get_metallic_roughness(HIPRTRenderData& render_data, float& metallic, float& roughness, const float2& texcoords, int metallic_texture_index, int roughness_texture_index, int metallic_roughness_texture_index)
{
    if (metallic_roughness_texture_index != -1)
    {
        ColorRGBA rgba;
    #ifdef __KERNELCC__
        int texture_width = render_data.buffers.material_textures_widths[metallic_roughness_texture_index];
        int texture_height = render_data.buffers.material_textures_heights[metallic_roughness_texture_index];

        int x = texcoords.x * texture_width;
        int y = (1.0f - texcoords.y) * texture_height;

        oroTextureObject_t texture = ((oroTextureObject_t*)render_data.buffers.material_textures)[metallic_roughness_texture_index];
        rgba = ColorRGBA(tex2D<float4>(reinterpret_cast<oroTextureObject_t>(texture), x, y));
    #else
        const ImageRGBA& texture = ((ImageRGBA*)render_data.buffers.material_textures)[metallic_roughness_texture_index];

        int y = texcoords.x * texture.width;
        int x = (1.0f - texcoords.y) * texture.height;

        x = hippt::clamp(0, texture.width - 1, x);
        y = hippt::clamp(0, texture.height - 1, y);

        rgba = texture[x * texture.width + y];
    #endif

        roughness = rgba.g;
        metallic = rgba.b;
    }
    else
    {
        get_material_property(render_data, metallic, texcoords, metallic_texture_index);
        get_material_property(render_data, roughness, texcoords, roughness_texture_index);
    }
}

HIPRT_HOST_DEVICE HIPRT_INLINE RendererMaterial get_intersection_material(HIPRTRenderData& render_data, HitInfo& closest_hit_info)
{
	int material_index = render_data.buffers.material_indices[closest_hit_info.primitive_index];

	RendererMaterial material = render_data.buffers.materials_buffer[material_index];

    get_material_property(render_data, material.emission, closest_hit_info.texcoords, material.emission_texture_index);
    get_material_property(render_data, material.base_color, closest_hit_info.texcoords, material.base_color_texture_index);

    get_metallic_roughness(render_data, material.metallic, material.roughness, closest_hit_info.texcoords, material.metallic_texture_index, material.roughness_texture_index, material.roughnes_metallic_texture_index);
    get_material_property(render_data, material.oren_nayar_sigma, closest_hit_info.texcoords, material.oren_sigma_texture_index);
    get_material_property(render_data, material.subsurface, closest_hit_info.texcoords, material.subsurface_texture_index);
    
    get_material_property(render_data, material.specular, closest_hit_info.texcoords, material.specular_texture_index);
    get_material_property(render_data, material.specular_tint, closest_hit_info.texcoords, material.specular_tint_texture_index);
    get_material_property(render_data, material.specular_color, closest_hit_info.texcoords, material.specular_color_texture_index);
    
    get_material_property(render_data, material.anisotropic, closest_hit_info.texcoords, material.anisotropic_texture_index);
    get_material_property(render_data, material.anisotropic_rotation, closest_hit_info.texcoords, material.anisotropic_rotation_texture_index);
    
    get_material_property(render_data, material.clearcoat, closest_hit_info.texcoords, material.clearcoat_texture_index);
    get_material_property(render_data, material.clearcoat_roughness, closest_hit_info.texcoords, material.clearcoat_roughness_texture_index);
    get_material_property(render_data, material.clearcoat_ior, closest_hit_info.texcoords, material.clearcoat_ior_texture_index);
    
    get_material_property(render_data, material.sheen, closest_hit_info.texcoords, material.sheen_texture_index);
    get_material_property(render_data, material.sheen_tint, closest_hit_info.texcoords, material.sheen_tint_color_texture_index);
    get_material_property(render_data, material.sheen_color, closest_hit_info.texcoords, material.sheen_color_texture_index);
    
    get_material_property(render_data, material.ior, closest_hit_info.texcoords, material.ior_texture_index);
    get_material_property(render_data, material.specular_transmission, closest_hit_info.texcoords, material.specular_transmission_texture_index);

    // If the oren nayar microfacet normal standard deviation is spatially varying on the
    // surface, we'll need to make sure that the A and B precomputed coefficient are actually
    // precomputed according to that standard deviation
    if (material.oren_sigma_texture_index != -1)
        material.precompute_oren_nayar();

    // Same for the anisotropic, recomputing the precomputed alpha_x and alpha_y if necessary
    if (material.roughness_texture_index != -1 || material.roughnes_metallic_texture_index || material.anisotropic_texture_index != -1 && material.anisotropic > 0.0f)
        material.precompute_anisotropic();

	return material;
}

#endif
