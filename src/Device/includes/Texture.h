/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_TEXTURE_H
#define DEVICE_TEXTURE_H

#include "Device/includes/FixIntellisense.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/RenderData.h"

#ifndef __KERNELCC__
#include "Image/Image.h"
#endif

HIPRT_HOST_DEVICE HIPRT_INLINE float luminance(ColorRGB pixel)
{
    return 0.3086f * pixel.r + 0.6094f * pixel.g + 0.0820f * pixel.b;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float luminance(ColorRGBA pixel)
{
    return 0.3086f * pixel.r + 0.6094f * pixel.g + 0.0820f * pixel.b;
}


HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA sample_texture_rgba(const void* texture_buffer, int texture_index, int2 texture_dims, bool is_srgb, float2 uv)
{
    ColorRGBA rgba;

#ifdef __KERNELCC__
    // We're doing the UV addressing oursevles since it seems to be broken in Orochi...
    // 
    // Sampling in repeat mode so we're just keeping the fractional part
    float u = uv.x - (int)uv.x;
    float v = uv.y - (int)uv.y;

    // For negative UVs, we also want to repeat and we want, for example, 
    // -0.1f to behave as 0.9f
    u = u < 0 ? 1.0f + u : u;
    v = v < 0 ? 1.0f + v : v;

    // Sampling with [0, 0] bottom-left convention
    v = 1.0f - v;

    rgba = ColorRGBA(tex2D<float4>(reinterpret_cast<const oroTextureObject_t*>(texture_buffer)[texture_index], u * (texture_dims.x - 1), v * (texture_dims.y - 1)));
#else
    const ImageRGBA& texture = reinterpret_cast<const ImageRGBA*>(texture_buffer)[texture_index];

    rgba = texture.sample(uv);
#endif

    // sRGB to linear conversion
    // Doing the conversion manually instead of using the hardware
    // because it's unavailable in Orochi (again) :(
    if (is_srgb)
        return pow(rgba, 2.2f);
    else
        return rgba;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_texture_rgb(const void* texture_buffer, int texture_index, int2 texture_dims, bool is_srgb, float2 uv)
{
    ColorRGBA rgba = sample_texture_rgba(texture_buffer, texture_index, texture_dims, is_srgb, uv);

    return ColorRGB(rgba.r, rgba.g, rgba.b);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_environment_map_texture(const WorldSettings& world_settings, float2 uv)
{
    const void* envmap_pointer;
#ifdef __KERNELCC__
    envmap_pointer = &world_settings.envmap;
#else
    envmap_pointer = world_settings.envmap;
#endif

    return sample_texture_rgb(envmap_pointer, 0, make_int2(world_settings.envmap_width, world_settings.envmap_height), /* is_srgb */ false, uv) * world_settings.envmap_intensity;
}

template <typename T>
HIPRT_HOST_DEVICE HIPRT_INLINE T uv_interpolate(int vertex_A_index, int vertex_B_index, int vertex_C_index, T* data, float2 uv)
{
    return data[vertex_B_index] * uv.x + data[vertex_C_index] * uv.y + data[vertex_A_index] * (1.0f - uv.x - uv.y);
}

template <typename T>
HIPRT_HOST_DEVICE HIPRT_INLINE T uv_interpolate(int* vertex_indices, int primitive_index, T* data, float2 uv)
{
    int vertex_A_index = vertex_indices[primitive_index * 3 + 0];
    int vertex_B_index = vertex_indices[primitive_index * 3 + 1];
    int vertex_C_index = vertex_indices[primitive_index * 3 + 2];

    return uv_interpolate(vertex_A_index, vertex_B_index, vertex_C_index, data, uv);
}


#endif
