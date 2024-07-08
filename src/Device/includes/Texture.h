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

HIPRT_HOST_DEVICE HIPRT_INLINE float luminance(ColorRGB32F pixel)
{
    return 0.3086f * pixel.r + 0.6094f * pixel.g + 0.0820f * pixel.b;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float luminance(ColorRGBA32F pixel)
{
    return 0.3086f * pixel.r + 0.6094f * pixel.g + 0.0820f * pixel.b;
}

#ifdef __KERNELCC__
// Dummy usings so that the GPU compiler doesn't complain that Image8Bit / Image32Bit don't exist.
// It's okay to dummy use them as int because they are not used on the GPU side anyway, this is
// purely for the compiler to be happy
using Image8Bit = int;
using Image32Bit = int;
#endif

// Templated here so that the CPU can cast the texture_buffer into Image8Bit or Image32Bit
// for proper sampling in unsigned char or float respectively.
// This template argument isn't used on the GPU and that's why Image8Bit and Image32Bit
// are being defined as 'ints'
template <typename ImageType = Image8Bit>
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F sample_texture_rgba(const void* texture_buffer, int texture_index, int2 texture_dims, bool is_srgb, float2 uv)
{
    ColorRGBA32F rgba;

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

    rgba = ColorRGBA32F(tex2D<float4>(reinterpret_cast<const oroTextureObject_t*>(texture_buffer)[texture_index], u * (texture_dims.x - 1), v * (texture_dims.y - 1)));
#else
    const ImageType& texture = reinterpret_cast<const ImageType*>(texture_buffer)[texture_index];

    rgba = texture.sample_rgba32f(uv);
#endif

    // sRGB to linear conversion
    // Doing the conversion manually instead of using the hardware
    // because it's unavailable in Orochi (again) :(
    if (is_srgb)
        return pow(rgba, 2.2f);
    else
        return rgba;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_texture_rgb_8bits(const void* texture_buffer, int texture_index, int2 texture_dims, bool is_srgb, float2 uv)
{
    ColorRGBA32F rgba = sample_texture_rgba<Image8Bit>(texture_buffer, texture_index, texture_dims, is_srgb, uv);

    return ColorRGB32F(rgba.r, rgba.g, rgba.b);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_texture_rgb_32bits(const void* texture_buffer, int texture_index, int2 texture_dims, bool is_srgb, float2 uv)
{
    ColorRGBA32F rgba = sample_texture_rgba<Image32Bit>(texture_buffer, texture_index, texture_dims, is_srgb, uv);

    return ColorRGB32F(rgba.r, rgba.g, rgba.b);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_environment_map_texture(const WorldSettings& world_settings, float2 uv)
{
    const void* envmap_pointer;
#ifdef __KERNELCC__
    envmap_pointer = &world_settings.envmap;
#else
    envmap_pointer = world_settings.envmap;
#endif

    return sample_texture_rgb_32bits(envmap_pointer, 0, make_int2(world_settings.envmap_width, world_settings.envmap_height), /* is_srgb */ false, uv) * world_settings.envmap_intensity;
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
