/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_TEXTURE_H
#define DEVICE_TEXTURE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/TriangleStructures.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/RenderData.h"

#ifndef __KERNELCC__
#include "Image/Image.h"
#endif

#ifdef __KERNELCC__
// Dummy usings so that the GPU compiler doesn't complain that Image8Bit / Image32Bit don't exist.
// It's okay to dummy use them as int because they are not used on the GPU side anyway, this is
// purely for the compiler to be happy
using Image8Bit = int;
using Image32Bit = int;
#endif

/**
 * Templated here so that the CPU can cast the texture_buffer into Image8Bit or Image32Bit
 * for proper sampling in unsigned char or float respectively.
 * This template argument isn't used on the GPU and that's why Image8Bit and Image32Bit
 * are being defined as 'ints'
 * 
 * If 'flip_uv_y' is true, then UV (0, 0) is the bottom left corner of the texture
 * and the texture must use a wrapping address mode for the V coordinate.
 * 
 * If 'flip_uv_y' is true, then the UV coordinates are just used as is
 */ 
template <typename ImageType = Image8Bit>
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F sample_texture_rgba(const void* texture_buffer, int texture_index, bool is_srgb, float2 uv, bool flip_uv_y = true)
{
    ColorRGBA32F rgba;

#ifdef __KERNELCC__
    // We're doing the UV addressing ourselves since it seems to be broken in Orochi...
    float u = uv.x;
    float v = uv.y;
    if (flip_uv_y)
        v = -v;

    if (reinterpret_cast<const oroTextureObject_t*>(texture_buffer)[texture_index] == 0)
        return ColorRGBA32F(0.0f);

    rgba = ColorRGBA32F(tex2D<float4>(reinterpret_cast<const oroTextureObject_t*>(texture_buffer)[texture_index], u, v));
#else
    const ImageType& texture = reinterpret_cast<const ImageType*>(texture_buffer)[texture_index];

    rgba = texture.sample_rgba32f(uv);
#endif

    // sRGB to linear conversion
    // Doing the conversion manually instead of using the hardware
    // because it's unavailable in Orochi (again) :(
    if (is_srgb)
        return intrin_pow(rgba, 2.2f);
    else
        return rgba;
}

/**
 * If 'flip_uv_y' is true, then UV(0, 0) is the bottom left corner of the texture
 * and the texture must use a wrapping address mode for the V coordinate.
 *
 * If 'flip_uv_y' is true, then the UV coordinates are just used as is
 * 
 * 'flip_uv_y' should basically be set to true in most cases and.
 * It should be set to false if your texture addressing mode isn't 'warping'
 * or when you know what you're doing and why you need to have it to false
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_texture_rgb_8bits(const void* texture_buffer, int texture_index, bool is_srgb, float2 uv, bool flip_uv_y = true)
{
    ColorRGBA32F rgba = sample_texture_rgba<Image8Bit>(texture_buffer, texture_index, is_srgb, uv, flip_uv_y);

    return ColorRGB32F(rgba.r, rgba.g, rgba.b);
}

/**
 * Samples a texture given by indexing the texture array 'texture_buffer' with 'texture_buffer[texture_index]'.
 * 
 * To read from a single texture, pass the pointer to the texture in 'texture_buffer' and
 * pass texture_index = 0
 * 
 * Not that on the GPU, 'texture_buffer' must be of type oroTextureObject_t*, i.e. it's a pointer on oroTextureObject_t
 * this means that if the pointer is set in RenderData with OrochiTexture::get_device_texture() on the CPU, then
 * &get_device_texture() must be passed to this function for 'texture_buffer'
 * 
 * If 'flip_uv_y' is true, then UV(0, 0) is the bottom left corner of the texture
 * and the texture must use a wrapping address mode for the V coordinate.
 *
 * If 'flip_uv_y' is true, then the UV coordinates are just used as is
 * 
 * 'flip_uv_y' should basically be set to true in most cases and.
 * It should be set to false if your texture addressing mode isn't 'warping'
 * or when you know what you're doing and why you need to have it to false
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_texture_rgb_32bits(const void* texture_buffer, int texture_index, bool is_srgb, float2 uv, bool flip_uv_y = true)
{
    ColorRGBA32F rgba = sample_texture_rgba<Image32Bit>(texture_buffer, texture_index, is_srgb, uv, flip_uv_y);

    return ColorRGB32F(rgba.r, rgba.g, rgba.b);
}

#ifdef __KERNELCC__
/**
 * Bilinearly samples around x & y on the layer z of a 3D texture configured for
 * nearest neighbor sampling
 * 
 * uv is supposed to be in [0, 1] already
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F internal_bilinear_sample_on_3D_texture(const oroTextureObject_t texture, int3 ires, float2 uv, int z)
{
    // Reference: https://iquilezles.org/articles/hwinterpolation/

    float2 res_f = make_float2(ires.x, ires.y);

    float2 st = (uv - 0.5f / res_f) * res_f;
    int2 i = make_int2(floorf(st.x), floorf(st.y));
    float2 w = make_float2(hippt::fract(st.x), hippt::fract(st.y));

    ColorRGBA32F a = ColorRGBA32F(tex3D<float4>(texture, i.x + 0, i.y + 0, z));
    ColorRGBA32F b = ColorRGBA32F(tex3D<float4>(texture, i.x + 1, i.y + 0, z));
    ColorRGBA32F c = ColorRGBA32F(tex3D<float4>(texture, i.x + 0, i.y + 1, z));
    ColorRGBA32F d = ColorRGBA32F(tex3D<float4>(texture, i.x + 1, i.y + 1, z));

    return hippt::lerp(hippt::lerp(a, b, w.x), hippt::lerp(c, d, w.x), w.y);
}
#endif

/**
 * This function samples a 3D texture given in the 'texture' parameter
 * This parameter should be an oroTextureObject_t on the GPU, not a
 * pointer 'oroTextureObject_t*' as is the case for 'sample_texture_rgb_32bits'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_texture_3D_rgb_32bits(void* texture, int3 texture_dims, float3 uvw, bool hardware_interpolation = false)
{
    if (texture == nullptr)
        return ColorRGB32F(0.0f);

#ifdef __KERNELCC__
    // Sampling in repeat mode so we're just keeping the fractional part
    float u = uvw.x;
    if (u != 1.0f)
        // Only doing that if u != 1.0f because if we actually have
        // uv.x == 1.0f, then subtracting static_cast<int>(uv.x) will
        // give us 0.0f even though we actually want 1.0f (which is correct).
        // 
        // Basically, 1.0f gets transformed into 0.0f even though 1.0f is a correct
        // U coordinate which needs not to be wrapped
        u = hippt::fract(uvw.x);

    float v = uvw.y;
    if (v != 1.0f)
        // Same for v
        v = hippt::fract(uvw.y);

    float w = uvw.z;
    if (w != 1.0f)
        // Same for w
        w = hippt::fract(uvw.z);

    // For negative UVs, we also want to repeat and we want, for example, 
    // -0.1f to behave as 0.9f
    u = u < 0 ? 1.0f + u : u;
    v = v < 0 ? 1.0f + v : v;
    w = w < 0 ? 1.0f + w : w;

    // Sampling with [0, 0] bottom-left convention
    v = 1.0f - v;

    if (hardware_interpolation)
    {
        float x = (u * (texture_dims.x - 1));
        float y = (v * (texture_dims.y - 1));
        float z = (w * (texture_dims.z - 1));

        return ColorRGB32F(ColorRGBA32F(tex3D<float4>(reinterpret_cast<oroTextureObject_t>(texture), x, y, z)));
    }
    else
    {
        float z = (w * (texture_dims.z - 1));

        // Whether or not we need to interpolate with layer z+1 or z-1
        bool z_layer_up = hippt::fract(w * texture_dims.z) > 0.5f;
        int z0 = z;
        int z1 = z_layer_up ? z0 + 1 : z0 - 1;

        ColorRGBA32F rgba0 = internal_bilinear_sample_on_3D_texture(reinterpret_cast<oroTextureObject_t>(texture), texture_dims, make_float2(u, v), z0);
        ColorRGBA32F rgba1 = internal_bilinear_sample_on_3D_texture(reinterpret_cast<oroTextureObject_t>(texture), texture_dims, make_float2(u, v), z1);

        return ColorRGB32F(hippt::lerp(rgba0, rgba1, w));
    }
#else
    const Image32Bit3D& image = *reinterpret_cast<const Image32Bit3D*>(texture);
    ColorRGBA32F rgba = image.sample_rgba32f(uvw);

    return ColorRGB32F(rgba);
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_environment_map_texture(const WorldSettings& world_settings, float2 uv)
{
#if EnvmapSamplingDoBilinearFiltering == KERNEL_OPTION_TRUE
    float x = uv.x * (world_settings.envmap_width - 1);
    float y = uv.y * (world_settings.envmap_height - 1);

    float x_frac = hippt::fract(x);
    float y_frac = hippt::fract(y);

    int x_i = (int)x;
    int x_i_1 = hippt::min(x_i + 1, (int)world_settings.envmap_width - 1);
    int y_i = (int)y;
    int y_i_1 = hippt::min(y_i + 1, (int)world_settings.envmap_height - 1);

    int index_x0y0 = x_i + y_i * world_settings.envmap_width;
    int index_x1y0 = x_i_1 + y_i * world_settings.envmap_width;
    int index_x0y1 = x_i + y_i_1 * world_settings.envmap_width;
    int index_x1y1 = x_i_1 + y_i_1 * world_settings.envmap_width;

    ColorRGB32F color_x0y0 = world_settings.envmap[index_x0y0].unpack() * world_settings.envmap_intensity;
    ColorRGB32F color_x1y0 = world_settings.envmap[index_x1y0].unpack() * world_settings.envmap_intensity;
    ColorRGB32F color_x0y1 = world_settings.envmap[index_x0y1].unpack() * world_settings.envmap_intensity;
    ColorRGB32F color_x1y1 = world_settings.envmap[index_x1y1].unpack() * world_settings.envmap_intensity;

    return hippt::lerp(hippt::lerp(color_x0y0, color_x1y0, x_frac), hippt::lerp(color_x0y1, color_x1y1, x_frac), y_frac);
#else
    int x = uv.x * (world_settings.envmap_width - 1);
    int y = uv.y * (world_settings.envmap_height - 1);
    int index = x + y * world_settings.envmap_width;

    return world_settings.envmap[index].unpack() * world_settings.envmap_intensity;
#endif
}

/**
 * Given the indices of the vertices of a triangle, interpolates the vertices data found
 * in the 'data' buffer passed as argument as the given UV coordinates
 */
template <typename T>
HIPRT_HOST_DEVICE HIPRT_INLINE T uv_interpolate(int vertex_A_index, int vertex_B_index, int vertex_C_index, T* data, float2 uv)
{
    return data[vertex_B_index] * uv.x + data[vertex_C_index] * uv.y + data[vertex_A_index] * (1.0f - uv.x - uv.y);
}

/**
 * Given the indices of the vertices of a triangle, interpolates the vertices data found
 * in the 'data' buffer passed as argument as the given UV coordinates
 */
template <typename T>
HIPRT_HOST_DEVICE HIPRT_INLINE T uv_interpolate(TriangleIndices triangle_vertex_indices, T* data, float2 uv)
{
    return uv_interpolate(triangle_vertex_indices.x, triangle_vertex_indices.y, triangle_vertex_indices.z, data, uv);
}

/**
 * Just a simple "specialization" for when we're interpolating texcoords
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float2 uv_interpolate(TriangleTexcoords texcoords, float2 uv)
{
    return texcoords.y * uv.x + texcoords.z * uv.y + texcoords.x * (1.0f - uv.x - uv.y);
}

/**
 * Same as the overloads above but you can call this one when you don't already have the vertex indices
 * from the place you're calling this function from. This function will then fetch the vertex indices again
 */
template <typename T>
HIPRT_HOST_DEVICE HIPRT_INLINE T uv_interpolate(int* vertex_indices, int primitive_index, T* data, float2 uv)
{
    int vertex_A_index = vertex_indices[primitive_index * 3 + 0];
    int vertex_B_index = vertex_indices[primitive_index * 3 + 1];
    int vertex_C_index = vertex_indices[primitive_index * 3 + 2];

    return uv_interpolate(vertex_A_index, vertex_B_index, vertex_C_index, data, uv);
}

#endif
