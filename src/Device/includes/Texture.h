/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_TEXTURE_H
#define DEVICE_TEXTURE_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Color.h"

HIPRT_HOST_DEVICE HIPRT_INLINE float luminance(ColorRGB pixel)
{
    return 0.3086f * pixel.r + 0.6094f * pixel.g + 0.0820f * pixel.b;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float luminance(ColorRGBA pixel)
{
    return 0.3086f * pixel.r + 0.6094f * pixel.g + 0.0820f * pixel.b;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_texture_pixel(void* texture_pointer, int width, int x, int y)
{
#ifdef __KERNELCC__
    float4 color = tex2D<float4>(reinterpret_cast<oroTextureObject_t>(texture_pointer), x, y);
    return ColorRGB(color.x, color.y, color.z);
#else
    ColorRGBA pixel = reinterpret_cast<ColorRGBA*>(texture_pointer)[y * width + x];
    return ColorRGB(pixel.r, pixel.g, pixel.b);
#endif
}

#endif
