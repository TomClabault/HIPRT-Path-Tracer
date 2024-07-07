/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef UTILS_H
#define UTILS_H

#include "HostDeviceCommon/Color.h"

#include <string>

class ImageRGB32F;

class Utils
{
public:

    static std::vector<unsigned char> tonemap_hdr_image(const ImageRGB32F& hdr_image, int sample_number, float gamma, float exposure);
    static std::vector<unsigned char> tonemap_hdr_image(const std::vector<ColorRGB32F>& hdr_image, int sample_number, float gamma, float exposure);
    static std::vector<unsigned char> tonemap_hdr_image(const std::vector<float>& hdr_image, int sample_number, float gamma, float exposure);
    static std::vector<unsigned char> tonemap_hdr_image(const float* hdr_image, size_t size, int sample_number, float gamma, float exposure);

    static std::string file_to_string(const char* filepath);

    /*
     * A blend factor of 1 gives only the noisy image. 0 only the denoised image
     */
    static ImageRGB32F OIDN_denoise(const ImageRGB32F& image, int width, int height, float blend_factor);

    /**
     * Breaks the debugger when calling this function as if a breakpoint was hit. 
     * Useful to be able to inspect the callstack at a given point in the program
     */
    static void debugbreak();
};

#endif
