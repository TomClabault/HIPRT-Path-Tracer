#ifndef UTILS_H
#define UTILS_H

#include "Image/image.h"

#include <string>

/*
 * Defines a rectangular region on a skysphere / image
 */

class Utils
{
public:

    static Image read_image_float(const std::string& filepath, int& image_width, int& image_height, bool flipY = true);
    static std::vector<unsigned char> tonemap_hdr_image(const std::vector<float>& hdr_image, int sample_number, float gamma, float exposure);

    static std::vector<float> compute_env_map_cdf(const Image& skysphere);
    static std::vector<ImageBin> importance_split_skysphere(const Image& skysphere, int minimum_bin_area = 1024, float minimum_bin_radiance = 1000000);
    static std::vector<ImageBin> importance_split_skysphere(const Image& skysphere, ImageBin current_region, float current_radiance, int minimum_bin_area, float minimum_bin_radiance);

    static void write_env_map_bins_to_file(const std::string& filepath, Image skysphere_data, const std::vector<ImageBin>& skysphere_importance_bins);

    /*
     * A blend factor of 1 gives only the noisy image. 0 only the denoised image
     */
    static Image OIDN_denoise(const Image& noisy_image, float blend_factor);
};

#endif
