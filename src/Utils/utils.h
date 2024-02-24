#ifndef UTILS_H
#define UTILS_H

#include "Image/image.h"
#include "Kernels/includes/hiprt_color.h"

#include <string>

/*
 * Defines a rectangular region on a skysphere / image
 */

class Utils
{
public:

    static std::vector<HIPRTColor> read_image_float(const std::string& filepath, int& image_width, int& image_height, bool flipY = true);
    static std::vector<unsigned char> tonemap_hdr_image(const std::vector<HIPRTColor>& hdr_image, int sample_number, float gamma, float exposure);
    static std::vector<unsigned char> tonemap_hdr_image(const std::vector<float>& hdr_image, int sample_number, float gamma, float exposure);
    static std::vector<unsigned char> tonemap_hdr_image(const float* hdr_image, size_t size, int sample_number, float gamma, float exposure);

    static float luminance_of_pixel(const std::vector<HIPRTColor>& skysphere, int width, int x, int y);
    static float luminance_of_area(const std::vector<HIPRTColor>& skysphere, int width, int start_x, int start_y, int stop_x, int stop_y);
    static float luminance_of_area(const std::vector<HIPRTColor>& skysphere, int width, int height, const ImageBin& area);
    static std::vector<float> compute_env_map_cdf(const std::vector<HIPRTColor>& skysphere, int width, int height);
    static std::vector<ImageBin> importance_split_skysphere(const std::vector<HIPRTColor>& skysphere, int height, int width, int minimum_bin_area = 1024, float minimum_bin_radiance = 1000000);
    static std::vector<ImageBin> importance_split_skysphere(const std::vector<HIPRTColor>& skysphere, int width, int height, ImageBin current_region, float current_radiance, int minimum_bin_area, float minimum_bin_radiance);

    static void write_env_map_bins_to_file(const std::string& filepath, std::vector<HIPRTColor> skysphere_data, int width, int height, const std::vector<ImageBin>& skysphere_importance_bins);

    /*
     * A blend factor of 1 gives only the noisy image. 0 only the denoised image
     */
    static std::vector<HIPRTColor> OIDN_denoise(const std::vector<HIPRTColor>& noisy_image, int width, int height, float blend_factor);
};

#endif
