/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef UTILS_H
#define UTILS_H

#include "HostDeviceCommon/Color.h"
#include "Image/Image.h"

#include <sstream>
#include <string>

class Utils
{
public:

    static std::vector<unsigned char> tonemap_hdr_image(const Image32Bit& hdr_image, int sample_number, float gamma, float exposure);
    static std::vector<unsigned char> tonemap_hdr_image(const std::vector<ColorRGB32F>& hdr_image, int sample_number, float gamma, float exposure);
    static std::vector<unsigned char> tonemap_hdr_image(const std::vector<float>& hdr_image, int sample_number, float gamma, float exposure);
    static std::vector<unsigned char> tonemap_hdr_image(const float* hdr_image, size_t size, int sample_number, float gamma, float exposure);

    static void compute_alias_table(const std::vector<float>& input, std::vector<float>& out_probas, std::vector<int>& out_alias, float* out_luminance_total_sum);
    static void compute_alias_table(const std::vector<float>& input, float in_input_total_sum, std::vector<float>& out_probas, std::vector<int>& out_alias);

    static std::string file_to_string(const char* filepath);
    static void get_current_date_string(std::stringstream& ss);

    static void* get_volume_handle_for_file(const char* filePath);
    static bool is_file_on_ssd(const char* file_path);
    static std::string open_file_dialog(const char* filter_patterns[], int filter_count);

	static float compute_image_mse(const Image32Bit& reference, const Image32Bit& subject);
	static float compute_image_rmse(const Image32Bit& reference, const Image32Bit& subject);
    static float compute_image_weighted_median_FLIP(const Image32Bit& reference, const Image32Bit& subject, float** out_error_map);

    static void copy_u8_image_data_to_clipboard(const std::vector<unsigned char>& data, int width, int height);
    static void copy_image_to_clipboard(const Image8Bit& image);
    static void copy_image_to_clipboard(const Image32Bit& image);

    /*
     * A blend factor of 1 gives only the noisy image. 0 only the denoised image
     */
    static Image32Bit OIDN_denoise(const Image32Bit& image, int width, int height, float blend_factor);

    /**
     * Breaks the debugger when calling this function as if a breakpoint was hit. 
     * Useful to be able to inspect the callstack at a given point in the program
     */
    static void debugbreak();

#ifdef _WIN32
    enum AddEnvVarError
    {
		ADD_ENV_VAR_ERROR_NONE = 0, // All good
        ADD_ENV_VAR_ERROR_NOT_FOUND, // Given env var not found
        ADD_ENV_VAR_ERROR_VALUE_TOO_LONG, // The value of the environment variable exceeds MAX_PATH
        ADD_ENV_VAR_ERROR_UNKNOWN // Unhandled error value
    };
	static AddEnvVarError windows_add_ENV_var_to_PATH(const wchar_t* env_var_name);
#endif
};

#endif
