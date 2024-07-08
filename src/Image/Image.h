/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMAGE_H
#define IMAGE_H

#include "HostDeviceCommon/Color.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include <string>
#include <type_traits>

struct ImageBin
{
    int x0, x1;
    int y0, y1;
};

class Image8Bit
{
public:
    Image8Bit() : width(0), height(0), channels(0) {}
    Image8Bit(int width, int height, int channels);
    Image8Bit(unsigned char* data, int width, int height, int channels);
    Image8Bit(const std::vector<unsigned char>& data, int width, int height, int channels);
    Image8Bit(const std::string& filepath);
    Image8Bit(const char* filepath);

    static Image8Bit read_image(const std::string& filepath, int output_channels, bool flipY);
    static Image8Bit read_image_hdr(const std::string& filepath, int output_channels, bool flipY);

    bool write_image_png(const char* filename, const bool flipY = true) const;
    bool write_image_hdr(const char* filename, const bool flipY = true) const;

    float luminance_of_pixel(int x, int y) const;
    float luminance_of_area(int start_x, int start_y, int stop_x, int stop_y) const;
    float luminance_of_area(const ImageBin& area) const;

    ColorRGBA32F sample_rgba32f(float2 uv) const;

    void set_data(const std::vector<unsigned char>& data);
    const std::vector<unsigned char>& data() const;
    std::vector<unsigned char>& data();

    const unsigned char& operator[](int index) const;
    unsigned char& operator[](int index);

    void compute_cdf();
    std::vector<float> compute_get_cdf();
    std::vector<float>& get_cdf();

    size_t byte_size() const;

    int width, height, channels;

protected:
    std::vector<unsigned char> m_pixel_data;

    std::vector<float> m_cdf;
    bool m_cdf_computed = false;
};

class Image32Bit
{
public:
    Image32Bit() {}
    Image32Bit(int width, int height, int channels);
    Image32Bit(float* data, int width, int height, int channels);
    Image32Bit(const std::vector<float>& data, int width, int height, int channels);
    Image32Bit(const std::string& filepath);
    Image32Bit(const char* filepath);

    static Image32Bit read_image(const std::string& filepath, int output_channels, bool flipY);
    static Image32Bit read_image_hdr(const std::string& filepath, int output_channels, bool flipY);

    bool write_image_png(const char* filename, const bool flipY = true) const;
    bool write_image_hdr(const char* filename, const bool flipY = true) const;

    float luminance_of_pixel(int x, int y) const;
    float luminance_of_area(int start_x, int start_y, int stop_x, int stop_y) const;
    float luminance_of_area(const ImageBin& area) const;

    ColorRGB32F* get_data_as_ColorRGB32F();
    ColorRGB32F get_pixel_ColorRGB32F(int pixel_index) const;
    ColorRGBA32F* get_data_as_ColorRGBA32F();
    ColorRGBA32F get_pixel_ColorRGBA32F(int pixel_index) const;

    ColorRGBA32F sample_rgba32f(float2 uv) const;

    void set_data(const std::vector<float>& data);
    const std::vector<float>& data() const;
    std::vector<float>& data();

    const float& operator[](int index) const;
    float& operator[](int index);

    void compute_cdf();
    std::vector<float> compute_get_cdf();
    std::vector<float>& get_cdf();

    size_t byte_size() const;

    int width, height, channels;

protected:
    std::vector<float> m_pixel_data;

    std::vector<float> m_cdf;
    bool m_cdf_computed = false;
};

#endif
