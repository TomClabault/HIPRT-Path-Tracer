/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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
    Image8Bit(const unsigned char* data, int width, int height, int channels);
    Image8Bit(const std::vector<unsigned char>& data, int width, int height, int channels);

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

    std::vector<float> compute_cdf() const;

    size_t byte_size() const;

    /** 
     * Returns true if all the pixels of the texture are the same color
     * False otherwise
     *
     * A threshold can be given to assume that a color is equal to another
     * if the R, G and B channels of the two colors are each within 'threshold'
     * distance
     */ 
     bool is_constant_color(int threshold = 0) const;

     /**
      * Returns true if all pixels of the image have 1.0f alpha channel.
      * Returns true if the texture has less than 4 channels
      * 
      * Returns false otherwise
      */
     bool is_fully_opaque() const;

    /**
     * Frees the data of this image and sets its width, height and channels back to 0
     */
    void free();

    int width, height, channels;

protected:
    std::vector<unsigned char> m_pixel_data;
};

class Image32Bit
{
public:
    Image32Bit() {}
    Image32Bit(int width, int height, int channels);
    Image32Bit(const float* data, int width, int height, int channels);
    Image32Bit(const std::vector<float>& data, int width, int height, int channels);
    Image32Bit(Image8Bit image, int channels = -1);

    static Image32Bit read_image(const std::string& filepath, int output_channels, bool flipY);
    static Image32Bit read_image_hdr(const std::string& filepath, int output_channels, bool flipY);
    static Image32Bit read_image_exr(const std::string& filepath, bool flipY);

    bool write_image_png(const char* filename, const bool flipY = true) const;
    bool write_image_hdr(const char* filename, const bool flipY = true) const;

    Image32Bit to_linear_rgb() const;

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

    std::vector<float> compute_cdf() const;
    void compute_alias_table(std::vector<float>& out_probas, std::vector<int>& out_alias, float* out_luminance_total_sum = nullptr) const;

    float compute_luminance_sum() const;

    size_t byte_size() const;

    /** 
     * Returns true if all the pixels of the texture are the same color
     * False otherwise
     *
     * A threshold can be given to assume that a color is equal to another
     * if the R, G and B channels of the two colors are each within 'threshold'
     * distance
     */
    bool is_constant_color(float threshold) const;

    /**
     * Frees the data of this image and sets its width, height and channels back to 0
     */
    void free();

    int width = 0, height = 0, channels = 0;

protected:
    std::vector<float> m_pixel_data;
};

class Image32Bit3D
{
public:
    Image32Bit3D();
    Image32Bit3D(const std::vector<Image32Bit> images);

    ColorRGBA32F sample_rgba32f(float3 uvw) const;

    int width, height, depth, channels;

private:
    std::vector<Image32Bit> m_images;
};

#endif
