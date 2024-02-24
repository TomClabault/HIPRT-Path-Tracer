#ifndef IMAGE_H
#define IMAGE_H

#include <Kernels/includes/hiprt_color.h>

struct ImageBin
{
    int x0, x1;
    int y0, y1;
};

class Image
{
public:
    Image() {}
    Image(int width, int height) : width(width), height(height), m_pixel_data(width* height) {}
    Image(const std::vector<HIPRTColor>& data, int width, int height) : width(width), height(height), m_pixel_data(data) {}

    size_t byte_size() const;

    float luminance_of_pixel(int x, int y) const;
    float luminance_of_area(int start_x, int start_y, int stop_x, int stop_y) const;
    float luminance_of_area(const ImageBin& area) const;

    bool write_image_png(const char* filename, const bool flipY) const;
    bool write_image_hdr(const char* filename, const bool flipY) const;

    const HIPRTColor& operator[](int index) const;

    int width, height;
protected:

    std::vector<HIPRTColor> m_pixel_data;
};

#endif
