
#include <cfloat>

#include "Kernels/includes/hiprt_color.h"
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

inline float clamp( const float x, const float min, const float max )
{
    if(x < min) return min;
    else if(x > max) return max;
    else return x;
}

bool write_image_png(const std::vector<HIPRTColor>& image, int width, int height, const char *filename, const bool flipY )
{
    if(image.size() == 0)
        return false;
    
    std::vector<unsigned char> tmp(image.size() * 3);
    for(unsigned i= 0, offset= 0; i < image.size(); i++, offset+= 3)
    {
        HIPRTColor pixel= image[i] * 255;
        tmp[offset   ]= clamp(pixel.r, 0, 255);
        tmp[offset +1]= clamp(pixel.g, 0, 255);
        tmp[offset +2]= clamp(pixel.b, 0, 255);
    }
    
    stbi_flip_vertically_on_write(flipY);
    return stbi_write_png(filename, width, height, 3, tmp.data(), width * 3) != 0;
}

bool write_image_hdr(const std::vector<HIPRTColor>& image, int width, int height, const char *filename, const bool flipY )
{
    if(image.size() == 0)
        return false;
    
    stbi_flip_vertically_on_write(flipY);
    return stbi_write_hdr(filename, width, height, 3, reinterpret_cast<const float*>(image.data())) != 0;
}
