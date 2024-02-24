#ifndef _IMAGE_IO_H
#define _IMAGE_IO_H

#include <Kernels/includes/hiprt_color.h>

bool write_image_png( const std::vector<HIPRTColor>& image, int width, int height, const char *filename, const bool flipY= true );
bool write_image_hdr(const std::vector<HIPRTColor>& image, int width, int height, const char *filename, const bool flipY= true );

#endif
