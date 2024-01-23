#ifndef _IMAGE_IO_H
#define _IMAGE_IO_H

//! \addtogroup image utilitaires pour manipuler des images
///@{


#include "image.h"

//! charge une image .bmp .tga .jpeg .png ou .hdr
Image read_image( const char *filename, const bool flipY= true );

//! enregistre une image au format .png
bool write_image( const Image& image, const char *filename, const bool flipY= true );
//! enregistre une image au format .png
bool write_image_png( const Image& image, const char *filename, const bool flipY= true );
//! enregistre une image au format .bmp
bool write_image_bmp( const Image& image, const char *filename, const bool flipY= true );
//! enregistre une image au format .hdr
bool write_image_hdr( const Image& image, const char *filename, const bool flipY= true );

//! raccourci pour write_image_png(tone(image, range(image)), "image.png")
bool write_image_preview( const Image& image, const char *filename, const bool flipY= true, const float gamma= float(2.2));

//! transformation gamma : rgb lineaire vers srgb
Image gamma( const Image& image, const float g= float(2.2) );
//! transformation gamma : srgb vers rgb lineaire
Image inverse_gamma( const Image& image, const float g= float(2.2) );

//! evalue l'exposition d'une image.
float range( const Image& image );
//! correction de l'exposition d'une image + transformation gamma.
Image tone( const Image& image, const float saturation, const float gamma= float(2.2) );

///@}
#endif
