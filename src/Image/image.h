
#ifndef _IMAGE_H
#define _IMAGE_H

#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>

#include "color.h"

struct ImageBin
{
    int x0, x1;
    int y0, y1;
};

//! \addtogroup image utilitaires pour manipuler des images
///@{

//! \file
//! manipulation simplifiee d'images

//! representation d'une image.
class Image
{
protected:
    std::vector<Color> m_pixels;
    int m_width;
    int m_height;

public:
    Image( ) : m_pixels(), m_width(0), m_height(0) {}
    Image( const int w, const int h, const Color& color= Color::Black() ) : m_pixels(w*h, color), m_width(w), m_height(h) {}
    
    /*! renvoie une reference sur la couleur d'un pixel de l'image.
    permet de modifier et/ou de connaitre la couleur d'un pixel :
    \code
    Image image(512, 512);
    
    image(10, 10)= Red();               // le pixel (10, 10) devient rouge
    image(0, 0)= image(10, 10);         // le pixel (0, 0) recupere la couleur du pixel (10, 10)
    \endcode
    */
    Color& operator() ( const int x, const int y )
    {
        return m_pixels[offset(x, y)];
    }
    
    //! renvoie la couleur d'un pixel de l'image (image non modifiable).
    Color operator() ( const int x, const int y ) const
    {
        return m_pixels[offset(x, y)];
    }
    
    //! renvoie une reference sur le ieme pixel de l'image.
    Color& operator() ( const size_t offset )
    {
        assert(offset < m_pixels.size());
        return m_pixels[offset];
    }
    
    //! renvoie lacouleur du ieme pixel de l'image.
    Color operator() ( const size_t offset ) const
    {
        assert(offset < m_pixels.size());
        return m_pixels[offset];
    }

    Color operator[](int index) const
    {
        return m_pixels[index];
    }

    Color& operator[](int index)
    {
        return m_pixels[index];
    }

    float luminance_of_pixel(int x, int y) const
    {
        Color pixel = m_pixels[offset(x, y)];

        return 0.3086 * pixel.r + 0.6094 * pixel.g + 0.0820 * pixel.b;
    }

    float luminance_of_area(int start_x, int start_y, int stop_x, int stop_y) const
    {
        float luminance = 0.0f;

        for (int x = start_x; x < stop_x; x++)
            for (int y = start_y; y < stop_y; y++)
                luminance += luminance_of_pixel(x, y);

        return luminance;
    }

    float luminance_of_area(const ImageBin& region) const
    {
        return luminance_of_area(region.x0, region.y0, region.x1, region.y1);
    }
    
    //! renvoie la couleur interpolee a la position (x, y) [0 .. width]x[0 .. height].
    Color sample_bilinear( const float x, const float y ) const
    {
        // interpolation bilineaire 
        float u= x - std::floor(x);
        float v= y - std::floor(y);
        int ix= x;
        int iy= y;
        return (*this)(ix, iy)    * ((1 - u) * (1 - v))
            + (*this)(ix+1, iy)   * (u       * (1 - v))
            + (*this)(ix, iy+1)   * ((1 - u) * v)
            + (*this)(ix+1, iy+1) * (u       * v);
    }

    Color sample_floor( const float x, const float y ) const
    {
        // interpolation bilineaire
        float u = std::floor(x);
        float v = std::floor(y);

        return (*this)(u, v);
    }
    
    //! renvoie la couleur interpolee aux coordonnees normalisees (x, y) [0 .. 1]x[0 .. 1].
    Color texture_bilinear( const float x, const float y ) const
    {
        return sample_bilinear(x * m_width, y * m_height);
    }

    Color texture_floor( const float x, const float y ) const
    {
        return sample_floor(x * m_width, y * m_height);
    }
    
    //! renvoie un const pointeur sur le stockage des couleurs des pixels.
    const float *data( ) const
    {
        assert(!m_pixels.empty());
        return (const float *) m_pixels.data();
    }
    
    //! renvoie un pointeur sur le stockage des couleurs des pixels.
    float *data( )
    {
        assert(!m_pixels.empty());
        return (float *) m_pixels.data();
    }

    Color* color_data( )
    {
        return m_pixels.data();
    }
    
    //! renvoie la largeur de l'image.
    int width( ) const { return m_width; }
    //! renvoie la hauteur de l'image.
    int height( ) const { return m_height; }
    //! renvoie le nombre de pixels de l'image.
    unsigned size( ) const { return m_width * m_height; }
    
    //! renvoie l'indice du pixel (x, y) [0 .. width]x[0 .. height].
    //! renvoie le pixel le plus proche si (x, y) est en dehors de l'image...
    unsigned offset( const int x, const int y ) const
    {
        int px= x;
        if(px < 0) px= 0;
        if(px > m_width-1) px= m_width-1;
        int py= y;
        if(py < 0) py= 0;
        if(py > m_height-1) py= m_height-1;
        
        unsigned p= py * m_width + px;
        assert(p < m_pixels.size());
        return p;
    }
};

///@}
#endif
