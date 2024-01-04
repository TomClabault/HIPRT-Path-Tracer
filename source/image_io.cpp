
#include <cfloat>

#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Color gamma( const Color& color, const float g )
{
    return Color(std::pow(color.r, g), std::pow(color.g, g), std::pow(color.b, g), color.a);
}

Image gamma( const Image& image, const float g= float(2.2) )
{
    Image tmp(image.width(), image.height());
    
    float invg= 1 / g;
    for(unsigned i= 0; i < image.size(); i++)
        tmp(i)= gamma(image(i), invg);
    
    return tmp;
}

Image inverse_gamma( const Image& image, const float g= float(2.2) )
{
    Image tmp(image.width(), image.height());
    
    for(unsigned i= 0; i < image.size(); i++)
        tmp(i)= gamma(image(i), g);
    
    return tmp;
}

float range( const Image& image )
{
    float gmin= FLT_MAX;
    float gmax= 0;
    for(unsigned i= 0; i < image.size(); i++)
    {
        Color color= image(i);
        float g= color.r + color.g + color.b;
        
        if(g < gmin) gmin= g;
        if(g > gmax) gmax= g;
    }
    
    int bins[100] = {};
    for(unsigned i= 0; i < image.size(); i++)
   {
        Color color= image(i);
        float g= color.r + color.g + color.b;
        
        int b= (g - gmin) * 100 / (gmax - gmin);
        if(b >= 99) b= 99;
        if(b < 0) b= 0;
        bins[b]++;
    }
    
    float qbins= 0;
    for(unsigned i= 0; i < 100; i++)
    {
        qbins= qbins + float(bins[i]) / float(image.size());
        if(qbins > .75f)
            return gmin + float(i+1) / 100 * (gmax - gmin);
    }
    
    return gmax;
}


Image tone( const Image& image, const float saturation, const float gamma )
{
    Image tmp(image.width(), image.height());
    
    float invg= 1 / gamma;
    float k= 1 / std::pow(saturation, invg);
    for(unsigned i= 0; i < image.size(); i++)
    {
        Color color= image(i);
        if(std::isnan(color.r) || std::isnan(color.g) || std::isnan(color.b))
            // marque les pixels pourris avec une couleur improbable...            
            color= Color(1, 0, 1);
        else
            // sinon transformation gamma rgb -> srgb
            color= Color(k * std::pow(color.r, invg), k * std::pow(color.g, invg), k * std::pow(color.b, invg));
        
        tmp(i)= Color(color, 1);
    }
    
    return tmp;
}

Image read_image( const char *filename, const bool flipY )
{
    stbi_set_flip_vertically_on_load(flipY);
    
    if(!stbi_is_hdr(filename))
    {
        int width, height, channels;
        unsigned char *data= stbi_load(filename, &width, &height, &channels, 4);
        if(!data)
        {
            printf("[error] loading '%s'...\n", filename);
            return {};
        }
        
        Image image(width, height);
        for(unsigned i= 0, offset= 0; i < image.size(); i++, offset+= 4)
        {
            Color pixel= Color( 
                data[offset], 
                data[offset + 1],
                data[offset + 2],
                data[offset + 3]) / 255;
            image(i)= pixel;
        }
        
        stbi_image_free(data);
        return image;
        
        // \todo utiliser stbi_loadf() dans tous les cas, + parametres de conversion
        //     stbi_ldr_to_hdr_scale(1.0f);
        //     stbi_ldr_to_hdr_gamma(2.2f);
    }
    else
    {
        int width, height, channels;
        float *data= stbi_loadf(filename, &width, &height, &channels, 4);
        if(!data)
        {
            printf("[error] loading '%s'...\n", filename);
            return {};
        }
        
        Image image(width, height);
        for(unsigned i= 0, offset= 0; i < image.size(); i++, offset+= 4)
        {
            Color pixel= Color( 
                data[offset], 
                data[offset + 1],
                data[offset + 2],
                data[offset + 3]);
            image(i)= pixel;
        }
        
        stbi_image_free(data);
        return image;
    }
    
    return {};
}

inline float clamp( const float x, const float min, const float max )
{
    if(x < min) return min;
    else if(x > max) return max;
    else return x;
}


bool write_image_png( const Image& image, const char *filename, const bool flipY )
{
    if(image.size() == 0)
        return false;
    
    std::vector<unsigned char> tmp(image.width()*image.height()*4);
    for(unsigned i= 0, offset= 0; i < image.size(); i++, offset+= 4)
    {
        Color pixel= image(i) * 255;
        tmp[offset   ]= clamp(pixel.r, 0, 255);
        tmp[offset +1]= clamp(pixel.g, 0, 255);
        tmp[offset +2]= clamp(pixel.b, 0, 255);
        tmp[offset +3]= clamp(pixel.a, 0, 255);
    }
    
    stbi_flip_vertically_on_write(flipY);
    return stbi_write_png(filename, image.width(), image.height(), 4, tmp.data(), image.width() * 4) != 0;
}

bool write_image( const Image& image, const char *filename, const bool flipY )
{
    return write_image_png(image, filename, flipY );
}

bool write_image_bmp( const Image& image, const char *filename, const bool flipY )
{
    if(image.size() == 0)
        return false;
    
    std::vector<unsigned char> tmp(image.width()*image.height()*4);
    for(unsigned i= 0, offset= 0; i < image.size(); i++, offset+= 4)
    {
        Color pixel= image(i) * 255;
        tmp[offset   ]= pixel.r;
        tmp[offset +1]= pixel.g;
        tmp[offset +2]= pixel.b;
        tmp[offset +3]= pixel.a;
    }
    
    stbi_flip_vertically_on_write(flipY);
    return stbi_write_bmp(filename, image.width(), image.height(), 4, tmp.data()) != 0;
}

bool write_image_hdr( const Image& image, const char *filename, const bool flipY )
{
    if(image.size() == 0)
        return false;
    
    stbi_flip_vertically_on_write(flipY);
    return stbi_write_hdr(filename, image.width(), image.height(), 4, image.data()) != 0;
}

bool write_image_preview( const Image& image, const char *filename, const bool flipY, const float g )
{
    if(image.size() == 0)
        return false;
    
    Image tmp= tone(image, range(image), g);
    return write_image_png(tmp, filename, flipY);
}

