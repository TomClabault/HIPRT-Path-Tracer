
#ifndef _COLOR_H
#define _COLOR_H

#include "vec.h"

#include <array>
#include <cmath>
#include <iostream>

struct Color
{
    //! constructeur par defaut.
    Color( ) : r(0.f), g(0.f), b(0.f), a(1.f) {}
    explicit Color( const std::array<float, 3>& rgb) : r(rgb[0]), g(rgb[1]), b(rgb[2]), a(1.0f) {}
    explicit Color( const Vector& vec) : r(vec.x), g(vec.y), b(vec.z), a(1.0f) {}
    explicit Color( const float _r, const float _g, const float _y, const float _x= 1.f ) : r(_r), g(_g), b(_y), a(_x) {}
    explicit Color( const float _value ) : r(_value), g(_value), b(_value), a(1.f) {}
    
    //! cree une couleur avec les memes composantes que color, mais remplace sa composante alpha (color.r, color.g, color.b, alpha).
    Color( const Color& color, const float alpha ) : r(color.r), g(color.g), b(color.b), a(alpha) {}  // remplace alpha.

    inline bool operator==(const Color& other)
    {
        return r == other.r && g == other.g && b == other.b;
    }

    inline bool operator!=(const Color& other)
    {
        return !(*this == other);
    }

    inline Color& operator=(const Vector& vec)
    {
        r = vec.x;
        g = vec.y;
        b = vec.z;

        return *this;
    }

    inline Color& operator+=(const Color &other)
    {
        r += other.r;
        g += other.g;
        b += other.b;

        return *this;
    }

    inline Color& operator*=(const Color &other)
    {
        r *= other.r;
        g *= other.g;
        b *= other.b;

        return *this;
    }

    inline Color& operator*=(float k)
    {
        r *= k;
        g *= k;
        b *= k;

        return *this;
    }

    inline Color& operator/=(const float k)
    {
        r /= k;
        g /= k;
        b /= k;

        return *this;
    }

    inline float luminance() const
    {
        return 0.3086f * r + 0.6094f * g + 0.0820f * b;
    }

    inline float power( ) const
    {
        return (r+g+b) / 3;
    }

    inline float max( ) const
    {
        return std::max(r, std::max(g, std::max(b, float(0))));
    }

    static Color Black()
    {
        return Color(0, 0, 0);
    }

    static Color White()
    {
        return Color(1, 1, 1);
    }

    static Color Red()
    {
        return Color(1, 0, 0);
    }

    static Color Green()
    {
        return Color(0, 1, 0);
    }

    static Color Blue()
    {
        return Color(0, 0, 1);
    }

    static Color Yellow()
    {
        return Color(1, 1, 0);
    }

    friend std::ostream& operator << (std::ostream& os, const Color& color);
    
    float r, g, b, a;
};


inline Color operator+ ( const Color& a, const Color& b )
{
    return Color(a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a);
}

inline Color operator- ( const Color& c )
{
    return Color(-c.r, -c.g, -c.b, -c.a);
}

inline Color operator- ( const Color& a, const Color& b )
{
    return a + (-b);
}

inline Color operator* ( const Color& a, const Color& b )
{
    return Color(a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a);
}

inline Color operator* ( const float k, const Color& c )
{
    return Color(c.r * k, c.g * k, c.b * k, c.a * k);
}

inline Color operator* ( const Color& c, const float k )
{
    return k * c;
}

inline Color operator/ ( const Color& a, const Color& b )
{
    return Color(a.r / b.r, a.g / b.g, a.b / b.b, a.a / b.a);
}

inline Color operator/ ( const float k, const Color& c )
{
    return Color(k / c.r, k / c.g, k / c.b, k / c.a);
}

inline Color operator/ ( const Color& c, const float k )
{
    float kk= 1 / k;
    return kk * c;
}

inline Color exp(const Color &col)
{
    return Color(std::exp(col.r), std::exp(col.g), std::exp(col.b), col.a);
}

inline Color pow(const Color &col, float k)
{
    return Color(std::pow(col.r, k), std::pow(col.g, k), std::pow(col.b, k), col.a);
}

///@}
#endif
