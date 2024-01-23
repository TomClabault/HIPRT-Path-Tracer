#include <algorithm>

#include "color.h"

std::ostream& operator << (std::ostream& os, const Color& color)
{
    os << "Color[" << color.r << ", " << color.g << ", " << color.b << "]";
    return os;
}
