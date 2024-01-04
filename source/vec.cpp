
#include <algorithm>
#include <cmath>

#include "vec.h"


Point Origin( )
{
    return Point(0, 0, 0);
}


float distance( const Point& a, const Point& b )
{
    return length(a - b);
}

float distance2( const Point& a, const Point& b )
{
    return length2(a - b);
}

Point center( const Point& a, const Point& b )
{
    return Point((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
}


Point min( const Point& a, const Point& b )
{ 
    return Point( std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z) ); 
}

Point max( const Point& a, const Point& b ) 
{ 
    return Point( std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z) ); 
}
