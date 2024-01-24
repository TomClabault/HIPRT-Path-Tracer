#include "triangle.h"

Point Triangle::bbox_centroid() const
{
    return (min(m_a, min(m_b, m_c)) + max(m_a, max(m_b, m_c))) / 2;
}

float Triangle::area() const
{
    return length(cross(m_b - m_a, m_c - m_a)) / 2;
}

Point& Triangle::operator[] (int i)
{
    return *((&m_a) + i);
}

const Point& Triangle::operator[] (int i) const
{
    return *((&m_a) + i);
}
