#include "ray.h"

std::ostream& operator <<(std::ostream& os, const Ray& ray)
{
    os << "Ray(" << ray.origin << ", " << ray.direction << ")";
    return os;
}
