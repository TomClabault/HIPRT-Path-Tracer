/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Ray.h"

std::ostream& operator <<(std::ostream& os, const Ray& ray)
{
    os << "Ray(" << ray.origin << ", " << ray.direction << ")";
    return os;
}
