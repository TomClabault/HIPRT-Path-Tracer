/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RAY_H
#define RAY_H

#include "Maths/vec.h"

struct Ray
{
    Ray(Point origin, Point direction_point) : origin(origin), direction(normalize(direction_point - origin)) {}
    Ray(Point origin, Vector direction) : origin(origin), direction(direction) {}

    friend std::ostream& operator << (std::ostream& os, const Ray& ray);

    Point origin;
    Vector direction;
};

#endif
