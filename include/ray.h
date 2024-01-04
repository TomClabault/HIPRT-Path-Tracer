#ifndef RAY_H
#define RAY_H

#include "vec.h"

enum RayState
{
    BOUNCE,
    MISSED,
    TERMINATED
};

struct Ray
{
    Ray(Point origin, Point direction_point) : origin(origin), direction(normalize(direction_point - origin)) {}
    Ray(Point origin, Vector direction) : origin(origin), direction(direction) {}

    friend std::ostream& operator << (std::ostream& os, const Ray& ray);

    Point origin;
    Vector direction;
};

#endif
