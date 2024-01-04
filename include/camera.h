
#ifndef CAMERA_H
#define CAMERA_H

#include "mat.h"

#define _USE_MATH_DEFINES
#include <math.h>

struct Camera
{
    static const Transform DEFAULT_COORDINATES_SYSTEM;
    static const Camera CORNELL_BOX_CAMERA, GANESHA_CAMERA, ITE_ORB_CAMERA, PBRT_DRAGON_CAMERA, MIS_CAMERA;

    Camera() : view_matrix(DEFAULT_COORDINATES_SYSTEM)
    {
        fov = 45;
        full_fov_radians = fov / 180.0f * (float)M_PI;
        fov_dist = 1.0f / std::tan(full_fov_radians / 2.0f);
    }

    /**
     * @brief Camera
     * @param full_fov In degrees
     * @param transformation
     */
    Camera(float full_fov, Transform transformation = Identity()) : view_matrix(transformation * DEFAULT_COORDINATES_SYSTEM)
    {
        fov = full_fov;
        full_fov_radians = fov / 180.0f * M_PI;
        fov_dist = 1.0f / std::tan(full_fov_radians / 2.0f);
    }

    Transform view_matrix;

    //Full FOV, not half
    float fov = 45;
    float full_fov_radians = fov / 180.0f * M_PI;
    float fov_dist = 1.0f / std::tan(full_fov_radians / 2.0f);
};

#endif
