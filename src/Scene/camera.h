
#ifndef CAMERA_H
#define CAMERA_H

#include "Maths/mat.h"

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

    Camera(Point position, Point look_at, Vector up_vector, float full_degrees_fov)
    {
        fov = full_degrees_fov;
        full_fov_radians = fov / 180.0f * M_PI;
        fov_dist = 1.0f / std::tan(full_fov_radians / 2.0f);

        Vector x_axis, y_axis, z_axis;
        z_axis = normalize(position - look_at); // Positive z-axis
        x_axis = normalize(-cross(z_axis, normalize(up_vector)));
        y_axis = normalize(cross(z_axis, x_axis));
        view_matrix = Transform(
            x_axis.x, y_axis.x, z_axis.x, position.x,
            x_axis.y, y_axis.y, z_axis.y, position.y,
            x_axis.z, y_axis.z, z_axis.z, position.z,
            0, 0, 0, 1
        );
    }

    Transform view_matrix;

    //Full FOV, not half
    float fov = 45;
    float full_fov_radians = fov / 180.0f * M_PI;
    float fov_dist = 1.0f / std::tan(full_fov_radians / 2.0f);
};

#endif
