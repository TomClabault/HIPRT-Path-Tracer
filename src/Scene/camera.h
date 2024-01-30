#ifndef CAMERA_H
#define CAMERA_H

#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

struct Camera
{
    static const glm::mat4x4 DEFAULT_COORDINATES_SYSTEM;

    Camera();

    /**
     * @brief Camera
     * @param full_fov In degrees
     * @param transformation
     */
    Camera(float full_fov, glm::mat4x4 transformation = glm::mat4x4(1.0f));

    Camera(glm::vec3 position, glm::vec3 look_at, glm::vec3 up_vector, float full_degrees_fov);

    glm::mat4x4 view_matrix;

    //Full FOV, not half
    float fov = 45;
    float full_fov_radians = fov / 180.0f * M_PI;
    float fov_dist = 1.0f / std::tan(full_fov_radians / 2.0f);
};

#endif
