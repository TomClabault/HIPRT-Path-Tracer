#ifndef CAMERA_H
#define CAMERA_H

#include "Kernels/includes/HIPRT_camera.h"

#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"
#include "glm/gtc/quaternion.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

struct Camera
{
    static const glm::mat4x4 DEFAULT_COORDINATES_SYSTEM;

    Camera();

    HIPRTCamera to_hiprt();
    glm::mat4x4 get_view_matrix() const;
    glm::vec3 get_view_direction() const;

    glm::mat4x4 projection_matrix;
    glm::vec3 translation = glm::vec3(0, 0, 0);
    glm::quat rotation = glm::quat(glm::vec3(0, 0, 0));
};

#endif
