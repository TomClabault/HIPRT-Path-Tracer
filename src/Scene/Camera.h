/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef CAMERA_H
#define CAMERA_H

#include "HostDeviceCommon/Camera.h"

#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"
#include "glm/gtc/quaternion.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

struct Camera
{
    Camera();

    HIPRTCamera to_hiprt();
    glm::mat4x4 get_view_matrix() const;
    glm::vec3 get_view_direction() const;

    glm::mat4x4 projection_matrix;
    float vertical_fov;
    float near_plane, far_plane;

    glm::vec3 translation = glm::vec3(0, 0, 0);
    glm::quat rotation = glm::quat(glm::vec3(0, 0, 0));
};

#endif
