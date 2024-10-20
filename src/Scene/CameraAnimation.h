/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef CAMERA_ANIMATION_H
#define CAMERA_ANIMATION_H

#include "Scene/CameraRotationType.h"

#include "glm/vec3.hpp"

class Camera;
class GPURenderer;

class CameraAnimation
{
public:
    void set_camera(Camera* camera);

    void animation_step(GPURenderer* renderer);
    void do_rotation_animation(GPURenderer* renderer);

    // Public attributes here because we want them to be
    // easily accessible and having to use getter/setters
    // everywhere is a pain in the butt
    bool animate = false;

    // If true, the camera will rotate around 'm_rotate_around_point'
    // with 'm_rotation_duration' as the speed target when 'animate' is
    // set to true
    bool m_do_rotation_animation = false;
    CameraRotationType m_rotation_type = CameraRotationType::SECONDS_PER_ROTATION;

    glm::vec3 m_rotate_around_point = glm::vec3(0.0f, 0.0f, 0.0f);
    // Rotation speed in number of rotations around the object per second
    float m_rotation_value = 8.0f;

private:
    Camera* m_camera = nullptr;
};

#endif
