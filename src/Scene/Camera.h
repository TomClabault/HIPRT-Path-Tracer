/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef CAMERA_H
#define CAMERA_H

#include "HostDeviceCommon/HIPRTCamera.h"
#include "Scene/BoundingBox.h"

#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"
#include "glm/gtc/quaternion.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

/**
 * Camera class meant for being manipulated through used interactions
 * etc... (hence the attributes translation and rotation for example)
 * 
 * The curated camera class that is meant for being used by the shaders is HIPRTCamera
 */
struct Camera
{
    // Variable used when calling Camera::auto_adjust_speed().
    // This is a time in seconds that represents how long it will take for the camera
    // to traverse the scene along its largest extent if the user holds the 'W' key for example.
    //
    // Note that this may be a little scuffed for scenes that are very elongated.
    static constexpr float SCENE_CROSS_TIME = 5.0f;

    Camera();

    HIPRTCamera to_hiprt();
    glm::mat4x4 get_view_matrix() const;
    glm::vec3 get_view_direction() const;

    void set_aspect(float new_aspect);

    /**
     * The given FOV must be in radians
     */
    void set_FOV(float new_fov);

    /**
     * Adjusts the speed attributes of this camera so that the camera
     */
    void auto_adjust_speed(const BoundingBox& scene_bounding_box);

    void translate(glm::vec3 translation_vec);
    /**
     * Basically a handy function for translating a certain distance in the direction
     * the camera is looking at
     */
    void zoom(float offset);
    void rotate(glm::vec3 rotation_angles);

    glm::mat4x4 projection_matrix;

    // Whether or not to jitter rays direction for anti-aliasing during the rendering
    bool do_jittering = false;

    // Vertical FOV in radians
    float vertical_fov = M_PI / 2.0f;
    float near_plane = 0.1f;
    float far_plane = 1000.0f;
    // Aspect ratio
    float aspect = 16.0f / 9.0f;

    // Camera movement speed. In world unit per second
    float camera_movement_speed = 1.0f;
    // Multiplier on the camera speed that the user can manipulate through the UI
    float user_movement_speed_multiplier = 1.0f;

    glm::vec3 translation = glm::vec3(0, 0, 0);
    glm::quat rotation = glm::quat(glm::vec3(0, 0, 0));
};

#endif
