/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Scene/Camera.h"

Camera::Camera()
{
    translation = glm::vec3(0.0f, 0.0f, 0.0f);
    rotation = glm::quat(glm::vec3(0.0f, 0.0f, 0.0f));
}

HIPRTCamera Camera::to_hiprt()
{
    HIPRTCamera hiprt_cam;

    glm::mat4x4 view_matrix = get_view_matrix();
    glm::mat4x4 view_matrix_inv = glm::inverse(view_matrix);
    glm::mat4x4 projection_matrix_inv = glm::inverse(projection_matrix);

    hiprt_cam.inverse_view = *reinterpret_cast<float4x4*>(&view_matrix_inv);
    hiprt_cam.inverse_projection = *reinterpret_cast<float4x4*>(&projection_matrix_inv);
    hiprt_cam.do_jittering = do_jittering;

    return hiprt_cam;
}

glm::mat4x4 Camera::get_view_matrix() const
{
    glm::mat4x4 view_matrix = glm::inverse(glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(glm::normalize(rotation)));

    return view_matrix;
}

glm::vec3 Camera::get_view_direction() const
{
    glm::mat4x4 view_mat = get_view_matrix();

    return glm::vec3(view_mat[0][2], view_mat[1][2], view_mat[2][2]);
}

void Camera::auto_adjust_speed(const BoundingBox& scene_bounding_box)
{
    camera_movement_speed = scene_bounding_box.get_max_extent() / Camera::SCENE_CROSS_TIME;
}
