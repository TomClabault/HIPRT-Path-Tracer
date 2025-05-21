/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Scene/Camera.h"

HIPRTCamera Camera::to_hiprt(int render_width, int render_height)
{
    HIPRTCamera hiprt_cam;

    glm::mat4x4 view_matrix = get_view_matrix();
    glm::mat4x4 view_matrix_inv = glm::inverse(view_matrix);
    glm::mat4x4 projection_matrix_inv = glm::inverse(projection_matrix);

    hiprt_cam.inverse_view = *reinterpret_cast<float4x4*>(&view_matrix_inv);
    hiprt_cam.inverse_projection = *reinterpret_cast<float4x4*>(&projection_matrix_inv);

    glm::mat4x4 view_projection = view_matrix * projection_matrix;
    hiprt_cam.view_projection = *reinterpret_cast<float4x4*>(&view_projection);

    glm::vec4 position_glm = glm::vec4(0, 0, 0, 1) * view_matrix_inv;
    hiprt_cam.position = make_float3(position_glm.x, position_glm.y, position_glm.z);

    hiprt_cam.vertical_fov = vertical_fov;
    hiprt_cam.sensor_width = render_width;
    hiprt_cam.sensor_height = render_height;

    hiprt_cam.do_jittering = do_jittering;

    return hiprt_cam;
}

glm::mat4x4 Camera::get_view_matrix() const
{
    // For our FPS camera, we want to translate first and then rotate
    // (so that we rotate around the current position of the camera = FPS camera).
    // 
    // Because our matrix multiplication in the shaders is by the right
    // (we multiply point by the right: M * point), we want the translation matrix 
    // on the right on the view matrix construction. We're inverting (or conjugating) 
    // the translation (or the rotation) because we want to construct a 
    // world-to-view matrix so we effectively need to reverse the transformations. 
    // 
    // For example, if the world space position of the camera is (5, 0, 0), 
    // then its translation is (5, 0, 0), obviously. If we now multiply a point 
    // with world space coordinates (5, 0, 0) by the view matrix, we're supposed to
    // get (0, 0, 0) for the point coordinates in view space since the point is at the 
    // camera's position. This is why we need to add (-5, 0, 0) to the point's position. 
    // We need to apply the inverse translation to bring from world space to view space. 
    // 
    // Same for the rotation
    //
    // We transpose the result because glm is column major in memory. We want row major.
    // Our convention is right-multiplying row major matrix by a column vector/point (which is on the right)
    glm::mat4x4 view_matrix = glm::transpose(glm::mat4_cast(glm::conjugate(glm::normalize(m_rotation))) * glm::translate(glm::mat4(1.0f), -m_translation));

    return view_matrix;
}

void Camera::set_aspect(float new_aspect)
{
    aspect = new_aspect;

    // Recomputing the projection matrix with the new aspect
    projection_matrix = glm::perspective(vertical_fov, new_aspect, near_plane, far_plane);
}

void Camera::set_FOV_radians(float new_fov)
{
    vertical_fov = new_fov;

    // Recomputing the projection matrix with the new FOV
    projection_matrix = glm::perspective(new_fov, aspect, near_plane, far_plane);
}

void Camera::auto_adjust_speed(const BoundingBox& scene_bounding_box)
{
    if (scene_bounding_box.get_max_extent() > 1.0e35f)
        // Probably an empty scene, we can't adjust the camera speed based on the scene
        return;

    camera_movement_speed = scene_bounding_box.get_max_extent() / Camera::SCENE_CROSS_TIME;
}

void Camera::translate(glm::vec3 translation_vec)
{
    m_translation = m_translation + translation_vec * glm::conjugate(m_rotation);
}

void Camera::translate(float3 translation_vec)
{
    translate(glm::vec3(translation_vec.x, translation_vec.y, translation_vec.z));
}

void Camera::zoom(float offset)
{
    glm::vec3 zoom_translation(0, 0, offset);
    m_translation = m_translation + zoom_translation * glm::conjugate(m_rotation);
}

/**
 * Reference:
 * 
 * https://stackoverflow.com/questions/12435671/quaternion-lookat-function
 */
void Camera::look_at_object(const BoundingBox& object_bounding_box)
{
    float3 object_center = object_bounding_box.get_center();
    float3 new_camera_position = object_center;
    new_camera_position += object_bounding_box.get_max_extent() * make_float3(3.0f, 0.0f, 0.0f);
    new_camera_position += object_bounding_box.get_max_extent() * make_float3(0.0f, 1.0f, 0.0f);

    glm::vec3 new_position_glm = glm::vec3(new_camera_position.x, new_camera_position.y, new_camera_position.z);
    glm::vec3 object_center_glm = glm::vec3(object_center.x, object_center.y, object_center.z);

    m_translation = new_position_glm;

    // Forward vector is looking away from the target since we're using
    // the "camera looking down -Z" convention
    glm::vec3 look_at_vector = glm::normalize(m_translation - object_center_glm);
    glm::vec3 right_axis = glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), look_at_vector);
    glm::vec3 rotated_up_axis = glm::cross(look_at_vector, right_axis);

    glm::mat3x3 rot_mat = glm::mat3x3(right_axis, rotated_up_axis, look_at_vector);
    m_rotation = glm::quat(rot_mat);
}

void Camera::rotate(glm::vec3 rotation_angles_rad)
{
    glm::quat qx = glm::angleAxis(rotation_angles_rad.x, glm::vec3(1.0f, 0.0f, 0.0f));
    glm::quat qy = glm::angleAxis(rotation_angles_rad.y, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::quat qz = glm::angleAxis(rotation_angles_rad.z, glm::vec3(0.0f, 0.0f, 1.0f));

    glm::quat new_orientation = glm::normalize(qy * m_rotation * qx * qz);
    m_rotation = new_orientation;
}

void Camera::rotate(float3 rotation_angles_rad)
{
    rotate(glm::vec3(rotation_angles_rad.x, rotation_angles_rad.y, rotation_angles_rad.z));
}

void Camera::rotate_around_point(const float3& point, const float3& angles_rad)
{
    glm::quat rotation_quat_x = glm::angleAxis(angles_rad.x, glm::vec3(1.0f, 0.0f, 0.0f));
    glm::quat rotation_quat_y = glm::angleAxis(angles_rad.y, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::quat rotation_quat_z = glm::angleAxis(angles_rad.z, glm::vec3(0.0f, 0.0f, 1.0f));

    glm::mat4x4 rotation_mat_x = glm::mat4_cast(rotation_quat_x);
    glm::mat4x4 rotation_mat_y = glm::mat4_cast(rotation_quat_y);
    glm::mat4x4 rotation_mat_z = glm::mat4_cast(rotation_quat_z);

    glm::vec3 point_glm = glm::vec3(point.x, point.y, point.z);
    glm::mat4x4 rotation_mat = rotation_mat_z * rotation_mat_y * rotation_mat_x;
    m_translation = rotation_mat * glm::vec4(m_translation - point_glm, 1.0f);
    m_translation += point_glm;

    glm::mat4x4 rot_mat = glm::mat4_cast(m_rotation);
    rot_mat = rotation_mat * rot_mat;
    m_rotation = glm::quat(rot_mat);
}

void Camera::rotate_around_point(const glm::vec3& point, const float3& angles_rad)
{
    rotate_around_point(make_float3(point.x, point.y, point.z), angles_rad);
}
