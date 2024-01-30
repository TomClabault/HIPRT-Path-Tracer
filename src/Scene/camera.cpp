#include "camera.h"

const glm::mat4x4 Camera::DEFAULT_COORDINATES_SYSTEM = glm::mat4x4(
     1.0f, 0.0f, 0.0f, 0.0f,
     0.0f, 1.0f, 0.0f, 0.0f,
     0.0f, 0.0f, -1.0f, 0.0f,
     0.0f, 0.0f, 0.0f, 1.0f
 );

Camera::Camera() : view_matrix(DEFAULT_COORDINATES_SYSTEM)
{
    fov = 45;
    full_fov_radians = fov / 180.0f * (float)M_PI;
    fov_dist = 1.0f / std::tan(full_fov_radians / 2.0f);
}

Camera::Camera(float full_fov, glm::mat4x4 transformation) : view_matrix(transformation* DEFAULT_COORDINATES_SYSTEM)
{
    fov = full_fov;
    full_fov_radians = fov / 180.0f * M_PI;
    fov_dist = 1.0f / std::tan(full_fov_radians / 2.0f);
}

Camera::Camera(glm::vec3 position, glm::vec3 look_at, glm::vec3 up_vector, float full_degrees_fov)
{
    fov = full_degrees_fov;
    full_fov_radians = fov / 180.0f * M_PI;
    fov_dist = 1.0f / std::tan(full_fov_radians / 2.0f);

    glm::vec3 x_axis, y_axis, z_axis;
    z_axis = normalize(position - look_at); // Positive z-axis
    x_axis = normalize(-cross(z_axis, normalize(up_vector)));
    y_axis = normalize(cross(z_axis, x_axis));
    view_matrix = glm::mat4x4(
        x_axis.x, x_axis.y, x_axis.z, 0,
        y_axis.x, y_axis.y, y_axis.z, 0,
        z_axis.x, z_axis.y, z_axis.z, 0,
        position.x, position.y, position.z, 1
    );
}
