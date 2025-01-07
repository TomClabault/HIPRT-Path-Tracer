/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Scene/Camera.h"
#include "Scene/CameraAnimation.h"

void CameraAnimation::set_camera(Camera* camera)
{
    m_camera = camera;
}

void CameraAnimation::animation_step(GPURenderer* renderer, float delta_time)
{
    // We can step the animation either if we're not accumulating or
    // if we're accumulating and we're allowed to step the animations
    bool can_step_animation = false;
    can_step_animation |= renderer->get_render_settings().accumulate && renderer->get_animation_state().can_step_animation;
    can_step_animation |= !renderer->get_render_settings().accumulate;

    if (animate && renderer->get_animation_state().do_animations && can_step_animation)
    {
        do_rotation_animation(delta_time);
    }
}

void CameraAnimation::do_rotation_animation(float delta_time)
{
    if (m_do_rotation_animation)
    {
        float rotation_angle_y_deg = 0.0f;

        // Modifying the camera's properties based on the chosen rotation type
        switch (m_rotation_type)
        {
            case SECONDS_PER_ROTATION:
                // Converting 'm_rotation_value' so that the camera
                // rotates at such a speed that it will rotate 360.0f
                // degrees in 'm_rotation_value' seconds
                rotation_angle_y_deg = 360.0f / m_rotation_value * (delta_time / 1000.0f);
                break;

            case DEGREES_PER_FRAME:
                // For 'DEGREES_PER_FRAME', 'm_rotation_value' is already
                // in degrees so we can just use that value
                rotation_angle_y_deg = m_rotation_value;
                break;

            default:
                break;
        }

        float rotation_angle_y_rad = rotation_angle_y_deg / 180.0f * M_PI;

        m_camera->rotate_around_point(m_rotate_around_point, make_float3(0.0f, rotation_angle_y_rad, 0.0f));
    }
}
