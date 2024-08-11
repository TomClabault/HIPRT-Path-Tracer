/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/RenderWindowMouseInteractor.h"
#include "UI/RenderWindow.h"

void RenderWindowMouseInteractor::glfw_mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse)
    {
        RenderWindow* render_window = reinterpret_cast<RenderWindow*>(glfwGetWindowUserPointer(window));

        yoffset = std::copysignf(1.0f, yoffset);

        // Because the mouse scroll isn't a continuous input, we can't use the delta time of the application reliably
        // to scale the speed of the zoom in the scene so we're hardcoding an arbitrary 12.0f here that proved to be
        // okay good
        render_window->update_renderer_view_zoom(static_cast<float>(-yoffset / 12.0f), false);
    }
}

bool RenderWindowMouseInteractor::is_interacting()
{
    return  m_interacting_left_button || m_interacting_right_button;
}

void RenderWindowMouseInteractor::set_interacting_left_button(bool interacting)
{
    m_interacting_left_button = interacting;
}

void RenderWindowMouseInteractor::set_interacting_right_button(bool interacting)
{
    m_interacting_right_button = interacting;
}

bool RenderWindowMouseInteractor::is_interacting_right_button()
{
    return m_interacting_right_button;
}

bool RenderWindowMouseInteractor::is_interacting_left_button()
{
    return m_interacting_left_button;
}
