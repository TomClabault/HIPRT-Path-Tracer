/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/Interaction/RenderWindowMouseInteractor.h"
#include "UI/RenderWindow.h"

void RenderWindowMouseInteractor::glfw_mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    ImGuiIO& io = ImGui::GetIO();
    void* user_pointer = glfwGetWindowUserPointer(window);

    // If it is the render window that is hovered, we're going to move the camera so we take
    // the inputs
    RenderWindow* render_window = reinterpret_cast<RenderWindow*>(user_pointer);
    bool render_window_hovered = render_window->get_imgui_renderer()->get_imgui_render_window().is_hovered();
    bool imgui_want_mouse = io.WantCaptureMouse && !render_window_hovered;
    if (!imgui_want_mouse)
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
