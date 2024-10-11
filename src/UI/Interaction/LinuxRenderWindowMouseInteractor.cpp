/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/Interaction/LinuxRenderWindowMouseInteractor.h"
#include "UI/RenderWindow.h"

#include "GLFW/glfw3.h"

#include "imgui.h"

void LinuxRenderWindowMouseInteractor::glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    void* user_pointer = glfwGetWindowUserPointer(window);
    RenderWindow* render_window = reinterpret_cast<RenderWindow*>(user_pointer);

    std::shared_ptr<RenderWindowMouseInteractor> interactor_instance_ptr = render_window->get_mouse_interactor();
    LinuxRenderWindowMouseInteractor* interactor_instance = static_cast<LinuxRenderWindowMouseInteractor*>(interactor_instance_ptr.get());

    // If it is the render window that is hovered, we're going to move the camera so we take
    // the inputs
    bool render_window_hovered = render_window->get_imgui_renderer()->get_imgui_render_window().is_hovered();
    bool imgui_wants_mouse = ImGui::GetIO().WantCaptureMouse && !render_window_hovered;

    switch (button)
    {
    case GLFW_MOUSE_BUTTON_LEFT:
        interactor_instance->set_interacting_left_button((action == GLFW_PRESS) && !imgui_wants_mouse);

        break;

    case GLFW_MOUSE_BUTTON_RIGHT:
        interactor_instance->set_interacting_right_button((action == GLFW_PRESS) && !imgui_wants_mouse);

        break;
    }

    if (action == GLFW_PRESS)
        // If the user just clicked over the render window, setting the boolean to true, false otherwise
        interactor_instance->render_window_hovered_on_click = render_window_hovered;
    else if (action == GLFW_RELEASE)
        interactor_instance->render_window_hovered_on_click = false;

    bool interacting = interactor_instance->is_interacting();
    if (interacting)
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    else
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void LinuxRenderWindowMouseInteractor::glfw_mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos)
{
    ImGuiIO& io = ImGui::GetIO();
    void* user_pointer = glfwGetWindowUserPointer(window);

    // If it is the render window that is hovered, we're going to move the camera so we take
    // the inputs
    RenderWindow* render_window = reinterpret_cast<RenderWindow*>(user_pointer);
    std::shared_ptr<RenderWindowMouseInteractor> interactor_instance_ptr = render_window->get_mouse_interactor();
    LinuxRenderWindowMouseInteractor* interactor_instance = static_cast<LinuxRenderWindowMouseInteractor*>(interactor_instance_ptr.get());

    // If the render window was hovered when the user clicked, then the user is trying to move the camera
    bool render_window_hovered_when_clicked = interactor_instance->render_window_hovered_on_click;

    bool render_window_hovered = render_window->get_imgui_renderer()->get_imgui_render_window().is_hovered();
    bool imgui_wants_mouse = io.WantCaptureMouse && !render_window_hovered && !render_window_hovered_when_clicked;

    if (!imgui_wants_mouse)
    {
        float xposf = static_cast<float>(xpos);
        float yposf = static_cast<float>(ypos);

        if (render_window_hovered_when_clicked)
        {
            std::pair<float, float> old_position = render_window->get_cursor_position();
            if (old_position.first == -1 && old_position.second == -1)
                // If this is the first position of the cursor, nothing to do
                ;
            else
            {
                // Computing the difference in movement
                std::pair<float, float> difference = std::make_pair(xposf - old_position.first, yposf - old_position.second);
                static int counter = 0;

                if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
                    render_window->update_renderer_view_translation(-difference.first, difference.second, true);

                if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
                    render_window->update_renderer_view_rotation(-difference.first, -difference.second);
            }

        }

        // Updating the position
        render_window->set_cursor_position(std::make_pair(xposf, yposf));
    }
}

void LinuxRenderWindowMouseInteractor::set_callbacks(GLFWwindow* window)
{
    glfwSetCursorPosCallback(window, LinuxRenderWindowMouseInteractor::glfw_mouse_cursor_callback);
    glfwSetMouseButtonCallback(window, LinuxRenderWindowMouseInteractor::glfw_mouse_button_callback);
	glfwSetScrollCallback(window, RenderWindowMouseInteractor::glfw_mouse_scroll_callback);
}