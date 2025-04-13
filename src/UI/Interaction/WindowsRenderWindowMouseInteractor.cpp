/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/Interaction/WindowsRenderWindowMouseInteractor.h"
#include "UI/RenderWindow.h"

#include "GLFW/glfw3.h"
#include "imgui.h"

void WindowsRenderWindowMouseInteractor::glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	void* user_pointer = glfwGetWindowUserPointer(window);
	RenderWindow* render_window = reinterpret_cast<RenderWindow*>(user_pointer);

	std::shared_ptr<RenderWindowMouseInteractor> interactor_instance = render_window->get_mouse_interactor();
	std::shared_ptr<WindowsRenderWindowMouseInteractor> windows_interactor = std::dynamic_pointer_cast<WindowsRenderWindowMouseInteractor>(interactor_instance);

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

	bool is_mouse_pressed = interactor_instance->is_interacting();
	if (is_mouse_pressed)
	{
		double current_x, current_y;
		glfwGetCursorPos(window, &current_x, &current_y);
		windows_interactor->m_grab_cursor_position = std::make_pair(static_cast<float>(current_x), static_cast<float>(current_y));

		windows_interactor->m_just_pressed = true;
	}
	else
		windows_interactor->m_just_pressed = false;
}

void WindowsRenderWindowMouseInteractor::glfw_mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos)
{
	void* user_pointer = glfwGetWindowUserPointer(window);
	RenderWindow* render_window = reinterpret_cast<RenderWindow*>(user_pointer);

	std::shared_ptr<RenderWindowMouseInteractor> interactor_instance = render_window->get_mouse_interactor();
	std::shared_ptr<WindowsRenderWindowMouseInteractor> windows_interactor = std::dynamic_pointer_cast<WindowsRenderWindowMouseInteractor>(interactor_instance);

	ImGuiIO& io = ImGui::GetIO();

	// If it is the render window that is hovered, we're going to move the camera so we take
	// the inputs
	bool render_window_hovered = render_window->get_imgui_renderer()->get_imgui_render_window().is_hovered();
	bool imgui_want_mouse = io.WantCaptureMouse && !render_window_hovered;
	if (!imgui_want_mouse)
	{
		if (windows_interactor->m_just_pressed)
		{
			// We want to skip the frame where the mouse is being repositioned to
			// the center of the screen because if the cursor wasn't at the center,
			// we're going to consider to delta from the old position to the center as
			// the moving having moved but it's not the case. The user didn't move the
			// mouse, it's us forcing it in the center of the viewport
			windows_interactor->m_just_pressed = false;

			return;
		}

		RenderWindow* render_window = reinterpret_cast<RenderWindow*>(glfwGetWindowUserPointer(window));

		float xposf = static_cast<float>(xpos);
		float yposf = static_cast<float>(ypos);

		std::pair<float, float> grab_position = windows_interactor->m_grab_cursor_position;
		if (grab_position.first == -1 && grab_position.second == -1)
			// If this is the first position of the cursor, nothing to do
			;
		else
		{
			// Computing the difference in movement
			std::pair<float, float> difference = std::make_pair(xposf - grab_position.first, yposf - grab_position.second);

			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
				render_window->update_renderer_view_translation(-difference.first / 300.0f, difference.second / 300.0f, false);

			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
				render_window->update_renderer_view_rotation(-difference.first, -difference.second);
		}

		// Updating the position
		if (interactor_instance->is_interacting())
			// Locking the cursor in place as long as we're moving the camera
			glfwSetCursorPos(window, grab_position.first, grab_position.second);
	}
}

void WindowsRenderWindowMouseInteractor::set_callbacks(GLFWwindow* window)
{
    glfwSetCursorPosCallback(window, WindowsRenderWindowMouseInteractor::glfw_mouse_cursor_callback);
    glfwSetMouseButtonCallback(window, WindowsRenderWindowMouseInteractor::glfw_mouse_button_callback);
	glfwSetScrollCallback(window, RenderWindowMouseInteractor::glfw_mouse_scroll_callback);
}
