/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/WindowsRenderWindowMouseInteractor.h"
#include "UI/RenderWindow.h"

#include "GLFW/glfw3.h"
#include "imgui.h"

bool WindowsRenderWindowMouseInteractor::m_interacting_left_button = false;
bool WindowsRenderWindowMouseInteractor::m_interacting_right_button = false;
bool WindowsRenderWindowMouseInteractor::m_just_pressed = false;
std::pair<float, float> WindowsRenderWindowMouseInteractor::m_grab_cursor_position = std::make_pair<float, float>(0.0f, 0.0f);

void WindowsRenderWindowMouseInteractor::glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	bool imgui_wants_mouse = ImGui::GetIO().WantCaptureMouse;

	switch (button)
	{
	case GLFW_MOUSE_BUTTON_LEFT:
		m_interacting_left_button = (action == GLFW_PRESS && !imgui_wants_mouse);

		break;

	case GLFW_MOUSE_BUTTON_RIGHT:
		m_interacting_right_button = (action == GLFW_PRESS && !imgui_wants_mouse);

		break;
	}

	RenderWindow* render_window = reinterpret_cast<RenderWindow*>(glfwGetWindowUserPointer(window));

	bool is_mouse_pressed = m_interacting_left_button || m_interacting_right_button;
	if (is_mouse_pressed)
	{
		double current_x, current_y;
		glfwGetCursorPos(window, &current_x, &current_y);
		m_grab_cursor_position = std::make_pair(static_cast<float>(current_x), static_cast<float>(current_y));

		m_just_pressed = true;
	}
	else
		m_just_pressed = false;

	render_window->set_render_low_resolution(is_mouse_pressed);
}

void WindowsRenderWindowMouseInteractor::glfw_mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos)
{
	ImGuiIO& io = ImGui::GetIO();
	if (!io.WantCaptureMouse)
	{
		if (m_just_pressed)
		{
			// We want to skip the frame where the mouse is being repositioned to
			// the center of the screen because if the cursor wasn't at the center,
			// we're going to consider to delta from the old position to the center as
			// the moving having moved but it's not the case. The user didn't move the
			// mouse, it's us forcing it in the center of the viewport
			m_just_pressed = false;

			return;
		}

		RenderWindow* render_window = reinterpret_cast<RenderWindow*>(glfwGetWindowUserPointer(window));

		float xposf = static_cast<float>(xpos);
		float yposf = static_cast<float>(ypos);

		std::pair<float, float> old_position = m_grab_cursor_position;
		if (old_position.first == -1 && old_position.second == -1)
			;
		// If this is the first position of the cursor, nothing to do
		else
		{
			// Computing the difference in movement
			std::pair<float, float> difference = std::make_pair(xposf - old_position.first, yposf - old_position.second);

			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
				render_window->update_renderer_view_translation(-difference.first, difference.second);

			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
				render_window->update_renderer_view_rotation(-difference.first, -difference.second);
		}

		// Updating the position
		if (render_window->is_interacting())
			// Locking the cursor in place as long as we're moving the camera
			glfwSetCursorPos(window, old_position.first, old_position.second);
	}
}

void WindowsRenderWindowMouseInteractor::set_callbacks(GLFWwindow* window)
{
    glfwSetCursorPosCallback(window, WindowsRenderWindowMouseInteractor::glfw_mouse_cursor_callback);
    glfwSetMouseButtonCallback(window, WindowsRenderWindowMouseInteractor::glfw_mouse_button_callback);
	glfwSetScrollCallback(window, RenderWindowMouseInteractor::glfw_mouse_scroll_callback);
}
