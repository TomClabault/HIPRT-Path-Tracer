/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/RenderWindow.h"
#include "UI/Interaction/RenderWindowKeyboardInteractor.h"

void RenderWindowKeyboardInteractor::glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	void* user_pointer = glfwGetWindowUserPointer(window);
	RenderWindow* render_window = reinterpret_cast<RenderWindow*>(user_pointer);

	// We still want to process the inputs if we're hovering the render window because then
	// we *are* trying to move the camera with the keyboard
	bool render_window_hovered = render_window->get_imgui_renderer()->get_imgui_render_window().is_hovered();

	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureKeyboard && !render_window_hovered && !(action == GLFW_RELEASE))
		// We always want to handle release key otherwise we could press a key while
		// hovering the render window and then release the with our mouse over another window
		// --> not hovering the render window --> the key won't be released and the camera
		// will keep moving
		return;

	RenderWindowKeyboardInteractor& interactor_instance = render_window->get_keyboard_interactor();
	switch (key)
	{
	case GLFW_KEY_W:
	case GLFW_KEY_Z:
		interactor_instance.m_z_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	case GLFW_KEY_A:
	case GLFW_KEY_Q:
		interactor_instance.m_q_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	case GLFW_KEY_S:
		interactor_instance.m_s_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);

		break;

	case GLFW_KEY_D:
		interactor_instance.m_d_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	case GLFW_KEY_SPACE:
		interactor_instance.m_space_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	case GLFW_KEY_LEFT_SHIFT:
		interactor_instance.m_lshift_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	default:
		break;
	}
}

void RenderWindowKeyboardInteractor::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;
}

void RenderWindowKeyboardInteractor::set_callbacks(GLFWwindow* window)
{
	glfwSetKeyCallback(window, RenderWindowKeyboardInteractor::glfw_key_callback);
}

void RenderWindowKeyboardInteractor::poll_keyboard_inputs()
{
	float zoom = 0.0f;
	std::pair<float, float> translation = { 0.0f, 0.0f };

	if (m_z_pressed)
		zoom += 1.0f;
	if (m_q_pressed)
		translation.first += 1.0f;
	if (m_s_pressed)
		zoom -= 1.0f;
	if (m_d_pressed)
		translation.first -= 1.0f;
	if (m_space_pressed)
		translation.second += 1.0f;
	if (m_lshift_pressed)
		translation.second -= 1.0f;

	m_render_window->update_renderer_view_translation(-translation.first, translation.second, true);
	m_render_window->update_renderer_view_zoom(-zoom, true);
}

bool RenderWindowKeyboardInteractor::is_interacting()
{
	return m_z_pressed || m_q_pressed || m_s_pressed || m_d_pressed || m_space_pressed || m_lshift_pressed;
}
