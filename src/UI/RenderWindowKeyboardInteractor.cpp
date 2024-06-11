/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/RenderWindow.h"
#include "UI/RenderWindowKeyboardInteractor.h"

bool RenderWindowKeyboardInteractor::m_z_pressed = false;
bool RenderWindowKeyboardInteractor::m_q_pressed = false;
bool RenderWindowKeyboardInteractor::m_s_pressed = false;
bool RenderWindowKeyboardInteractor::m_d_pressed = false;
bool RenderWindowKeyboardInteractor::m_space_pressed = false;
bool RenderWindowKeyboardInteractor::m_lshift_pressed = false;

void RenderWindowKeyboardInteractor::glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	switch (key)
	{
	case GLFW_KEY_W:
	case GLFW_KEY_Z:
		m_z_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	case GLFW_KEY_A:
	case GLFW_KEY_Q:
		m_q_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	case GLFW_KEY_S:
		m_s_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);

		break;

	case GLFW_KEY_D:
		m_d_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	case GLFW_KEY_SPACE:
		m_space_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
		break;

	case GLFW_KEY_LEFT_SHIFT:
		m_lshift_pressed = (action == GLFW_PRESS) || (action == GLFW_REPEAT);
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
	std::pair<float, float> translation;
	if (m_z_pressed)
		zoom += 1.0f;
	if (m_q_pressed)
		translation.first += 36.0f;
	if (m_s_pressed)
		zoom -= 1.0f;
	if (m_d_pressed)
		translation.first -= 36.0f;
	if (m_space_pressed)
		translation.second += 36.0f;
	if (m_lshift_pressed)
		translation.second -= 36.0f;

	m_render_window->update_renderer_view_translation(-translation.first, translation.second);
	m_render_window->update_renderer_view_zoom(-zoom);
}
