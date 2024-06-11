/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDER_WINDOW_KEYBOARD_INTERACTOR_H
#define RENDER_WINDOW_KEYBOARD_INTERACTOR_H

struct GLFWwindow;
class RenderWindow;

class RenderWindowKeyboardInteractor
{
public:
	void set_render_window(RenderWindow* render_window);
	void set_callbacks(GLFWwindow* window);

	static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

	/**
	 * Looks at the key states of the interactor and manipulates
	 * the render window to reflect on these pressed keys.
	 */
	void poll_keyboard_inputs();

private:
	static bool m_z_pressed;
	static bool m_q_pressed;
	static bool m_s_pressed;
	static bool m_d_pressed;
	static bool m_space_pressed;
	static bool m_lshift_pressed;

	RenderWindow* m_render_window;
};

#endif
