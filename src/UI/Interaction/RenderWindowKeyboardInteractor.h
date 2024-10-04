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
	 * the queue_frame_for_render window to reflect on these pressed keys.
	 */
	void poll_keyboard_inputs();

	/**
	 * Returns true if any key (only keys relevant to this interactor) is currently being held down.
	 * Returns false otherwise.
	 */
	bool is_interacting();

protected:
	bool m_z_pressed = false;
	bool m_q_pressed = false;
	bool m_s_pressed = false;
	bool m_d_pressed = false;
	bool m_space_pressed = false;
	bool m_lshift_pressed = false;

	RenderWindow* m_render_window = nullptr;
};

#endif
