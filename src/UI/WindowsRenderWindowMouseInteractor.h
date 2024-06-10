/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef WINDOWS_RENDER_WINDOW_MOUSE_INTERACTOR_H
#define WINDOWS_RENDER_WINDOW_MOUSE_INTERACTOR_H

#include "UI/RenderWindowMouseInteractor.h"

struct GLFWwindow;

class WindowsRenderWindowMouseInteractor : public RenderWindowMouseInteractor
{
public:
    void set_callbacks(GLFWwindow* window);

private:
    static void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void glfw_mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos);

    static bool m_interacting_left_button;
    static bool m_interacting_right_button;
    static bool m_just_pressed;
};

#endif
