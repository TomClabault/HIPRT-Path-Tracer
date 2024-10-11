/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef LINUX_RENDER_WINDOW_MOUSE_INTERACTOR_H
#define LINUX_RENDER_WINDOW_MOUSE_INTERACTOR_H

#include "UI/Interaction/RenderWindowMouseInteractor.h"

struct GLFWwindow;

class LinuxRenderWindowMouseInteractor : public RenderWindowMouseInteractor
{
public:
    void set_callbacks(GLFWwindow* window);

private:
    static void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void glfw_mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos);

    bool render_window_hovered_on_click = false;
};

#endif
