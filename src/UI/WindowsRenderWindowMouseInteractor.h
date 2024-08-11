/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef WINDOWS_RENDER_WINDOW_MOUSE_INTERACTOR_H
#define WINDOWS_RENDER_WINDOW_MOUSE_INTERACTOR_H

#include "UI/RenderWindowMouseInteractor.h"

#include <utility>

struct GLFWwindow;

class WindowsRenderWindowMouseInteractor : public RenderWindowMouseInteractor
{
public:
    void set_callbacks(GLFWwindow* window);

protected:
    bool m_just_pressed;

private:
    static void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void glfw_mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos);

    // Position of the mouse when the user first clicked the viewport.
    // Used to put the cursor back in place to allow infinite mouse movements
    std::pair<float, float> m_grab_cursor_position = { 0.0f, 0.0f };

};

#endif
