/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDER_WINDOW_MOUSE_INTERACTOR_H
#define RENDER_WINDOW_MOUSE_INTERACTOR_H

struct GLFWwindow;

/**
 * This class is derived in a LinuxRenderWindowMouseInteractor and a WindowsRenderWindowMouseInteractor.
 * 
 * This is because GLFW_CURSOR_DISABLED seems to be buggued on Windows. There's some kind of annoying 
 * cursor jumping happening. 
 * 
 * Therefore, a "custom" solution that forcefully repositions the cursor at its previous position 
 * has been implemented to provide unlimited movement (otherwise the cursor would end up getting 
 * out of the window). On Linux, it seemed that glfwSetCursorPosition had no effect (my cursor 
 * wasn't being repositionned at all during my testing). However, on Linux, there's no cursor 
 * jumping with GLFW_CURSOR_DISABLED so we can use that (using the same implementation as Windows 
 * is broken anyways because, again, glfwSetCursorPosition, which is used by the Windows implementation, 
 * has no effect). 
 * 
 * This is why we have 2 different implementations:
 * 
 *  - Windows uses a replacement implementation to GLFW_CURSOR_DISABLED that manually
 *      repositions the cursor using glfwSetCursorPosition
 *  - Linux uses GLFW_CURSOR_DISABLED (we could have used only the Windows implementation but
 *      glfwSetCursorPosition doesn't seemed to be working during my testing on Linux)
 */
class RenderWindowMouseInteractor
{
public:
    virtual void set_callbacks(GLFWwindow* window) {}

    /**
     * Returns true if either to left mouse button or the right
     * mouse button is currently held down
     */
    bool is_interacting();

    void set_interacting_left_button(bool interacting); 
    void set_interacting_right_button(bool interacting);

    bool is_interacting_right_button();
    bool is_interacting_left_button();

protected:
    static void glfw_mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

private:
    // Is the mouse left button beind held down?
    bool m_interacting_left_button = false;
    // Is the mouse right button beind held down?
    bool m_interacting_right_button = false;
};

#endif
