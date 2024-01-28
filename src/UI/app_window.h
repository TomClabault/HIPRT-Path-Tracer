#ifndef APP_WINDOW_H
#define APP_WINDOW_H

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "Utils/commandline_arguments.h"
#include "Renderer/renderer.h"

class AppWindow
{
public:
	static constexpr int DISPLAY_TEXTURE_UNIT = 1;





	AppWindow(int width, int height);
	~AppWindow();

	static void APIENTRY gl_debug_output_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
	void resize_frame(int pixels_width, int pixels_height);

	void setup_display_program();
	void setup_renderer(const CommandLineArguments& arguments);
	void set_renderer_scene(Scene& scene);
	void display(const std::vector<Color>& image_data);
	void display(OrochiBuffer<float>& orochi_buffer);

	void run();
	void quit();

private:
	int m_width, m_height;

	Renderer m_renderer;

	GLuint m_display_program;
	GLuint m_display_texture;
	GLFWwindow* m_window;
};

#endif