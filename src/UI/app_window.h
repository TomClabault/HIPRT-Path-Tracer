#ifndef APP_WINDOW_H
#define APP_WINDOW_H

#include "UI/application_settings.h"

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

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
	void change_resolution_scaling(float new_scaling);

	int get_width();
	int get_height();

	Renderer& get_renderer();
	void setup_display_program();
	void set_renderer_scene(Scene& scene);
	void update_renderer_view_translation(float translation_x, float translation_y);
	void update_renderer_view_zoom(float offset);
	void update_renderer_view_rotation(float offset_x, float offset_y);
	void increment_frame_number();
	void reset_frame_number();

	std::pair<float, float> get_cursor_position();
	void set_cursor_position(std::pair<float, float> new_position);

	void display(const std::vector<Color>& image_data);
	void display(OrochiBuffer<float>& orochi_buffer);
	void display_imgui();

	void run();
	void quit();

private:
	int m_viewport_width, m_viewport_height;
	int m_frame_number = 0;
	std::pair<float, float> m_cursor_position;
	std::chrono::high_resolution_clock::time_point m_startRenderTime;

	ApplicationSettings m_application_settings;

	Renderer m_renderer;

	GLuint m_display_program;
	GLuint m_display_texture;
	GLFWwindow* m_window;
};

#endif