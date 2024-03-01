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
#include "Renderer/open_image_denoiser.h"

class AppWindow
{
public:
	static constexpr int DISPLAY_TEXTURE_UNIT = 1;

	struct DisplaySettings
	{
		bool display_normals;
		bool scale_by_frame_number;
		bool do_tonemapping;
	};



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
	void increment_sample_number();
	void reset_sample_number();
	void reset_frame_number();

	std::pair<float, float> get_cursor_position();
	void set_cursor_position(std::pair<float, float> new_position);

	void setup_display_program(GLuint program, const AppWindow::DisplaySettings& display_settings);
	void display(const void* data, const AppWindow::DisplaySettings& display_settings = { false, true, true });
	template <typename T>
	void display(const std::vector<T>& orochi_buffer, const AppWindow::DisplaySettings& display_settings = { false, true, true });
	template <typename T>
	void display(const OrochiBuffer<T>& orochi_buffer, const AppWindow::DisplaySettings& display_settings = { false, true, true });

	void display_imgui();

	void run();
	void quit();

private:
	int m_viewport_width, m_viewport_height;
	int m_sample_number = 0;
	std::pair<float, float> m_cursor_position;
	std::chrono::high_resolution_clock::time_point m_startRenderTime;

	ApplicationSettings m_application_settings;

	Renderer m_renderer;
	OpenImageDenoiser m_denoiser;

	GLuint m_display_program;
	GLuint m_display_texture;
	GLFWwindow* m_window;
};

template <typename T>
void AppWindow::display(const std::vector<T>& pixels_data, const AppWindow::DisplaySettings& display_settings)
{
	display(pixels_data.data(), display_settings);
}

template <typename T>
void AppWindow::display(const OrochiBuffer<T>& orochi_buffer, const AppWindow::DisplaySettings& display_settings)
{	
	display(orochi_buffer.download_pixels().data(), display_settings);
	//setup_display_program(m_display_program, display_settings);

	//// TODO
	//// This is very sub optimal and should absolutely be replaced by a 
	//// buffer that is shared between hiprt and opengl
	//// Unfortunately, this is unavailable in Orochi so we would have
	//// to switch to HIP (or CUDA but it doesn't support AMD) to get access
	//// to the hipGraphicsMapResources() functions family
	//std::vector<T> pixels_data = orochi_buffer.download_pixels();

	//glBindTexture(GL_TEXTURE_2D, m_display_texture);
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_renderer.m_render_width, m_renderer.m_render_height, GL_RGB, GL_FLOAT, pixels_data.data());

	//glDrawArrays(GL_TRIANGLES, 0, 6);
}

#endif