#ifndef RENDER_WINDOW_H
#define RENDER_WINDOW_H

#include "UI/application_settings.h"
#include "UI/display_settings.h"

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "Renderer/renderer.h"
#include "Renderer/open_image_denoiser.h"
#include "Image/image_writer.h"
#include "Utils/commandline_arguments.h"

class RenderWindow
{
public:
	static constexpr int DISPLAY_TEXTURE_UNIT = 1;
	static constexpr int DISPLAY_COMPUTE_IMAGE_UNIT = 2;





	RenderWindow(int width, int height);
	~RenderWindow();

	static void APIENTRY gl_debug_output_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
	void resize_frame(int pixels_width, int pixels_height);
	void change_resolution_scaling(float new_scaling);

	int get_width();
	int get_height();
	void set_interacting(bool is_interacting);

	ApplicationSettings& get_application_settings();
	const ApplicationSettings& get_application_settings() const;
	Renderer& get_renderer();

	void setup_display_program();
	void set_renderer_scene(Scene& scene);
	void update_renderer_view_translation(float translation_x, float translation_y);
	void update_renderer_view_zoom(float offset);
	void update_renderer_view_rotation(float offset_x, float offset_y);
	void increment_sample_number();
	void reset_sample_number();
	void reset_frame_number();

	DisplaySettings get_display_settings();
	void set_display_settings(DisplaySettings settings);

	std::pair<float, float> get_cursor_position();
	void set_cursor_position(std::pair<float, float> new_position);

	void setup_display_uniforms(GLuint program);
	void display(const void* data);
	template <typename T>
	void display(const std::vector<T>& orochi_buffer);
	template <typename T>
	void display(const OrochiBuffer<T>& orochi_buffer);

	void show_render_settings_panel();
	void show_denoiser_panel();
	void show_post_process_panel();
	void display_imgui();

	void run();
	void quit();

private:
	int m_viewport_width, m_viewport_height;
	std::pair<float, float> m_cursor_position;
	bool m_interacting;

	std::chrono::high_resolution_clock::time_point m_startRenderTime;

	ApplicationSettings m_application_settings;
	DisplaySettings m_display_settings;

	Renderer m_renderer;
	RenderSettings& m_render_settings;
	OpenImageDenoiser m_denoiser;

	ImageWriter m_image_writer;

	GLuint m_display_program;
	GLuint m_display_texture;
	GLFWwindow* m_window;
};

template <typename T>
void RenderWindow::display(const std::vector<T>& pixels_data)
{
	display(pixels_data.data());
}

template <typename T>
void RenderWindow::display(const OrochiBuffer<T>& orochi_buffer)
{	
	display(orochi_buffer.download_pixels().data());
}

#endif