/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDER_WINDOW_H
#define RENDER_WINDOW_H

#include "OpenGL/OpenGLProgram.h"
#include "Renderer/open_image_denoiser.h"
#include "Renderer/renderer.h"
#include "UI/application_settings.h"
#include "UI/DisplayTextureType.h"
#include "UI/DisplayView.h"
#include "UI/Screenshoter.h"
#include "Utils/commandline_arguments.h"

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


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

	void create_display_programs();
	void select_display_program(DisplayView display_view);

	/*
	 * This function ensures that the display texture is of the proper format
	 * for the display view selected.
	 * 
	 * For example, if the user decided to display normals in the viewport, we'll need
	 * the display texture to be a float3 (RGB32F) texture. If the user is displaying
	 * the adaptative sampling heatmap, we'll only need an integer texture.
	 * 
	 * This function deletes/recreates the texture everytime its required format changes
	 * (i.e. when the current texture was a float3 and we asked for an integer texture) 
	 because we don't want to keep every single possible texture in VRAM. This may cause
	 * a (very) small stutter but that's probably expected since we're asking for a different view
	 * to show up in the viewport
	 */
	void recreate_display_texture_from_display_view(DisplayView display_view);
	void recreate_display_texture(DisplayTextureType texture_type, int width, int height);
	void upload_data_to_display_texture(const void* data, GLenum format, GLenum type);
	void update_program_uniforms(OpenGLProgram& program);
	void update_renderer_view_translation(float translation_x, float translation_y);
	void update_renderer_view_zoom(float offset);
	void update_renderer_view_rotation(float offset_x, float offset_y);

	void increment_sample_number();
	void reset_render();


	std::pair<float, float> get_cursor_position();
	void set_cursor_position(std::pair<float, float> new_position);

	void display(const void* data);
	template <typename T>
	void display(const std::vector<T>& orochi_buffer);
	template <typename T>
	void display(const OrochiBuffer<T>& orochi_buffer);
	template <typename T>
	void display(OpenGLInteropBuffer<T>& buffer);

	void show_render_settings_panel();
	void show_objects_panel();
	void show_denoiser_panel();
	void show_post_process_panel();
	void draw_imgui();

	void run();
	bool render();
	void quit();

private:
	int m_viewport_width, m_viewport_height;
	// Current mouse cursor position within the window. Used to compute mouse
	// mouse delta movement by comparing the new mouse position with this variable
	std::pair<float, float> m_cursor_position;
	// Is the user interacting with the camera (rotating, zooming, ...)? 
	bool m_interacting;

	// Timer started at the first sample. Used to time how long the render has been running
	// for so far
	std::chrono::high_resolution_clock::time_point m_start_render_time;

	ApplicationSettings m_application_settings;

	// Set to true if some settings of the render changed and we need
	// to restart rendering from sample 0
	bool m_render_dirty = false;
	Renderer m_renderer;
	HIPRTRenderSettings& m_render_settings;
	OpenImageDenoiser m_denoiser;

	Screenshoter m_screenshoter;

	OpenGLProgram m_active_display_program;
	OpenGLProgram m_default_display_program;
	OpenGLProgram m_normal_display_program;
	OpenGLProgram m_albedo_display_program;
	OpenGLProgram m_adaptative_sampling_display_program;
	// We don't need a VAO because we're hardcoding our fullscreen
	// quad vertices in our vertex shader but we still need an empty/fake
	// VAO for NVIDIA drivers to avoid errors
	GLuint m_vao;
	// Texture used by the display program to draw on the fullscreen quad.
	// This texture should be the same resolution as the render resolution, 
	// it has nothing to do with the resolution of the viewport
	GLuint m_display_texture = -1;
	// Format of the texel of the texture used by the display program
	// This is useful because we have several types of programs using several
	// types of textures. For example, displaying normals on the screen requires float3 textures
	// whereas displaying a heatmap requires only a texture whose texels are scalar (floats or ints)
	DisplayTextureType m_display_texture_type = DisplayTextureType::UNINITIALIZED;
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
	display(orochi_buffer.download_data().data());
}

template <typename T>
void RenderWindow::display(OpenGLInteropBuffer<T>& buffer)
{
	buffer.unmap();
	buffer.unpack_to_texture(m_display_texture, GL_TEXTURE0 + RenderWindow::DISPLAY_TEXTURE_UNIT, m_renderer.m_render_width, m_renderer.m_render_height, m_display_texture_type);

	update_program_uniforms(m_active_display_program);

	// Binding an empty VAO here (empty because we're hardcoding our full-screen quad vertices
	// in our vertex shader) because this is required on NVIDIA drivers
	glBindVertexArray(m_vao);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

#endif
