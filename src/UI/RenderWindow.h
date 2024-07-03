/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDER_WINDOW_H
#define RENDER_WINDOW_H

#include "OpenGL/OpenGLProgram.h"
#include "Renderer/OpenImageDenoiser.h"
#include "Renderer/GPURenderer.h"
#include "UI/ApplicationSettings.h"
#include "UI/DisplayTextureType.h"
#include "UI/DisplayView.h"
#include "UI/ImGuiRenderer.h"
#include "UI/PerformanceMetricsComputer.h"
#include "UI/RenderWindowKeyboardInteractor.h"
#include "UI/RenderWindowMouseInteractor.h"
#include "UI/Screenshoter.h"
#include "Utils/CommandlineArguments.h"

#include "GL/glew.h"
#include "GLFW/glfw3.h"

class RenderWindow
{
public:
	static constexpr int DISPLAY_TEXTURE_UNIT_1 = 1;
	static constexpr int DISPLAY_TEXTURE_UNIT_2 = 2;
	static constexpr int DISPLAY_COMPUTE_IMAGE_UNIT = 3;



	RenderWindow(int width, int height);
	~RenderWindow();

	void init_glfw(int width, int height);
	void init_gl(int width, int height);
	void init_imgui();

	static void APIENTRY gl_debug_output_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
	void resize_frame(int pixels_width, int pixels_height);
	void change_resolution_scaling(float new_scaling);

	int get_width();
	int get_height();
	void set_render_low_resolution(bool on_or_off);
	bool is_interacting();

	std::shared_ptr<ApplicationSettings> get_application_settings();
	std::shared_ptr<GPURenderer> get_renderer();
	std::shared_ptr<OpenImageDenoiser> get_denoiser();
	std::shared_ptr<PerformanceMetricsComputer> get_performance_metrics();
	std::shared_ptr<Screenshoter> get_screenshoter();

	void create_display_programs();
	void change_display_view(DisplayView display_view);

	void upload_data_to_display_texture(GLuint display_texture, const void* data, GLenum format, GLenum type);
	void update_program_uniforms(OpenGLProgram& program);
	void update_renderer_view_translation(float translation_x, float translation_y);
	void update_renderer_view_zoom(float offset);
	void update_renderer_view_rotation(float offset_x, float offset_y);

	void increment_sample_number();
	/**
	 * Returns true if the renderer is not sampling the image anymore. 
	 * This can be the case if all pixels have converged according to
	 * adaptive sampling or if the maximum number of samples specified by
	 * the user has been reached, etc...
	 */
	bool is_rendering_done();
	void reset_render();
	void set_render_dirty(bool render_dirty);

	float get_current_render_time();
	float get_samples_per_second();

	/**
	 * Uploads raw data or a buffer to display_texture_1 and draws on
	 * the fullscreen quad with the active shader program
	 */
	void display(const void* data);
	template <typename T>
	void display(const std::vector<T>& orochi_buffer);
	template <typename T>
	void display(const OrochiBuffer<T>& orochi_buffer);
	template <typename T>
	void display(std::shared_ptr<OpenGLInteropBuffer<T>> buffer);

	/**
	 * Uploads data from buffer_1 and buffer_2 to display_texture_1 and
	 * display_texture_2 respectively and draws on the fullscreen quad with
	 * the active shader program.
	 * 
	 * Blend override allows the overriding of the blend factor of the two images. 
	 *	* -1.0f is no override. 
	 *	* 0.0f forces the display of only the data of buffer_1. 
	 *	* 1.0f forces the display of only the data of buffer_2.
	 */
	template <typename T>
	void display_blend(std::shared_ptr<OpenGLInteropBuffer<T>> buffer_1, std::shared_ptr<OpenGLInteropBuffer<T>> buffer_2, float blend_override = -1.0f);

	void run();
	void render();
	void quit();

	std::pair<float, float> get_cursor_position()
	{
		return m_cursor_position;
	}

void set_cursor_position(std::pair<float, float> new_cursor_position)
{
	m_cursor_position = new_cursor_position;
}

private:
	/*
	 * This function ensures that the display texture is of the proper format
	 * for the display view selected.
	 *
	 * For example, if the user decided to display normals in the viewport, we'll need
	 * the display texture to be a float3 (RGB32F) texture. If the user is displaying
	 * the adaptive sampling heatmap, we'll only need an integer texture.
	 *
	 * This function deletes/recreates the texture everytime its required format changes
	 * (i.e. when the current texture was a float3 and we asked for an integer texture)
	 because we don't want to keep every single possible texture in VRAM. This may cause
	 * a (very) small stutter but that's probably expected since we're asking for a different view
	 * to show up in the viewport
	 */
	void internal_recreate_display_textures_from_display_view(DisplayView display_view);
	void internal_recreate_display_texture(std::pair<GLuint, DisplayTextureType>& display_texture, GLenum display_texture_unit, DisplayTextureType new_texture_type, int width, int height);

	int m_viewport_width, m_viewport_height;

	// Timer started at the first sample. Used to time how long the render has been running
	// for so far
	std::chrono::high_resolution_clock::time_point m_start_cpu_frame_time, m_stop_cpu_frame_time;
	// How long the current render has been running for in milliseconds
	float m_current_render_time_ms = 0.0f;
	float m_samples_per_second = 0.0f;

	std::shared_ptr<ApplicationSettings> m_application_settings;

	// Set to true if some settings of the render changed and we need
	// to restart rendering from sample 0
	bool m_render_dirty = false;
	std::shared_ptr<GPURenderer> m_renderer;
	std::shared_ptr<OpenImageDenoiser> m_denoiser;
	std::shared_ptr<PerformanceMetricsComputer> m_perf_metrics;
	std::shared_ptr<Screenshoter> m_screenshoter;

	// We don't need a VAO because we're hardcoding our fullscreen
	// quad vertices in our vertex shader but we still need an empty/fake
	// VAO for NVIDIA drivers to avoid errors
	GLuint m_vao;

	OpenGLProgram m_active_display_program;
	OpenGLProgram m_default_display_program;
	OpenGLProgram m_blend_2_display_program;
	OpenGLProgram m_normal_display_program;
	OpenGLProgram m_albedo_display_program;
	OpenGLProgram m_adaptive_sampling_display_program;


	// Display textures & their display type
	// 
	// The display type is the format of the texel of the texture used by the display program.
	// This is useful because we have several types of programs using several
	// types of textures. For example, displaying normals on the screen requires float3 textures
	// whereas displaying a heatmap requires only a texture whose texels are scalar (floats or ints).
	// This means that, depending on the display view selected, we're going to have to use the proper
	// OpenGL texture format type and that's what the DisplayTextureType is for
	// 
	// The textures should be the same resolution as the render resolution.
	// They have nothing to do with the resolution of the viewport.
	// 
	// The first texture is used by the display program to draw on the fullscreen quad.
	// Also used as the first blending texture when a blending display view is selected
	std::pair<GLuint, DisplayTextureType> m_display_texture_1 = { -1, DisplayTextureType::UNINITIALIZED };
	// Second display texture.
	// Used as the second texture for blending when a blending display view is selected
	// (used by the denoiser blending for example)
	std::pair<GLuint, DisplayTextureType> m_display_texture_2 = { -1, DisplayTextureType::UNINITIALIZED };

	GLFWwindow* m_glfw_window;
	// Needs to be a unique_ptr because we're using polymorphism for the Linux/Windows implementation here
	std::unique_ptr<RenderWindowMouseInteractor> m_mouse_interactor;
	RenderWindowKeyboardInteractor m_keyboard_interactor;
	ImGuiRenderer m_imgui_renderer;

	std::pair<float, float> m_cursor_position;
};

template <typename T>
void RenderWindow::display(const std::vector<T>& pixels_data)
{
	display(pixels_data.data());
}

template <typename T>
void RenderWindow::display(const OrochiBuffer<T>& orochi_buffer)
{
	display(orochi_buffer.download_data());
}

template <typename T>
void RenderWindow::display(std::shared_ptr<OpenGLInteropBuffer<T>> buffer)
{
	buffer->unmap();
	buffer->unpack_to_texture(m_display_texture_1.first, GL_TEXTURE0 + RenderWindow::DISPLAY_TEXTURE_UNIT_1, m_renderer->m_render_width, m_renderer->m_render_height, m_display_texture_1.second);

	update_program_uniforms(m_active_display_program);

	// Binding an empty VAO here (empty because we're hardcoding our full-screen quad vertices
	// in our vertex shader) because this is required on NVIDIA drivers
	glBindVertexArray(m_vao);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

template <typename T>
void RenderWindow::display_blend(std::shared_ptr<OpenGLInteropBuffer<T>> buffer_1, std::shared_ptr<OpenGLInteropBuffer<T>> buffer_2, float blend_override /* = -1.0f */)
{
	buffer_1->unmap();
	buffer_1->unpack_to_texture(m_display_texture_1.first, GL_TEXTURE0 + RenderWindow::DISPLAY_TEXTURE_UNIT_1, m_renderer->m_render_width, m_renderer->m_render_height, m_display_texture_1.second);

	buffer_2->unmap();
	buffer_2->unpack_to_texture(m_display_texture_2.first, GL_TEXTURE0 + RenderWindow::DISPLAY_TEXTURE_UNIT_2, m_renderer->m_render_width, m_renderer->m_render_height, m_display_texture_2.second);

	update_program_uniforms(m_active_display_program);
	if (blend_override != -1.0f)
		m_active_display_program.set_uniform("u_blend_factor", blend_override);

	// Binding an empty VAO here (empty because we're hardcoding our full-screen quad vertices
	// in our vertex shader) because this is required on NVIDIA drivers
	glBindVertexArray(m_vao);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

#endif
