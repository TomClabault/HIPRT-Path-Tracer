/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDER_WINDOW_H
#define RENDER_WINDOW_H

#include "OpenGL/OpenGLProgram.h"
#include "Renderer/OpenImageDenoiser.h"
#include "Renderer/GPURenderer.h"
#include "Renderer/Baker/GPUBaker.h"
#include "UI/ApplicationSettings.h"
#include "UI/ApplicationState.h"
#include "UI/DisplayView/DisplayTextureType.h"
#include "UI/DisplayView/DisplayViewEnum.h"
#include "UI/DisplayView/DisplayViewSystem.h"
#include "UI/ImGui/ImGuiRenderer.h"
#include "UI/PerformanceMetricsComputer.h"
#include "UI/Interaction/RenderWindowKeyboardInteractor.h"
#include "UI/Interaction/RenderWindowMouseInteractor.h"
#include "UI/Screenshoter.h"
#include "Utils/CommandlineArguments.h"

#include "GL/glew.h"
#include "GLFW/glfw3.h"

class RenderWindow
{
public:
	static const std::string PERF_METRICS_CPU_DISPLAY_TIME_KEY;

	RenderWindow(int width, int height, std::shared_ptr<HIPRTOrochiCtx> hiprt_oro_ctx);
	~RenderWindow();

	void init_glfw(int window_width, int window_height);
	void init_gl(int width, int height);

	static void APIENTRY gl_debug_output_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
	void resize(int pixels_width, int pixels_height);
	void change_resolution_scaling(float new_scaling);

	int get_width();
	int get_height();
	bool is_interacting();

	RenderWindowKeyboardInteractor& get_keyboard_interactor();
	std::shared_ptr<RenderWindowMouseInteractor> get_mouse_interactor();

	std::shared_ptr<ApplicationSettings> get_application_settings();
	std::shared_ptr<GPURenderer> get_renderer();
	std::shared_ptr<GPUBaker> get_baker();
	std::shared_ptr<OpenImageDenoiser> get_denoiser();
	std::shared_ptr<PerformanceMetricsComputer> get_performance_metrics();
	std::shared_ptr<Screenshoter> get_screenshoter();
	std::shared_ptr<ImGuiRenderer> get_imgui_renderer();

	std::shared_ptr<DisplayViewSystem> get_display_view_system();

	/**
	 * Translates the camera along its X and Y axis by translation_x and translation_y respectively.
	 * 
	 * If scale_translation is true, translation_x and translation_y are scaled by the delta time of
	 * the application and the camera speed before the translation is applied to the camera. You may
	 * want to set scale_translation to false when translation the camera with a mouse pan and scale it
	 * manually by a predefined arbitrary factor
 	 */
	void update_renderer_view_translation(float translation_x, float translation_y, bool scale_translation);
	void update_renderer_view_zoom(float offset, bool scale_delta_time);
	void update_renderer_view_rotation(float offset_x, float offset_y);

	/**
	 * Returns true if the renderer is not sampling the image anymore. 
	 * This can be the case if all pixels have converged according to
	 * adaptive sampling or if the maximum number of samples specified by
	 * the user has been reached, etc...
	 */
	bool is_rendering_done();
	bool needs_viewport_refresh();
	/**
	 * Computes how long to wait between each viewport
	 * refresh given the current render time
	 * 
	 * For a total current render time less than 1s, this returns 0.0f for example, 
	 * meaning that the viewport should always immediately be refreshed.
	 * 
	 * ...
	 * 
	 * See the function for the rest of the delays
	 */
	float get_viewport_refresh_delay_ms();
	/**
	 * Returns the time in milliseconds before the next viewport refresh when rendering offline.
	 * 
	 * Returns 0.0f if not rendering offline (i.e. rendering realtime)
	 */
	float get_time_ms_before_viewport_refresh();
	void reset_render();
	void set_render_dirty(bool render_dirty);
	/**
	 * When offline rendering, the viewport isn't always refreshed at each frame because 
	 * refreshing consumes resources 
	 */
	void set_force_viewport_refresh(bool force_viewport_refresh);

	float get_current_render_time();
	float get_samples_per_second();
	/**
	 * Computes the number of samples per second as seen from the render window. "As seen by the render window"
	 * means that the GPU stall percentage is taken into account for example
	 */
	float compute_samples_per_second();
	/**
	 * Returns how long the GPU has to stall for before next frame according to the application settings
	 */
	float compute_GPU_stall_duration();

	/**
	 * Returns the time in milliseconds of the last UI frame (decoupled from the renderer's frame)
	 */
	float get_UI_delta_time();

	void run();
	void render();
	/**
	 * Denoises the color framebuffer if necessary (according to ImGui
	 * parameters such as denoiser sample skip, only denoise when rendering done, ...)
	 * 
	 * Returns true if denoising occured and if the application needs to display the denoised data.
	 * False otherwise.
	 * 
	 * That return value can be usedf to decide whether or not we need to upload to denoised data
	 * to OpenGL or not (we need to upload it if something new was denoised AND the application
	 * wants to display the denoised data. This condition corresponds exactly to the returned value)
	 */
	bool denoise();

	// TODO: why is this not in linux mouse interactor only and why is it in renderwindow ?
	std::pair<float, float> get_cursor_position()
	{
		return m_cursor_position;
	}

	void set_cursor_position(std::pair<float, float> new_cursor_position)
	{
		m_cursor_position = new_cursor_position;
	}

private:
	int m_viewport_width, m_viewport_height;

	/**
	 * Denoises the renderer framebuffer using the OpenGL interop buffers.
	 *
	 * Using the OpenGL interop buffers greatly improves the performance of the
	 * denoiser but they seem to be a bit slower to manipulate from the path
	 * tracing shaders --> a bit slower path tracing performance.
	 * 
	 * This may only be true on AMD GPUs though where the OpenGL interop driver implementation
	 * seems to be pretty bad
	 * 
	 * This function should not be called directly. ::denoise() should be called
	 * and it will take care of calling the right function depending on the value 
	 * of ApplicationSettings::denoiser_use_interop_buffers
	 */
	float denoise_interop_buffers();

	/**
	 * Denoises the renderer framebuffer not using the OpenGL interop buffers.
	 * 
	 * This makes the denoising more expensive but increases path tracing performance
	 * a bit (on AMD GPUs at least)
	 * 
	 * This function should not be called directly. ::denoise() should be called
	 * and it will take care of calling the right function depending on the value 
	 * of ApplicationSettings::denoiser_use_interop_buffers
	 */
	float denoise_no_interop_buffers();

	// All the settings of the application (that can, for the most part, be controlled
	// through ImGui)
	std::shared_ptr<ApplicationSettings> m_application_settings;
	std::shared_ptr<ApplicationState> m_application_state;

	std::shared_ptr<GPURenderer> m_renderer;
	std::shared_ptr<GPUBaker> m_gpu_baker;
	std::shared_ptr<OpenImageDenoiser> m_denoiser;
	std::shared_ptr<PerformanceMetricsComputer> m_perf_metrics;
	std::shared_ptr<Screenshoter> m_screenshoter;

	// Encapsulates the handling of display views
	std::shared_ptr<DisplayViewSystem> m_display_view_system;

	GLFWwindow* m_glfw_window;
	std::shared_ptr<ImGuiRenderer> m_imgui_renderer;

	std::shared_ptr<RenderWindowMouseInteractor> m_mouse_interactor;
	RenderWindowKeyboardInteractor m_keyboard_interactor;

	std::pair<float, float> m_cursor_position;
};

#endif
