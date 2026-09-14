/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DISPLAY_VIEW_SYSTEM_H
#define DISPLAY_VIEW_SYSTEM_H

#include "Renderer/GPURenderer.h"
#include "OpenGL/OpenGLInteropBuffer.h"
#include "OpenGL/OpenGLProgram.h"
#include "UI/DisplayView/DisplayTextureType.h"
#include "UI/DisplayView/DisplayViewEnum.h"

#include <memory>

class RenderWindow;

class DisplayViewSystem
{
public:
	// Texture unit used for the final device-side display framebuffer.
	static constexpr int DISPLAY_TEXTURE_UNIT_1 = 1;
	// Texture unit reserved for the compute shader screenshoter
	static constexpr int DISPLAY_COMPUTE_IMAGE_UNIT = 3;

	DisplayViewSystem(std::shared_ptr<GPURenderer> renderer, RenderWindow* render_window);
	~DisplayViewSystem();

	void configure_framebuffer();
	void resize_framebuffer();

	DisplayViewType get_current_display_view_type();

	/**
	 * Applies queued changes (such as changing the display view for example)
	 *
	 * Returns true if the display view was changed. False otherwise
	 */
	bool update_selected_display_view();

	/**
	 * Displays the final device-side post-process output onto the viewport
	 */
	void display();

	/**
	 * Queues a change of display view that will take effect upon calling update()
	 */
	void queue_display_view_change(DisplayViewType display_view);

	void resize(int new_render_width, int new_render_height);

	/**
	 * Binds the final device-side display buffer to a legacy arbitrary display program.
	 */
	static void update_display_program_uniforms(const DisplayViewSystem* display_view_system,
												std::shared_ptr<OpenGLProgram> program,
												std::shared_ptr<GPURenderer> renderer,
												std::shared_ptr<ApplicationSettings> application_settings);

	void upload_final_display_buffer();

	// Kept as a compatibility entry point for the legacy compute screenshot path.
	void upload_relevant_buffers_to_texture();

private:
	template <typename T>
	void internal_upload_buffer_to_texture(std::shared_ptr<OpenGLInteropBuffer<T>> buffer, GLuint display_texture, int texture_unit);

	// Every display view now consumes the final device-side RGB32F framebuffer.
	void internal_resize_display_texture(int width, int height);

	/**
	 * Automatically changes the display view used if some conditions are met (or not met).
	 *
	 * For example, if the current display view is "GMoN Blend" but the user disables GMoN, we
	 * don't want to keep using the GMoN blend view so this function will change it automatically
	 */
	void handle_automatic_display_view_changes();

	// Display view currently being used by the device-side post-process pass.
	DisplayViewType m_current_display_view_type = DisplayViewType::UNDEFINED;
	// If != UNDEFINED, remembers the display view before an auto-switch, to restore later
	DisplayViewType m_saved_display_view = DisplayViewType::UNDEFINED;

	// If != UNDEFINED, then someone has requested a display view change and the display view change will be applied upon calling update().
	// Keeping the selection queued ensures that the device-side post-process pass and the OpenGL upload switch together at a frame boundary.
	DisplayViewType m_queued_display_view_change = DisplayViewType::UNDEFINED;

	// The texture should be the same resolution as the render resolution. It has nothing to do with the resolution of the viewport.
	// This is the texture used by the trivial OpenGL program to draw the final device-side post-process output.
	GLuint m_display_texture						 = 0;
	std::shared_ptr<OpenGLProgram> m_display_program = nullptr;

	// We don't need a VAO because we're hardcoding our fullscreen
	// quad vertices in our vertex shader but we still need an empty/fake
	// VAO for NVIDIA drivers to avoid errors
	GLuint m_vao;

	// Framebuffer we're drawing. We're not directly drawing to the back buffer because we
	// want ImGui to do the drawing in one of its ImGui window
	GLuint m_framebuffer;

public:
	GLuint m_fbo_texture;

private:
	std::shared_ptr<GPURenderer> m_renderer = nullptr;
	RenderWindow* m_render_window			= nullptr;
};

template <typename T>
void DisplayViewSystem::internal_upload_buffer_to_texture(std::shared_ptr<OpenGLInteropBuffer<T>> buffer, GLuint display_texture, int texture_unit)
{
	if (buffer == nullptr)
		return;

	buffer->unmap();
	DisplayTextureType texture_type = DisplayTextureType::FLOAT3;
	buffer->unpack_to_GL_texture(display_texture, GL_TEXTURE0 + texture_unit, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y,
								 texture_type);
}

#endif // #ifndef DISPLAY_VIEW_SYSTEM_H
