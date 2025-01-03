/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DISPLAY_VIEW_SYSTEM_H
#define DISPLAY_VIEW_SYSTEM_H

#include "Renderer/GPURenderer.h"
#include "OpenGL/OpenGLInteropBuffer.h"
#include "UI/DisplayView/DisplayView.h"
#include "UI/DisplayView/DisplayViewEnum.h"

#include <unordered_map>

class RenderWindow;

class DisplayViewSystem
{
public:
	// Default texture unit for displaying most of things
	static constexpr int DISPLAY_TEXTURE_UNIT_1 = 1;
	// Second display texture used when we want to blend between two displays
	static constexpr int DISPLAY_TEXTURE_UNIT_2 = 2;
	// Texture unit reserved for the compute shader screenshoter
	static constexpr int DISPLAY_COMPUTE_IMAGE_UNIT = 3;

	DisplayViewSystem(std::shared_ptr<GPURenderer> renderer, RenderWindow* render_window);
	~DisplayViewSystem();

	void configure_framebuffer();
	void resize_framebuffer();

	DisplayViewType get_current_display_view_type();
	const DisplayView* get_current_display_view() const;
	std::shared_ptr<OpenGLProgram> get_active_display_program();
	DisplaySettings& get_display_settings();

	/**
	 * Applies queued changes (such as changing the display view for example)
	 * 
	 * Returns true if the display view was changed. False otherwise
	 */
	bool update_selected_display_view();

	/**
	 * Returns true if the current display view needs the adaptive sampling buffers for
	 * displaying
	 */
	bool current_display_view_needs_adaptive_sampling_buffers();

	/**
	 * Displays the currently active texture view onto the viewport
	 */
	void display();

	/**
	 * Queues a change of display view that will take effect upon calling update()
	 */
	void queue_display_view_change(DisplayViewType display_view);

	bool get_render_low_resolution() const;
	/**
	 * Sets whether or not thje next display() call will display the given texture
	 * as a low render resolution texture or not
	 */
	void set_render_low_resolution(bool low_resolution_or_not);

	void resize(int new_render_width, int new_render_height);

	/**
	 * Updates the uniforms of an arbitrary input program given the state of the renderer and the applications settings given
	 */
	static void update_display_program_uniforms(const DisplayViewSystem* display_view_system, std::shared_ptr<OpenGLProgram> program, std::shared_ptr<GPURenderer> renderer, std::shared_ptr<ApplicationSettings> application_settings);

	/**
	 * Updates the uniforms of the display program currently used by this display view system
	 */
	void update_current_display_program_uniforms();
	void upload_relevant_buffers_to_texture();

private:
	template <typename T>
	void internal_upload_buffer_to_texture(std::shared_ptr<OpenGLInteropBuffer<T>> buffer, const std::pair<GLuint, DisplayTextureType>& display_texture, int texture_unit);

	template<typename T>
	void internal_upload_buffer_to_texture(std::shared_ptr<OrochiBuffer<T>> buffer, const std::pair<GLuint, DisplayTextureType>& display_texture, int texture_unit);

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
	void internal_recreate_display_textures_from_display_view(DisplayViewType display_view);
	void internal_recreate_display_texture(std::pair<GLuint, DisplayTextureType>& display_texture, GLenum display_texture_unit, DisplayTextureType new_texture_type, int width, int height);





	// All the different display view that can be used for displaying
	std::unordered_map<DisplayViewType, DisplayView> m_display_views;
	// Index within the vector of the display view currently being used
	DisplayView* m_current_display_view = nullptr;
	// If != UNDEFINED, then someone has requested a display view change and the display view change will be applied upon calling update().
	// Why is this necessary and why not just change the DisplayView directly?
	//		- Picture this scenario: we're currently displaying the default display view.
	//		- The display view is immediately changed to the AdaptiveSamplingMap view.
	//		- These two display views use different display texture types. The default display view
	//			uses a float3 texture type whereas the AdaptiveSamplingMap view uses a int texture type
	//		- Changing the display view will thus trigger a display texture re-creation (to change the type of the texture)
	//		- This texture re-creation means that the current texture (which has just been recreated) contains no data
	//			and data needs to be uploaded to it. However, data is only uploaded when a kernel frame render is completed
	//			(and not at every RenderWindow run() loop iteration).
	//		- If our current (asynchronous) kernel frame isn't completed, then we will keep displaying --> we will display
	//			with a texture that didn't get new data uploaded to it --> black viewport
	//		- This is why we need to queue the change so that the texture change is only made when a kernel frame is completed.
	DisplayViewType m_queued_display_view_change = DisplayViewType::UNDEFINED;

	// Whether or not the DisplayView used is going to be displaying at low resolution or not
	bool m_displaying_low_resolution = false;

	// Display textures & their display type
	// 
	// The display type is the format of the texel of the texture used by the display program.
	// This is useful because we have several types of programs using several
	// types of textures. For example, displaying normals on the screen requires float3 textures
	// whereas displaying a heatmap requires only a texture whose texels are scalar (floats or ints).
	// This means that, depending on the display view selected, we're going to have to use the proper
	// OpenGL texture format type and that's what the DisplayTextureType is for.
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
	DisplaySettings m_display_settings;
	std::shared_ptr<GPURenderer> m_renderer = nullptr;
	RenderWindow* m_render_window = nullptr;
};

template<typename T>
void DisplayViewSystem::internal_upload_buffer_to_texture(std::shared_ptr<OpenGLInteropBuffer<T>> buffer, const std::pair<GLuint, DisplayTextureType>& display_texture, int texture_unit)
{
	if (buffer == nullptr)
		return;

	buffer->unmap();
	buffer->unpack_to_GL_texture(display_texture.first, GL_TEXTURE0 + texture_unit, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, display_texture.second);
}

template<typename T>
void DisplayViewSystem::internal_upload_buffer_to_texture(std::shared_ptr<OrochiBuffer<T>> buffer, const std::pair<GLuint, DisplayTextureType>& display_texture, int texture_unit)
{
	if (buffer == nullptr)
		return;

	buffer->unpack_to_GL_texture(display_texture.first, GL_TEXTURE0 + texture_unit, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, display_texture.second);
}

#endif
