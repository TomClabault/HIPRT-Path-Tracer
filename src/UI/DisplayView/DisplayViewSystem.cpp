/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/ApplicationSettings.h"
#include "UI/DisplayView/DisplayViewSystem.h"
#include "UI/ImGui/ImGuiLogWindow.h"
#include "UI/RenderWindow.h"
#include "Utils/Utils.h"

extern ImGuiLogger g_imgui_logger;

DisplayViewSystem::DisplayViewSystem(std::shared_ptr<GPURenderer> renderer, RenderWindow* render_window)
{
	m_renderer		= renderer;
	m_render_window = render_window;

	// Creating the texture that will contain the final device-side display data
	// to be displayed by the shader.
	glGenTextures(1, &m_display_texture_1.first);

	// This empty VAO is necessary on NVIDIA drivers even though
	// we're hardcoding our full screen quad in the vertex shader
	glCreateVertexArrays(1, &m_vao);

	OpenGLShader fullscreen_quad_vertex_shader	 = OpenGLShader(GLSL_SHADERS_DIRECTORY "/fullscreen_quad.vert", OpenGLShader::VERTEX_SHADER);
	OpenGLShader default_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/default_display.frag", OpenGLShader::FRAGMENT_SHADER);

	// Making shared_ptr<OpenGLProgram>s here because multiple display views may share the same OpenGLProgram
	std::shared_ptr<OpenGLProgram> default_display_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, default_display_fragment_shader);
	// The device display post-process pass owns the transforms for these views; they only need the trivial copy program here.
	std::shared_ptr<OpenGLProgram> gmon_blend_display_program	   = default_display_program;
	std::shared_ptr<OpenGLProgram> normal_display_program		   = default_display_program;
	std::shared_ptr<OpenGLProgram> albedo_display_program		   = default_display_program;
	std::shared_ptr<OpenGLProgram> white_furnace_threshold_program = default_display_program;
	std::shared_ptr<OpenGLProgram> denoised_blend_display_program  = default_display_program;

	// Creating all the display views
	DisplayView default_display_view		 = DisplayView(DisplayViewType::DEFAULT, default_display_program);
	DisplayView gmon_blend_display_view		 = DisplayView(DisplayViewType::GMON_BLEND, gmon_blend_display_program);
	DisplayView denoise_blend_display_view	 = DisplayView(DisplayViewType::DENOISED_BLEND, denoised_blend_display_program);
	DisplayView normals_display_view		 = DisplayView(DisplayViewType::DISPLAY_DENOISER_NORMALS, normal_display_program);
	DisplayView albedo_display_view			 = DisplayView(DisplayViewType::DISPLAY_DENOISER_ALBEDO, albedo_display_program);
	DisplayView white_furnace_threshold_view = DisplayView(DisplayViewType::WHITE_FURNACE_THRESHOLD, white_furnace_threshold_program);

	// Adding the display views to the map
	m_display_views[DisplayViewType::DEFAULT]				   = default_display_view;
	m_display_views[DisplayViewType::GMON_BLEND]			   = gmon_blend_display_view;
	m_display_views[DisplayViewType::DENOISED_BLEND]		   = denoise_blend_display_view;
	m_display_views[DisplayViewType::DISPLAY_DENOISER_NORMALS] = normals_display_view;
	m_display_views[DisplayViewType::DISPLAY_DENOISER_ALBEDO]  = albedo_display_view;
	m_display_views[DisplayViewType::WHITE_FURNACE_THRESHOLD]  = white_furnace_threshold_view;

	// Denoiser blend by default if denoising enabled. Default view otherwise
	DisplayViewType default_display_view_type = DisplayViewType::DEFAULT;
	if (m_render_window->get_application_settings()->enable_denoising)
		default_display_view_type = DisplayViewType::DENOISED_BLEND;
	else if (m_renderer->gmon_used())
		default_display_view_type = DisplayViewType::GMON_BLEND;
	else
		default_display_view_type = DisplayViewType::DEFAULT;

	queue_display_view_change(default_display_view_type);
	configure_framebuffer();
}

DisplayViewSystem::~DisplayViewSystem()
{
	glDeleteTextures(1, &m_display_texture_1.first);
	glDeleteVertexArrays(1, &m_vao);
}

void DisplayViewSystem::configure_framebuffer()
{
	glCreateFramebuffers(1, &m_framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);

	// Creating the texture for drawing to the FBO
	glGenTextures(1, &m_fbo_texture);
	glBindTexture(GL_TEXTURE_2D, m_fbo_texture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_render_window->get_width(), m_render_window->get_height(), 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

	// GL_NEAREST because we don't want to linearly interpolate between those beautiful pixels, THAT'S DISGUSTING!
	// We want maximum monte carlo noise crispiness!
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fbo_texture, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
	{
		// Procedes with a victory dance: Dance dance dance dance
		return;
	}
	else
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Incomplete framebuffer in DisplayViewSystem!");

		Debug::debugbreak();
		std::exit(1);
	}
}

void DisplayViewSystem::resize_framebuffer()
{
	glDeleteTextures(1, &m_fbo_texture);
	glGenTextures(1, &m_fbo_texture);
	glBindTexture(GL_TEXTURE_2D, m_fbo_texture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_render_window->get_width(), m_render_window->get_height(), 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

	// GL_NEAREST because we don't want to linearly interpolate between those beautiful pixels, THAT'S DISGUSTING!
	// We want maximum monte carlo noise crispiness!
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fbo_texture, 0);
}

bool DisplayViewSystem::update_selected_display_view()
{
	if (m_queued_display_view_change != DisplayViewType::UNDEFINED)
	{
		// Adjusting the denoiser setting according to the selected view
		// so if the user just selected the denoiser blend display view,
		// enabling the denoising
		//
		// If the user changed the view and this is not the denoiser blend view,
		// this disables denoising
		// m_render_window->get_application_settings()->enable_denoising = m_queued_display_view_change == DisplayViewType::DENOISED_BLEND;

		m_current_display_view = &m_display_views[m_queued_display_view_change];

		internal_recreate_display_textures_from_display_view(m_queued_display_view_change);

		m_queued_display_view_change = DisplayViewType::UNDEFINED;

		return true;
	}

	handle_automatic_display_view_changes();

	return false;
}

void DisplayViewSystem::handle_automatic_display_view_changes()
{
	DisplayViewType current_type = m_current_display_view->get_display_view_type();
	bool gmon_available			 = m_renderer->gmon_used();

	if (current_type == DisplayViewType::GMON_BLEND && !gmon_available)
	{
		m_saved_display_view = DisplayViewType::GMON_BLEND;
		queue_display_view_change(DisplayViewType::DEFAULT);
		update_selected_display_view();
	}
	else if (current_type == DisplayViewType::DEFAULT && m_saved_display_view == DisplayViewType::GMON_BLEND && gmon_available)
	{
		m_saved_display_view = DisplayViewType::UNDEFINED;
		queue_display_view_change(DisplayViewType::GMON_BLEND);
		update_selected_display_view();
	}
	else if (current_type != DisplayViewType::DEFAULT && current_type != DisplayViewType::GMON_BLEND)
		m_saved_display_view = DisplayViewType::UNDEFINED;
}

void DisplayViewSystem::display()
{
	glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);

	// Binding an empty VAO here (empty because we're hardcoding our full-screen quad vertices
	// in our vertex shader) because this is required on NVIDIA drivers
	glBindVertexArray(m_vao);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

DisplayViewType DisplayViewSystem::get_current_display_view_type()
{
	if (m_current_display_view == nullptr)
		return DisplayViewType::UNDEFINED;

	return m_current_display_view->get_display_view_type();
}

const DisplayView* DisplayViewSystem::get_current_display_view() const
{
	return m_current_display_view;
}

std::shared_ptr<OpenGLProgram> DisplayViewSystem::get_active_display_program()
{
	return m_current_display_view->get_display_program();
}

DisplaySettings& DisplayViewSystem::get_display_settings()
{
	return m_display_settings;
}

void DisplayViewSystem::update_display_program_uniforms(const DisplayViewSystem*,
														std::shared_ptr<OpenGLProgram> program,
														std::shared_ptr<GPURenderer>,
														std::shared_ptr<ApplicationSettings>)
{
	program->use();
	// Averaging, debug-view transforms, and tone mapping are performed by the final device pass.
	program->set_uniform("u_texture", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
}

void DisplayViewSystem::update_current_display_program_uniforms()
{
	DisplayViewSystem::update_display_program_uniforms(this, get_active_display_program(), m_renderer, m_render_window->get_application_settings());
}

void DisplayViewSystem::upload_relevant_buffers_to_texture()
{
	internal_upload_buffer_to_texture(m_renderer->get_display_post_process_interop_framebuffer(), m_display_texture_1,
									  DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
}

void DisplayViewSystem::queue_display_view_change(DisplayViewType display_view)
{
	m_queued_display_view_change = display_view;
}

void DisplayViewSystem::set_render_low_resolution(bool low_resolution_or_not)
{
	m_displaying_low_resolution = low_resolution_or_not;
}

bool DisplayViewSystem::get_render_low_resolution() const
{
	return m_displaying_low_resolution;
}

void DisplayViewSystem::resize(int new_render_width, int new_render_height)
{
	resize_framebuffer();
	internal_recreate_display_texture(m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1, m_display_texture_1.second, new_render_width,
									  new_render_height);
}

void DisplayViewSystem::internal_recreate_display_textures_from_display_view(DisplayViewType display_view)
{
	DisplayTextureType texture_1_type_needed = DisplayTextureType::UNINITIALIZED;

	switch (display_view)
	{
	case DisplayViewType::DEFAULT:
	case DisplayViewType::GMON_BLEND:
	case DisplayViewType::DISPLAY_DENOISER_NORMALS:
	case DisplayViewType::DISPLAY_DENOISER_ALBEDO:
	case DisplayViewType::WHITE_FURNACE_THRESHOLD:
	case DisplayViewType::DENOISED_BLEND:
		texture_1_type_needed = DisplayTextureType::FLOAT3;
		break;

	default:
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR,
								"Unhandled display texture type in 'internal_recreate_display_textures_from_display_view'");

		Debug::debugbreak();

		break;
	}

	if (m_display_texture_1.second != texture_1_type_needed)
		internal_recreate_display_texture(m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1, texture_1_type_needed,
										  m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y);
}

void DisplayViewSystem::internal_recreate_display_texture(
	std::pair<GLuint, DisplayTextureType>& display_texture, GLenum display_texture_unit, DisplayTextureType new_texture_type, int width, int height)
{
	bool freeing = false;
	if (new_texture_type == DisplayTextureType::UNINITIALIZED)
	{
		if (display_texture.second != DisplayTextureType::UNINITIALIZED)
		{
			// If the texture was valid before and we've given UNINITIALIZED as the new type, this means
			// that we're not using the texture anymore. We're going to queue_resize the texture to 1x1,
			// essentially freeing it but without really destroying the OpenGL object
			width = height = 1;

			// Not changing the texture type, just resizing
			new_texture_type = display_texture.second;

			freeing = true;
		}
		else
			// Else, the texture is already UNINITIALIZED
			return;
	}

	GLint internal_format = new_texture_type.get_gl_internal_format();
	GLenum format		  = new_texture_type.get_gl_format();
	GLenum type			  = new_texture_type.get_gl_type();

	// Making sure the buffer isn't bound
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glActiveTexture(GL_TEXTURE0 + display_texture_unit);
	glBindTexture(GL_TEXTURE_2D, display_texture.first);
	glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	if (freeing)
		// If we just freed the texture, setting it as UNINITIALIZED so that it is basically invalidated
		// and will be recreated correctly next time
		display_texture.second = DisplayTextureType::UNINITIALIZED;
	else
		display_texture.second = new_texture_type;
}
