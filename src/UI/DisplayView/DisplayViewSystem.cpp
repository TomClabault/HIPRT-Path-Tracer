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
	glGenTextures(1, &m_display_texture);
	internal_resize_display_texture(m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y);

	// This empty VAO is necessary on NVIDIA drivers even though
	// we're hardcoding our full screen quad in the vertex shader
	glCreateVertexArrays(1, &m_vao);

	OpenGLShader fullscreen_quad_vertex_shader	 = OpenGLShader(GLSL_SHADERS_DIRECTORY "/fullscreen_quad.vert", OpenGLShader::VERTEX_SHADER);
	OpenGLShader default_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/default_display.frag", OpenGLShader::FRAGMENT_SHADER);

	// All display views use this trivial copy program. The device display post-process pass owns the view transforms.
	m_display_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, default_display_fragment_shader);
	m_display_program->use();
	m_display_program->set_uniform("u_texture", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);

	// Denoiser blend by default if denoising is enabled, GMoN blend when available, and the default view otherwise.
	DisplayViewType default_display_view_type = DisplayViewType::DEFAULT;
	if (m_render_window->get_application_settings()->enable_denoising)
		default_display_view_type = DisplayViewType::DENOISED_BLEND;
	else if (m_renderer->gmon_used())
		default_display_view_type = DisplayViewType::GMON_BLEND;

	queue_display_view_change(default_display_view_type);
	configure_framebuffer();
}

DisplayViewSystem::~DisplayViewSystem()
{
	glDeleteTextures(1, &m_display_texture);
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

	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Incomplete framebuffer in DisplayViewSystem!");

	Debug::debugbreak();
	std::exit(1);
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

		m_current_display_view_type	 = m_queued_display_view_change;
		m_queued_display_view_change = DisplayViewType::UNDEFINED;

		return true;
	}

	handle_automatic_display_view_changes();

	return false;
}

void DisplayViewSystem::handle_automatic_display_view_changes()
{
	if (m_current_display_view_type == DisplayViewType::UNDEFINED)
		return;

	bool gmon_available = m_renderer->gmon_used();

	if (m_current_display_view_type == DisplayViewType::GMON_BLEND && !gmon_available)
	{
		m_saved_display_view = DisplayViewType::GMON_BLEND;
		queue_display_view_change(DisplayViewType::DEFAULT);
		update_selected_display_view();
	}
	else if (m_current_display_view_type == DisplayViewType::DEFAULT && m_saved_display_view == DisplayViewType::GMON_BLEND && gmon_available)
	{
		m_saved_display_view = DisplayViewType::UNDEFINED;
		queue_display_view_change(DisplayViewType::GMON_BLEND);
		update_selected_display_view();
	}
	else if (m_current_display_view_type != DisplayViewType::DEFAULT && m_current_display_view_type != DisplayViewType::GMON_BLEND)
		m_saved_display_view = DisplayViewType::UNDEFINED;
}

void DisplayViewSystem::display()
{
	m_display_program->use();
	glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);

	// Binding an empty VAO here (empty because we're hardcoding our full-screen quad vertices
	// in our vertex shader) because this is required on NVIDIA drivers
	glBindVertexArray(m_vao);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

DisplayViewType DisplayViewSystem::get_current_display_view_type()
{
	return m_current_display_view_type;
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

void DisplayViewSystem::upload_final_display_buffer()
{
	internal_upload_buffer_to_texture(m_renderer->get_display_post_process_interop_framebuffer(), m_display_texture, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
}

void DisplayViewSystem::upload_relevant_buffers_to_texture()
{
	// The legacy compute screenshot path still calls this name; it now uploads the same final buffer.
	upload_final_display_buffer();
}

void DisplayViewSystem::queue_display_view_change(DisplayViewType display_view)
{
	m_queued_display_view_change = display_view;
}

void DisplayViewSystem::resize(int new_render_width, int new_render_height)
{
	resize_framebuffer();
	internal_resize_display_texture(new_render_width, new_render_height);
}

void DisplayViewSystem::internal_resize_display_texture(int width, int height)
{
	DisplayTextureType texture_type = DisplayTextureType::FLOAT3;
	GLint internal_format			= texture_type.get_gl_internal_format();
	GLenum format					= texture_type.get_gl_format();
	GLenum type						= texture_type.get_gl_type();

	// Making sure the buffer isn't bound
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glActiveTexture(GL_TEXTURE0 + DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
	glBindTexture(GL_TEXTURE_2D, m_display_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}
