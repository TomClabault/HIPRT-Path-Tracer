/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
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
	m_renderer = renderer;
	m_render_window = render_window;

	// Creating the texture that will contain the path traced data to be displayed
	// by the shader.
	glGenTextures(1, &m_display_texture_1.first);
	glGenTextures(1, &m_display_texture_2.first);

	// This empty VAO is necessary on NVIDIA drivers even though
	// we're hardcoding our full screen quad in the vertex shader
	glCreateVertexArrays(1, &m_vao);

	OpenGLShader fullscreen_quad_vertex_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/fullscreen_quad.vert", OpenGLShader::VERTEX_SHADER);
	OpenGLShader default_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/default_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader blend_2_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/blend_2_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader normal_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/normal_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader albedo_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/albedo_display.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader adaptive_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/heatmap_int.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader pixel_converged_display_fragment_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/boolmap_int.frag", OpenGLShader::FRAGMENT_SHADER);
	OpenGLShader white_furnace_threshold_shader = OpenGLShader(GLSL_SHADERS_DIRECTORY "/white_furnace_threshold.frag", OpenGLShader::FRAGMENT_SHADER);

	// Making shared_ptr<OpenGLProgram>s here because multiple display views may share the same OpenGLProgram
	std::shared_ptr<OpenGLProgram> default_display_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, default_display_fragment_shader);
	std::shared_ptr<OpenGLProgram> blend_2_display_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, blend_2_display_fragment_shader);
	std::shared_ptr<OpenGLProgram> normal_display_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, normal_display_fragment_shader);
	std::shared_ptr<OpenGLProgram> albedo_display_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, albedo_display_fragment_shader);
	std::shared_ptr<OpenGLProgram> pixel_convergence_heatmap_display_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, adaptive_display_fragment_shader);
	std::shared_ptr<OpenGLProgram> pixel_converged_display_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, pixel_converged_display_fragment_shader);
	std::shared_ptr<OpenGLProgram> white_furnace_threshold_program = std::make_shared<OpenGLProgram>(fullscreen_quad_vertex_shader, white_furnace_threshold_shader);

	// Creating all the display views
	DisplayView default_display_view = DisplayView(DisplayViewType::DEFAULT, default_display_program);
	DisplayView gmon_blend_display_view = DisplayView(DisplayViewType::GMON_BLEND, blend_2_display_program);
	DisplayView denoise_blend_display_view = DisplayView(DisplayViewType::DENOISED_BLEND, blend_2_display_program);
	DisplayView normals_display_view = DisplayView(DisplayViewType::DISPLAY_DENOISER_NORMALS, normal_display_program);
	DisplayView albedo_display_view = DisplayView(DisplayViewType::DISPLAY_DENOISER_ALBEDO, albedo_display_program);
	DisplayView pixel_convergence_heatmap_display_view = DisplayView(DisplayViewType::PIXEL_CONVERGENCE_HEATMAP, pixel_convergence_heatmap_display_program);
	DisplayView pixel_converged_display_view = DisplayView(DisplayViewType::PIXEL_CONVERGED_MAP, pixel_converged_display_program);
	DisplayView white_furnace_threshold_view = DisplayView(DisplayViewType::WHITE_FURNACE_THRESHOLD, white_furnace_threshold_program);

	// Adding the display views to the map
	m_display_views[DisplayViewType::DEFAULT] = default_display_view;
	m_display_views[DisplayViewType::GMON_BLEND] = gmon_blend_display_view;
	m_display_views[DisplayViewType::DENOISED_BLEND] = denoise_blend_display_view;
	m_display_views[DisplayViewType::DISPLAY_DENOISER_NORMALS] = normals_display_view;
	m_display_views[DisplayViewType::DISPLAY_DENOISER_ALBEDO] = albedo_display_view;
	m_display_views[DisplayViewType::PIXEL_CONVERGENCE_HEATMAP] = pixel_convergence_heatmap_display_view;
	m_display_views[DisplayViewType::PIXEL_CONVERGED_MAP] = pixel_converged_display_view;
	m_display_views[DisplayViewType::WHITE_FURNACE_THRESHOLD] = white_furnace_threshold_view;

	// Denoiser blend by default if denoising enabled. Default view otherwise
	DisplayViewType default_display_view_type = DisplayViewType::DEFAULT;
	if (m_render_window->get_application_settings()->enable_denoising)
		default_display_view_type = DisplayViewType::DENOISED_BLEND;
	else if (m_renderer->is_using_gmon())
		default_display_view_type = DisplayViewType::GMON_BLEND;
	else 
		default_display_view_type = DisplayViewType::DEFAULT;

	queue_display_view_change(default_display_view_type);
	configure_framebuffer();
}

DisplayViewSystem::~DisplayViewSystem()
{
	glDeleteTextures(1, &m_display_texture_1.first);
	glDeleteTextures(1, &m_display_texture_2.first);
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

		Utils::debugbreak();
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
	if (current_display_view_needs_adaptive_sampling_buffers()
	&& !m_render_window->get_renderer()->get_render_settings().has_access_to_adaptive_sampling_buffers())
		// If the adaptive sampling heatmap is selected as the current view but
		// the adaptive sampling buffers are no longer available (after a change
		// to ImGui for example), we need to switch out of the adaptive sampling
		// view because we don't have the buffers to display it anymore
		m_queued_display_view_change = DisplayViewType::DEFAULT;

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
	
	return false;
}

bool DisplayViewSystem::current_display_view_needs_adaptive_sampling_buffers()
{
	return get_current_display_view_type() == DisplayViewType::PIXEL_CONVERGENCE_HEATMAP
		|| get_current_display_view_type() == DisplayViewType::PIXEL_CONVERGED_MAP;
}

void DisplayViewSystem::display()
{
	TracyGpuZone("Display");

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

void DisplayViewSystem::update_display_program_uniforms(const DisplayViewSystem* display_view_system, std::shared_ptr<OpenGLProgram> program, std::shared_ptr<GPURenderer> renderer, std::shared_ptr<ApplicationSettings> application_settings)
{
	const DisplayView* display_view = display_view_system->get_current_display_view();
	const DisplaySettings& display_settings = display_view_system->m_display_settings;
	
	HIPRTRenderSettings render_settings = renderer->get_render_settings();
	render_settings.sample_number = std::max(1u, render_settings.sample_number); 

	bool display_low_resolution = display_view_system->get_render_low_resolution();
	int render_low_resolution_scaling = display_low_resolution ? render_settings.render_low_resolution_scaling : 1;

	program->use();

	switch (display_view->get_display_view_type())
	{
	case DisplayViewType::DEFAULT:
	{
		int sample_number;
		if (application_settings->enable_denoising && application_settings->last_denoised_sample_count != -1)
			// If we have denoising enabled, the viewport may not be updated at each frame.
			// This means that we may be displaying the same denoised buffer for multiple frame
			// and that same denoised buffer is only going to have a given amount of samples accumulated
			// in it so we must you that number of samples for displaying otherwise things are going
			// to be too dark because we're going to be dividing the data of the denoised buffer by a
			// sample count that doesn't match
			sample_number = application_settings->last_denoised_sample_count;
		else if (renderer->is_using_gmon())
			sample_number = renderer->get_gmon_render_pass().get_last_recomputed_sample_count();
		else
			sample_number = render_settings.sample_number;

		program->set_uniform("u_texture", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		program->set_uniform("u_sample_number", sample_number);
		program->set_uniform("u_do_tonemapping", display_settings.do_tonemapping);
		program->set_uniform("u_resolution_scaling", render_low_resolution_scaling);
		program->set_uniform("u_gamma", display_settings.tone_mapping_gamma);
		program->set_uniform("u_exposure", display_settings.tone_mapping_exposure);

		break;
	}

	case DisplayViewType::WHITE_FURNACE_THRESHOLD:
	{
		int sample_number;
		if (application_settings->enable_denoising && application_settings->last_denoised_sample_count != -1)
			sample_number = application_settings->last_denoised_sample_count;
		else
			sample_number = render_settings.sample_number;

		program->set_uniform("u_texture", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		program->set_uniform("u_sample_number", sample_number);
		program->set_uniform("u_do_tonemapping", display_settings.do_tonemapping);
		program->set_uniform("u_resolution_scaling", render_low_resolution_scaling);
		program->set_uniform("u_gamma", display_settings.tone_mapping_gamma);
		program->set_uniform("u_exposure", display_settings.tone_mapping_exposure);
		program->set_uniform("u_use_low_threshold", display_settings.white_furnace_display_use_low_threshold);
		program->set_uniform("u_use_high_threshold", display_settings.white_furnace_display_use_high_threshold);

		break;
	}

	case DisplayViewType::GMON_BLEND:
	{
		int gmon_sample_number = renderer->get_gmon_render_pass().get_last_recomputed_sample_count();
		int default_sample_number = render_settings.sample_number;

		program->set_uniform("u_blend_factor", renderer->get_gmon_render_pass().get_gmon_data().gmon_blend_factor);
		program->set_uniform("u_texture_1", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		program->set_uniform("u_texture_2", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_2);
		program->set_uniform("u_sample_number_1", default_sample_number);
		program->set_uniform("u_sample_number_2", gmon_sample_number);
		program->set_uniform("u_do_tonemapping", display_settings.do_tonemapping);
		program->set_uniform("u_resolution_scaling", render_low_resolution_scaling);
		program->set_uniform("u_gamma", display_settings.tone_mapping_gamma);
		program->set_uniform("u_exposure", display_settings.tone_mapping_exposure);

		break;
	}

	case DisplayViewType::DENOISED_BLEND:
	{
		int noisy_sample_number;
		int denoised_sample_number;

		noisy_sample_number = render_settings.sample_number;
		denoised_sample_number = application_settings->last_denoised_sample_count;

		if (display_settings.blend_override != -1.0f)
			program->set_uniform("u_blend_factor", display_settings.blend_override);
		else
			program->set_uniform("u_blend_factor", display_settings.denoiser_blend);
		program->set_uniform("u_texture_1", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		program->set_uniform("u_texture_2", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_2);
		program->set_uniform("u_sample_number_1", noisy_sample_number);
		program->set_uniform("u_sample_number_2", denoised_sample_number);
		program->set_uniform("u_do_tonemapping", display_settings.do_tonemapping);
		program->set_uniform("u_resolution_scaling", render_low_resolution_scaling);
		program->set_uniform("u_gamma", display_settings.tone_mapping_gamma);
		program->set_uniform("u_exposure", display_settings.tone_mapping_exposure);

		break;
	}

	case DisplayViewType::DISPLAY_DENOISER_ALBEDO:
		program->set_uniform("u_texture", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		program->set_uniform("u_resolution_scaling", render_low_resolution_scaling);

		break;

	case DisplayViewType::DISPLAY_DENOISER_NORMALS:
		program->set_uniform("u_texture", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		program->set_uniform("u_resolution_scaling", render_low_resolution_scaling);
		program->set_uniform("u_do_tonemapping", display_settings.do_tonemapping);
		program->set_uniform("u_gamma", display_settings.tone_mapping_gamma);
		program->set_uniform("u_exposure", display_settings.tone_mapping_exposure);

		break;

	case DisplayViewType::PIXEL_CONVERGENCE_HEATMAP:
	{
		std::vector<ColorRGB32F> color_stops = { ColorRGB32F(0.0f, 0.0f, 1.0f), ColorRGB32F(0.0f, 1.0f, 0.0f), ColorRGB32F(1.0f, 0.0f, 0.0f) };

		// If we don't have adaptive sampling enabled, we want to display the convergence
		// of pixels as soon as possible so we set the min_val to 1. Otherwise, if we're using
		// adaptive sampling, we only have the convergence information after the minimum
		// adaptive sampling samples have been reached so we set that as the min_val
		float min_val = render_settings.enable_adaptive_sampling ? (float)render_settings.adaptive_sampling_min_samples : 1;
		float max_val = std::max((float)render_settings.sample_number, min_val);

		program->set_uniform("u_texture", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		program->set_uniform("u_resolution_scaling", render_low_resolution_scaling);
		program->set_uniform("u_color_stops", 3, (float*)color_stops.data());
		program->set_uniform("u_nb_stops", 3);
		program->set_uniform("u_min_val", min_val);
		program->set_uniform("u_max_val", max_val);

		break;
	}

	case DisplayViewType::PIXEL_CONVERGED_MAP:
	{
		float min_val = render_settings.enable_adaptive_sampling ? (float)render_settings.adaptive_sampling_min_samples : 1;

		// If a pixel has a lower sample count than the threshold val, then it has converged
		float threshold_val = std::max((float)render_settings.sample_number, min_val);

		program->set_uniform("u_texture", DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		program->set_uniform("u_resolution_scaling", render_low_resolution_scaling);
		program->set_uniform("u_threshold_val", threshold_val);
		break;
	}

	case DisplayViewType::UNDEFINED:
		break;
	}
}

void DisplayViewSystem::update_current_display_program_uniforms()
{
	DisplayViewSystem::update_display_program_uniforms(this, get_active_display_program(), m_renderer, m_render_window->get_application_settings());
}

void DisplayViewSystem::upload_relevant_buffers_to_texture()
{
	DisplayViewType current_display_view_type = get_current_display_view_type();

	switch (current_display_view_type)
	{
	case DisplayViewType::GMON_BLEND:
		internal_upload_buffer_to_texture(m_renderer->get_default_interop_framebuffer(), m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		internal_upload_buffer_to_texture(m_renderer->get_color_interop_framebuffer(), m_display_texture_2, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_2);
		break;

	case DisplayViewType::DENOISED_BLEND:
		internal_upload_buffer_to_texture(m_renderer->get_color_interop_framebuffer(), m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		internal_upload_buffer_to_texture(m_renderer->get_denoised_interop_framebuffer(), m_display_texture_2, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_2);
		break;

	case DisplayViewType::DISPLAY_DENOISER_ALBEDO:
		if (m_render_window->get_application_settings()->denoiser_use_interop_buffers)
			internal_upload_buffer_to_texture(m_renderer->get_denoiser_albedo_AOV_interop_buffer(), m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		else
			internal_upload_buffer_to_texture(m_renderer->get_denoiser_albedo_AOV_no_interop_buffer(), m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);

		break;

	case DisplayViewType::DISPLAY_DENOISER_NORMALS:
		if (m_render_window->get_application_settings()->denoiser_use_interop_buffers)
			internal_upload_buffer_to_texture(m_renderer->get_denoiser_normals_AOV_interop_buffer(), m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		else
			internal_upload_buffer_to_texture(m_renderer->get_denoiser_normals_AOV_no_interop_buffer(), m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);

		break;

	case DisplayViewType::PIXEL_CONVERGED_MAP:
	case DisplayViewType::PIXEL_CONVERGENCE_HEATMAP:
		internal_upload_buffer_to_texture(m_renderer->get_pixels_converged_sample_count_buffer(), m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		break;

	case DisplayViewType::DEFAULT:
	case DisplayViewType::WHITE_FURNACE_THRESHOLD:
	default:
		internal_upload_buffer_to_texture(m_renderer->get_default_interop_framebuffer(), m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1);
		break;
	}
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
	internal_recreate_display_texture(m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1, m_display_texture_1.second, new_render_width, new_render_height);
	internal_recreate_display_texture(m_display_texture_2, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_2, m_display_texture_2.second, new_render_width, new_render_height);
}

void DisplayViewSystem::internal_recreate_display_textures_from_display_view(DisplayViewType display_view)
{
	DisplayTextureType texture_1_type_needed = DisplayTextureType::UNINITIALIZED;
	DisplayTextureType texture_2_type_needed = DisplayTextureType::UNINITIALIZED;

	switch (display_view)
	{
	case DisplayViewType::DEFAULT:
	case DisplayViewType::DISPLAY_DENOISER_NORMALS:
	case DisplayViewType::DISPLAY_DENOISER_ALBEDO:
	case DisplayViewType::WHITE_FURNACE_THRESHOLD:
		texture_1_type_needed = DisplayTextureType::FLOAT3;
		break;

	case DisplayViewType::PIXEL_CONVERGENCE_HEATMAP:
	case DisplayViewType::PIXEL_CONVERGED_MAP:
		texture_1_type_needed = DisplayTextureType::INT;
		break;

	case DisplayViewType::GMON_BLEND:
	case DisplayViewType::DENOISED_BLEND:
		texture_1_type_needed = DisplayTextureType::FLOAT3;
		texture_2_type_needed = DisplayTextureType::FLOAT3;
		break;

	default:
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Unhandled display texture type in 'internal_recreate_display_textures_from_display_view'");

		Utils::debugbreak();

		break;
	}

	if (m_display_texture_1.second != texture_1_type_needed)
		internal_recreate_display_texture(m_display_texture_1, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_1, texture_1_type_needed, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y);

	if (m_display_texture_2.second != texture_2_type_needed)
		internal_recreate_display_texture(m_display_texture_2, DisplayViewSystem::DISPLAY_TEXTURE_UNIT_2, texture_2_type_needed, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y);
}

void DisplayViewSystem::internal_recreate_display_texture(std::pair<GLuint, DisplayTextureType>& display_texture, GLenum display_texture_unit, DisplayTextureType new_texture_type, int width, int height)
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
	GLenum format = new_texture_type.get_gl_format();
	GLenum type = new_texture_type.get_gl_type();

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
